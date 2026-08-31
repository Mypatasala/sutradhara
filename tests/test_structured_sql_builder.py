"""
Category A -- deterministic builder correctness. No LLM involved anywhere in
this file: every plan is hand-constructed. These tests are the actual
"determinism" proof (canonical plan -> byte-identical SQL) and must always
pass in CI, unlike the live-model reliability tests in
test_live_structured_reliability.py.
"""

import pytest

from src.agents.query_plan import (
    ComparisonFilter,
    DisplayField,
    Entity,
    ExtremeSelector,
    FilterField,
    GroupingDimension,
    Operation,
    PercentageSpec,
    QueryPlan,
    RelativeDate,
    SortField,
    SortSpec,
)
from src.agents.query_normalizer import normalize
from src.retrieval.structured_sql_builder import StructuredSQLBuilder


def test_count_students_by_class_matches_expected_sql():
    """The exact motivating case: 'how many students are in each class'.
    This SQL must never be ambiguous, must join both class_sections and
    school_classes, and must group by the real per-section id."""
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_CLASS)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT CONCAT_WS(' - ', school_classes.name, class_sections.name) AS class_name, COUNT(*) AS count "
        "FROM students JOIN class_sections ON students.section_id = class_sections.id "
        "JOIN school_classes ON class_sections.school_class_id = school_classes.id "
        "GROUP BY class_sections.id, school_classes.name, class_sections.name "
        "ORDER BY school_classes.level, class_sections.name"
    )


def test_plain_count_students():
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM students"


def test_attendance_percentage():
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT (COUNT(CASE WHEN attendance.status = 'present' THEN 1 END) * 100.0 / COUNT(*)) AS percentage FROM attendance"
    )


def test_homework_pending_count():
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM homework WHERE homework.status = 'pending'"


def test_report_cards_latest_list_with_sort_and_limit():
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        sort=SortSpec(field=SortField.ISSUE_DATE, direction="desc"), limit=1,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT report_cards.term, report_cards.overall_grade, report_cards.overall_percentage FROM report_cards "
        "ORDER BY report_cards.issue_date DESC LIMIT 1"
    )


def test_course_schedule_subject_lookup_filter():
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT courses.name, course_schedule.start_time, course_schedule.end_time, course_schedule.room "
        "FROM course_schedule JOIN courses ON course_schedule.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )


def test_students_grade_filter_exact_sql():
    """The exact Issue 1 motivating case: 'List all students in Grade 5.'
    No display_fields needed -- STUDENTS' own default_display_fields
    (first_name, last_name) already provide a meaningful, privacy-safe
    identification result with no hallucination required."""
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.GRADE, value="5")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.GRADE: "5"}))
    assert sql == "SELECT students.first_name, students.last_name FROM students WHERE students.grade = '5'"


def test_distinct_flag_applied():
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        display_fields=[DisplayField.SUBJECT_NAME], distinct=True,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql.startswith("SELECT DISTINCT courses.name FROM course_schedule")


def test_day_of_week_filter():
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.DAY_OF_WEEK, value="Friday")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert "WHERE course_schedule.day_of_week = 'Friday'" in sql


def test_users_list_default_display_fields_never_includes_password():
    plan = QueryPlan(entity=Entity.USERS, operation=Operation.LIST)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert "password" not in sql.lower()
    assert sql == "SELECT users.first_name, users.last_name, users.email, users.phone, users.department FROM users"


# ── Determinism proof: equivalent-but-differently-shaped plans converge ────

def test_semantic_equivalence_filter_order_and_casing():
    plan_a = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")],
    )
    plan_b = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="PENDING")],
    )
    sql_a = StructuredSQLBuilder.build(normalize(plan_a, {}))
    sql_b = StructuredSQLBuilder.build(normalize(plan_b, {}))
    assert sql_a == sql_b


def test_semantic_equivalence_display_field_order():
    plan_a = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        display_fields=[DisplayField.OVERALL_GRADE, DisplayField.TERM],
    )
    plan_b = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        display_fields=[DisplayField.TERM, DisplayField.OVERALL_GRADE],
    )
    sql_a = StructuredSQLBuilder.build(normalize(plan_a, {}))
    sql_b = StructuredSQLBuilder.build(normalize(plan_b, {}))
    assert sql_a == sql_b


# ── BY_STUDENT grouping / AGGREGATE_VALUE sort / extreme (no SQL emitted) ──

def test_attendance_percentage_by_student_matches_verified_schema():
    """Uses the real production schema's join (attendance.student_id ->
    students.id directly, students owns first_name/last_name itself -- see
    query_registry.py's ATTENDANCE.supported_groupings comment for how this
    was verified against my_patasala's actual migration DDL, correcting an
    earlier assumption sourced from a stale demo fixture)."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT CONCAT_WS(' ', students.first_name, students.last_name) AS student_name, "
        "school_classes.name AS class_name, class_sections.name AS section_name, "
        "(COUNT(CASE WHEN attendance.status = 'present' THEN 1 END) * 100.0 / COUNT(*)) AS percentage "
        "FROM attendance JOIN students ON attendance.student_id = students.id "
        "LEFT JOIN class_sections ON students.section_id = class_sections.id "
        "LEFT JOIN school_classes ON class_sections.school_class_id = school_classes.id "
        "GROUP BY students.id, students.first_name, students.last_name, "
        "class_sections.id, class_sections.name, school_classes.id, school_classes.name"
    )


def test_aggregate_value_sort_uses_operation_derived_alias_not_registry_column():
    """SortField.AGGREGATE_VALUE is a sentinel, not looked up in
    meta.sort_field_columns -- the builder must sort by its own just-built
    'percentage' alias. This represents "5 students with the lowest
    attendance" -- an EXPLICIT count, never an arbitrary default."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"), limit=5,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql.endswith(
        "GROUP BY students.id, students.first_name, students.last_name, "
        "class_sections.id, class_sections.name, school_classes.id, school_classes.name "
        "ORDER BY percentage ASC LIMIT 5"
    )


@pytest.mark.parametrize(
    "n,direction_word,sql_direction",
    [(5, "asc", "ASC"), (3, "desc", "DESC"), (1, "asc", "ASC"), (17, "desc", "DESC")],
)
def test_explicit_n_ranking_preserves_the_exact_stated_number(n, direction_word, sql_direction):
    """Regression pinning the extreme-vs-explicit-N distinction at the
    builder level, independent of what the model actually produces: an
    explicit-N plan (sort=aggregate_value + limit=N) must never be silently
    converted to extreme -- it must produce LIMIT with that EXACT number,
    for a range of stated Ns, and extreme must never appear in the SQL at
    all (there is no SQL representation of it -- see the module docstring)."""
    from src.agents.query_plan import SortField, SortSpec

    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction=direction_word), limit=n,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql.endswith(f"ORDER BY percentage {sql_direction} LIMIT {n}")
    assert plan.extreme is None


def test_extreme_no_number_never_produces_a_limit_clause():
    """The mirror case: a no-number extreme plan must never contain LIMIT or
    ORDER BY -- proving the two ranking shapes remain structurally distinct
    at the builder level regardless of what the model chooses upstream."""
    from src.agents.query_plan import ExtremeSelector

    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert "LIMIT" not in sql
    assert "ORDER BY" not in sql


def test_extreme_plan_produces_the_same_flat_sql_as_plain_grouped_query():
    """Authorization regression: plan.extreme must add NO SQL of its own --
    same base table, same single JOIN, same GROUP BY, no ORDER BY, no LIMIT,
    no nested subquery -- so the existing, unmodified AliasAwareFilterInjector
    authorizes it exactly as it does any other grouped query (single
    occurrence of the target table in the outer FROM/JOIN scope). See
    structured_sql_builder.py's module docstring for the live-traced reason
    a nested MIN/MAX subquery was rejected."""
    from src.agents.query_plan import ExtremeSelector

    plan_with_extreme = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST,
    )
    plan_without_extreme = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
    )
    sql_with = StructuredSQLBuilder.build(normalize(plan_with_extreme, {}))
    sql_without = StructuredSQLBuilder.build(normalize(plan_without_extreme, {}))

    assert sql_with == sql_without
    assert sql_with.count("SELECT") == 1  # exactly one query, no nested subquery
    assert " ORDER BY " not in sql_with
    assert " LIMIT " not in sql_with
    assert "FROM attendance JOIN students" in sql_with  # target table appears exactly once, in the outer FROM


def test_semantic_equivalence_lookup_filter_casing():
    plan_a = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    plan_b = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="MATHEMATICS")],
    )
    resolved = {FilterField.SUBJECT: "Mathematics"}
    sql_a = StructuredSQLBuilder.build(normalize(plan_a, resolved))
    sql_b = StructuredSQLBuilder.build(normalize(plan_b, resolved))
    assert sql_a == sql_b
