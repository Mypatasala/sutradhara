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
    NumericField,
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


def test_attendance_list_default_display_joins_students_for_the_name():
    """Part B: ATTENDANCE's new LIST capability. Default shape (no
    display_fields specified) must show student name + the record's own
    date/status, requiring a join to students even with group_by=NONE --
    exercises EntityMeta.list_joins, the new mechanism for LIST-only
    required joins (see query_registry.py's EntityMeta.list_joins
    docstring)."""
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.LIST)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT students.first_name, students.last_name, attendance.date, attendance.status "
        "FROM attendance JOIN students ON attendance.student_id = students.id"
    )


def test_attendance_count_with_last_30_days_has_no_status_filter_unless_asked():
    """Regression test for the exact previously-observed defect: a plain
    COUNT of attendance records with no status named must never gain a
    status=present filter -- filters=[] means every status."""
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, date_range=RelativeDate.LAST_30_DAYS)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert "status" not in sql
    assert sql.startswith("SELECT COUNT(*) AS count FROM attendance WHERE attendance.date BETWEEN")


def test_attendance_percentage_with_last_30_days():
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, date_range=RelativeDate.LAST_30_DAYS,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert "attendance.date BETWEEN" in sql
    assert "attendance.status = 'present'" in sql


def test_attendance_list_with_last_30_days_exact_sql():
    from datetime import date, timedelta
    today = date.today()
    start = today - timedelta(days=29)
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.LIST, date_range=RelativeDate.LAST_30_DAYS)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT students.first_name, students.last_name, attendance.date, attendance.status "
        "FROM attendance JOIN students ON attendance.student_id = students.id "
        f"WHERE attendance.date BETWEEN '{start.isoformat()}' AND '{today.isoformat()}'"
    )


def test_attendance_list_with_explicit_status_filter_still_qualified():
    """A status filter IS honored when the question actually names one --
    this proves the fix for the over-invented filter isn't achieved by
    disabling status filtering altogether."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="absent")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT students.first_name, students.last_name, attendance.date, attendance.status "
        "FROM attendance JOIN students ON attendance.student_id = students.id "
        "WHERE attendance.status = 'absent'"
    )


def test_homework_list_default_display_is_title_subject_status_no_join():
    """LIST display-shape fix (2026-09-10): HOMEWORK previously had no
    display_field_columns/default_display_fields at all -- this proves the
    fix produces valid, join-free SQL, replacing the invalid
    "SELECT  FROM homework" the entity would have emitted before this
    registry addition (verified directly against the pre-fix registry
    during investigation)."""
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.LIST)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT homework.title, homework.subject, homework.status FROM homework"
    assert "JOIN" not in sql


def test_homework_pending_count():
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM homework WHERE homework.status = 'pending'"


def test_homework_completed_count_exact_sql():
    """STATUS vocabulary fix (2026-09-11): "completed" is a real
    HomeworkStatus value, previously wrongly rejected by the validator --
    this proves the builder now produces the exact expected SQL for it."""
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="completed")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM homework WHERE homework.status = 'completed'"


def test_homework_subject_lookup_filter_produces_no_join():
    """homework.subject is native to homework's own row (no courses/
    subjects join, unlike COURSE_SCHEDULE.SUBJECT below) -- the generated
    SQL must have zero JOIN clauses."""
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == "SELECT COUNT(*) AS count FROM homework WHERE homework.subject = 'Mathematics'"
    assert "JOIN" not in sql


def test_homework_subject_and_status_combined_still_no_join():
    """The exact motivating case: 'How many math homework assignments are
    pending?' -- STATUS (enum) and SUBJECT (lookup) combined, still zero
    joins and a plain COUNT(*)."""
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[
            ComparisonFilter(field=FilterField.STATUS, value="pending"),
            ComparisonFilter(field=FilterField.SUBJECT, value="mathematics"),
        ],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT COUNT(*) AS count FROM homework "
        "WHERE homework.status = 'pending' AND homework.subject = 'Mathematics'"
    )
    assert "JOIN" not in sql


def test_assignments_plain_count():
    plan = QueryPlan(entity=Entity.ASSIGNMENTS, operation=Operation.COUNT)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM assignments"


def test_assignments_status_filter_count_produces_no_join():
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="overdue")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM assignments WHERE assignments.status = 'overdue'"
    assert "JOIN" not in sql


def test_assignments_list_default_display_is_title_and_status_no_join():
    """ASSIGNMENTS.LIST default shape: title + status, both native columns
    on assignments' own row -- must produce zero joins (unlike ATTENDANCE's
    LIST, which needs a join for the student's name; ASSIGNMENTS
    deliberately exposes no student-identifying display field this phase,
    see query_registry.py's ASSIGNMENTS entry)."""
    plan = QueryPlan(entity=Entity.ASSIGNMENTS, operation=Operation.LIST)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT assignments.title, assignments.status FROM assignments"
    assert "JOIN" not in sql


def test_assignments_count_by_status_group_by():
    plan = QueryPlan(entity=Entity.ASSIGNMENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT assignments.status AS status, COUNT(*) AS count FROM assignments GROUP BY assignments.status"
    )
    assert "JOIN" not in sql
    assert sql.count("SELECT") == 1


def test_examinations_plain_count():
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.COUNT)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM examinations"
    assert "JOIN" not in sql


def test_examinations_list_default_display_is_title_status_no_join():
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.LIST)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT examinations.title, examinations.status FROM examinations"
    assert "JOIN" not in sql


def test_examinations_status_filter_count_exact_sql():
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="completed")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM examinations WHERE examinations.status = 'completed'"
    assert "JOIN" not in sql


def test_examinations_subject_lookup_filter_count_exact_sql():
    """Phase 1 (2026-09-11): examinations.course_id reaches courses.name via
    a real join -- exactly one JOIN to courses, no nested subquery, no
    duplicate SELECT."""
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT COUNT(*) AS count FROM examinations JOIN courses ON examinations.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    assert sql.count("JOIN") == 1
    assert sql.count("SELECT") == 1


def test_examinations_subject_lookup_filter_list_exact_sql():
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT examinations.title, examinations.status FROM examinations "
        "JOIN courses ON examinations.course_id = courses.id WHERE courses.name = 'Mathematics'"
    )
    assert sql.count("JOIN") == 1
    assert sql.count("SELECT") == 1


def test_assignments_subject_lookup_filter_count_exact_sql():
    """Phase 2 (2026-09-10): assignments.course_id reaches courses.name via
    a real join (unlike HOMEWORK.SUBJECT's own native-column, join-free
    filter) -- exactly one JOIN to courses, no nested subquery, no
    duplicate SELECT."""
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT COUNT(*) AS count FROM assignments JOIN courses ON assignments.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    assert sql.count("JOIN") == 1
    assert sql.count("SELECT") == 1


def test_assignments_subject_lookup_filter_list_exact_sql():
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT assignments.title, assignments.status FROM assignments "
        "JOIN courses ON assignments.course_id = courses.id WHERE courses.name = 'Mathematics'"
    )
    assert sql.count("JOIN") == 1
    assert sql.count("SELECT") == 1


def test_assignments_subject_filter_combined_with_by_status_grouping():
    """SUBJECT filter + BY_STATUS grouping: the SUBJECT lookup filter
    (mathematics) narrows the population via the courses JOIN, BY_STATUS
    then breaks that narrowed population down by assignments.status (a
    native, join-free column) -- still exactly one JOIN, no nested
    subquery, no duplicate SELECT."""
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
        group_by=GroupingDimension.BY_STATUS,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT assignments.status AS status, COUNT(*) AS count FROM assignments "
        "JOIN courses ON assignments.course_id = courses.id "
        "WHERE courses.name = 'Mathematics' GROUP BY assignments.status"
    )
    assert sql.count("JOIN") == 1
    assert sql.count("SELECT") == 1


def test_courses_plain_count():
    plan = QueryPlan(entity=Entity.COURSES, operation=Operation.COUNT)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM courses"
    assert "JOIN" not in sql


def test_courses_list_default_display_is_name_code_credits_no_join():
    """COURSES.LIST default shape: name, code, credits, all native columns
    on courses' own row -- must produce zero application-level joins."""
    plan = QueryPlan(entity=Entity.COURSES, operation=Operation.LIST)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT courses.name, courses.code, courses.credits FROM courses"
    assert "JOIN" not in sql


def test_homework_subject_filter_combined_with_by_status_grouping():
    """'How many math homework assignments are pending vs graded?' -- the
    SUBJECT lookup filter (mathematics) narrows the population, BY_STATUS
    then breaks that narrowed population down by homework.status. Both are
    join-free (homework.subject and homework.status are both native columns
    on homework's own row), so the combination stays a single flat query --
    no JOIN, no nested subquery -- exactly as each piece already proved
    independently in test_homework_subject_and_status_combined_still_no_join
    (a STATUS filter) and test_homework_count_by_status_group_by (a BY_STATUS
    grouping with no filter at all)."""
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
        group_by=GroupingDimension.BY_STATUS,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.SUBJECT: "Mathematics"}))
    assert sql == (
        "SELECT homework.status AS status, COUNT(*) AS count FROM homework "
        "WHERE homework.subject = 'Mathematics' GROUP BY homework.status"
    )
    assert "JOIN" not in sql
    assert sql.count("SELECT") == 1


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


def test_course_schedule_sorted_by_start_time():
    """2026-09-10: EntityMeta.sort_field_columns already registered
    SortField.START_TIME for COURSE_SCHEDULE -- this proves the existing
    generic sort-handling code (no builder change) produces the expected
    chronological ORDER BY, exactly the shape 'Show today's schedule in
    order' now maps to in the prompt.

    Note: the default display fields include courses.name (SUBJECT_NAME)
    with no JOIN present -- this is pre-existing COURSE_SCHEDULE LIST
    behavior, unrelated to sorting and unchanged by this task (no prior
    test exercised a plain, filter-less COURSE_SCHEDULE LIST to have
    caught it); asserted here exactly as produced, not modified."""
    plan = QueryPlan(entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST, sort=SortSpec(field=SortField.START_TIME, direction="asc"))
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT courses.name, course_schedule.start_time, course_schedule.end_time, course_schedule.room "
        "FROM course_schedule ORDER BY course_schedule.start_time ASC"
    )


def test_course_schedule_sorted_by_start_time_with_limit():
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        sort=SortSpec(field=SortField.START_TIME, direction="asc"), limit=1,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT courses.name, course_schedule.start_time, course_schedule.end_time, course_schedule.room "
        "FROM course_schedule ORDER BY course_schedule.start_time ASC LIMIT 1"
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


# ── USERS COUNT + ROLE lookup filter (P0-1) ────────────────────────────────

def test_users_count_plain():
    plan = QueryPlan(entity=Entity.USERS, operation=Operation.COUNT)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM users"


def test_users_count_with_role_filter_exact_sql():
    """The exact P0-1 motivating case: 'how many teachers are there' -- must
    join users -> user_roles -> roles and filter on the real, existence-
    checked roles.name value, using the same generic COUNT(*) path every
    other entity's COUNT already goes through."""
    plan = QueryPlan(
        entity=Entity.USERS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ROLE, value="teacher")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.ROLE: "TEACHER"}))
    assert sql == (
        "SELECT COUNT(*) AS count FROM users "
        "JOIN user_roles ON users.id = user_roles.user_id "
        "JOIN roles ON user_roles.role_id = roles.id "
        "WHERE roles.name = 'TEACHER'"
    )


def test_users_list_still_byte_identical_after_count_and_role_filter_added():
    """Regression: adding COUNT + the ROLE lookup filter to USERS must not
    leak any join or display-field change into USERS' existing LIST shape."""
    plan = QueryPlan(entity=Entity.USERS, operation=Operation.LIST)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT users.first_name, users.last_name, users.email, users.phone, users.department FROM users"


# ── Previously-orphaned grouping dimensions (P0-2) ─────────────────────────

def test_attendance_count_by_status_group_by():
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT attendance.status AS status, COUNT(*) AS count FROM attendance GROUP BY attendance.status"
    )


def test_homework_count_by_status_group_by():
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT homework.status AS status, COUNT(*) AS count FROM homework GROUP BY homework.status"
    )


def test_homework_count_by_subject_group_by():
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.COUNT, group_by=GroupingDimension.BY_SUBJECT)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT homework.subject AS subject, COUNT(*) AS count FROM homework GROUP BY homework.subject"
    )
    assert "JOIN" not in sql
    assert sql.count("SELECT") == 1


def test_course_schedule_count_by_day_of_week_group_by():
    plan = QueryPlan(entity=Entity.COURSE_SCHEDULE, operation=Operation.COUNT, group_by=GroupingDimension.BY_DAY_OF_WEEK)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT course_schedule.day_of_week AS day_of_week, COUNT(*) AS count "
        "FROM course_schedule GROUP BY course_schedule.day_of_week"
    )


def test_report_cards_count_by_term_group_by():
    plan = QueryPlan(entity=Entity.REPORT_CARDS, operation=Operation.COUNT, group_by=GroupingDimension.BY_TERM)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT report_cards.term AS term, COUNT(*) AS count FROM report_cards GROUP BY report_cards.term"
    )


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


# ── Explicit date/date-range, Phase 2 (SQL builder) ──────────────────────────

def test_explicit_date_range_produces_exact_between_clause():
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date="2026-08-01", explicit_end_date="2026-08-15",
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM attendance WHERE attendance.date BETWEEN '2026-08-01' AND '2026-08-15'"


def test_explicit_single_day_produces_same_day_between():
    """A single explicit day is expressed as start == end -- must still
    produce a BETWEEN clause with identical bounds, same shape as TODAY/
    YESTERDAY's relative-date single-day case."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date="2026-08-15", explicit_end_date="2026-08-15",
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT COUNT(*) AS count FROM attendance WHERE attendance.date BETWEEN '2026-08-15' AND '2026-08-15'"


def test_explicit_date_works_for_report_cards():
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        explicit_start_date="2026-08-01", explicit_end_date="2026-08-31",
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT COUNT(*) AS count FROM report_cards "
        "WHERE report_cards.issue_date BETWEEN '2026-08-01' AND '2026-08-31'"
    )


def test_explicit_date_sql_escaping_defense_in_depth():
    """The validator's strict YYYY-MM-DD regex already makes a quote-
    breaking string unreachable here in practice, but the builder must
    still escape defensively, exactly like every other model-supplied
    string value in this file (see the filter loop in build()) -- this
    test bypasses the validator on purpose to prove the builder's own
    escaping is real, not merely assumed, without relying on validator
    behavior to prevent SQL injection at this layer."""
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT)
    plan = plan.model_copy(update={
        "explicit_start_date": "2026-08-01' OR '1'='1",
        "explicit_end_date": "2026-08-15",
    })
    sql = StructuredSQLBuilder.build(plan)  # deliberately unnormalized/unvalidated, see docstring above
    # The raw, unescaped injection string must never appear verbatim --
    # every one of its 4 single quotes must have been doubled.
    assert "2026-08-01' OR '1'='1" not in sql
    assert sql.count("'") == 2 * 4 + 4  # 4 original quotes doubled (8) + the 4 BETWEEN-literal delimiter quotes


def test_relative_date_sql_unchanged_when_explicit_dates_absent():
    """Regression: a plan using only date_range (no explicit fields at all)
    must produce byte-identical SQL to before Phase 2's builder change."""
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, date_range=RelativeDate.LAST_30_DAYS)
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql.startswith("SELECT COUNT(*) AS count FROM attendance WHERE attendance.date BETWEEN")
    assert "explicit" not in sql.lower()


# ── REPORT_CARDS AVERAGE, Phase 2 (SQL builder) ──────────────────────────────
# The sole approved target: report_cards.overall_percentage. SUM/GPA/any
# other entity remain out of scope -- see NumericField's and
# EntityMeta.numeric_agg_fields' own docstrings in query_plan.py/
# query_registry.py for the full Phase 1 architecture this builds on.

def test_report_cards_term_filter_count_exact_sql():
    """Phase 1 (2026-09-10): report_cards.term is native to report_cards'
    own row -- COUNT+TERM filter must produce zero application-level
    JOIN, exactly like HOMEWORK.SUBJECT's own no-join filter."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.TERM, value="term 2")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.TERM: "Term 2"}))
    assert sql == "SELECT COUNT(*) AS count FROM report_cards WHERE report_cards.term = 'Term 2'"
    assert "JOIN" not in sql


def test_report_cards_term_filter_list_exact_sql():
    """LIST+TERM: report_cards.term is native to report_cards' own row, and
    LIST's default display fields (term, overall_grade, overall_percentage)
    are all native columns too -- must produce zero application-level
    JOIN, exactly like the COUNT+TERM case above."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.TERM, value="term 2")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.TERM: "Term 2"}))
    assert sql == (
        "SELECT report_cards.term, report_cards.overall_grade, report_cards.overall_percentage "
        "FROM report_cards WHERE report_cards.term = 'Term 2'"
    )
    assert "JOIN" not in sql


def test_report_cards_term_filter_combined_with_by_term_grouping_still_no_join():
    """TERM filter + BY_TERM grouping is a structurally coherent (if
    redundant) combination -- both reuse the exact same report_cards.term
    column, still zero joins, still a single flat query."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.TERM, value="term 2")],
        group_by=GroupingDimension.BY_TERM,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.TERM: "Term 2"}))
    assert sql == (
        "SELECT report_cards.term AS term, COUNT(*) AS count FROM report_cards "
        "WHERE report_cards.term = 'Term 2' GROUP BY report_cards.term"
    )
    assert "JOIN" not in sql
    assert sql.count("SELECT") == 1


def test_report_cards_term_filter_combined_with_average_still_no_join():
    """TERM filter + AVERAGE(overall_percentage): the real motivating case
    ('What is the average grade in Term 2?') -- report_cards' own natural-
    key uniqueness (student_id, term, academic_year) already rules out
    fanout for the unfiltered AVERAGE (see the existing BY_TERM average
    test above); narrowing to one term via a native, join-free filter
    changes nothing about that guarantee."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
        filters=[ComparisonFilter(field=FilterField.TERM, value="term 2")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.TERM: "Term 2"}))
    assert sql == (
        "SELECT AVG(report_cards.overall_percentage) AS average FROM report_cards "
        "WHERE report_cards.term = 'Term 2'"
    )
    assert "JOIN" not in sql


def test_report_cards_academic_year_filter_count_exact_sql():
    """Phase 2 (2026-09-11): report_cards.academic_year is native to
    report_cards' own row -- COUNT+ACADEMIC_YEAR filter must produce zero
    application-level JOIN, identical shape to TERM above."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.ACADEMIC_YEAR: "2025-2026"}))
    assert sql == "SELECT COUNT(*) AS count FROM report_cards WHERE report_cards.academic_year = '2025-2026'"
    assert "JOIN" not in sql


def test_report_cards_academic_year_filter_list_exact_sql():
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.ACADEMIC_YEAR: "2025-2026"}))
    assert sql == (
        "SELECT report_cards.term, report_cards.overall_grade, report_cards.overall_percentage "
        "FROM report_cards WHERE report_cards.academic_year = '2025-2026'"
    )
    assert "JOIN" not in sql


def test_report_cards_academic_year_and_term_filters_combined_still_no_join():
    """ACADEMIC_YEAR + TERM: two INDEPENDENT native-column filters, both
    ANDed into a single WHERE clause -- still zero joins, still a single
    flat query, mirroring HOMEWORK's own SUBJECT+STATUS combined-filter
    proof."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[
            ComparisonFilter(field=FilterField.TERM, value="term 2"),
            ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026"),
        ],
    )
    sql = StructuredSQLBuilder.build(
        normalize(plan, {FilterField.TERM: "Term 2", FilterField.ACADEMIC_YEAR: "2025-2026"})
    )
    assert sql == (
        "SELECT COUNT(*) AS count FROM report_cards "
        "WHERE report_cards.academic_year = '2025-2026' AND report_cards.term = 'Term 2'"
    )
    assert "JOIN" not in sql
    assert sql.count("SELECT") == 1


def test_report_cards_academic_year_filter_combined_with_average_still_no_join():
    """ACADEMIC_YEAR filter + AVERAGE(overall_percentage): report_cards'
    own natural-key uniqueness (student_id, term, academic_year) already
    rules out fanout for the unfiltered AVERAGE; narrowing to one academic
    year via a native, join-free filter changes nothing about that
    guarantee, identical reasoning to TERM's own average-combination test
    above."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026")],
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {FilterField.ACADEMIC_YEAR: "2025-2026"}))
    assert sql == (
        "SELECT AVG(report_cards.overall_percentage) AS average FROM report_cards "
        "WHERE report_cards.academic_year = '2025-2026'"
    )
    assert "JOIN" not in sql


def test_report_cards_average_ungrouped_exact_sql():
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == "SELECT AVG(report_cards.overall_percentage) AS average FROM report_cards"


def test_report_cards_average_by_term_exact_sql():
    """Also the fan-out regression case: BY_TERM's GroupingPath has
    joins=[] (report_cards.term is a plain column on report_cards' own base
    row -- see query_registry.py's BY_TERM comment), so the generated SQL
    must contain NO JOIN clause at all. This is the actual proof the
    average cannot be distorted by row duplication: there is no join
    present that could ever multiply a report_cards row, not merely an
    added DISTINCT/dedup mechanism papering over one that exists."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE, group_by=GroupingDimension.BY_TERM,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT report_cards.term AS term, AVG(report_cards.overall_percentage) AS average "
        "FROM report_cards GROUP BY report_cards.term"
    )
    assert "JOIN" not in sql


def test_report_cards_average_sort_aggregate_value_resolves_to_average_alias():
    """SortField.AGGREGATE_VALUE is a sentinel resolved against whatever
    this build just aliased -- must resolve to "average", not "count" or
    "percentage", exactly like the existing PERCENTAGE+BY_STUDENT case."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE, group_by=GroupingDimension.BY_TERM,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="desc"), limit=3,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT report_cards.term AS term, AVG(report_cards.overall_percentage) AS average "
        "FROM report_cards GROUP BY report_cards.term ORDER BY average DESC LIMIT 3"
    )


def test_report_cards_average_extreme_produces_same_flat_sql_as_plain_grouped_query():
    """Authorization regression, mirroring
    test_extreme_plan_produces_the_same_flat_sql_as_plain_grouped_query
    above exactly, for AVERAGE instead of PERCENTAGE: plan.extreme must add
    NO SQL of its own -- same base table, no JOIN at all here, same GROUP
    BY, no ORDER BY, no LIMIT, no nested subquery -- so the existing,
    unmodified AliasAwareFilterInjector authorizes it exactly as it does
    any other grouped query."""
    plan_with_extreme = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE, group_by=GroupingDimension.BY_TERM,
        aggregate_target=NumericField.OVERALL_PERCENTAGE, extreme=ExtremeSelector.LOWEST,
    )
    plan_without_extreme = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE, group_by=GroupingDimension.BY_TERM,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    sql_with = StructuredSQLBuilder.build(normalize(plan_with_extreme, {}))
    sql_without = StructuredSQLBuilder.build(normalize(plan_without_extreme, {}))

    assert sql_with == sql_without
    assert sql_with.count("SELECT") == 1  # exactly one query, no nested subquery
    assert " ORDER BY " not in sql_with
    assert " LIMIT " not in sql_with
    assert "JOIN" not in sql_with
    assert sql_with == (
        "SELECT report_cards.term AS term, AVG(report_cards.overall_percentage) AS average "
        "FROM report_cards GROUP BY report_cards.term"
    )


def test_report_cards_average_with_filters_and_date_range():
    """AVERAGE composes normally with the existing WHERE-clause machinery
    (filters, date_range) -- no special-casing needed, same as COUNT/
    PERCENTAGE."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
        date_range=RelativeDate.THIS_YEAR,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql.startswith("SELECT AVG(report_cards.overall_percentage) AS average FROM report_cards WHERE report_cards.issue_date BETWEEN")


def test_report_cards_average_with_explicit_date_range():
    """AVERAGE composes normally with explicit_start_date/explicit_end_date
    too, not just date_range -- the builder's explicit-date branch is
    entirely operation-agnostic (see the elif chain in build()), so this is
    the same generic WHERE-clause machinery every other operation already
    uses with explicit dates (see test_explicit_date_range_produces_exact_
    between_clause for the COUNT case this mirrors). Bounds are inclusive
    on both ends, same BETWEEN semantics every relative-date value already
    uses -- no different rounding/exclusivity for AVERAGE."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
        explicit_start_date="2026-08-01", explicit_end_date="2026-08-31",
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert sql == (
        "SELECT AVG(report_cards.overall_percentage) AS average FROM report_cards "
        "WHERE report_cards.issue_date BETWEEN '2026-08-01' AND '2026-08-31'"
    )
    assert "JOIN" not in sql
    assert sql.count("SELECT") == 1


def test_sum_still_raises_not_implemented():
    """Regression: SUM remains an explicit, deliberate failure -- Phase 2
    only implements AVERAGE. This plan cannot pass QueryPlanValidator
    (SUM is not in any entity's supported_operations), so this test
    exercises the builder function directly and unvalidated, exactly
    mirroring how the pre-existing NotImplementedError contract was
    documented and tested before this phase."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.SUM,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    with pytest.raises(NotImplementedError):
        StructuredSQLBuilder.build(plan)


def test_existing_count_percentage_list_behavior_unaffected_by_average_addition():
    """Regression: REPORT_CARDS' pre-existing COUNT/LIST SQL (and, on a
    different entity, PERCENTAGE) must remain byte-identical after adding
    the AVERAGE branch -- confirms the new elif branch didn't disturb the
    existing operation dispatch."""
    count_plan = QueryPlan(entity=Entity.REPORT_CARDS, operation=Operation.COUNT, group_by=GroupingDimension.BY_TERM)
    count_sql = StructuredSQLBuilder.build(normalize(count_plan, {}))
    assert count_sql == (
        "SELECT report_cards.term AS term, COUNT(*) AS count FROM report_cards GROUP BY report_cards.term"
    )

    list_plan = QueryPlan(entity=Entity.REPORT_CARDS, operation=Operation.LIST)
    list_sql = StructuredSQLBuilder.build(normalize(list_plan, {}))
    assert list_sql == (
        "SELECT report_cards.term, report_cards.overall_grade, report_cards.overall_percentage FROM report_cards"
    )

    pct_plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
    )
    pct_sql = StructuredSQLBuilder.build(normalize(pct_plan, {}))
    assert pct_sql == (
        "SELECT (COUNT(CASE WHEN attendance.status = 'present' THEN 1 END) * 100.0 / COUNT(*)) AS percentage "
        "FROM attendance"
    )
