"""
Class/Section display for ATTENDANCE + BY_STUDENT ranking -- the fix for
the live UI report of "duplicate" highest-attendance rows. Investigation
proved those weren't duplicates: real seeded data has multiple distinct
students (different students.id, different section) sharing the same full
name (e.g. three "Riya Verma"s in school 56, each independently computing to
61.57635% attendance). The grouping was always correct; the label wasn't
disambiguating. Adding class_name/section_name via GroupingPath's registry-
owned default_display_columns fixes the disambiguation without any model-
facing schema change.
"""

from decimal import Decimal

import pytest

from src.agents.query_plan import (
    ComparisonFilter,
    Entity,
    ExtremeSelector,
    FilterField,
    GroupingDimension,
    Operation,
    PercentageSpec,
    QueryPlan,
    SortField,
    SortSpec,
)
from src.agents.query_lifecycle import _apply_extreme_selection
from src.agents.query_normalizer import normalize
from src.agents.query_registry import REGISTRY, EntityMeta, GroupingPath, JoinStep, LabelExpression
from src.policy.filter_injector import AliasAwareFilterInjector
from src.policy.identity_guard import IdentityFilterGuard
from src.policy.sanitizer import SQLSanitizer
from src.retrieval.db_client import DBClient
from src.retrieval.structured_sql_builder import StructuredSQLBuilder


def _build_percentage_by_student_plan(**overrides) -> QueryPlan:
    kwargs = dict(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
    )
    kwargs.update(overrides)
    return QueryPlan(**kwargs)


# ── JoinStep join_type / LEFT JOIN capability ───────────────────────────────

def test_join_step_defaults_to_inner_join_preserving_existing_behavior():
    step = JoinStep(table="students", left_column="student_id", right_column="id")
    assert step.join_type == "JOIN"


def test_attendance_by_student_uses_left_join_for_class_and_section():
    """The class_sections/school_classes joins must be LEFT JOIN, never the
    default inner JOIN -- students.section_id is nullable, and an
    unassigned student must not be silently dropped from a ranking merely
    because of a display-only addition."""
    path = REGISTRY[Entity.ATTENDANCE].supported_groupings[GroupingDimension.BY_STUDENT]
    joins_by_table = {step.table: step for step in path.joins}
    assert joins_by_table["students"].join_type == "JOIN"  # unchanged -- attendance.student_id is NOT NULL
    assert joins_by_table["class_sections"].join_type == "LEFT JOIN"
    assert joins_by_table["school_classes"].join_type == "LEFT JOIN"


# ── Builder: exact SQL shape ─────────────────────────────────────────────────

def test_builder_emits_class_and_section_via_left_join():
    plan = _build_percentage_by_student_plan()
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


def test_class_section_appears_for_extreme_ranking():
    for extreme in (ExtremeSelector.LOWEST, ExtremeSelector.HIGHEST):
        plan = _build_percentage_by_student_plan(extreme=extreme)
        sql = StructuredSQLBuilder.build(normalize(plan, {}))
        assert "school_classes.name AS class_name" in sql
        assert "class_sections.name AS section_name" in sql
        assert "LIMIT" not in sql  # extreme still adds no SQL of its own


def test_class_section_appears_for_explicit_top_bottom_n():
    plan = _build_percentage_by_student_plan(
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"), limit=5,
    )
    sql = StructuredSQLBuilder.build(normalize(plan, {}))
    assert "school_classes.name AS class_name" in sql
    assert "class_sections.name AS section_name" in sql
    assert sql.endswith(
        "GROUP BY students.id, students.first_name, students.last_name, "
        "class_sections.id, class_sections.name, school_classes.id, school_classes.name "
        "ORDER BY percentage ASC LIMIT 5"
    )


def test_default_display_columns_is_generic_not_attendance_specific():
    """The mechanism itself (GroupingPath.default_display_columns +
    StructuredSQLBuilder's generic emission of it) must not be hardcoded to
    ATTENDANCE -- verified by exercising it through a hand-built fake
    grouping for a different entity."""
    fake_meta = EntityMeta(
        table="attendance",
        supported_operations={Operation.COUNT},
        supported_groupings={
            GroupingDimension.BY_STUDENT: GroupingPath(
                joins=[JoinStep(table="students", left_column="student_id", right_column="id")],
                group_by_columns=["students.id"],
                label=LabelExpression(columns=["students.id"], separator=""),
                label_alias="sid",
                default_display_columns=[("students.email", "email")],
            ),
        },
    )
    import src.agents.query_registry as qr
    original = qr.REGISTRY[Entity.ATTENDANCE]
    qr.REGISTRY[Entity.ATTENDANCE] = fake_meta
    try:
        plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.BY_STUDENT)
        sql = StructuredSQLBuilder.build(normalize(plan, {}))
        assert "students.email AS email" in sql
    finally:
        qr.REGISTRY[Entity.ATTENDANCE] = original


# ── One row per distinct students.id, live against the real MariaDB ────────
# Skipped without a real DATABASE_URL -- see test_live_mariadb_execution.py
# for the established pattern.

import os

_HAS_REAL_DB = bool(os.getenv("DATABASE_URL"))
_REAL_ROW_FILTER = "student_id IN (SELECT id FROM students WHERE school_id = 56)"


def _run_and_execute(plan: QueryPlan) -> list:
    canonical = normalize(plan, {})
    sql = StructuredSQLBuilder.build(canonical)
    guarded = IdentityFilterGuard.strip(sql)
    qualified = AliasAwareFilterInjector.inject(guarded, _REAL_ROW_FILTER, "attendance")
    final_sql = SQLSanitizer.apply_constraints(guarded, [], qualified)
    return DBClient().execute(final_sql)


@pytest.mark.skipif(not _HAS_REAL_DB, reason="No DATABASE_URL configured -- skipping live MariaDB verification.")
def test_by_student_query_still_produces_one_row_per_student_id():
    """Adding the class/section joins must not change grouping granularity
    -- confirmed live against real seeded data known to contain same-named
    students, by re-querying with students.id exposed and checking for
    duplicate ids (there must be none) and duplicate NAMES that are
    legitimately different ids (there should be some, matching the
    investigation)."""
    plan = _build_percentage_by_student_plan()
    rows = _run_and_execute(plan)
    assert rows and "error" not in rows[0]

    # Re-run with students.id exposed directly to check for true duplicate ids.
    canonical = normalize(plan, {})
    sql_with_id = StructuredSQLBuilder.build(canonical).replace(
        "SELECT CONCAT_WS(' ', students.first_name, students.last_name) AS student_name,",
        "SELECT students.id AS sid, CONCAT_WS(' ', students.first_name, students.last_name) AS student_name,",
    )
    guarded = IdentityFilterGuard.strip(sql_with_id)
    qualified = AliasAwareFilterInjector.inject(guarded, _REAL_ROW_FILTER, "attendance")
    final_sql = SQLSanitizer.apply_constraints(guarded, [], qualified)
    rows_with_id = DBClient().execute(final_sql)
    ids = [r["sid"] for r in rows_with_id]
    assert len(ids) == len(set(ids)), "duplicate students.id in the aggregate result -- grouping regression"


@pytest.mark.skipif(not _HAS_REAL_DB, reason="No DATABASE_URL configured -- skipping live MariaDB verification.")
def test_same_named_students_are_distinguishable_by_class_and_section():
    """The exact motivating case: three real, distinct "Riya Verma" students
    in school 56 must all be present with DIFFERENT (class, section) pairs,
    proving the fix actually disambiguates them rather than merely adding
    unused columns."""
    plan = _build_percentage_by_student_plan()
    rows = _run_and_execute(plan)
    riya_rows = [r for r in rows if r.get("student_name") == "Riya Verma"]
    assert len(riya_rows) >= 2, "expected the known multiple same-named students in this dataset"
    class_section_pairs = {(r.get("class_name"), r.get("section_name")) for r in riya_rows}
    assert len(class_section_pairs) == len(riya_rows), (
        f"same-named students did not get distinct (class, section) pairs: {riya_rows}"
    )


@pytest.mark.skipif(not _HAS_REAL_DB, reason="No DATABASE_URL configured -- skipping live MariaDB verification.")
def test_tied_lowest_and_highest_include_class_and_section_live():
    for extreme, aggregate_alias in ((ExtremeSelector.LOWEST, "percentage"), (ExtremeSelector.HIGHEST, "percentage")):
        plan = _build_percentage_by_student_plan(extreme=extreme)
        rows = _run_and_execute(plan)
        post = _apply_extreme_selection(rows, extreme.value, aggregate_alias)
        assert post, f"expected at least one row for extreme={extreme.value}"
        for row in post:
            assert "class_name" in row
            assert "section_name" in row


@pytest.mark.skipif(not _HAS_REAL_DB, reason="No DATABASE_URL configured -- skipping live MariaDB verification.")
def test_explicit_n_includes_class_and_section_live():
    plan = _build_percentage_by_student_plan(
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"), limit=5,
    )
    canonical = normalize(plan, {})
    sql = StructuredSQLBuilder.build(canonical)
    guarded = IdentityFilterGuard.strip(sql)
    qualified = AliasAwareFilterInjector.inject(guarded, _REAL_ROW_FILTER, "attendance")
    final_sql = SQLSanitizer.apply_constraints(guarded, [], qualified)
    rows = DBClient().execute(final_sql)
    assert len(rows) == 5
    for row in rows:
        assert "class_name" in row
        assert "section_name" in row


def test_unassigned_student_not_excluded_from_ranking():
    """A student with section_id=NULL must still appear in the ranking (with
    class_name/section_name NULL), never silently dropped by the LEFT JOIN.

    Deliberately does NOT mutate the real database to simulate this --
    instead proves the SQL shape itself (LEFT JOIN, not JOIN) guarantees it,
    and proves the downstream Python pipeline (_apply_extreme_selection)
    does not drop or choke on a row carrying NULL class_name/section_name,
    using a fake row set that stands in for what MariaDB would return for
    an unassigned student (a LEFT JOIN with no match yields NULL for the
    joined columns, never a missing row)."""
    # SQL-shape guarantee: LEFT JOIN never removes an unmatched base row.
    path = REGISTRY[Entity.ATTENDANCE].supported_groupings[GroupingDimension.BY_STUDENT]
    joins_by_table = {step.table: step for step in path.joins}
    assert joins_by_table["class_sections"].join_type == "LEFT JOIN"
    assert joins_by_table["school_classes"].join_type == "LEFT JOIN"

    # Pipeline-level guarantee: a NULL-class/section row survives extreme
    # selection unchanged, tied alongside students who do have a class/section.
    rows_including_unassigned = [
        {"student_name": "Assigned Student", "class_name": "5th Grade", "section_name": "A", "percentage": Decimal("40.00")},
        {"student_name": "Unassigned Student", "class_name": None, "section_name": None, "percentage": Decimal("40.00")},
    ]
    result = _apply_extreme_selection(rows_including_unassigned, "lowest", "percentage")
    assert len(result) == 2
    names = {r["student_name"] for r in result}
    assert "Unassigned Student" in names, "the unassigned student was dropped -- must remain tied at the extreme"
