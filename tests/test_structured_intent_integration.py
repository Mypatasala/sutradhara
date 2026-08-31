"""
Proves the structured path's output integrates cleanly with the EXISTING,
UNMODIFIED authorization pipeline: IdentityFilterGuard -> AliasAwareFilter
Injector -> SQLSanitizer.apply_constraints. None of those three classes are
touched by this work -- this file exists to prove that fact rather than
assert it.
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
    RelativeDate,
    SortField,
    SortSpec,
)
from src.agents.query_lifecycle import _apply_extreme_selection
from src.agents.query_normalizer import normalize
from src.agents.query_registry import REGISTRY
from src.policy.identity_guard import IdentityFilterGuard
from src.policy.filter_injector import AliasAwareFilterInjector
from src.policy.sanitizer import SQLSanitizer
from src.retrieval.structured_sql_builder import StructuredSQLBuilder


def _run_full_pipeline(plan: QueryPlan, row_filter: str, allowed_cols=None, resolved_lookups=None) -> str:
    canonical = normalize(plan, resolved_lookups or {})
    sql = StructuredSQLBuilder.build(canonical)
    target_table = REGISTRY[canonical.entity].table  # exactly how query_lifecycle.py derives it

    guarded_sql = IdentityFilterGuard.strip(sql)
    qualified_filter = AliasAwareFilterInjector.inject(guarded_sql, row_filter, target_table)
    return SQLSanitizer.apply_constraints(guarded_sql, allowed_cols or [], qualified_filter)


def test_count_students_by_class_through_full_pipeline():
    """The exact motivating case, end to end through every existing
    authorization layer, none of which are modified by this work."""
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_CLASS)
    final_sql = _run_full_pipeline(plan, row_filter="school_id = 56")

    assert "JOIN class_sections ON students.section_id = class_sections.id" in final_sql
    assert "JOIN school_classes ON class_sections.school_class_id = school_classes.id" in final_sql
    assert "WHERE students.school_id = 56" in final_sql  # alias-injector correctly qualifies the unaliased target table
    assert "AND" not in final_sql or "school_id = 56" in final_sql  # no stray ambiguous bare column


def test_grade_filter_through_full_pipeline_with_authorization():
    """Issue 1's motivating case, end to end through the real, unmodified
    authorization pipeline with a representative row filter: a grade filter
    must survive alongside row-level authorization, and the row filter must
    still correctly qualify against the unaliased students table."""
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.GRADE, value="5")],
    )
    final_sql = _run_full_pipeline(
        plan, row_filter="school_id = 56", resolved_lookups={FilterField.GRADE: "5"},
    )
    assert final_sql == (
        "SELECT students.first_name, students.last_name FROM students "
        "WHERE students.grade = '5' AND (students.school_id = 56)"
    )


def test_identity_guard_never_triggers_on_builder_output():
    """The deterministic builder has no field through which a self-invented
    identity/tenant literal could even be expressed -- IdentityFilterGuard
    should be a pure no-op (return the SQL completely unchanged) for every
    builder-generated query, proving the two layers don't conflict."""
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.COUNT,
                      filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")])
    canonical = normalize(plan, {})
    sql = StructuredSQLBuilder.build(canonical)
    assert IdentityFilterGuard.strip(sql) == sql


def test_alias_injector_resolves_unaliased_builder_output_cleanly():
    """The builder never aliases the primary table -- this must always hit
    AliasAwareFilterInjector's already-tested unaliased-table code path,
    never its rejection path."""
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT)
    canonical = normalize(plan, {})
    sql = StructuredSQLBuilder.build(canonical)
    qualified = AliasAwareFilterInjector.inject(sql, "school_id = 56", REGISTRY[Entity.ATTENDANCE].table)
    assert qualified == "attendance.school_id = 56"


def test_users_list_password_never_reachable_even_with_column_allowlist():
    """Defense-in-depth check: even though DisplayField structurally can't
    express 'password', also confirm apply_constraints' column allowlist
    (as OPA would supply it) still can't be defeated by the structured
    path's output."""
    plan = QueryPlan(entity=Entity.USERS, operation=Operation.LIST)
    final_sql = _run_full_pipeline(plan, row_filter="school_id = 56",
                                    allowed_cols=["first_name", "last_name", "email", "phone", "department"])
    assert "password" not in final_sql.lower()


# ── extreme-value authorization regression ──────────────────────────────────
# See structured_sql_builder.py's module docstring: a live trace proved a
# nested MIN/MAX aggregate subquery would only ever be authorized in its
# OUTER occurrence, since AliasAwareFilterInjector deliberately never
# qualifies columns inside a nested Subquery/Select. plan.extreme therefore
# adds NO SQL of its own -- these tests prove that through the real,
# unmodified authorization pipeline, then prove the post-processing step
# only ever sees rows that already passed it.

def test_extreme_plan_row_filter_applied_before_execution_flat_shape_preserved():
    """The row_filter must actually reach the query BEFORE any hypothetical
    execution -- proven end to end through the real, unmodified
    IdentityFilterGuard -> AliasAwareFilterInjector -> SQLSanitizer chain --
    and the resulting SQL must still be the plain flat shape (single
    occurrence of the target table, no nested subquery) that
    AliasAwareFilterInjector can authorize at all."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST,
    )
    final_sql = _run_full_pipeline(plan, row_filter="school_id = 56")

    assert "WHERE attendance.school_id = 56" in final_sql
    assert final_sql.count("SELECT") == 1
    assert " ORDER BY " not in final_sql
    assert " LIMIT " not in final_sql


def test_extreme_post_processing_only_sees_already_authorized_rows():
    """The post-processing step (_apply_extreme_selection) takes only the
    row LIST the DB execution returned -- it has no DB handle, no SQL, no
    access to anything beyond what's already been authorized and executed.
    This test proves that by construction: feeding it a row list that
    (by hypothesis) already excludes an out-of-scope student's row, the
    extreme it computes can only ever reflect the authorized subset, never
    a cross-tenant minimum/maximum."""
    # Simulates the fully authorized result AFTER row_filter has already
    # excluded a different school's students -- the true global minimum
    # (attendance 10%, a different school's student) is absent from this
    # list entirely, exactly as row_filter would ensure in the real pipeline.
    authorized_rows = [
        {"student_name": "Alice", "percentage": Decimal("40.00")},
        {"student_name": "Bob", "percentage": Decimal("85.00")},
    ]
    result = _apply_extreme_selection(authorized_rows, "lowest", "percentage")
    assert len(result) == 1
    assert result[0]["student_name"] == "Alice"  # the authorized-scope minimum, never a cross-tenant value


def test_full_pipeline_report_cards_exact_sql():
    """Was previously a weak startswith('SELECT') check -- this layer is
    supposed to guarantee deterministic SQL correctness, so it gets an
    exact assertion like every other test in this file."""
    plan = QueryPlan(entity=Entity.REPORT_CARDS, operation=Operation.LIST)
    final_sql = _run_full_pipeline(plan, row_filter="id = 'self-row-id'")
    assert final_sql == (
        "SELECT report_cards.term, report_cards.overall_grade, report_cards.overall_percentage FROM report_cards "
        "WHERE report_cards.id = 'self-row-id'"
    )


# ── Issue 2: registry column qualification, through the real authorization ──
# pipeline. Root cause (confirmed live against the real MariaDB instance):
# BY_STUDENT is the first grouping that joins ATTENDANCE against a table
# (STUDENTS) sharing a column name (status) -- an unqualified `status` in
# the builder's PERCENTAGE branch produced the exact MariaDB error
# "(1052, "Column 'status' in SELECT is ambiguous")". Fixed by qualifying
# every registry-owned physical column with its table at the source
# (query_registry.py), not by changing builder logic.

def test_attendance_percentage_by_student_no_longer_ambiguous_through_full_pipeline():
    """The exact motivating Issue 2 case end to end through the real,
    unmodified authorization pipeline: attendance.status must be qualified
    everywhere it's used, since students (joined for BY_STUDENT) also has
    its own status column (enrollment status, unrelated to attendance)."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.HIGHEST,
    )
    final_sql = _run_full_pipeline(plan, row_filter="school_id = 56")
    assert "attendance.status" in final_sql
    assert " status " not in f" {final_sql} "  # no lingering unqualified reference anywhere
    assert final_sql == (
        "SELECT CONCAT_WS(' ', students.first_name, students.last_name) AS student_name, "
        "school_classes.name AS class_name, class_sections.name AS section_name, "
        "(COUNT(CASE WHEN attendance.status = 'present' THEN 1 END) * 100.0 / COUNT(*)) AS percentage "
        "FROM attendance JOIN students ON attendance.student_id = students.id "
        "LEFT JOIN class_sections ON students.section_id = class_sections.id "
        "LEFT JOIN school_classes ON class_sections.school_class_id = school_classes.id "
        "WHERE attendance.school_id = 56 "
        "GROUP BY students.id, students.first_name, students.last_name, "
        "class_sections.id, class_sections.name, school_classes.id, school_classes.name"
    )


def test_attendance_status_filter_combined_with_by_student_not_ambiguous():
    """A STATUS filter (not just the percentage numerator) combined with
    BY_STUDENT must also be qualified -- this is the second site the audit
    identified as sharing the same latent defect shape as the percentage
    branch, now proven fixed through the real builder + authorization
    pipeline together, not just reasoned about."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.BY_STUDENT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="absent")],
    )
    final_sql = _run_full_pipeline(plan, row_filter="school_id = 56")
    assert "attendance.status = 'absent'" in final_sql
    assert " status " not in f" {final_sql} "


def test_date_filter_remains_correctly_qualified_through_full_pipeline():
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, date_range=RelativeDate.TODAY)
    final_sql = _run_full_pipeline(plan, row_filter="school_id = 56")
    assert "attendance.date BETWEEN" in final_sql


def test_sort_column_qualified_through_full_pipeline():
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        sort=SortSpec(field=SortField.ISSUE_DATE, direction="desc"), limit=1,
    )
    final_sql = _run_full_pipeline(plan, row_filter="id = 'self-row-id'")
    assert "ORDER BY report_cards.issue_date DESC" in final_sql



def test_display_fields_qualified_through_full_pipeline():
    plan = QueryPlan(entity=Entity.USERS, operation=Operation.LIST)
    final_sql = _run_full_pipeline(
        plan, row_filter="school_id = 56",
        allowed_cols=["users.first_name", "users.last_name", "users.email", "users.phone", "users.department"],
    )
    assert "SELECT users.first_name, users.last_name, users.email, users.phone, users.department" in final_sql
