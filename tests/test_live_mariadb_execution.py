"""
Category B (live-dependent, non-gating in CI) -- executes the deterministic
builder's SQL against the REAL configured MariaDB instance, not a mock or
SQLite fixture. This is the only way to actually prove Issue 2's ambiguous-
column defect is fixed: unit tests assert the SQL *text* is qualified, but
only a real MariaDB execution can prove the query is genuinely free of the
live "Column 'status' in SELECT is ambiguous" (1052) error that was
reproduced against this exact instance during the investigation.

Skips gracefully if DATABASE_URL isn't configured to a real database (e.g.
CI with no live MariaDB), so it never blocks a run that has no live DB.
"""

import os

import pytest

from src.agents.query_plan import (
    ComparisonFilter,
    FilterField,
    Entity,
    ExtremeSelector,
    GroupingDimension,
    Operation,
    PercentageSpec,
    QueryPlan,
)
from src.agents.query_normalizer import normalize
from src.policy.filter_injector import AliasAwareFilterInjector
from src.policy.identity_guard import IdentityFilterGuard
from src.policy.sanitizer import SQLSanitizer
from src.retrieval.db_client import DBClient
from src.retrieval.structured_sql_builder import StructuredSQLBuilder

_HAS_REAL_DB = bool(os.getenv("DATABASE_URL"))


# The REAL production OPA row_filter shape for attendance (my_patasala's
# policy/opa/admin.rego) -- NOT a bare "school_id = ..." filter, since
# `attendance` has no direct school_id column at all (verified against
# V1__baseline.sql: attendance's columns are id/date/notes/status/
# course_id/student_id/marked_by). Using the real shape here, not a
# simplified stand-in, so this test proves the fix against what production
# actually sends, not a convenient fiction.
_REAL_ATTENDANCE_ROW_FILTER = "student_id IN (SELECT id FROM students WHERE school_id = 56)"


def _run_through_pipeline_and_execute(plan: QueryPlan, row_filter: str) -> list:
    canonical = normalize(plan, {})
    sql = StructuredSQLBuilder.build(canonical)
    guarded = IdentityFilterGuard.strip(sql)
    qualified = AliasAwareFilterInjector.inject(guarded, row_filter, "attendance")
    final_sql = SQLSanitizer.apply_constraints(guarded, [], qualified)
    return DBClient().execute(final_sql)


@pytest.mark.skipif(not _HAS_REAL_DB, reason="No DATABASE_URL configured -- skipping live MariaDB execution check.")
def test_attendance_percentage_by_student_executes_without_ambiguous_column_error():
    """The exact motivating Issue 2 case: 'Which students have the highest
    attendance?'. Before the fix, this exact SQL shape produced MariaDB
    error (1052, "Column 'status' in SELECT is ambiguous") -- reproduced
    live during the investigation. This must now execute cleanly."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.HIGHEST,
    )
    result = _run_through_pipeline_and_execute(plan, row_filter=_REAL_ATTENDANCE_ROW_FILTER)
    assert not (result and "error" in result[0]), f"Query failed: {result}"
    assert "ambiguous" not in str(result).lower()


@pytest.mark.skipif(not _HAS_REAL_DB, reason="No DATABASE_URL configured -- skipping live MariaDB execution check.")
def test_attendance_status_filter_with_by_student_executes_without_ambiguous_column_error():
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.BY_STUDENT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="absent")],
    )
    result = _run_through_pipeline_and_execute(plan, row_filter=_REAL_ATTENDANCE_ROW_FILTER)
    assert not (result and "error" in result[0]), f"Query failed: {result}"
    assert "ambiguous" not in str(result).lower()
