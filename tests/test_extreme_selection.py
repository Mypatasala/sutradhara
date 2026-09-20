"""
Deterministic, post-authorization extreme-value tie resolution --
_apply_extreme_selection in query_lifecycle.py. No LLM, no DB, no graph: this
operates purely on the row list a DB execution would have already returned,
which is exactly the point (see structured_sql_builder.py's module docstring
for why this happens here rather than in SQL).
"""

from decimal import Decimal

from src.agents.query_lifecycle import _apply_extreme_selection


def test_lowest_extreme_multiple_students_tied_all_returned():
    data = [
        {"student_name": "Alice", "percentage": Decimal("40.00")},
        {"student_name": "Bob", "percentage": Decimal("40.00")},
        {"student_name": "Carol", "percentage": Decimal("90.00")},
    ]
    result = _apply_extreme_selection(data, "lowest", "percentage")
    assert {row["student_name"] for row in result} == {"Alice", "Bob"}
    assert len(result) == 2


def test_highest_extreme_multiple_students_tied_all_returned():
    data = [
        {"student_name": "Alice", "percentage": Decimal("40.00")},
        {"student_name": "Bob", "percentage": Decimal("95.00")},
        {"student_name": "Carol", "percentage": Decimal("95.00")},
    ]
    result = _apply_extreme_selection(data, "highest", "percentage")
    assert {row["student_name"] for row in result} == {"Bob", "Carol"}
    assert len(result) == 2


def test_lowest_extreme_single_winner_returns_exactly_one_row():
    data = [
        {"student_name": "Alice", "percentage": Decimal("40.00")},
        {"student_name": "Bob", "percentage": Decimal("95.00")},
    ]
    result = _apply_extreme_selection(data, "lowest", "percentage")
    assert len(result) == 1
    assert result[0]["student_name"] == "Alice"


def test_no_arbitrary_limit_all_tied_rows_survive_regardless_of_count():
    """Never an implicit LIMIT 1 or any other invented cap -- every row tied
    at the true extreme must survive, even with many ties."""
    data = [{"student_name": f"Student {i}", "percentage": Decimal("50.00")} for i in range(10)]
    result = _apply_extreme_selection(data, "lowest", "percentage")
    assert len(result) == 10


def test_extreme_none_returns_data_unchanged():
    """A plan with no extreme (ordinary grouped query, or explicit top-N via
    sort+limit) must never have this step touch its rows."""
    data = [{"student_name": "Alice", "percentage": Decimal("40.00")}]
    assert _apply_extreme_selection(data, None, None) == data


def test_extreme_exact_decimal_comparison_no_rounding():
    """MariaDB returns the PERCENTAGE expression as decimal.Decimal (verified
    live against the real driver) -- two rows whose values are genuinely
    different but would round to the same displayed value must NOT be
    treated as tied."""
    data = [
        {"student_name": "Alice", "percentage": Decimal("40.001")},
        {"student_name": "Bob", "percentage": Decimal("40.004")},
    ]
    result = _apply_extreme_selection(data, "lowest", "percentage")
    assert len(result) == 1
    assert result[0]["student_name"] == "Alice"


def test_extreme_count_operation_uses_int_comparison():
    data = [
        {"class_name": "3A", "count": 12},
        {"class_name": "3B", "count": 12},
        {"class_name": "4A", "count": 30},
    ]
    result = _apply_extreme_selection(data, "highest", "count")
    assert {row["class_name"] for row in result} == {"4A"}


# ── Empty / degenerate result handling ──────────────────────────────────────

def test_empty_result_returns_empty_without_calling_min_or_max():
    """Must not raise ValueError from min()/max() on an empty sequence."""
    result = _apply_extreme_selection([], "lowest", "percentage")
    assert result == []


def test_error_carrying_result_returned_unchanged():
    """A DB-execution error payload (see DBClient.execute) must pass through
    untouched -- there's no aggregate field to extract an extreme from."""
    data = [{"error": "Only SELECT queries are permitted."}]
    result = _apply_extreme_selection(data, "lowest", "percentage")
    assert result == data


def test_missing_field_in_all_rows_returns_data_unchanged():
    """Defensive: if extreme_field somehow isn't present in any row, there's
    no value to compute a min/max from -- must not raise, must not silently
    drop all rows."""
    data = [{"student_name": "Alice"}]
    result = _apply_extreme_selection(data, "lowest", "percentage")
    assert result == data


def test_lowest_average_grouped_by_term_reduces_to_the_minimum_term():
    """This function is generic over the extreme_field NAME -- confirms it
    works identically for "average" (REPORT_CARDS' aggregate alias) as it
    already does for "percentage"/"count", exactly the field name
    query_lifecycle.py's aggregate_alias ternary now resolves AVERAGE to.
    'Which term has the lowest average grade?' with three terms, no ties."""
    data = [
        {"term": "Term 1", "average": Decimal("85.00")},
        {"term": "Term 2", "average": Decimal("72.50")},
        {"term": "Term 3", "average": Decimal("90.00")},
    ]
    result = _apply_extreme_selection(data, "lowest", "average")
    assert len(result) == 1
    assert result[0]["term"] == "Term 2"


def test_highest_average_grouped_by_term_all_ties_included():
    data = [
        {"term": "Term 1", "average": Decimal("85.00")},
        {"term": "Term 2", "average": Decimal("85.00")},
        {"term": "Term 3", "average": Decimal("70.00")},
    ]
    result = _apply_extreme_selection(data, "highest", "average")
    assert {row["term"] for row in result} == {"Term 1", "Term 2"}
    assert len(result) == 2


# ── COUNT + grouped + extreme (2026-09-07 audit follow-up) ──────────────────
# Same generic-over-field-name proof as the "average" tests above, now for
# "count" -- the aggregate_alias query_lifecycle.py resolves for
# ATTENDANCE.BY_STATUS/REPORT_CARDS.BY_TERM/COURSE_SCHEDULE.BY_DAY_OF_WEEK.

def test_highest_count_grouped_by_status_reduces_to_the_maximum_status():
    """'Which status has the most attendance records?' with four statuses,
    no ties."""
    data = [
        {"status": "present", "count": 120},
        {"status": "absent", "count": 15},
        {"status": "late", "count": 8},
        {"status": "excused", "count": 3},
    ]
    result = _apply_extreme_selection(data, "highest", "count")
    assert len(result) == 1
    assert result[0]["status"] == "present"


def test_lowest_count_grouped_by_term_all_ties_included():
    """'Which term had the fewest report cards?' with a tie between two
    terms -- both must survive, no arbitrary single-row cap."""
    data = [
        {"term": "Term 1", "count": 30},
        {"term": "Term 2", "count": 12},
        {"term": "Term 3", "count": 12},
    ]
    result = _apply_extreme_selection(data, "lowest", "count")
    assert {row["term"] for row in result} == {"Term 2", "Term 3"}
    assert len(result) == 2


def test_highest_count_grouped_by_day_of_week_single_winner():
    """'Which day has the most classes scheduled?'"""
    data = [
        {"day_of_week": "Monday", "count": 6},
        {"day_of_week": "Tuesday", "count": 4},
        {"day_of_week": "Wednesday", "count": 6},
        {"day_of_week": "Thursday", "count": 2},
    ]
    result = _apply_extreme_selection(data, "highest", "count")
    assert {row["day_of_week"] for row in result} == {"Monday", "Wednesday"}
    assert len(result) == 2
