"""
Deterministic unit tests for _resolve_relative_date, including the new
LAST_30_DAYS member added to fix a live-traced incident: with no vocabulary
entry matching "the last 30 days", the structured-output model silently
substituted LAST_WEEK, scoping "last 30 days" questions to only 7 days with
no error or indication anything was wrong.

Uses a fixed reference date (2026-09-02, a Wednesday) so every boundary is
exact and reproducible, not dependent on the day this test happens to run.
"""

from datetime import date

from src.retrieval.structured_sql_builder import _resolve_relative_date
from src.agents.query_plan import RelativeDate

REFERENCE_DATE = date(2026, 9, 2)  # a Wednesday


def test_last_30_days_is_a_true_rolling_30_day_window():
    """Today plus the preceding 29 days = exactly 30 distinct calendar
    dates, inclusive of today -- NOT a calendar-month approximation."""
    start, end = _resolve_relative_date(RelativeDate.LAST_30_DAYS, today=REFERENCE_DATE)
    assert end == REFERENCE_DATE
    assert start == date(2026, 8, 4)
    assert (end - start).days == 29  # inclusive range spans 30 calendar dates
    assert (end - start).days + 1 == 30


def test_last_30_days_boundary_crosses_month_and_year():
    """Boundary case: the 30-day window crossing a year boundary must still
    resolve correctly (plain date arithmetic, not calendar-month logic)."""
    start, end = _resolve_relative_date(RelativeDate.LAST_30_DAYS, today=date(2027, 1, 5))
    assert end == date(2027, 1, 5)
    assert start == date(2026, 12, 7)
    assert (end - start).days + 1 == 30


def test_last_30_days_never_equals_this_month_or_last_month():
    """LAST_30_DAYS must be a genuinely distinct window from THIS_MONTH/
    LAST_MONTH, never silently collapsing to either -- a calendar month can
    be 28-31 days and start on a different day entirely."""
    last_30 = _resolve_relative_date(RelativeDate.LAST_30_DAYS, today=REFERENCE_DATE)
    this_month = _resolve_relative_date(RelativeDate.THIS_MONTH, today=REFERENCE_DATE)
    last_month = _resolve_relative_date(RelativeDate.LAST_MONTH, today=REFERENCE_DATE)
    assert last_30 != this_month
    assert last_30 != last_month


def test_last_30_days_never_equals_last_week():
    """The exact literal regression this fixes: "last 30 days" must never
    resolve to the same window as LAST_WEEK (7 days)."""
    last_30 = _resolve_relative_date(RelativeDate.LAST_30_DAYS, today=REFERENCE_DATE)
    last_week = _resolve_relative_date(RelativeDate.LAST_WEEK, today=REFERENCE_DATE)
    assert last_30 != last_week
    start, end = last_30
    assert (end - start).days + 1 == 30
    lw_start, lw_end = last_week
    assert (lw_end - lw_start).days + 1 == 7


# ── Every pre-existing RelativeDate member must remain byte-for-byte ────────
# unchanged -- pinned against fixed expected values so any accidental
# behavior change (not just LAST_30_DAYS's own correctness) is caught.

def test_today_unchanged():
    assert _resolve_relative_date(RelativeDate.TODAY, today=REFERENCE_DATE) == (REFERENCE_DATE, REFERENCE_DATE)


def test_this_week_unchanged():
    assert _resolve_relative_date(RelativeDate.THIS_WEEK, today=REFERENCE_DATE) == (date(2026, 8, 31), date(2026, 9, 6))


def test_last_week_unchanged():
    assert _resolve_relative_date(RelativeDate.LAST_WEEK, today=REFERENCE_DATE) == (date(2026, 8, 24), date(2026, 8, 30))


def test_this_month_unchanged():
    assert _resolve_relative_date(RelativeDate.THIS_MONTH, today=REFERENCE_DATE) == (date(2026, 9, 1), date(2026, 9, 30))


def test_last_month_unchanged():
    assert _resolve_relative_date(RelativeDate.LAST_MONTH, today=REFERENCE_DATE) == (date(2026, 8, 1), date(2026, 8, 31))


def test_this_year_unchanged():
    assert _resolve_relative_date(RelativeDate.THIS_YEAR, today=REFERENCE_DATE) == (date(2026, 1, 1), date(2026, 12, 31))


def test_last_year_unchanged():
    assert _resolve_relative_date(RelativeDate.LAST_YEAR, today=REFERENCE_DATE) == (date(2025, 1, 1), date(2025, 12, 31))


# ── YESTERDAY (P1, 2026-09-05) ──────────────────────────────────────────────

def test_yesterday_is_the_single_day_before_today():
    assert _resolve_relative_date(RelativeDate.YESTERDAY, today=REFERENCE_DATE) == (date(2026, 9, 1), date(2026, 9, 1))


def test_yesterday_crosses_month_boundary():
    assert _resolve_relative_date(RelativeDate.YESTERDAY, today=date(2026, 9, 1)) == (date(2026, 8, 31), date(2026, 8, 31))


def test_yesterday_crosses_year_boundary():
    assert _resolve_relative_date(RelativeDate.YESTERDAY, today=date(2027, 1, 1)) == (date(2026, 12, 31), date(2026, 12, 31))


def test_yesterday_never_equals_today():
    yesterday = _resolve_relative_date(RelativeDate.YESTERDAY, today=REFERENCE_DATE)
    today = _resolve_relative_date(RelativeDate.TODAY, today=REFERENCE_DATE)
    assert yesterday != today


# ── LAST_7_DAYS (P1, 2026-09-05) ─────────────────────────────────────────────

def test_last_7_days_is_a_true_rolling_7_day_window():
    """Today plus the preceding 6 days = exactly 7 distinct calendar dates,
    inclusive of today -- NOT the previous calendar Monday-Sunday."""
    start, end = _resolve_relative_date(RelativeDate.LAST_7_DAYS, today=REFERENCE_DATE)
    assert end == REFERENCE_DATE
    assert start == date(2026, 8, 27)
    assert (end - start).days + 1 == 7


def test_last_7_days_boundary_crosses_month_and_year():
    start, end = _resolve_relative_date(RelativeDate.LAST_7_DAYS, today=date(2027, 1, 5))
    assert end == date(2027, 1, 5)
    assert start == date(2026, 12, 30)
    assert (end - start).days + 1 == 7


def test_last_7_days_never_equals_last_week():
    """The exact literal regression this fixes: "last 7 days" must never
    resolve to the same window as LAST_WEEK -- LAST_WEEK is the previous
    calendar Monday-Sunday (excludes today entirely); LAST_7_DAYS is a
    rolling window ending today."""
    last_7 = _resolve_relative_date(RelativeDate.LAST_7_DAYS, today=REFERENCE_DATE)
    last_week = _resolve_relative_date(RelativeDate.LAST_WEEK, today=REFERENCE_DATE)
    assert last_7 != last_week
    start, end = last_7
    assert (end - start).days + 1 == 7


def test_last_7_days_never_equals_this_week():
    """THIS_WEEK is the current calendar Monday-Sunday -- a different window
    from a rolling 7-day-ending-today range whenever today isn't Sunday."""
    last_7 = _resolve_relative_date(RelativeDate.LAST_7_DAYS, today=REFERENCE_DATE)
    this_week = _resolve_relative_date(RelativeDate.THIS_WEEK, today=REFERENCE_DATE)
    assert last_7 != this_week


def test_all_time_raises_before_reaching_here():
    """ALL_TIME is filtered out by the caller (StructuredSQLBuilder.build)
    before this function is ever invoked -- confirms that contract still
    holds (raises rather than silently returning a bogus range)."""
    import pytest
    from datetime import date as _date
    with pytest.raises(ValueError):
        _resolve_relative_date(RelativeDate.ALL_TIME, today=REFERENCE_DATE)
