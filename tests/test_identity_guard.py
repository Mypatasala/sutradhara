import pytest

from src.policy.identity_guard import IdentityFilterGuard, IdentityFilterRejected
from src.policy.sanitizer import SQLSanitizer


# ── Safe stripping: reachable purely through AND ────────────────────────────

def test_strip_single_blocked_predicate_is_entire_where():
    sql = "SELECT first_name FROM users WHERE id = 'abc'"
    result = IdentityFilterGuard.strip(sql)
    assert "WHERE" not in result.upper()
    assert "abc" not in result


def test_strip_business_condition_and_blocked_predicate():
    sql = "SELECT first_name FROM users WHERE status = 'active' AND id = 'abc'"
    result = IdentityFilterGuard.strip(sql)
    assert "status = 'active'" in result
    assert "id = 'abc'" not in result
    assert "abc" not in result


def test_strip_parenthesized_and_combination():
    sql = "SELECT first_name FROM users WHERE (status = 'active' AND id = 'abc')"
    result = IdentityFilterGuard.strip(sql)
    assert "status = 'active'" in result
    assert "abc" not in result


def test_strip_multiple_blocked_predicates_joined_only_by_and():
    sql = "SELECT first_name FROM users WHERE id = 'abc' AND email = 'x@example.com'"
    result = IdentityFilterGuard.strip(sql)
    assert "WHERE" not in result.upper()
    assert "abc" not in result
    assert "x@example.com" not in result


def test_strip_alias_qualified_blocked_column():
    sql = "SELECT u.first_name FROM users u WHERE u.id = 'abc'"
    result = IdentityFilterGuard.strip(sql)
    assert "WHERE" not in result.upper()
    assert "abc" not in result


def test_strip_alias_qualified_within_and():
    sql = "SELECT u.first_name FROM users u WHERE u.status = 'active' AND u.id = 'abc'"
    result = IdentityFilterGuard.strip(sql)
    assert "status = 'active'" in result
    assert "abc" not in result


def test_strip_three_way_and_only_middle_blocked():
    sql = "SELECT first_name FROM users WHERE status = 'active' AND id = 'abc' AND department = 'sales'"
    result = IdentityFilterGuard.strip(sql)
    assert "status = 'active'" in result
    assert "department = 'sales'" in result
    assert "abc" not in result


def test_strip_no_blocked_predicate_returns_sql_unchanged():
    sql = "SELECT first_name FROM users WHERE status = 'active'"
    result = IdentityFilterGuard.strip(sql)
    assert result == sql


# ── Must reject: OR / nested OR / subquery / NOT / CASE / unsupported shapes ─

def test_reject_business_or_blocked():
    sql = "SELECT first_name FROM users WHERE status = 'active' OR id = 'abc'"
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_blocked_and_a_or_b():
    sql = "SELECT first_name FROM users WHERE (id = 'abc' AND status = 'active') OR department = 'sales'"
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_a_and_blocked_or_b():
    sql = "SELECT first_name FROM users WHERE status = 'active' AND (id = 'abc' OR department = 'sales')"
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_blocked_predicate_in_subquery():
    sql = (
        "SELECT first_name FROM users WHERE id IN "
        "(SELECT user_id FROM teacher_profiles WHERE id = 'abc')"
    )
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_blocked_predicate_negated():
    sql = "SELECT first_name FROM users WHERE NOT (id = 'abc')"
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_blocked_predicate_in_case():
    sql = (
        "SELECT first_name FROM users WHERE "
        "CASE WHEN id = 'abc' THEN true ELSE false END"
    )
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_blocked_predicate_in_join_on():
    sql = (
        "SELECT u.first_name FROM users u "
        "JOIN teacher_profiles tp ON tp.user_id = u.id AND u.id = 'abc' "
        "WHERE u.status = 'active'"
    )
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_blocked_predicate_in_having():
    sql = "SELECT department FROM users GROUP BY department HAVING id = 'abc'"
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_parse_failure_does_not_fall_back():
    sql = "SELECT FROM WHERE (((("
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_reject_non_select_top_level_shape_with_blocked_predicate():
    sql = (
        "WITH ranked AS (SELECT * FROM users) "
        "SELECT * FROM ranked WHERE id = 'abc'"
    )
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


# ── Must remain untouched: legitimate business predicates ───────────────────

def test_untouched_status_predicate():
    sql = "SELECT * FROM homework WHERE status = 'pending'"
    assert IdentityFilterGuard.strip(sql) == sql


def test_untouched_day_of_week_predicate():
    sql = "SELECT * FROM course_schedule WHERE day_of_week = 'Friday'"
    assert IdentityFilterGuard.strip(sql) == sql


def test_untouched_section_id_predicate():
    sql = "SELECT * FROM class_sections WHERE section_id = 5"
    # section_id is not on the blocklist (student_id/teacher_id/user_id/etc. are) -- untouched.
    assert IdentityFilterGuard.strip(sql) == sql


def test_untouched_course_id_predicate():
    sql = "SELECT * FROM course_schedule WHERE course_id = 7"
    assert IdentityFilterGuard.strip(sql) == sql


def test_untouched_identity_like_text_inside_string_literal():
    sql = "SELECT * FROM homework WHERE status = 'my id is 123'"
    assert IdentityFilterGuard.strip(sql) == sql


def test_untouched_few_shot_report_card_pattern():
    sql = (
        "SELECT term, academic_year, overall_grade, overall_percentage, "
        "class_teacher_name, remarks, issue_date FROM report_cards "
        "ORDER BY issue_date DESC LIMIT 1"
    )
    assert IdentityFilterGuard.strip(sql) == sql


def test_untouched_few_shot_timetable_pattern():
    sql = (
        "SELECT c.name, cs.start_time, cs.end_time, cs.room FROM course_schedule cs "
        "JOIN courses c ON cs.course_id = c.id WHERE cs.day_of_week = 'Friday' "
        "ORDER BY cs.start_time"
    )
    assert IdentityFilterGuard.strip(sql) == sql


def test_untouched_few_shot_aggregate_pattern():
    sql = "SELECT COUNT(*) FROM homework WHERE status = 'pending'"
    assert IdentityFilterGuard.strip(sql) == sql


# ── Additional required semantic tests ───────────────────────────────────────

def test_status_and_id_stripped():
    sql = "SELECT * FROM users WHERE status = 'active' AND id = 'x'"
    result = IdentityFilterGuard.strip(sql)
    assert "status = 'active'" in result
    assert "'x'" not in result


def test_status_or_id_rejected():
    sql = "SELECT * FROM users WHERE status = 'active' OR id = 'x'"
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


def test_grouped_or_status_and_id_stripped():
    # (status='active' OR status='pending') AND id='x'
    # The blocked leaf (id='x') is reachable from the WHERE root via a single
    # AND, with an OR living entirely on the OTHER side -- that OR contains
    # no blocked predicate itself, so it is left completely untouched, and
    # only the id='x' leaf is removed.
    sql = "SELECT * FROM users WHERE (status = 'active' OR status = 'pending') AND id = 'x'"
    result = IdentityFilterGuard.strip(sql)
    assert "status = 'active'" in result
    assert "status = 'pending'" in result
    assert "'x'" not in result


def test_ambiguous_case_fails_closed_not_broadened():
    """Proves the guard never silently broadens access: on an ambiguous shape
    it raises rather than returning ANY rewritten SQL (safe or otherwise)."""
    sql = "SELECT * FROM users WHERE status = 'active' AND (id = 'x' OR department = 'sales')"
    with pytest.raises(IdentityFilterRejected):
        IdentityFilterGuard.strip(sql)


# ── End-to-end: guard output feeding into the existing apply_constraints ────

def test_end_to_end_stripped_predicate_then_opa_filter_injected():
    sql = "SELECT id, email FROM users WHERE status = 'active' AND id = 'fabricated-uuid'"
    guarded = IdentityFilterGuard.strip(sql)
    final_sql = SQLSanitizer.apply_constraints(guarded, ["id", "email"], "school_id = 56")
    assert "fabricated-uuid" not in final_sql
    assert "status = 'active'" in final_sql
    assert "school_id = 56" in final_sql


def test_end_to_end_fully_stripped_where_then_opa_filter_via_no_where_path():
    sql = "SELECT id, email FROM users WHERE id = 'fabricated-uuid'"
    guarded = IdentityFilterGuard.strip(sql)
    assert "WHERE" not in guarded.upper()
    final_sql = SQLSanitizer.apply_constraints(guarded, ["id", "email"], "school_id = 56")
    assert "fabricated-uuid" not in final_sql
    assert " WHERE school_id = 56" in final_sql
    # Must use the plain "WHERE <filter>" path, not "AND (...)", since the
    # guard left no WHERE clause behind for apply_constraints to extend.
    assert "AND (school_id = 56)" not in final_sql


def test_end_to_end_opa_row_filter_never_touched_even_when_self_only():
    """OPA's own trusted row_filter can legitimately BE an identity-shaped
    equality predicate (e.g. a self-only role's `id = 'real-user-id'`) --
    the guard must never see or alter it, since it only ever runs on the
    LLM's own `sql` argument, never on row_filter."""
    sql = "SELECT id, email FROM users WHERE status = 'active'"
    guarded = IdentityFilterGuard.strip(sql)
    assert guarded == sql  # nothing to strip in the LLM's own SQL here
    opa_row_filter = "id = 'real-user-id'"
    final_sql = SQLSanitizer.apply_constraints(guarded, ["id", "email"], opa_row_filter)
    assert "id = 'real-user-id'" in final_sql
    assert "status = 'active'" in final_sql


def test_end_to_end_rejected_sql_never_reaches_apply_constraints():
    sql = "SELECT * FROM users WHERE status = 'active' OR id = 'fabricated-uuid'"
    with pytest.raises(IdentityFilterRejected):
        guarded = IdentityFilterGuard.strip(sql)
        SQLSanitizer.apply_constraints(guarded, [], "school_id = 56")  # unreachable
