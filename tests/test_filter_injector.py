import pytest

from src.policy.filter_injector import AliasAwareFilterInjector, FilterInjectionRejected
from src.policy.sanitizer import SQLSanitizer


# ── Alias resolution: successful cases ──────────────────────────────────────

def test_unaliased_target_table():
    sql = "SELECT name, COUNT(*) AS class_size FROM class_sections JOIN students ON class_sections.id = students.section_id GROUP BY class_sections.id"
    result = AliasAwareFilterInjector.inject(sql, "id IN (SELECT id FROM class_sections WHERE school_id = 56)", "class_sections")
    assert result == "class_sections.id IN (SELECT id FROM class_sections WHERE school_id = 56)"


def test_aliased_target_table():
    sql = "SELECT c.name, COUNT(*) AS class_size FROM class_sections c JOIN students s ON c.id = s.section_id GROUP BY c.id"
    result = AliasAwareFilterInjector.inject(sql, "id IN (SELECT id FROM class_sections WHERE school_id = 56)", "class_sections")
    assert result == "c.id IN (SELECT id FROM class_sections WHERE school_id = 56)"


def test_three_way_join_resolves_correct_table():
    sql = (
        "SELECT * FROM students s "
        "JOIN class_sections cs ON s.section_id = cs.id "
        "JOIN school_classes sc ON cs.school_class_id = sc.id"
    )
    result = AliasAwareFilterInjector.inject(sql, "id IN (SELECT id FROM school_classes WHERE school_id = 56)", "school_classes")
    assert result == "sc.id IN (SELECT id FROM school_classes WHERE school_id = 56)"


def test_bare_column_filter_qualified():
    sql = "SELECT * FROM students s WHERE s.status = 'active'"
    result = AliasAwareFilterInjector.inject(sql, "school_id = 56", "students")
    assert result == "s.school_id = 56"


def test_self_only_literal_filter_qualified():
    sql = "SELECT id, email FROM users u"
    result = AliasAwareFilterInjector.inject(sql, "id = 'real-user-id'", "users")
    assert result == "u.id = 'real-user-id'"


def test_composite_or_filter_both_disjuncts_qualified():
    sql = "SELECT * FROM homework hw"
    row_filter = (
        "(student_id IN (SELECT id FROM students WHERE school_id = 56) "
        "OR course_id IN (SELECT c.id FROM courses c JOIN class_sections cs ON c.section_id = cs.id WHERE cs.school_id = 56))"
    )
    result = AliasAwareFilterInjector.inject(sql, row_filter, "homework")
    assert "hw.student_id IN" in result
    assert "hw.course_id IN" in result
    # subquery internals must stay untouched -- still reference their own
    # real table names, not "hw."
    assert "SELECT id FROM students WHERE school_id = 56" in result
    assert "courses AS c JOIN class_sections AS cs" in result or "courses c JOIN class_sections cs" in result


def test_already_qualified_column_left_untouched():
    sql = "SELECT * FROM users u"
    result = AliasAwareFilterInjector.inject(sql, "u.school_id = 56", "users")
    assert result == "u.school_id = 56"


def test_empty_filter_returns_unchanged_without_parsing():
    sql = "SELECT * FROM users u"
    result = AliasAwareFilterInjector.inject(sql, "", "users")
    assert result == ""


# ── Fail-closed cases ────────────────────────────────────────────────────────

def test_reject_self_join_ambiguous():
    sql = "SELECT * FROM students s1 JOIN students s2 ON s1.id = s2.id"
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject(sql, "school_id = 56", "students")


def test_reject_target_table_absent_from_outer_scope():
    sql = "SELECT * FROM users WHERE status = 'active'"
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject(sql, "school_id = 56", "class_sections")


def test_reject_target_table_only_inside_nested_subquery():
    sql = "SELECT * FROM students s WHERE s.section_id IN (SELECT id FROM class_sections WHERE name = 'A')"
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject(sql, "id IN (SELECT id FROM class_sections WHERE school_id = 56)", "class_sections")


def test_reject_malformed_sql():
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject("SELECT FROM WHERE ((((", "school_id = 56", "students")


def test_reject_malformed_row_filter():
    sql = "SELECT * FROM students s"
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject(sql, "school_id = ", "students")


def test_reject_non_select_top_level_shape():
    sql = "WITH ranked AS (SELECT * FROM users) SELECT * FROM ranked"
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject(sql, "school_id = 56", "ranked")


def test_reject_derived_table_in_from_position():
    sql = "SELECT * FROM (SELECT * FROM class_sections) AS sub"
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject(sql, "id IN (SELECT id FROM class_sections WHERE school_id = 56)", "class_sections")


# ── End-to-end: qualified filter feeding into the existing apply_constraints ─

def test_end_to_end_class_sections_aliased_join_no_longer_ambiguous():
    """Reproduces the exact live failure: llama3.2 aliased class_sections as
    'c' when asked 'how many students are in each class', producing
    'Column school_id in WHERE is ambiguous' with the old bare-filter
    approach. Proves the full pipeline (inject -> apply_constraints) now
    produces valid, unambiguous SQL."""
    sql = (
        "SELECT c.name AS class_name, COUNT(*) AS class_size "
        "FROM class_sections c JOIN students s ON c.id = s.section_id "
        "GROUP BY c.id, c.name"
    )
    row_filter = "id IN (SELECT id FROM class_sections WHERE school_id = 56)"
    qualified = AliasAwareFilterInjector.inject(sql, row_filter, "class_sections")
    final_sql = SQLSanitizer.apply_constraints(sql, [], qualified)

    assert "WHERE c.id IN (SELECT id FROM class_sections WHERE school_id = 56)" in final_sql
    # the ambiguous bare form must never appear
    assert "WHERE id IN" not in final_sql


def test_end_to_end_class_sections_grouped_by_id_keeps_repeated_names_distinct():
    """Regression for a result-shaping bug found after the alias fix:
    class_sections.name alone ("A", "B") repeats across different grades, so
    a query that joins in school_classes for a combined label but still
    GROUPs BY the real per-section id (not just the name) must keep every
    section as its own row -- proven here structurally (the GROUP BY key
    includes class_sections.id, not just the human-readable name columns),
    matching the corrected few-shot in intent_agent.py. Verified separately
    against real seed data (school_id=56, MariaDB) that this exact query
    returns 9 distinct labelled rows, e.g. "3rd Grade - A" and "1st Grade -
    A", never collapsing same-lettered sections from different grades."""
    sql = (
        "SELECT CONCAT(school_classes.name, ' - ', class_sections.name) AS class_name, "
        "COUNT(*) AS class_size FROM class_sections "
        "JOIN school_classes ON class_sections.school_class_id = school_classes.id "
        "JOIN students ON students.section_id = class_sections.id "
        "GROUP BY class_sections.id, school_classes.name, class_sections.name"
    )
    row_filter = "id IN (SELECT id FROM class_sections WHERE school_id = 56)"
    qualified = AliasAwareFilterInjector.inject(sql, row_filter, "class_sections")
    final_sql = SQLSanitizer.apply_constraints(sql, [], qualified)

    assert "WHERE class_sections.id IN (SELECT id FROM class_sections WHERE school_id = 56)" in final_sql
    assert "CONCAT(school_classes.name, ' - ', class_sections.name)" in final_sql
    # the GROUP BY key must include the real per-section id, not just the
    # (repeatable) name columns -- otherwise two different "A" sections in
    # different grades would collapse into a single aggregated row.
    assert "GROUP BY class_sections.id, school_classes.name, class_sections.name" in final_sql


def test_end_to_end_unaliased_join_also_qualified_correctly():
    sql = (
        "SELECT class_sections.name AS class_name, COUNT(*) AS class_size "
        "FROM class_sections JOIN students ON class_sections.id = students.section_id "
        "GROUP BY class_sections.id, class_sections.name"
    )
    row_filter = "id IN (SELECT id FROM class_sections WHERE school_id = 56)"
    qualified = AliasAwareFilterInjector.inject(sql, row_filter, "class_sections")
    final_sql = SQLSanitizer.apply_constraints(sql, [], qualified)

    assert "WHERE class_sections.id IN (SELECT id FROM class_sections WHERE school_id = 56)" in final_sql


def test_end_to_end_nested_subquery_only_target_rejection():
    sql = "SELECT * FROM students s WHERE s.section_id IN (SELECT id FROM class_sections WHERE name = 'A')"
    with pytest.raises(FilterInjectionRejected):
        AliasAwareFilterInjector.inject(sql, "id IN (SELECT id FROM class_sections WHERE school_id = 56)", "class_sections")


def test_end_to_end_subquery_filter_qualification_preserves_inner_scope():
    sql = "SELECT hw.* FROM homework hw"
    row_filter = "student_id IN (SELECT id FROM students WHERE school_id = 56)"
    qualified = AliasAwareFilterInjector.inject(sql, row_filter, "homework")
    final_sql = SQLSanitizer.apply_constraints(sql, [], qualified)

    assert "hw.student_id IN (SELECT id FROM students WHERE school_id = 56)" in final_sql


def test_end_to_end_self_only_literal_filter_still_works():
    sql = "SELECT id, email FROM users u WHERE u.status = 'active'"
    qualified = AliasAwareFilterInjector.inject(sql, "id = 'real-user-id'", "users")
    final_sql = SQLSanitizer.apply_constraints(sql, [], qualified)

    assert "u.id = 'real-user-id'" in final_sql
    assert "u.status = 'active'" in final_sql


def test_end_to_end_superuser_empty_filter_unaffected():
    sql = "SELECT * FROM users u"
    qualified = AliasAwareFilterInjector.inject(sql, "", "users")
    final_sql = SQLSanitizer.apply_constraints(sql, [], qualified)

    assert final_sql == sql  # nothing injected at all, exactly as before


def test_end_to_end_teacher_profiles_outer_column_qualified_subquery_untouched():
    """Exact production filter shape (admin/principal/teacher.rego's
    teacher_profiles rule): the outer column (user_id, belonging to
    teacher_profiles itself) must be alias-qualified, while the nested
    subquery -- which targets a DIFFERENT table (users) than the one being
    authorized -- must remain completely untouched."""
    sql = "SELECT tp.designation FROM teacher_profiles tp"
    row_filter = "user_id IN (SELECT id FROM users WHERE school_id = 56)"
    qualified = AliasAwareFilterInjector.inject(sql, row_filter, "teacher_profiles")
    final_sql = SQLSanitizer.apply_constraints(sql, [], qualified)

    assert "WHERE tp.user_id IN (SELECT id FROM users WHERE school_id = 56)" in final_sql
    # the nested subquery's own columns must stay bare, never qualified with
    # the outer alias -- they belong to `users`, not `teacher_profiles`
    assert "tp.id" not in final_sql
    assert "tp.school_id" not in final_sql
