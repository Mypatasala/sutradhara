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


def test_courses_section_id_subquery_filter_qualified():
    """COURSES Phase 1 (2026-09-10): my_patasala's OPA policy authorizes
    "courses" via a bare-column subquery filter identical across
    admin/teacher/student/parent.rego -- proves the already-generic
    AliasAwareFilterInjector qualifies it correctly for target_table=
    "courses" too, with no injector code change required. Test coverage
    only, per the COURSES Phase 1 authorization requirement."""
    sql = "SELECT COUNT(*) AS count FROM courses"
    row_filter = "section_id IN (SELECT id FROM class_sections WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "courses")
    assert result == "courses.section_id IN (SELECT id FROM class_sections WHERE school_id = 56)"
    # subquery internals must stay untouched -- still reference their own
    # real table name, not "courses."
    assert "SELECT id FROM class_sections WHERE school_id = 56" in result


def test_courses_section_id_subquery_filter_qualified_with_subject_where():
    """COURSES.SUBJECT (2026-09-11): re-runs the same admin/teacher/
    student/parent.rego shape now against SQL that already has its own
    SUBJECT WHERE clause, proving the added filter does not change the
    injector's alias-resolution target at all -- the same
    section_id-anchored authorization is unaffected."""
    sql = "SELECT COUNT(*) AS count FROM courses WHERE courses.name = 'Mathematics'"
    row_filter = "section_id IN (SELECT id FROM class_sections WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "courses")
    assert result == "courses.section_id IN (SELECT id FROM class_sections WHERE school_id = 56)"
    assert "SELECT id FROM class_sections WHERE school_id = 56" in result


def test_report_cards_student_id_subquery_filter_qualified_with_own_where_clause():
    """REPORT_CARDS TERM filter (Phase 1, 2026-09-10): this is the first
    time REPORT_CARDS gets a filter-produced WHERE clause of its own
    (report_cards.term = '...') composed alongside the injected
    authorization filter -- no existing test exercised report_cards' own
    OPA filter shape ("student_id IN (SELECT id FROM students WHERE
    school_id = %v)", identical across admin/teacher/student/parent/
    principal.rego) at all before this. Proves the already-generic
    AliasAwareFilterInjector qualifies it correctly for target_table=
    "report_cards" against SQL that already has its own WHERE clause, with
    no injector code change required."""
    sql = "SELECT COUNT(*) AS count FROM report_cards WHERE report_cards.term = 'Term 2'"
    row_filter = "student_id IN (SELECT id FROM students WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "report_cards")
    assert result == "report_cards.student_id IN (SELECT id FROM students WHERE school_id = 56)"
    # subquery internals must stay untouched -- still reference their own
    # real table name, not "report_cards."
    assert "SELECT id FROM students WHERE school_id = 56" in result


def test_report_cards_academic_year_and_term_combined_filter_qualified():
    """REPORT_CARDS ACADEMIC_YEAR filter (Phase 2, 2026-09-11): proves the
    already-generic AliasAwareFilterInjector still correctly qualifies
    report_cards' own OPA filter shape when the outer SQL already has TWO
    of its own WHERE predicates (term AND academic_year), not just one --
    no injector code change required."""
    sql = (
        "SELECT COUNT(*) AS count FROM report_cards "
        "WHERE report_cards.academic_year = '2025-2026' AND report_cards.term = 'Term 2'"
    )
    row_filter = "student_id IN (SELECT id FROM students WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "report_cards")
    assert result == "report_cards.student_id IN (SELECT id FROM students WHERE school_id = 56)"
    # subquery internals must stay untouched -- still reference their own
    # real table name, not "report_cards."
    assert "SELECT id FROM students WHERE school_id = 56" in result


def test_composite_or_filter_both_disjuncts_qualified_for_assignments():
    """ASSIGNMENTS Phase 1 (2026-09-08): my_patasala's OPA policy applies
    this exact OR-shaped filter identically to {"homework", "assignments"}
    (see admin/teacher/student/parent.rego) -- proves the already-generic
    AliasAwareFilterInjector qualifies it correctly for target_table=
    "assignments" too, with no injector code change required. Test coverage
    only, per the ASSIGNMENTS Phase 1 authorization requirement."""
    sql = "SELECT * FROM assignments a"
    row_filter = (
        "(student_id IN (SELECT id FROM students WHERE school_id = 56) "
        "OR course_id IN (SELECT c.id FROM courses c JOIN class_sections cs ON c.section_id = cs.id WHERE cs.school_id = 56))"
    )
    result = AliasAwareFilterInjector.inject(sql, row_filter, "assignments")
    assert "a.student_id IN" in result
    assert "a.course_id IN" in result
    # subquery internals must stay untouched -- still reference their own
    # real table names, not "a."
    assert "SELECT id FROM students WHERE school_id = 56" in result
    assert "courses AS c JOIN class_sections AS cs" in result or "courses c JOIN class_sections cs" in result


def test_composite_or_filter_qualified_for_assignments_with_subject_join_and_where():
    """ASSIGNMENTS SUBJECT filter (Phase 2, 2026-09-10): this is the first
    time an ASSIGNMENTS query has its OWN JOIN (to courses, for the SUBJECT
    filter) and its OWN WHERE clause (courses.name = '...') already present
    when the authorization filter is injected -- proves the already-generic
    AliasAwareFilterInjector still correctly resolves "assignments" as the
    target alias (distinguishing it from the ALSO-present "courses" table
    in the same FROM/JOIN scope) and still correctly qualifies BOTH
    disjuncts to assignments' own columns, leaving the courses/class_sections
    subquery internals -- and the pre-existing SUBJECT WHERE clause -- fully
    untouched. Test coverage only, no injector code change required."""
    sql = (
        "SELECT COUNT(*) AS count FROM assignments JOIN courses ON assignments.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    row_filter = (
        "(student_id IN (SELECT id FROM students WHERE school_id = 56) "
        "OR course_id IN (SELECT c.id FROM courses c JOIN class_sections cs ON c.section_id = cs.id WHERE cs.school_id = 56))"
    )
    result = AliasAwareFilterInjector.inject(sql, row_filter, "assignments")
    assert "assignments.student_id IN" in result
    assert "assignments.course_id IN" in result
    # subquery internals must stay untouched -- still reference their own
    # real table names, not "assignments."
    assert "SELECT id FROM students WHERE school_id = 56" in result
    assert (
        "courses AS c JOIN class_sections AS cs" in result
        or "courses c JOIN class_sections cs" in result
    )


def test_student_rego_assignments_filter_qualified_with_subject_join_and_where():
    """ASSIGNMENTS SUBJECT filter x student.rego (2026-09-10 Principal
    Engineer review follow-up): student.rego's ASSIGNMENTS filter uses a
    genuinely DIFFERENT shape from admin/teacher.rego -- a direct equality
    ("student_id = '%v'", not an IN-subquery) plus a nested "students s"
    join for the course-side disjunct (the actual current text of
    student.rego, verbatim, not an approximation). Proves the already-
    generic AliasAwareFilterInjector still correctly resolves "assignments"
    as the target alias against SQL that already has its own courses JOIN
    and courses.name WHERE clause, still correctly qualifies BOTH
    disjuncts to assignments' own columns, and leaves the nested subquery
    internals (including the "students s" join it introduces) fully
    untouched. Test coverage only, no injector code change required."""
    sql = (
        "SELECT COUNT(*) AS count FROM assignments JOIN courses ON assignments.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    row_filter = (
        "(student_id = '11111111-1111-1111-1111-111111111111' "
        "OR course_id IN (SELECT c.id FROM courses c JOIN class_sections cs ON c.section_id = cs.id "
        "JOIN students s ON s.section_id = cs.id WHERE s.id = '11111111-1111-1111-1111-111111111111'))"
    )
    result = AliasAwareFilterInjector.inject(sql, row_filter, "assignments")
    assert result == (
        "(assignments.student_id = '11111111-1111-1111-1111-111111111111' "
        "OR assignments.course_id IN (SELECT c.id FROM courses AS c JOIN class_sections AS cs "
        "ON c.section_id = cs.id JOIN students AS s ON s.section_id = cs.id "
        "WHERE s.id = '11111111-1111-1111-1111-111111111111'))"
    )
    # the pre-existing SUBJECT filter's own JOIN/WHERE clause on `sql` is
    # never touched by inject() at all (it only ever returns the qualified
    # row_filter fragment) -- confirmed by construction, not re-asserted
    # here as a no-op check on the untouched input string.


def test_parent_rego_assignments_filter_qualified_with_subject_join_and_where():
    """ASSIGNMENTS SUBJECT filter x parent.rego (2026-09-10 Principal
    Engineer review follow-up): parent.rego's ASSIGNMENTS filter uses yet
    another genuinely different shape from admin/teacher.rego -- both
    disjuncts route through a "guardians_legacy" lookup by email (the
    actual current text of parent.rego, verbatim, not an approximation).
    Proves the already-generic AliasAwareFilterInjector still correctly
    resolves "assignments" as the target alias against SQL that already
    has its own courses JOIN and courses.name WHERE clause, still
    correctly qualifies BOTH disjuncts to assignments' own columns, and
    leaves every nested subquery (including the two separate
    guardians_legacy lookups) fully untouched. Test coverage only, no
    injector code change required."""
    sql = (
        "SELECT COUNT(*) AS count FROM assignments JOIN courses ON assignments.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    row_filter = (
        "(student_id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com') "
        "OR course_id IN (SELECT c.id FROM courses c JOIN class_sections cs ON c.section_id = cs.id "
        "JOIN students s ON s.section_id = cs.id "
        "WHERE s.id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com')))"
    )
    result = AliasAwareFilterInjector.inject(sql, row_filter, "assignments")
    assert result == (
        "(assignments.student_id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com') "
        "OR assignments.course_id IN (SELECT c.id FROM courses AS c JOIN class_sections AS cs "
        "ON c.section_id = cs.id JOIN students AS s ON s.section_id = cs.id "
        "WHERE s.id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com')))"
    )
    # both guardians_legacy lookups (outer disjunct and nested inside the
    # course-side subquery) must stay untouched -- still reference their
    # own real table name, not "assignments."
    assert result.count("SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com'") == 2
    # the pre-existing SUBJECT filter's own JOIN/WHERE clause on `sql` is
    # never touched by inject() at all (it only ever returns the qualified
    # row_filter fragment).


def test_admin_teacher_principal_examinations_filter_qualified_with_subject_join_and_where():
    """EXAMINATIONS SUBJECT filter x admin/teacher/principal.rego (Phase 1,
    2026-09-11): the actual current examinations filter text, verbatim --
    a single bare student_id subquery, NOT an OR-shaped filter (unlike
    ASSIGNMENTS/HOMEWORK), since examinations.student_id is NOT NULL and
    the policy therefore needs no course-side fallback disjunct at all.
    Proves the already-generic AliasAwareFilterInjector still correctly
    resolves "examinations" as the target alias against SQL that already
    has its own courses JOIN and courses.name WHERE clause, and still
    correctly qualifies the bare column to examinations' own. Test
    coverage only, no injector code change required."""
    sql = (
        "SELECT COUNT(*) AS count FROM examinations JOIN courses ON examinations.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    row_filter = "student_id IN (SELECT id FROM students WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "examinations")
    assert result == "examinations.student_id IN (SELECT id FROM students WHERE school_id = 56)"
    # subquery internals must stay untouched -- still reference their own
    # real table name, not "examinations."
    assert "SELECT id FROM students WHERE school_id = 56" in result


def test_student_rego_examinations_filter_qualified_with_subject_join_and_where():
    """EXAMINATIONS SUBJECT filter x student.rego (Phase 1, 2026-09-11):
    the actual current student.rego examinations filter text, verbatim --
    a direct equality against the caller's own id, no subquery at all."""
    sql = (
        "SELECT COUNT(*) AS count FROM examinations JOIN courses ON examinations.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    row_filter = "student_id = '11111111-1111-1111-1111-111111111111'"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "examinations")
    assert result == "examinations.student_id = '11111111-1111-1111-1111-111111111111'"


def test_parent_rego_examinations_filter_qualified_with_subject_join_and_where():
    """EXAMINATIONS SUBJECT filter x parent.rego (Phase 1, 2026-09-11): the
    actual current parent.rego examinations filter text, verbatim -- a
    guardians_legacy lookup by email."""
    sql = (
        "SELECT COUNT(*) AS count FROM examinations JOIN courses ON examinations.course_id = courses.id "
        "WHERE courses.name = 'Mathematics'"
    )
    row_filter = "student_id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com')"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "examinations")
    assert result == (
        "examinations.student_id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com')"
    )
    # subquery internals must stay untouched -- still reference their own
    # real table name, not "examinations."
    assert "SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com'" in result


def test_admin_principal_absence_requests_filter_qualified_with_status_where():
    """ABSENCE_REQUESTS Phase 1 (2026-09-11): the actual current
    admin/principal.rego absence_requests filter text, verbatim -- the same
    single bare student_id subquery shape already proven for attendance/
    examinations/report_cards. Proves the already-generic
    AliasAwareFilterInjector still correctly resolves "absence_requests" as
    the target alias against SQL that already has its own STATUS WHERE
    clause. Test coverage only, no injector code change required."""
    sql = "SELECT COUNT(*) AS count FROM absence_requests WHERE absence_requests.status = 'pending'"
    row_filter = "student_id IN (SELECT id FROM students WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "absence_requests")
    assert result == "absence_requests.student_id IN (SELECT id FROM students WHERE school_id = 56)"
    assert "SELECT id FROM students WHERE school_id = 56" in result


def test_student_rego_absence_requests_filter_qualified():
    """ABSENCE_REQUESTS Phase 1 (2026-09-11): the actual current
    student.rego absence_requests filter text, verbatim -- direct equality
    against the caller's own id."""
    sql = "SELECT COUNT(*) AS count FROM absence_requests WHERE absence_requests.status = 'pending'"
    row_filter = "student_id = '11111111-1111-1111-1111-111111111111'"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "absence_requests")
    assert result == "absence_requests.student_id = '11111111-1111-1111-1111-111111111111'"


def test_parent_rego_absence_requests_filter_qualified():
    """ABSENCE_REQUESTS Phase 1 (2026-09-11): the actual current
    parent.rego absence_requests filter text, verbatim -- a
    guardians_legacy lookup by email."""
    sql = "SELECT COUNT(*) AS count FROM absence_requests WHERE absence_requests.status = 'pending'"
    row_filter = "student_id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com')"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "absence_requests")
    assert result == (
        "absence_requests.student_id IN (SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com')"
    )
    assert "SELECT student_id FROM guardians_legacy WHERE email = 'parent@example.com'" in result


def test_teacher_rego_absence_requests_nested_ownership_filter_qualified():
    """ABSENCE_REQUESTS Phase 1 (2026-09-11): teacher.rego's
    absence_requests filter is the actual current text, verbatim -- and is
    genuinely DIFFERENT from admin/principal's own shape (the first entity
    in this registry where the teacher authorization shape diverges from
    admin/principal). It reflects AttendanceController.
    getAbsenceRequestsByStatus's real per-teacher ownership check (routes
    through getAbsenceRequestsByStatusForTeacher, scoped to the caller's
    own sections only): a NESTED subquery (absence_requests -> students ->
    class_sections) checking primary_teacher_id OR secondary_teacher_id.
    Proves the already-generic AliasAwareFilterInjector still correctly
    resolves "absence_requests" as the target alias and qualifies ONLY the
    outermost bare student_id column to absence_requests.student_id,
    leaving BOTH nested subquery levels (students, then class_sections)
    fully untouched -- still referencing their own real table names, never
    "absence_requests.". Test coverage only, no injector or OPA code
    change required."""
    sql = "SELECT COUNT(*) AS count FROM absence_requests WHERE absence_requests.status = 'pending'"
    row_filter = (
        "student_id IN (SELECT id FROM students WHERE section_id IN (SELECT id FROM class_sections "
        "WHERE primary_teacher_id = '22222222-2222-2222-2222-222222222222' "
        "OR secondary_teacher_id = '22222222-2222-2222-2222-222222222222'))"
    )
    result = AliasAwareFilterInjector.inject(sql, row_filter, "absence_requests")
    assert result == (
        "absence_requests.student_id IN (SELECT id FROM students WHERE section_id IN "
        "(SELECT id FROM class_sections WHERE primary_teacher_id = "
        "'22222222-2222-2222-2222-222222222222' OR secondary_teacher_id = "
        "'22222222-2222-2222-2222-222222222222'))"
    )
    # both nested subquery levels must stay untouched -- still reference
    # their own real table names, not "absence_requests."
    assert "FROM students WHERE section_id IN" in result
    assert "FROM class_sections WHERE primary_teacher_id" in result


def test_admin_principal_teacher_profiles_filter_qualified_with_employment_type_where():
    """TEACHER_PROFILES Phase 1 (2026-09-11): the actual current
    admin.rego/principal.rego teacher_profiles filter text, verbatim --
    teacher_profiles has no school_id column of its own, so authorization
    joins through users via user_id. Proves the already-generic
    AliasAwareFilterInjector still correctly resolves "teacher_profiles" as
    the target alias against SQL that already has its own EMPLOYMENT_TYPE
    WHERE clause. Test coverage only, no injector code change required."""
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.employment_type = 'FULL_TIME'"
    )
    row_filter = "user_id IN (SELECT id FROM users WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "teacher_profiles")
    assert result == "teacher_profiles.user_id IN (SELECT id FROM users WHERE school_id = 56)"
    assert "SELECT id FROM users WHERE school_id = 56" in result


def test_teacher_rego_teacher_profiles_self_only_filter_qualified():
    """TEACHER_PROFILES Phase 1 (2026-09-11): the actual current
    teacher.rego teacher_profiles filter text, verbatim -- self-only, no
    same-school admin-level bypass (per teacher.rego's own comment: a
    teacher can only ever be "the current user" for their own id)."""
    sql = "SELECT COUNT(*) AS count FROM teacher_profiles"
    row_filter = "user_id = '22222222-2222-2222-2222-222222222222'"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "teacher_profiles")
    assert result == "teacher_profiles.user_id = '22222222-2222-2222-2222-222222222222'"


def test_superuser_teacher_profiles_empty_filter_remains_unfiltered_no_op():
    """TEACHER_PROFILES Phase 1 (2026-09-11): superuser.rego's
    teacher_profiles filter is the actual current text, verbatim --
    empty/unfiltered (same no-op shape already proven safe for USERS).
    Proves the injector's documented empty-filter no-op behavior (returns
    the filter unchanged -- nothing to qualify) applies here too, without
    accidentally injecting a WHERE clause that isn't authorized."""
    sql = "SELECT teacher_profiles.designation, teacher_profiles.department FROM teacher_profiles"
    result = AliasAwareFilterInjector.inject(sql, "", "teacher_profiles")
    assert result == ""


def test_parent_and_student_teacher_profiles_default_deny_sentinel_unqualified():
    """TEACHER_PROFILES Phase 1 (2026-09-11): the first entity in this
    registry with two roles unconditionally denied at once -- confirmed by
    directly re-reading parent.rego/student.rego, both of which have no
    "teacher_profiles" rule at all and therefore fall through to each
    file's own `default decision` ("authorized": false, "filter": "1=0").
    Since an unauthorized decision never reaches the SQL pipeline in the
    first place, there is no qualified SQL to assert for these roles --
    what IS testable and matters here is that the OPA default-deny
    sentinel itself ("1=0", a literal boolean expression with no bare
    column) is not accidentally rewritten by the generic injector into
    something that could ever resolve to true. This proves the deny
    semantics are preserved even if the sentinel were ever passed through
    by mistake, without needing to invoke OPA itself in this deterministic
    test."""
    sql = "SELECT teacher_profiles.designation, teacher_profiles.department FROM teacher_profiles"
    result = AliasAwareFilterInjector.inject(sql, "1=0", "teacher_profiles")
    assert result == "1 = 0"


# ── TEACHER_PROFILES DEPARTMENT lookup filter (2026-09-11) -- re-runs the
# same six authorization shapes now against SQL that already has its own
# DEPARTMENT WHERE clause, proving the added filter does not change the
# injector's alias-resolution target at all.

def test_admin_principal_teacher_profiles_filter_qualified_with_department_where():
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.department = 'Mathematics'"
    )
    row_filter = "user_id IN (SELECT id FROM users WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "teacher_profiles")
    assert result == "teacher_profiles.user_id IN (SELECT id FROM users WHERE school_id = 56)"
    assert "SELECT id FROM users WHERE school_id = 56" in result


def test_teacher_rego_teacher_profiles_self_only_filter_qualified_with_department_where():
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.department = 'Mathematics'"
    )
    row_filter = "user_id = '22222222-2222-2222-2222-222222222222'"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "teacher_profiles")
    assert result == "teacher_profiles.user_id = '22222222-2222-2222-2222-222222222222'"


def test_superuser_teacher_profiles_department_filtered_query_remains_unfiltered_no_op():
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.department = 'Mathematics'"
    )
    result = AliasAwareFilterInjector.inject(sql, "", "teacher_profiles")
    assert result == ""


def test_parent_and_student_teacher_profiles_department_filtered_default_deny_sentinel_unqualified():
    """Confirms the OPA default-deny sentinel is unaffected by the presence
    of a DEPARTMENT WHERE clause in the main query -- parent/student remain
    denied regardless of which filter the query would have applied."""
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.department = 'Mathematics'"
    )
    result = AliasAwareFilterInjector.inject(sql, "1=0", "teacher_profiles")
    assert result == "1 = 0"


# ── TEACHER_PROFILES DESIGNATION lookup filter (2026-09-11) -- re-runs the
# same six authorization shapes now against SQL that already has its own
# DESIGNATION WHERE clause, proving the added filter does not change the
# injector's alias-resolution target at all.

def test_admin_principal_teacher_profiles_filter_qualified_with_designation_where():
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.designation = 'Head Teacher'"
    )
    row_filter = "user_id IN (SELECT id FROM users WHERE school_id = 56)"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "teacher_profiles")
    assert result == "teacher_profiles.user_id IN (SELECT id FROM users WHERE school_id = 56)"
    assert "SELECT id FROM users WHERE school_id = 56" in result


def test_teacher_rego_teacher_profiles_self_only_filter_qualified_with_designation_where():
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.designation = 'Head Teacher'"
    )
    row_filter = "user_id = '22222222-2222-2222-2222-222222222222'"
    result = AliasAwareFilterInjector.inject(sql, row_filter, "teacher_profiles")
    assert result == "teacher_profiles.user_id = '22222222-2222-2222-2222-222222222222'"


def test_superuser_teacher_profiles_designation_filtered_query_remains_unfiltered_no_op():
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.designation = 'Head Teacher'"
    )
    result = AliasAwareFilterInjector.inject(sql, "", "teacher_profiles")
    assert result == ""


def test_parent_and_student_teacher_profiles_designation_filtered_default_deny_sentinel_unqualified():
    """Confirms the OPA default-deny sentinel is unaffected by the presence
    of a DESIGNATION WHERE clause in the main query -- parent/student
    remain denied regardless of which filter the query would have
    applied."""
    sql = (
        "SELECT COUNT(*) AS count FROM teacher_profiles "
        "WHERE teacher_profiles.designation = 'Head Teacher'"
    )
    result = AliasAwareFilterInjector.inject(sql, "1=0", "teacher_profiles")
    assert result == "1 = 0"


# ── USERS DEPARTMENT lookup filter (2026-09-11) -- re-runs the applicable
# USERS authorization shapes against SQL that already has its own
# DEPARTMENT WHERE clause, proving the added filter does not change the
# injector's alias-resolution target at all.

def test_admin_principal_users_filter_qualified_with_department_where():
    sql = "SELECT COUNT(*) AS count FROM users WHERE users.department = 'Mathematics'"
    result = AliasAwareFilterInjector.inject(sql, "school_id = 56", "users")
    assert result == "users.school_id = 56"


def test_teacher_and_parent_rego_users_self_only_filter_qualified_with_department_where():
    sql = "SELECT COUNT(*) AS count FROM users WHERE users.department = 'Mathematics'"
    result = AliasAwareFilterInjector.inject(sql, "id = '22222222-2222-2222-2222-222222222222'", "users")
    assert result == "users.id = '22222222-2222-2222-2222-222222222222'"


def test_superuser_users_department_filtered_query_remains_unfiltered_no_op():
    sql = "SELECT COUNT(*) AS count FROM users WHERE users.department = 'Mathematics'"
    result = AliasAwareFilterInjector.inject(sql, "", "users")
    assert result == ""


# ── USERS NAME sort (2026-09-11) -- re-runs the same authorization shapes
# now against SQL that already has its own ORDER BY clause, proving the
# added sort does not change the injector's alias-resolution target at all
# (no WHERE clause is added by a sort, but the existing display columns
# and ORDER BY must remain intact around the injected filter).

def test_admin_principal_users_filter_qualified_with_name_sort():
    sql = (
        "SELECT users.first_name, users.last_name, users.email, users.phone, users.department "
        "FROM users ORDER BY users.last_name ASC"
    )
    result = AliasAwareFilterInjector.inject(sql, "school_id = 56", "users")
    assert result == "users.school_id = 56"


def test_teacher_and_parent_rego_users_self_only_filter_qualified_with_name_sort():
    sql = (
        "SELECT users.first_name, users.last_name, users.email, users.phone, users.department "
        "FROM users ORDER BY users.last_name ASC"
    )
    result = AliasAwareFilterInjector.inject(sql, "id = '22222222-2222-2222-2222-222222222222'", "users")
    assert result == "users.id = '22222222-2222-2222-2222-222222222222'"


def test_superuser_users_name_sorted_query_remains_unfiltered_no_op():
    sql = (
        "SELECT users.first_name, users.last_name, users.email, users.phone, users.department "
        "FROM users ORDER BY users.last_name ASC"
    )
    result = AliasAwareFilterInjector.inject(sql, "", "users")
    assert result == ""


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
