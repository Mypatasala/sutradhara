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
from src.agents.query_registry import REGISTRY, EntityMeta
from src.agents.query_validator import QueryPlanValidator, QueryPlanValidationError
from src.agents.query_normalizer import normalize
from src.retrieval.structured_sql_builder import StructuredSQLBuilder


class FakeDB:
    """Returns a match only for a fixed set of (lowercased) values, to
    exercise both the found and not-found lookup-filter paths."""

    KNOWN = {
        "mathematics": "Mathematics", "5": "5", "10": "10", "teacher": "TEACHER",
        "term 2": "Term 2", "2025-2026": "2025-2026",
    }

    def execute(self, sql):
        for key, real in self.KNOWN.items():
            if f"'{key}'" in sql.lower():
                return [{"matched_value": real}]
        return []


@pytest.fixture()
def validator():
    return QueryPlanValidator(FakeDB())


def test_valid_plan_passes(validator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_CLASS)
    resolved = validator.validate(plan, school_id=56)
    assert resolved == {}


def test_unsupported_operation_rejected(validator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.PERCENTAGE)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_unsupported_grouping_rejected(validator):
    plan = QueryPlan(entity=Entity.REPORT_CARDS, operation=Operation.LIST, group_by=GroupingDimension.BY_CLASS)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_filter_field_not_applicable_to_entity_rejected(validator):
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="present")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_filter_value_not_in_allowed_set_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="on_leave")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_filter_value_case_insensitive_accepted(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="PRESENT")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_percentage_without_percentage_of_rejected(validator):
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_percentage_field_duplicated_in_filters_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        filters=[ComparisonFilter(field=FilterField.STATUS, value="absent")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_percentage_of_with_lookup_backed_field_rejected(validator):
    """Flattening ComparisonFilter removed the type-level 'numerator must be
    enum-backed' guarantee (PercentageSpec.numerator used to be typed
    EnumComparisonFilter specifically) -- this must now be an explicit
    validator rule instead, or a lookup-backed numerator would reach
    StructuredSQLBuilder and KeyError on meta.enum_filter_fields[...]."""
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.PERCENTAGE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.SUBJECT, value="Mathematics")),
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_date_range_on_entity_without_date_column_rejected(validator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, date_range=RelativeDate.THIS_MONTH)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_display_field_not_valid_for_entity_rejected(validator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.LIST, display_fields=[DisplayField.OVERALL_GRADE])
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_sort_field_not_valid_for_entity_rejected(validator):
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.LIST, sort=SortSpec(field=SortField.ISSUE_DATE))
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_homework_list_passes(validator):
    """LIST display-shape fix (2026-09-10): a plain HOMEWORK LIST plan,
    with no display_fields specified, must validate cleanly now that
    default_display_fields is populated (previously empty)."""
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.LIST)
    validator.validate(plan, school_id=56)  # must not raise


def test_course_schedule_start_time_sort_passes(validator):
    """COURSE_SCHEDULE.sort_field_columns already registers START_TIME --
    2026-09-10 prompt/test change makes this reachable, no registry/
    validator/builder change needed."""
    plan = QueryPlan(entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST, sort=SortSpec(field=SortField.START_TIME))
    validator.validate(plan, school_id=56)  # must not raise


def test_course_schedule_start_time_sort_still_rejected_for_unrelated_entity(validator):
    """Regression: START_TIME is registered only for COURSE_SCHEDULE --
    confirms documenting it in the prompt did not somehow leak sort
    eligibility into an entity that never registered it."""
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.LIST, sort=SortSpec(field=SortField.START_TIME))
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_lookup_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_lookup_filter_not_found_rejected(validator):
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="nonexistent subject")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


# ── HOMEWORK SUBJECT lookup filter -- homework.subject is native to ────────
# homework's own row (no courses/subjects join, unlike COURSE_SCHEDULE.
# SUBJECT above); reuses the same lookup-filter existence-check mechanism.

def test_homework_subject_lookup_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_homework_subject_lookup_filter_case_insensitive(validator):
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="MATHEMATICS")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_homework_subject_lookup_filter_not_found_rejected(validator):
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="nonexistent subject")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


class _HomeworkSubjectSchoolScopedDB:
    """Only returns a match when BOTH the subject value AND the exact
    `homework.school_id = <school_id>` clause appear in the SQL -- proves
    existence_check_join_path=[] combined with school_id_column=
    "homework.school_id" still correctly scopes the check to the caller's
    own school, not just any homework row anywhere (the cross-tenant case:
    a subject that exists at a DIFFERENT school must not resolve here)."""

    def execute(self, sql):
        if "'mathematics'" in sql.lower() and "homework.school_id = 56" in sql:
            return [{"matched_value": "Mathematics"}]
        return []


def test_homework_subject_lookup_filter_is_school_scoped():
    scoped_validator = QueryPlanValidator(_HomeworkSubjectSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    resolved = scoped_validator.validate(plan, school_id=56)  # must not raise
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_homework_subject_lookup_filter_cross_tenant_not_resolved():
    """The same subject value, validated for a DIFFERENT school_id, must
    not resolve -- confirms the existence check is genuinely school-scoped,
    not merely subject-name-matched."""
    scoped_validator = QueryPlanValidator(_HomeworkSubjectSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    with pytest.raises(QueryPlanValidationError):
        scoped_validator.validate(plan, school_id=99)


# ── ASSIGNMENTS SUBJECT lookup filter (Phase 2, 2026-09-10) -- unlike ─────
# HOMEWORK.SUBJECT above, assignments.course_id reaches courses.name via a
# real join (assignments has no native subject/course-name column of its
# own) -- "subject" and "course" are the same concept in this schema. The
# EXISTENCE CHECK must join courses -> class_sections (courses has no
# school_id column of its own), identical to COURSE_SCHEDULE.SUBJECT's own
# existence check below.

def test_assignments_subject_lookup_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_assignments_subject_lookup_filter_case_insensitive(validator):
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="MATHEMATICS")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_assignments_subject_lookup_filter_not_found_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="nonexistent subject")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


class _AssignmentsSubjectSchoolScopedDB:
    """Only returns a match when BOTH the subject value AND the exact
    `class_sections.school_id = <school_id>` clause appear in the SQL --
    proves existence_check_join_path=[JoinStep(class_sections, section_id,
    id)] combined with school_id_column="class_sections.school_id"
    correctly scopes the check to the caller's own school (courses has no
    school_id column of its own, unlike HOMEWORK's own school-scoped test
    above, which scopes directly via homework.school_id)."""

    def execute(self, sql):
        if "'mathematics'" in sql.lower() and "class_sections.school_id = 56" in sql:
            return [{"matched_value": "Mathematics"}]
        return []


def test_assignments_subject_lookup_filter_is_school_scoped():
    scoped_validator = QueryPlanValidator(_AssignmentsSubjectSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    resolved = scoped_validator.validate(plan, school_id=56)  # must not raise
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_assignments_subject_lookup_filter_cross_tenant_not_resolved():
    """The same subject value, validated for a DIFFERENT school_id, must
    not resolve -- confirms the existence check is genuinely school-scoped
    via class_sections.school_id, not merely subject-name-matched."""
    scoped_validator = QueryPlanValidator(_AssignmentsSubjectSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    with pytest.raises(QueryPlanValidationError):
        scoped_validator.validate(plan, school_id=99)


# ── EXAMINATIONS Phase 1 (2026-09-11) -- COUNT, LIST, STATUS filter, ──────
# SUBJECT filter only. examinations.course_id reaches courses.name via a
# real join, exactly like ASSIGNMENTS.SUBJECT above -- the same
# courses/class_sections existence-check shape is reused byte-for-byte.

def test_examinations_count_passes(validator):
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.COUNT)
    validator.validate(plan, school_id=56)  # must not raise


def test_examinations_list_passes(validator):
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.LIST)
    validator.validate(plan, school_id=56)  # must not raise


def test_examinations_status_filter_accepts_pending(validator):
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_examinations_status_filter_accepts_in_progress(validator):
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="in_progress")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_examinations_status_filter_accepts_completed(validator):
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="completed")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_examinations_status_filter_rejects_invalid_value(validator):
    """Regression: must never accept a TeacherExam.ExamStatus value
    (e.g. "draft"/"submitted"/"evaluated") -- Examination.ExamStatus is a
    completely separate, unrelated 3-value vocabulary."""
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="draft")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_examinations_subject_lookup_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_examinations_subject_lookup_filter_case_insensitive(validator):
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="MATHEMATICS")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_examinations_subject_lookup_filter_not_found_rejected(validator):
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="nonexistent subject")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


class _ExaminationsSubjectSchoolScopedDB:
    """Only returns a match when BOTH the subject value AND the exact
    `class_sections.school_id = <school_id>` clause appear in the SQL --
    proves existence_check_join_path=[JoinStep(class_sections, section_id,
    id)] combined with school_id_column="class_sections.school_id"
    correctly scopes the check to the caller's own school (courses has no
    school_id column of its own), identical to ASSIGNMENTS.SUBJECT's own
    school-scoped test above."""

    def execute(self, sql):
        if "'mathematics'" in sql.lower() and "class_sections.school_id = 56" in sql:
            return [{"matched_value": "Mathematics"}]
        return []


def test_examinations_subject_lookup_filter_is_school_scoped():
    scoped_validator = QueryPlanValidator(_ExaminationsSubjectSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    resolved = scoped_validator.validate(plan, school_id=56)  # must not raise
    assert resolved[FilterField.SUBJECT] == "Mathematics"


def test_examinations_subject_lookup_filter_cross_tenant_not_resolved():
    """The same subject value, validated for a DIFFERENT school_id, must
    not resolve -- confirms the existence check is genuinely school-scoped
    via class_sections.school_id, not merely subject-name-matched."""
    scoped_validator = QueryPlanValidator(_ExaminationsSubjectSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.EXAMINATIONS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    with pytest.raises(QueryPlanValidationError):
        scoped_validator.validate(plan, school_id=99)


def test_examinations_by_status_grouping_rejected(validator):
    """Scope guard: EXAMINATIONS registers no supported_groupings at all
    this phase."""
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_examinations_sort_rejected(validator):
    """Scope guard: EXAMINATIONS registers no sort_field_columns at all
    this phase."""
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.LIST, sort=SortSpec(field=SortField.ISSUE_DATE))
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_examinations_date_range_rejected(validator):
    """Scope guard: EXAMINATIONS registers no date_column at all this
    phase."""
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.LIST, date_range=RelativeDate.LAST_30_DAYS)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_examinations_average_rejected(validator):
    """Scope guard: EXAMINATIONS registers no numeric_agg_fields at all
    this phase -- obtained_marks/total_marks are deliberately unmodeled."""
    plan = QueryPlan(entity=Entity.EXAMINATIONS, operation=Operation.AVERAGE, aggregate_target=NumericField.OVERALL_PERCENTAGE)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


# ── REPORT_CARDS TERM lookup filter (Phase 1, 2026-09-10) -- report_cards. ─
# term is native to report_cards' own row (main_query_join_path=[], like
# HOMEWORK.SUBJECT above), but the EXISTENCE CHECK must join through
# students (report_cards has no school_id column of its own) -- see
# query_registry.py's REPORT_CARDS.lookup_filter_fields entry.

def test_report_cards_term_lookup_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.TERM, value="term 2")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.TERM] == "Term 2"


def test_report_cards_term_lookup_filter_case_insensitive(validator):
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.TERM, value="TERM 2")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.TERM] == "Term 2"


def test_report_cards_term_lookup_filter_not_found_rejected(validator):
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.TERM, value="nonexistent term")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


class _ReportCardsTermSchoolScopedDB:
    """Only returns a match when BOTH the term value AND the exact
    `students.school_id = <school_id>` clause appear in the SQL -- proves
    existence_check_join_path=[JoinStep(students, student_id, id)] combined
    with school_id_column="students.school_id" correctly scopes the check
    to the caller's own school (report_cards has no school_id column of its
    own, unlike HOMEWORK's school-scoped test above)."""

    def execute(self, sql):
        if "'term 2'" in sql.lower() and "students.school_id = 56" in sql:
            return [{"matched_value": "Term 2"}]
        return []


def test_report_cards_term_lookup_filter_is_school_scoped():
    scoped_validator = QueryPlanValidator(_ReportCardsTermSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.TERM, value="term 2")],
    )
    resolved = scoped_validator.validate(plan, school_id=56)  # must not raise
    assert resolved[FilterField.TERM] == "Term 2"


def test_report_cards_term_lookup_filter_cross_tenant_not_resolved():
    """The same term value, validated for a DIFFERENT school_id, must not
    resolve -- confirms the existence check is genuinely school-scoped via
    students.school_id, not merely term-text-matched."""
    scoped_validator = QueryPlanValidator(_ReportCardsTermSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.TERM, value="term 2")],
    )
    with pytest.raises(QueryPlanValidationError):
        scoped_validator.validate(plan, school_id=99)


# ── REPORT_CARDS ACADEMIC_YEAR lookup filter (Phase 2, 2026-09-11) -- ─────
# identical shape to TERM immediately above: report_cards.academic_year is
# native to report_cards' own row (main_query_join_path=[]), existence
# check joins through students for school-scoping (report_cards has no
# school_id column of its own) -- see query_registry.py's REPORT_CARDS.
# lookup_filter_fields entry.

def test_report_cards_academic_year_lookup_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.ACADEMIC_YEAR] == "2025-2026"


def test_report_cards_academic_year_lookup_filter_uses_case_insensitive_comparison():
    """Mirrors TERM's own case-insensitive lookup mechanism exactly.
    Real academic-year labels ("2025-2026") are digit-only in practice, so
    there is no visible casing to vary in the VALUE itself -- this instead
    directly inspects the generated existence-check SQL to confirm the
    LOWER(...)=LOWER(...) comparison (the actual mechanism TERM's own
    case-insensitivity relies on) is used here too, not a column-specific
    special case."""
    captured = []

    class _CapturingDB:
        def execute(self, sql):
            captured.append(sql)
            return [{"matched_value": "2025-2026"}]

    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026")],
    )
    QueryPlanValidator(_CapturingDB()).validate(plan, school_id=56)
    assert "LOWER(report_cards.academic_year) = LOWER('2025-2026')" in captured[0]


def test_report_cards_academic_year_lookup_filter_not_found_rejected(validator):
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="1999-2000")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


class _ReportCardsAcademicYearSchoolScopedDB:
    """Only returns a match when BOTH the academic_year value AND the exact
    `students.school_id = <school_id>` clause appear in the SQL -- proves
    existence_check_join_path=[JoinStep(students, student_id, id)] combined
    with school_id_column="students.school_id" correctly scopes the check
    to the caller's own school, identical to TERM's own school-scoped test
    above."""

    def execute(self, sql):
        if "'2025-2026'" in sql.lower() and "students.school_id = 56" in sql:
            return [{"matched_value": "2025-2026"}]
        return []


def test_report_cards_academic_year_lookup_filter_is_school_scoped():
    scoped_validator = QueryPlanValidator(_ReportCardsAcademicYearSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026")],
    )
    resolved = scoped_validator.validate(plan, school_id=56)  # must not raise
    assert resolved[FilterField.ACADEMIC_YEAR] == "2025-2026"


def test_report_cards_academic_year_lookup_filter_cross_tenant_not_resolved():
    """The same academic_year value, validated for a DIFFERENT school_id,
    must not resolve -- confirms the existence check is genuinely
    school-scoped via students.school_id, not merely text-matched."""
    scoped_validator = QueryPlanValidator(_ReportCardsAcademicYearSchoolScopedDB())
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ACADEMIC_YEAR, value="2025-2026")],
    )
    with pytest.raises(QueryPlanValidationError):
        scoped_validator.validate(plan, school_id=99)


# ── GRADE filter (students) -- Issue 1: verified as a per-school dynamic  ──
# value (my_patasala's PlatformGradeConfig lets each school define its own
# ordered grade-label list, e.g. ["1".."10"] or ["KG","1".."12"]), so this is
# a lookup-backed field (existence-checked, like SUBJECT), never a hardcoded
# enum allowed_values set.

def test_grade_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.GRADE, value="5")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.GRADE] == "5"


def test_grade_filter_another_valid_grade(validator):
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.GRADE, value="10")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_grade_filter_unsupported_grade_rejected(validator):
    """FakeDB only recognizes '5' and '10' as existing for this school (see
    KNOWN above) -- a grade this school doesn't have must be rejected by the
    existence check, exactly like an unrecognized subject."""
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.GRADE, value="99")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_grade_filter_is_not_confused_with_by_class_grouping(validator):
    """Regression for the exact Issue 1 failure mode: llama3.2 previously
    substituted group_by=BY_CLASS for a grade filter, which is invalid
    (operation=list + group_by requires an aggregate operation) -- a
    correctly-formed grade FILTER must validate with group_by left unset,
    proving the filter path is independent of and not reliant on grouping."""
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.GRADE, value="5")],
    )
    assert plan.group_by == GroupingDimension.NONE
    validator.validate(plan, school_id=56)  # must not raise despite group_by being unset


# ── ROLE filter (users) -- P0-1: role only reachable via users -> ─────────
# user_roles -> roles, and (like SUBJECT) existence-checked against real
# per-school data (whether the named role is actually assigned to a user at
# this school), never a hardcoded Python allowed-values set.

def test_role_filter_found_resolves_value(validator):
    plan = QueryPlan(
        entity=Entity.USERS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ROLE, value="teacher")],
    )
    resolved = validator.validate(plan, school_id=56)
    assert resolved[FilterField.ROLE] == "TEACHER"


def test_role_filter_not_found_rejected(validator):
    plan = QueryPlan(
        entity=Entity.USERS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.ROLE, value="nonexistent role")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_users_count_operation_passes(validator):
    plan = QueryPlan(entity=Entity.USERS, operation=Operation.COUNT)
    resolved = validator.validate(plan, school_id=56)
    assert resolved == {}


# ── Newly-supported grouping dimensions (P0-2: previously-orphaned) ───────

def test_attendance_by_status_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    validator.validate(plan, school_id=56)  # must not raise


def test_homework_status_filter_accepts_real_previously_rejected_value(validator):
    """STATUS vocabulary fix (2026-09-11): "completed" is a real value in
    the application's HomeworkStatus enum but was wrongly absent from the
    old, fabricated allowed_values set -- must now be accepted."""
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="completed")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_homework_status_filter_rejects_fake_previously_accepted_value(validator):
    """STATUS vocabulary fix (2026-09-11): "graded" was never a real
    HomeworkStatus value -- it does not exist in the application's Java
    enum or the DB's own ENUM column -- and was wrongly present in the
    old, fabricated allowed_values set. Must now be rejected."""
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="graded")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_homework_status_filter_still_accepts_real_pending_value(validator):
    """Regression: "pending" was the one value that overlapped between the
    old fabricated set and the real enum -- must remain accepted."""
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_homework_by_status_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    validator.validate(plan, school_id=56)  # must not raise


def test_homework_by_subject_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.COUNT, group_by=GroupingDimension.BY_SUBJECT)
    validator.validate(plan, school_id=56)  # must not raise


def test_assignments_count_passes(validator):
    plan = QueryPlan(entity=Entity.ASSIGNMENTS, operation=Operation.COUNT)
    validator.validate(plan, school_id=56)  # must not raise


def test_assignments_list_passes(validator):
    plan = QueryPlan(entity=Entity.ASSIGNMENTS, operation=Operation.LIST)
    validator.validate(plan, school_id=56)  # must not raise


def test_assignments_status_filter_passes(validator):
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="overdue")],
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_assignments_status_filter_invalid_value_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ASSIGNMENTS, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_assignments_by_status_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.ASSIGNMENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    validator.validate(plan, school_id=56)  # must not raise


def test_assignments_by_subject_grouping_still_rejected(validator):
    """Regression: ASSIGNMENTS never registered BY_SUBJECT grouping (Phase 2
    added a SUBJECT filter, not a grouping -- see query_registry.py's
    ASSIGNMENTS entry) -- confirms adding the SUBJECT filter did not
    somehow leak BY_SUBJECT grouping eligibility into it too."""
    plan = QueryPlan(entity=Entity.ASSIGNMENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_SUBJECT)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_course_schedule_by_day_of_week_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.COURSE_SCHEDULE, operation=Operation.COUNT, group_by=GroupingDimension.BY_DAY_OF_WEEK)
    validator.validate(plan, school_id=56)  # must not raise


def test_courses_count_passes(validator):
    plan = QueryPlan(entity=Entity.COURSES, operation=Operation.COUNT)
    validator.validate(plan, school_id=56)  # must not raise


def test_courses_list_passes(validator):
    plan = QueryPlan(entity=Entity.COURSES, operation=Operation.LIST)
    validator.validate(plan, school_id=56)  # must not raise


def test_courses_by_status_grouping_rejected(validator):
    """Regression: COURSES registers no supported_groupings at all this
    phase -- confirms BY_STATUS (a dimension other entities already use)
    was not somehow left reachable for COURSES."""
    plan = QueryPlan(entity=Entity.COURSES, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_courses_status_filter_rejected(validator):
    """Regression: COURSES registers no enum_filter_fields at all this
    phase -- STATUS is a generic FilterField reused by several other
    entities, so this confirms it was not accidentally left reachable for
    COURSES."""
    plan = QueryPlan(
        entity=Entity.COURSES, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="present")],
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_courses_sort_by_name_rejected(validator):
    """Regression: COURSES registers no sort_field_columns at all this
    phase -- SortField.NAME exists (registered for STUDENTS) but must not
    be reachable for COURSES."""
    plan = QueryPlan(entity=Entity.COURSES, operation=Operation.LIST, sort=SortSpec(field=SortField.NAME))
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_report_cards_by_term_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.REPORT_CARDS, operation=Operation.COUNT, group_by=GroupingDimension.BY_TERM)
    validator.validate(plan, school_id=56)  # must not raise


def test_students_by_status_grouping_still_rejected(validator):
    """Regression: STUDENTS never registered BY_STATUS -- confirms adding
    BY_STATUS to ATTENDANCE/HOMEWORK's registry entries did not somehow leak
    it into an entity that doesn't support it."""
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_attendance_by_subject_grouping_still_rejected(validator):
    """Regression: ATTENDANCE never registered BY_SUBJECT -- confirms adding
    BY_SUBJECT to HOMEWORK's registry entry did not somehow leak it into an
    entity that doesn't support it."""
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.BY_SUBJECT)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_multiple_failures_all_reported(validator):
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.PERCENTAGE,
        group_by=GroupingDimension.BY_STATUS,
    )
    with pytest.raises(QueryPlanValidationError) as exc_info:
        validator.validate(plan, school_id=56)
    # Both the unsupported operation AND the unsupported grouping should be
    # reported together, not just the first failure found.
    assert len(exc_info.value.reasons) >= 2


# ── LIST + grouping: must always be rejected (Principal Engineer Review    ──
# finding, 2026-08-30: operation=LIST, group_by=BY_CLASS/BY_SUBJECT reached
# the builder and produced invalid GROUP BY SQL, since a plain list of rows
# combined with a GROUP BY has no well-defined semantics). Fixed at the
# validation layer, not by patching the builder to guess a meaning.

def test_list_with_by_class_rejected(validator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.LIST, group_by=GroupingDimension.BY_CLASS)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_list_with_by_subject_rejected(validator):
    plan = QueryPlan(entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST, group_by=GroupingDimension.BY_SUBJECT)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


# ── extreme / aggregate_value sort validation ──────────────────────────────

def test_extreme_without_group_by_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST,
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_extreme_with_non_aggregate_operation_rejected(validator):
    """extreme with group_by set but operation=LIST is rejected -- LIST
    returns individual rows, so even with a grouping dimension present
    there's no aggregate value to be the extreme of. group_by is
    deliberately set here (BY_CLASS, valid for STUDENTS) to isolate this
    rule from the separate group_by==NONE rejection reason."""
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.LIST, group_by=GroupingDimension.BY_CLASS,
        extreme=ExtremeSelector.LOWEST,
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_extreme_combined_with_limit_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST, limit=5,
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_extreme_combined_with_sort_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST, sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"),
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_extreme_valid_plan_passes(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST,
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_aggregate_value_sort_without_group_by_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"),
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_aggregate_value_sort_with_explicit_limit_passes(validator):
    """Explicit top/bottom N -- "5 lowest" -- remains fully supported and
    distinct from extreme."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"), limit=5,
    )
    validator.validate(plan, school_id=56)  # must not raise


# ── AVERAGE aggregate_target, Phase 1 (validator-only) ──────────────────────
# SQL builder support does not exist yet -- StructuredSQLBuilder still
# raises NotImplementedError for average/sum by design. These tests only
# cover QueryPlanValidator's own fail-closed gate on aggregate_target.
# Approved scope: AVERAGE for report_cards.overall_percentage only; SUM
# remains rejected everywhere (never added to any EntityMeta.numeric_
# agg_fields); every other entity has no registered numeric target at all.

def test_report_cards_average_with_overall_percentage_passes(validator):
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_average_without_aggregate_target_rejected(validator):
    plan = QueryPlan(entity=Entity.REPORT_CARDS, operation=Operation.AVERAGE)
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_average_rejected_for_entity_with_no_registered_target(validator):
    """End-to-end (validate()) coupled case: STUDENTS neither supports
    operation=average nor registers any numeric_agg_fields, so BOTH the
    operation-support check and the target-registration check reject this
    plan -- this test alone cannot prove which one fired. That is
    necessarily true through the full validate() pipeline with only one
    NumericField value existing so far (registered exclusively on
    REPORT_CARDS): every entity that lacks a registered target also lacks
    operation=average in supported_operations today, so no real
    (entity, operation) pair isolates "target not registered" as the SOLE
    failure reason through validate() alone. See
    test_validate_aggregate_target_rejects_unregistered_target_in_isolation
    directly below for the independent, whitebox proof of that specific
    check -- this test stays as the end-to-end confirmation that the two
    checks compose correctly (either one is sufficient to reject)."""
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_validate_aggregate_target_rejects_unregistered_target_in_isolation(validator):
    """Whitebox: calls QueryPlanValidator._validate_aggregate_target()
    directly against a throwaway, non-REGISTRY EntityMeta that legitimately
    "supports" operation=average but registers zero numeric_agg_fields --
    proving the target-registration check fires independently of the
    operation-support check (which lives entirely in validate(), not in
    this method), without touching production REGISTRY, without adding a
    second NumericField/entity capability, and without enabling AVERAGE for
    any real entity. The fake EntityMeta exists only in this test's local
    scope and is never registered anywhere."""
    fake_meta = EntityMeta(table="fake_table_for_isolation_test_only", supported_operations={Operation.AVERAGE})
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    reasons: list = []
    validator._validate_aggregate_target(plan, fake_meta, reasons)
    assert reasons, "expected a rejection reason for a target not registered on this EntityMeta"
    assert "not a registered numeric aggregation field" in reasons[0]


def test_validate_aggregate_target_accepts_registered_target_in_isolation(validator):
    """Converse whitebox case: the same method, called directly, must NOT
    reject when the target IS registered on the (throwaway) EntityMeta --
    confirms the isolation test above is actually exercising the check's
    both branches, not just always failing."""
    fake_meta = EntityMeta(
        table="fake_table_for_isolation_test_only",
        supported_operations={Operation.AVERAGE},
        numeric_agg_fields={NumericField.OVERALL_PERCENTAGE: "fake_table_for_isolation_test_only.some_column"},
    )
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.AVERAGE,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    reasons: list = []
    validator._validate_aggregate_target(plan, fake_meta, reasons)
    assert reasons == []


def test_sum_rejected_even_with_valid_target(validator):
    """SUM is deliberately not enabled anywhere -- report_cards supports
    AVERAGE only, so a SUM plan with an otherwise-valid target must still
    be rejected on the operation-support check."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.SUM,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


@pytest.mark.parametrize("operation", [Operation.COUNT, Operation.LIST])
def test_aggregate_target_rejected_on_unrelated_operation(validator, operation):
    """An aggregation target must never be silently accepted/ignored on an
    operation it doesn't apply to -- explicit rejection, not silent
    tolerance, matching every other rule in this validator."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=operation,
        aggregate_target=NumericField.OVERALL_PERCENTAGE,
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


# ── Full operation x group_by compatibility matrix, every registered entity ─
#
# For every (entity, operation, group_by) combination, an independent,
# registry-facts-only oracle (NOT a re-implementation of the validator's own
# code) predicts whether it should validate. Combinations the oracle expects
# to pass are additionally run all the way through normalize() and
# StructuredSQLBuilder.build() -- proving the full validate -> normalize ->
# build chain succeeds for everything the registry claims to support, and
# that invalid combinations are rejected at validation, before ever reaching
# normalize/build.
#
# The oracle's aggregate-operation set is spelled out here independently,
# deliberately NOT imported from src.agents.query_plan.AGGREGATE_OPERATIONS.
# If that constant were ever wrong, importing it here would let the test
# validate itself against its own mistake instead of catching it.
_ORACLE_AGGREGATE_OPERATIONS = {Operation.COUNT, Operation.PERCENTAGE, Operation.AVERAGE, Operation.SUM}


@pytest.mark.parametrize("entity", list(Entity))
def test_operation_group_by_compatibility_matrix(entity, validator):
    meta = REGISTRY[entity]
    for operation in Operation:
        for group_by in GroupingDimension:
            kwargs = {"entity": entity, "operation": operation, "group_by": group_by}
            if operation == Operation.PERCENTAGE:
                if not meta.enum_filter_fields:
                    # This entity has no enum filter field to build a
                    # meaningful percentage_of from -- PERCENTAGE is already
                    # excluded from supported_operations for every such
                    # entity today, so this combination is out of scope for
                    # this matrix (covered instead by
                    # test_percentage_without_percentage_of_rejected).
                    continue
                field = next(iter(meta.enum_filter_fields))
                value = next(iter(meta.enum_filter_fields[field].allowed_values))
                kwargs["percentage_of"] = PercentageSpec(
                    numerator=ComparisonFilter(field=FilterField(field.value), value=value)
                )

            if operation in (Operation.AVERAGE, Operation.SUM):
                if not meta.numeric_agg_fields:
                    # No registered numeric aggregation target for this
                    # entity -- out of scope for this matrix (covered
                    # instead by the dedicated aggregate_target tests
                    # below: test_average_without_aggregate_target_rejected
                    # etc.), same "continue" pattern PERCENTAGE uses above
                    # for entities with no enum_filter_fields.
                    continue
                kwargs["aggregate_target"] = next(iter(meta.numeric_agg_fields))

            plan = QueryPlan(**kwargs)

            expected_valid = (
                operation in meta.supported_operations
                and (group_by == GroupingDimension.NONE or group_by in meta.supported_groupings)
                and (group_by == GroupingDimension.NONE or operation in _ORACLE_AGGREGATE_OPERATIONS)
            )

            if expected_valid:
                resolved = validator.validate(plan, school_id=56)  # must not raise
                canonical = normalize(plan, resolved)
                if operation == Operation.SUM:
                    # SUM remains a deliberate, explicit failure -- Phase 2
                    # (2026-09-07) only implemented AVERAGE.
                    # StructuredSQLBuilder still raises NotImplementedError
                    # for SUM by design (see its own comment). Not currently
                    # reachable via expected_valid=True (no entity lists SUM
                    # in supported_operations), but asserted defensively so
                    # this stays correct if that ever changes.
                    with pytest.raises(NotImplementedError):
                        StructuredSQLBuilder.build(canonical)
                else:
                    StructuredSQLBuilder.build(canonical)  # must not raise
            else:
                with pytest.raises(QueryPlanValidationError):
                    validator.validate(plan, school_id=56)


# ── Explicit date/date-range, Phase 1 (validator-only) ──────────────────────
# SQL builder and normalizer support are a separate, not-yet-started
# follow-up -- these tests only cover QueryPlanValidator's own fail-closed
# gate on explicit_start_date/explicit_end_date.

def test_explicit_single_day_passes(validator):
    """A single explicit day is expressed as start == end -- no third
    field."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date="2026-08-15", explicit_end_date="2026-08-15",
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_explicit_date_range_passes(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date="2026-08-01", explicit_end_date="2026-08-15",
    )
    validator.validate(plan, school_id=56)  # must not raise


def test_explicit_date_only_start_set_rejected(validator):
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, explicit_start_date="2026-08-01")
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_explicit_date_only_end_set_rejected(validator):
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.COUNT, explicit_end_date="2026-08-15")
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


@pytest.mark.parametrize("bad_value", [
    "08/15/2026",       # wrong separators/order
    "2026-8-15",        # non-zero-padded -- rejected for canonical-form reasons, not just parseability
    "not-a-date",
    "2026-15-08",       # month out of range
    "",
])
def test_explicit_date_malformed_string_rejected(validator, bad_value):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date=bad_value, explicit_end_date="2026-08-15",
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_explicit_date_impossible_calendar_date_rejected(validator):
    """2026-02-30 has the right shape but is not a real calendar date --
    strptime itself must reject it, no separate calendar-validity rule
    needed."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date="2026-02-30", explicit_end_date="2026-02-30",
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_explicit_date_inverted_range_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date="2026-08-15", explicit_end_date="2026-08-01",
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_explicit_date_mutually_exclusive_with_date_range_rejected(validator):
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT, date_range=RelativeDate.LAST_30_DAYS,
        explicit_start_date="2026-08-01", explicit_end_date="2026-08-15",
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_explicit_date_rejected_for_entity_without_date_column(validator):
    """STUDENTS has no date_column -- explicit dates can no more be scoped
    on it than date_range can (see the existing date_range/date_column
    rule this mirrors)."""
    plan = QueryPlan(
        entity=Entity.STUDENTS, operation=Operation.COUNT,
        explicit_start_date="2026-08-01", explicit_end_date="2026-08-15",
    )
    with pytest.raises(QueryPlanValidationError):
        validator.validate(plan, school_id=56)


def test_explicit_date_absent_preserves_existing_relative_date_behavior(validator):
    """Regression: a plan using only date_range (no explicit fields at all)
    must be completely unaffected by this phase's new rule."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT, date_range=RelativeDate.LAST_30_DAYS,
    )
    validator.validate(plan, school_id=56)  # must not raise
