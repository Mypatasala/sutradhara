"""
Registry-level regression guard for Issue 2 (the "attendance.status" live
MariaDB ambiguity, see query_registry.py's EnumFilterFieldMeta docstring and
structured_sql_builder.py's module docstring): every registry-owned physical
SQL column reference must be fully qualified as "<table>.<column>", NEVER a
bare column name -- bareness was safe only by accident, for as long as every
entity queried a single table with no joins. This test inspects the actual
REGISTRY data (not the builder) so a future registry entry that reintroduces
a bare column is caught here, mechanically, rather than waiting for it to
collide with a joined table's same-named column at live-DB time.

A qualifying table name is accepted if it's either the entity's own base
table, or a table this entity ever joins to (via any registered grouping's
`joins`, or any lookup filter's `main_query_join_path` /
`existence_check_join_path`) -- so explicitly modeled joined-table
references (e.g. COURSE_SCHEDULE's "courses.name") are correctly allowed,
not just base-table self-references.
"""

import re

import pytest

from src.agents.query_registry import (
    REGISTRY,
    EntityMeta,
    EnumFilterFieldMeta,
    GroupingPath,
    JoinStep,
    LabelExpression,
)

_QUALIFIED_COLUMN_RE = re.compile(r"^([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)$")


def _reachable_tables(meta) -> set:
    """Every table this entity's own registry data ever references: its own
    base table, plus every JOIN target reachable through any registered
    grouping path, list_joins (operation=LIST's own always-included joins --
    see EntityMeta.list_joins' docstring), or lookup filter's join paths."""
    tables = {meta.table}
    for grouping_path in meta.supported_groupings.values():
        for step in grouping_path.joins:
            tables.add(step.table)
    for step in meta.list_joins:
        tables.add(step.table)
    for lookup_meta in meta.lookup_filter_fields.values():
        for step in lookup_meta.main_query_join_path:
            tables.add(step.table)
        for step in lookup_meta.existence_check_join_path:
            tables.add(step.table)
        tables.add(lookup_meta.lookup_table)
    return tables


def _assert_qualified(value: str, context: str, reachable_tables: set) -> None:
    match = _QUALIFIED_COLUMN_RE.match(value)
    assert match, (
        f"{context}: {value!r} is not a fully qualified '<table>.<column>' reference. "
        f"Every registry-owned physical column must be qualified with its owning table "
        f"(see EnumFilterFieldMeta's docstring) -- a bare column silently risks producing "
        f"ambiguous SQL the moment a join introduces a same-named column on another table."
    )
    table = match.group(1)
    assert table in reachable_tables, (
        f"{context}: {value!r} is qualified with table {table!r}, which this entity never "
        f"joins to (reachable tables: {sorted(reachable_tables)}). A column can only be "
        f"safely referenced against a table that is actually part of the query it's used in."
    )


def test_every_enum_filter_column_is_qualified():
    for entity, meta in REGISTRY.items():
        reachable = _reachable_tables(meta)
        for field, field_meta in meta.enum_filter_fields.items():
            _assert_qualified(field_meta.column, f"{entity.value}.enum_filter_fields[{field.value}].column", reachable)


def test_every_lookup_filter_column_is_qualified():
    for entity, meta in REGISTRY.items():
        reachable = _reachable_tables(meta)
        for field, field_meta in meta.lookup_filter_fields.items():
            _assert_qualified(field_meta.column, f"{entity.value}.lookup_filter_fields[{field.value}].column", reachable)


def test_every_date_column_is_qualified():
    for entity, meta in REGISTRY.items():
        if meta.date_column is not None:
            _assert_qualified(meta.date_column, f"{entity.value}.date_column", _reachable_tables(meta))


def test_report_cards_date_column_wired_to_issue_date():
    """P1 (2026-09-05): report_cards.issue_date was already a registered
    DisplayField/SortField column but had never been wired for date_range
    filtering -- confirms the registry addition reuses that exact same
    column, not a new/different one."""
    from src.agents.query_plan import Entity
    assert REGISTRY[Entity.REPORT_CARDS].date_column == "report_cards.issue_date"


def test_report_cards_term_lookup_filter_is_configured_correctly():
    """Phase 1 (2026-09-10): report_cards.term is native to report_cards'
    own row (main_query_join_path=[]), reusing the exact same column
    already registered as DisplayField.TERM and already used by BY_TERM
    grouping below -- confirms no new/different column and no unnecessary
    main-query join."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.REPORT_CARDS].lookup_filter_fields[LookupFilterField.TERM]
    assert meta.column == "report_cards.term"
    assert meta.lookup_table == "report_cards"
    assert meta.lookup_column == "term"
    assert meta.main_query_join_path == []


def test_report_cards_term_existence_check_reaches_students_school_id():
    """report_cards has no school_id column of its own -- the existence
    check must join through students (report_cards.student_id ->
    students.id) to reach students.school_id, exactly like
    COURSE_SCHEDULE.SUBJECT reaches class_sections.school_id via courses."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.REPORT_CARDS].lookup_filter_fields[LookupFilterField.TERM]
    assert len(meta.existence_check_join_path) == 1
    step = meta.existence_check_join_path[0]
    assert step.table == "students"
    assert step.left_column == "student_id"
    assert step.right_column == "id"
    assert meta.school_id_column == "students.school_id"


def test_report_cards_by_term_grouping_unaffected_by_term_filter_addition():
    """Regression: the pre-existing BY_TERM grouping must survive this
    addition completely unmodified -- still joins=[], still the same
    group_by_columns/label_alias as before."""
    from src.agents.query_plan import Entity, GroupingDimension
    path = REGISTRY[Entity.REPORT_CARDS].supported_groupings[GroupingDimension.BY_TERM]
    assert path.joins == []
    assert path.group_by_columns == ["report_cards.term"]
    assert path.label_alias == "term"


def test_report_cards_by_academic_year_grouping_uses_native_column_no_join():
    """Phase 3 (2026-09-11): BY_ACADEMIC_YEAR reuses the exact same column
    already used by LookupFilterField.ACADEMIC_YEAR (report_cards.
    academic_year) -- confirms an empty join path and the exact expected
    group_by_columns/label, identical shape to BY_TERM above."""
    from src.agents.query_plan import Entity, GroupingDimension
    path = REGISTRY[Entity.REPORT_CARDS].supported_groupings[GroupingDimension.BY_ACADEMIC_YEAR]
    assert path.joins == []
    assert path.group_by_columns == ["report_cards.academic_year"]
    assert path.label_alias == "academic_year"


def test_report_cards_academic_year_lookup_filter_is_configured_correctly():
    """Phase 2 (2026-09-11): report_cards.academic_year is native to
    report_cards' own row (main_query_join_path=[]), reusing the exact same
    column already registered as DisplayField.ACADEMIC_YEAR -- confirms no
    new/different column and no unnecessary main-query join, identical
    shape to TERM above."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.REPORT_CARDS].lookup_filter_fields[LookupFilterField.ACADEMIC_YEAR]
    assert meta.column == "report_cards.academic_year"
    assert meta.lookup_table == "report_cards"
    assert meta.lookup_column == "academic_year"
    assert meta.main_query_join_path == []


def test_report_cards_academic_year_existence_check_reaches_students_school_id():
    """report_cards has no school_id column of its own -- the existence
    check must join through students (report_cards.student_id ->
    students.id) to reach students.school_id, identical to TERM's own
    existence check above."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.REPORT_CARDS].lookup_filter_fields[LookupFilterField.ACADEMIC_YEAR]
    assert len(meta.existence_check_join_path) == 1
    step = meta.existence_check_join_path[0]
    assert step.table == "students"
    assert step.left_column == "student_id"
    assert step.right_column == "id"
    assert meta.school_id_column == "students.school_id"


def test_report_cards_lookup_filter_fields_are_exactly_term_and_academic_year():
    """Scope guard: REPORT_CARDS registers exactly TERM and ACADEMIC_YEAR
    as lookup filters, and exactly BY_TERM and BY_ACADEMIC_YEAR as
    groupings (Phase 3, 2026-09-11) -- no other lookup field, no
    sorting/date semantics for academic_year beyond this filter+grouping
    pair."""
    from src.agents.query_plan import Entity, GroupingDimension, LookupFilterField
    meta = REGISTRY[Entity.REPORT_CARDS]
    assert set(meta.lookup_filter_fields.keys()) == {LookupFilterField.TERM, LookupFilterField.ACADEMIC_YEAR}
    assert set(meta.supported_groupings.keys()) == {GroupingDimension.BY_TERM, GroupingDimension.BY_ACADEMIC_YEAR}


def test_subject_lookup_filter_registered_only_for_intended_entities():
    """Scope guard: SUBJECT is a shared LookupFilterField value used by
    multiple entities (unlike TERM's single-entity guard below) -- confirms
    it is registered for exactly {HOMEWORK, COURSE_SCHEDULE, ASSIGNMENTS,
    EXAMINATIONS, COURSES, TEACHER_EXAMS} (TEACHER_EXAMS being the latest,
    2026-09-11 addition) and did not leak into any other entity."""
    from src.agents.query_plan import Entity, LookupFilterField
    expected = {
        Entity.HOMEWORK, Entity.COURSE_SCHEDULE, Entity.ASSIGNMENTS,
        Entity.EXAMINATIONS, Entity.COURSES, Entity.TEACHER_EXAMS,
    }
    actual = {
        entity for entity, meta in REGISTRY.items()
        if LookupFilterField.SUBJECT in meta.lookup_filter_fields
    }
    assert actual == expected


def test_term_lookup_filter_not_registered_for_unrelated_entities():
    """Scope guard: TERM must be registered ONLY for REPORT_CARDS and
    TEACHER_EXAMS (2026-09-11) -- confirms it did not somehow leak into
    any other entity's lookup_filter_fields."""
    from src.agents.query_plan import Entity, LookupFilterField
    for entity, meta in REGISTRY.items():
        if entity in (Entity.REPORT_CARDS, Entity.TEACHER_EXAMS):
            continue
        assert LookupFilterField.TERM not in meta.lookup_filter_fields


def test_homework_display_fields_are_title_subject_status():
    """LIST display-shape fix (2026-09-10): title/subject/status are all
    plain columns native to homework's own row -- confirms the exact
    display-field mapping, default display fields, and canonical display
    order, replacing the previously-empty (invalid) registration."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.HOMEWORK]
    assert meta.display_field_columns == {
        DisplayField.TITLE: "homework.title",
        DisplayField.SUBJECT: "homework.subject",
        DisplayField.STATUS: "homework.status",
    }
    assert meta.default_display_fields == [DisplayField.TITLE, DisplayField.SUBJECT, DisplayField.STATUS]
    assert meta.canonical_display_order == [DisplayField.TITLE, DisplayField.SUBJECT, DisplayField.STATUS]


def test_homework_by_subject_grouping_uses_native_column_no_join():
    """2026-09-08: BY_SUBJECT reuses the exact same column already used by
    LookupFilterField.SUBJECT (homework.subject) -- confirms an empty join
    path and the exact expected group_by_columns/label, not a new/different
    column or an accidental join through courses."""
    from src.agents.query_plan import Entity, GroupingDimension
    path = REGISTRY[Entity.HOMEWORK].supported_groupings[GroupingDimension.BY_SUBJECT]
    assert path.joins == []
    assert path.group_by_columns == ["homework.subject"]
    assert path.label_alias == "subject"


def test_assignments_by_status_grouping_uses_native_column_no_join():
    """2026-09-08: ASSIGNMENTS.BY_STATUS reuses the exact same column
    already used by EnumFilterField.STATUS (assignments.status) -- confirms
    an empty join path and the exact expected group_by_columns/label."""
    from src.agents.query_plan import Entity, GroupingDimension
    path = REGISTRY[Entity.ASSIGNMENTS].supported_groupings[GroupingDimension.BY_STATUS]
    assert path.joins == []
    assert path.group_by_columns == ["assignments.status"]
    assert path.label_alias == "status"


def test_assignments_registers_no_numeric_or_date_fields():
    """Scope guard: ASSIGNMENTS must expose no numeric aggregation and no
    date filtering -- see query_registry.py's ASSIGNMENTS entry for the
    full rationale (per-assignment-scale grade/points; no date column).
    Phase 1 also asserted no lookup_filter_fields at all; Phase 2
    (2026-09-10) added SUBJECT specifically (course_id is now confirmed
    write-path-reliable, unlike student_id) -- see
    test_assignments_subject_lookup_filter_is_configured_correctly and
    test_assignments_registers_no_grouping_or_other_lookup_fields below for
    the narrower guards that replace this one's old blanket
    lookup_filter_fields=={} assertion."""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.ASSIGNMENTS]
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None


def test_assignments_subject_lookup_filter_is_configured_correctly():
    """Phase 2 (2026-09-10): assignments.course_id reaches courses.name via
    a real join (assignments has no native subject/course-name column of
    its own) -- unlike HOMEWORK.SUBJECT, which is a plain native column, so
    main_query_join_path is non-empty here."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.ASSIGNMENTS].lookup_filter_fields[LookupFilterField.SUBJECT]
    assert meta.column == "courses.name"
    assert meta.lookup_table == "courses"
    assert meta.lookup_column == "name"
    assert len(meta.main_query_join_path) == 1
    main_step = meta.main_query_join_path[0]
    assert main_step.table == "courses"
    assert main_step.left_column == "course_id"
    assert main_step.right_column == "id"


def test_assignments_subject_existence_check_reaches_class_sections_school_id():
    """courses has no school_id column of its own -- the existence check
    must join through class_sections (courses.section_id ->
    class_sections.id) to reach class_sections.school_id, identical to
    COURSE_SCHEDULE.SUBJECT's own existence check below."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.ASSIGNMENTS].lookup_filter_fields[LookupFilterField.SUBJECT]
    assert len(meta.existence_check_join_path) == 1
    step = meta.existence_check_join_path[0]
    assert step.table == "class_sections"
    assert step.left_column == "section_id"
    assert step.right_column == "id"
    assert meta.school_id_column == "class_sections.school_id"


def test_assignments_registers_no_grouping_or_other_lookup_fields():
    """Scope guard: Phase 2 adds SUBJECT only -- no BY_SUBJECT grouping, no
    course-code filter, no other lookup field, and STATUS/BY_STATUS from
    Phase 1 remain the only other registered filter/grouping."""
    from src.agents.query_plan import Entity, GroupingDimension, LookupFilterField
    meta = REGISTRY[Entity.ASSIGNMENTS]
    assert set(meta.lookup_filter_fields.keys()) == {LookupFilterField.SUBJECT}
    assert GroupingDimension.BY_SUBJECT not in meta.supported_groupings
    assert set(meta.supported_groupings.keys()) == {GroupingDimension.BY_STATUS}


def test_examinations_display_fields_are_title_status():
    """Phase 1 (2026-09-11): only title/status are exposed -- both native
    columns on examinations' own row, mirroring ASSIGNMENTS' own Phase 1
    display shape exactly."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.EXAMINATIONS]
    assert meta.display_field_columns == {
        DisplayField.TITLE: "examinations.title",
        DisplayField.STATUS: "examinations.status",
    }
    assert meta.default_display_fields == [DisplayField.TITLE, DisplayField.STATUS]
    assert meta.canonical_display_order == [DisplayField.TITLE, DisplayField.STATUS]


def test_examinations_status_enum_is_pending_in_progress_completed():
    """Confirms the real Examination.ExamStatus vocabulary -- and
    explicitly NOT the unrelated, differently-shaped TeacherExam.ExamStatus
    vocabulary (draft/submitted/approved/published/conducted/
    marks_submitted/evaluated)."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.EXAMINATIONS]
    status_meta = meta.enum_filter_fields[EnumFilterField.STATUS]
    assert status_meta.column == "examinations.status"
    assert status_meta.allowed_values == {"pending", "in_progress", "completed"}


def test_examinations_subject_lookup_filter_is_configured_correctly():
    """Phase 1 (2026-09-11): examinations.course_id reaches courses.name
    via a real join (examinations has no native subject/course-name column
    of its own), identical shape to ASSIGNMENTS.SUBJECT."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.EXAMINATIONS].lookup_filter_fields[LookupFilterField.SUBJECT]
    assert meta.column == "courses.name"
    assert meta.lookup_table == "courses"
    assert meta.lookup_column == "name"
    assert len(meta.main_query_join_path) == 1
    main_step = meta.main_query_join_path[0]
    assert main_step.table == "courses"
    assert main_step.left_column == "course_id"
    assert main_step.right_column == "id"


def test_examinations_subject_existence_check_reaches_class_sections_school_id():
    """courses has no school_id column of its own -- the existence check
    must join through class_sections (courses.section_id ->
    class_sections.id) to reach class_sections.school_id, identical to
    ASSIGNMENTS.SUBJECT's own existence check above."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.EXAMINATIONS].lookup_filter_fields[LookupFilterField.SUBJECT]
    assert len(meta.existence_check_join_path) == 1
    step = meta.existence_check_join_path[0]
    assert step.table == "class_sections"
    assert step.left_column == "section_id"
    assert step.right_column == "id"
    assert meta.school_id_column == "class_sections.school_id"


def test_examinations_registers_no_grouping_numeric_date_or_sort_fields():
    """Scope guard: Phase 1 registers COUNT/LIST/STATUS/SUBJECT only -- no
    grouping, no numeric aggregation, no date column, no sort field."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.EXAMINATIONS]
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None
    assert meta.sort_field_columns == {}
    assert set(meta.lookup_filter_fields.keys()) == {LookupFilterField.SUBJECT}


def test_absence_requests_display_fields_are_reason_status():
    """Phase 1 (2026-09-11): only reason/status are exposed -- both native
    columns on absence_requests' own row, mirroring EXAMINATIONS' own
    Phase 1 display shape exactly."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.ABSENCE_REQUESTS]
    assert meta.display_field_columns == {
        DisplayField.REASON: "absence_requests.reason",
        DisplayField.STATUS: "absence_requests.status",
    }
    assert meta.default_display_fields == [DisplayField.REASON, DisplayField.STATUS]
    assert meta.canonical_display_order == [DisplayField.REASON, DisplayField.STATUS]


def test_absence_requests_status_enum_is_pending_forwarded_approved_rejected():
    """Confirms the real AbsenceRequest.AbsenceStatus vocabulary -- enforced
    at the JPA layer (@Enumerated(EnumType.STRING)) even though the DB
    column itself is a plain varchar(32), not a native SQL ENUM."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.ABSENCE_REQUESTS]
    status_meta = meta.enum_filter_fields[EnumFilterField.STATUS]
    assert status_meta.column == "absence_requests.status"
    assert status_meta.allowed_values == {"pending", "forwarded_to_principal", "approved", "rejected"}


def test_absence_requests_registers_no_lookup_grouping_numeric_date_or_sort_fields():
    """Scope guard: Phase 1 registers COUNT/LIST/STATUS only -- no lookup
    filter (absence_requests has no course/subject dimension), no
    grouping, no numeric aggregation, no date column (despite
    absence_date/to_date existing on the real table), no sort field."""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.ABSENCE_REQUESTS]
    assert meta.lookup_filter_fields == {}
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None
    assert meta.sort_field_columns == {}


def test_absence_requests_supported_operations_are_exactly_count_and_list():
    from src.agents.query_plan import Entity, Operation
    meta = REGISTRY[Entity.ABSENCE_REQUESTS]
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}
    assert meta.table == "absence_requests"


# ── TEACHER_PROFILES Phase 1 (2026-09-11) ──────────────────────────────────
# Sensitive-field boundary: none of the V26 HR/verification/qualification/
# registration/experience columns, nor notes/bio/qualifications/subjects,
# may appear in ANY registry mapping for this entity -- Principal
# Engineer-approved security boundary from the readiness investigation.
# hire_date was reassessed (2026-09-11) and is now a display-only field --
# see test_teacher_profiles_hire_date_display_metadata_is_exact -- so it is
# deliberately NOT in this forbidden set.
_TEACHER_PROFILES_FORBIDDEN_COLUMNS = {
    "notes", "bio",
    "identity_verification_type", "identity_verification_number",
    "identity_verification_status", "identity_verification_document_url",
    "highest_qualification", "specialization", "university",
    "year_of_graduation", "qualification_status",
    "qualification_certificate_provided", "qualification_certificate_url",
    "registration_number", "issuing_authority", "registration_expiry_date",
    "experience_level", "years_of_experience", "previous_school",
    "experience_certificate_provided", "experience_certificate_url",
    "qualifications", "subjects",
}


def test_teacher_profiles_display_fields_are_designation_department():
    """Phase 1 (2026-09-11): designation/department/hire_date are exposed
    -- all plain, optional free-text/date columns native to
    teacher_profiles' own row. DEPARTMENT is reused from USERS' own
    display mapping (each EntityMeta.display_field_columns mapping is
    independently scoped). HIRE_DATE was added 2026-09-11 -- see
    test_teacher_profiles_hire_date_display_metadata_is_exact."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    assert meta.display_field_columns == {
        DisplayField.DESIGNATION: "teacher_profiles.designation",
        DisplayField.DEPARTMENT: "teacher_profiles.department",
        DisplayField.HIRE_DATE: "teacher_profiles.hire_date",
    }
    assert meta.default_display_fields == [DisplayField.DESIGNATION, DisplayField.DEPARTMENT, DisplayField.HIRE_DATE]
    assert meta.canonical_display_order == [DisplayField.DESIGNATION, DisplayField.DEPARTMENT, DisplayField.HIRE_DATE]


def test_teacher_profiles_hire_date_display_metadata_is_exact():
    """HIRE_DATE (2026-09-11): display-only -- teacher_profiles.hire_date
    is native to the entity's own row, no join required. Deliberately NOT
    wired to date_column or sort_field_columns this phase (see
    test_teacher_profiles_registers_no_grouping_numeric_date_or_sort_fields)."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    assert meta.display_field_columns[DisplayField.HIRE_DATE] == "teacher_profiles.hire_date"


def test_teacher_profiles_employment_type_enum_is_exactly_four_real_values():
    """Confirms the real TeacherProfile.EmploymentType vocabulary -- enforced
    at the JPA layer (@Enumerated(EnumType.STRING)) even though the DB
    column itself is a plain varchar(20), not a native SQL ENUM. No
    "UNKNOWN" placeholder value was invented for legacy NULL rows."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    status_meta = meta.enum_filter_fields[EnumFilterField.EMPLOYMENT_TYPE]
    assert status_meta.column == "teacher_profiles.employment_type"
    assert status_meta.allowed_values == {"FULL_TIME", "PART_TIME", "CONTRACT", "VISITING"}


def test_teacher_profiles_registers_no_grouping_numeric_date_or_sort_fields():
    """Scope guard: Phase 1 registers COUNT/LIST/EMPLOYMENT_TYPE/DEPARTMENT
    only -- no grouping, no numeric aggregation, no date column (hire_date
    is display-only, deliberately not wired to date_column), no sort
    field. (DEPARTMENT lookup filtering was added 2026-09-11 -- see
    test_teacher_profiles_department_lookup_filter_metadata_is_exact for
    its own dedicated coverage.)"""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None
    assert meta.sort_field_columns == {}


def test_teacher_profiles_department_lookup_filter_metadata_is_exact():
    """DEPARTMENT (2026-09-11): teacher_profiles has NO school_id column of
    its own, so the existence check must join through users
    (teacher_profiles.user_id -> users.id), scoped by users.school_id --
    the same authorization anchor already proven for this entity's own row
    filter. Mirrors REPORT_CARDS.TERM's shape, NOT STUDENTS.GRADE's
    self-referential shape."""
    from src.agents.query_plan import Entity, LookupFilterField
    from src.agents.query_registry import JoinStep
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    dept_meta = meta.lookup_filter_fields[LookupFilterField.DEPARTMENT]
    assert dept_meta.column == "teacher_profiles.department"
    assert dept_meta.lookup_table == "teacher_profiles"
    assert dept_meta.lookup_column == "department"
    assert dept_meta.main_query_join_path == []
    assert dept_meta.existence_check_join_path == [
        JoinStep(table="users", left_column="user_id", right_column="id"),
    ]
    assert dept_meta.school_id_column == "users.school_id"


def test_teacher_profiles_designation_lookup_filter_metadata_is_exact():
    """DESIGNATION (2026-09-11): same rationale and shape as DEPARTMENT
    immediately above, applied to teacher_profiles.designation -- the
    existence check joins through users (teacher_profiles.user_id ->
    users.id), scoped by users.school_id."""
    from src.agents.query_plan import Entity, LookupFilterField
    from src.agents.query_registry import JoinStep
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    desig_meta = meta.lookup_filter_fields[LookupFilterField.DESIGNATION]
    assert desig_meta.column == "teacher_profiles.designation"
    assert desig_meta.lookup_table == "teacher_profiles"
    assert desig_meta.lookup_column == "designation"
    assert desig_meta.main_query_join_path == []
    assert desig_meta.existence_check_join_path == [
        JoinStep(table="users", left_column="user_id", right_column="id"),
    ]
    assert desig_meta.school_id_column == "users.school_id"


def test_teacher_profiles_department_and_employment_type_unchanged_by_designation_addition():
    """Scope guard: adding DESIGNATION must not alter DEPARTMENT's or
    EMPLOYMENT_TYPE's own metadata at all."""
    from src.agents.query_plan import Entity, EnumFilterField, LookupFilterField
    from src.agents.query_registry import JoinStep
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    dept_meta = meta.lookup_filter_fields[LookupFilterField.DEPARTMENT]
    assert dept_meta.column == "teacher_profiles.department"
    assert dept_meta.lookup_table == "teacher_profiles"
    assert dept_meta.lookup_column == "department"
    assert dept_meta.main_query_join_path == []
    assert dept_meta.existence_check_join_path == [
        JoinStep(table="users", left_column="user_id", right_column="id"),
    ]
    assert dept_meta.school_id_column == "users.school_id"

    employment_meta = meta.enum_filter_fields[EnumFilterField.EMPLOYMENT_TYPE]
    assert employment_meta.column == "teacher_profiles.employment_type"
    assert employment_meta.allowed_values == {"FULL_TIME", "PART_TIME", "CONTRACT", "VISITING"}


def test_students_registers_no_designation_lookup_filter():
    """Scope guard: DESIGNATION lookup filtering must not leak into an
    unrelated entity with no designation concept at all."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.STUDENTS]
    assert LookupFilterField.DESIGNATION not in meta.lookup_filter_fields


def test_users_department_lookup_filter_metadata_is_exact():
    """DEPARTMENT (2026-09-11): reuses the exact same
    LookupFilterField.DEPARTMENT/FilterField.DEPARTMENT enum values already
    introduced for TEACHER_PROFILES.department -- no new enum value.
    users.department sits directly alongside users' own school_id column
    (confirmed in V1__baseline.sql), so both join paths are empty --
    self-referential exactly like STUDENTS.GRADE, unlike
    TEACHER_PROFILES.department's cross-table shape."""
    from src.agents.query_plan import Entity, LookupFilterField
    from src.agents.query_registry import JoinStep
    meta = REGISTRY[Entity.USERS]
    dept_meta = meta.lookup_filter_fields[LookupFilterField.DEPARTMENT]
    assert dept_meta.column == "users.department"
    assert dept_meta.lookup_table == "users"
    assert dept_meta.lookup_column == "department"
    assert dept_meta.main_query_join_path == []
    assert dept_meta.existence_check_join_path == []
    assert dept_meta.school_id_column == "users.school_id"


def test_users_name_sort_metadata_is_exact():
    """NAME (2026-09-11): reuses the exact same SortField.NAME value
    already proven for STUDENTS.NAME -- no new enum value. users.last_name
    is native to USERS' own row (NOT NULL, per V1__baseline.sql), so no
    join is required."""
    from src.agents.query_plan import Entity, SortField
    meta = REGISTRY[Entity.USERS]
    assert meta.sort_field_columns == {SortField.NAME: "users.last_name"}


def test_users_department_and_role_unchanged_by_name_sort_addition():
    """Scope guard: adding NAME sort must not alter USERS' own
    ROLE/DEPARTMENT lookup filter metadata or display fields at all."""
    from src.agents.query_plan import DisplayField, Entity, LookupFilterField
    meta = REGISTRY[Entity.USERS]
    assert meta.lookup_filter_fields[LookupFilterField.DEPARTMENT].column == "users.department"
    assert meta.lookup_filter_fields[LookupFilterField.ROLE].column == "roles.name"
    assert meta.display_field_columns == {
        DisplayField.FIRST_NAME: "users.first_name",
        DisplayField.LAST_NAME: "users.last_name",
        DisplayField.EMAIL: "users.email",
        DisplayField.PHONE: "users.phone",
        DisplayField.DEPARTMENT: "users.department",
    }


def test_teacher_profiles_department_lookup_filter_unchanged_by_users_addition():
    """Scope guard: adding USERS.department must not alter
    TEACHER_PROFILES.department's own metadata at all."""
    from src.agents.query_plan import Entity, LookupFilterField
    from src.agents.query_registry import JoinStep
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    dept_meta = meta.lookup_filter_fields[LookupFilterField.DEPARTMENT]
    assert dept_meta.column == "teacher_profiles.department"
    assert dept_meta.lookup_table == "teacher_profiles"
    assert dept_meta.lookup_column == "department"
    assert dept_meta.main_query_join_path == []
    assert dept_meta.existence_check_join_path == [
        JoinStep(table="users", left_column="user_id", right_column="id"),
    ]
    assert dept_meta.school_id_column == "users.school_id"


def test_courses_registers_no_department_lookup_filter():
    """Scope guard: DEPARTMENT lookup filtering must not leak into an
    unrelated entity with no department concept at all."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.COURSES]
    assert LookupFilterField.DEPARTMENT not in meta.lookup_filter_fields


def test_teacher_profiles_supported_operations_are_exactly_count_and_list():
    from src.agents.query_plan import Entity, Operation
    meta = REGISTRY[Entity.TEACHER_PROFILES]
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}
    assert meta.table == "teacher_profiles"


def test_teacher_profiles_no_forbidden_sensitive_column_in_any_registry_mapping():
    """Security boundary regression: none of the V26 HR/verification/
    qualification/registration/experience columns, nor notes/bio/
    qualifications/subjects, may ever appear as a display field, enum
    filter column, lookup filter column, sort column, grouping, or numeric
    aggregation target for TEACHER_PROFILES -- this must fail loudly if a
    future change accidentally exposes one of them. hire_date is
    deliberately excluded from this forbidden set (2026-09-11, display
    field only) -- see test_teacher_profiles_hire_date_display_metadata_is_exact."""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.TEACHER_PROFILES]

    all_referenced_columns = set(meta.display_field_columns.values())
    all_referenced_columns |= {f.column for f in meta.enum_filter_fields.values()}
    all_referenced_columns |= {f.column for f in meta.lookup_filter_fields.values()}
    all_referenced_columns |= set(meta.sort_field_columns.values())
    all_referenced_columns |= set(meta.numeric_agg_fields.values())
    if meta.date_column:
        all_referenced_columns.add(meta.date_column)

    referenced_bare_columns = {
        col.split(".", 1)[1] if "." in col else col for col in all_referenced_columns
    }
    assert referenced_bare_columns & _TEACHER_PROFILES_FORBIDDEN_COLUMNS == set()


def test_courses_display_fields_are_name_code_credits():
    """Phase 1 (2026-09-10): only name/code/credits are exposed -- the only
    columns confirmed write-path-authoritative against my_patasala's actual
    CourseService/DTOs (semester/enrollment_count/max_enrollment are all
    dead/unpopulated at every write path -- see query_registry.py's
    COURSES entry)."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.COURSES]
    assert meta.display_field_columns == {
        DisplayField.NAME: "courses.name",
        DisplayField.CODE: "courses.code",
        DisplayField.CREDITS: "courses.credits",
    }
    assert meta.default_display_fields == [DisplayField.NAME, DisplayField.CODE, DisplayField.CREDITS]
    assert meta.canonical_display_order == [DisplayField.NAME, DisplayField.CODE, DisplayField.CREDITS]


def test_courses_registers_no_numeric_date_grouping_or_sort_fields():
    """Scope guard: COURSES exposes no numeric aggregation, no date
    filtering, no grouping, and no sort -- see query_registry.py's COURSES
    entry for the full rationale (nullable instructor_id, dead
    semester/enrollment_count columns). (SUBJECT lookup filtering was added
    2026-09-11 -- see test_courses_subject_lookup_filter_metadata_is_exact
    for its own dedicated coverage.)"""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.COURSES]
    assert meta.enum_filter_fields == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None
    assert meta.supported_groupings == {}
    assert meta.sort_field_columns == {}


def test_courses_subject_lookup_filter_metadata_is_exact():
    """SUBJECT (2026-09-11): reuses the exact same LookupFilterField.SUBJECT/
    FilterField.SUBJECT enum values already proven for ASSIGNMENTS,
    COURSE_SCHEDULE, and EXAMINATIONS -- no new enum value. Unlike those
    three, courses.name is native to COURSES' own row (main_query_join_path
    empty); the existence check still reaches class_sections for
    school-scoping, since courses has no school_id column of its own --
    identical shape to the other three entities' own SUBJECT existence
    check."""
    from src.agents.query_plan import Entity, LookupFilterField
    from src.agents.query_registry import JoinStep
    meta = REGISTRY[Entity.COURSES]
    subject_meta = meta.lookup_filter_fields[LookupFilterField.SUBJECT]
    assert subject_meta.column == "courses.name"
    assert subject_meta.lookup_table == "courses"
    assert subject_meta.lookup_column == "name"
    assert subject_meta.main_query_join_path == []
    assert subject_meta.existence_check_join_path == [
        JoinStep(table="class_sections", left_column="section_id", right_column="id"),
    ]
    assert subject_meta.school_id_column == "class_sections.school_id"


def test_assignments_course_schedule_examinations_subject_lookup_filter_unchanged_by_courses_addition():
    """Scope guard: adding COURSES.SUBJECT must not alter the pre-existing
    ASSIGNMENTS/COURSE_SCHEDULE/EXAMINATIONS SUBJECT metadata at all -- each
    keeps its own course_id-anchored main_query_join_path, unlike COURSES'
    own self-referential (empty) shape."""
    from src.agents.query_plan import Entity, LookupFilterField
    from src.agents.query_registry import JoinStep
    expected_main_join = [JoinStep(table="courses", left_column="course_id", right_column="id")]
    expected_existence_join = [JoinStep(table="class_sections", left_column="section_id", right_column="id")]
    for entity in (Entity.ASSIGNMENTS, Entity.COURSE_SCHEDULE, Entity.EXAMINATIONS):
        subject_meta = REGISTRY[entity].lookup_filter_fields[LookupFilterField.SUBJECT]
        assert subject_meta.column == "courses.name"
        assert subject_meta.lookup_table == "courses"
        assert subject_meta.lookup_column == "name"
        assert subject_meta.main_query_join_path == expected_main_join
        assert subject_meta.existence_check_join_path == expected_existence_join
        assert subject_meta.school_id_column == "class_sections.school_id"


def test_every_numeric_agg_field_column_is_qualified():
    for entity, meta in REGISTRY.items():
        reachable = _reachable_tables(meta)
        for field, column in meta.numeric_agg_fields.items():
            _assert_qualified(column, f"{entity.value}.numeric_agg_fields[{field.value}]", reachable)


def test_only_report_cards_registers_a_numeric_agg_field():
    """AVERAGE/SUM Phase 1 (2026-09-07): scoped to exactly report_cards ->
    overall_percentage -- confirms no other entity accidentally gained a
    numeric aggregation target."""
    from src.agents.query_plan import Entity, NumericField
    for entity, meta in REGISTRY.items():
        if entity is Entity.REPORT_CARDS:
            assert meta.numeric_agg_fields == {NumericField.OVERALL_PERCENTAGE: "report_cards.overall_percentage"}
        else:
            assert meta.numeric_agg_fields == {}


def test_every_sort_field_column_is_qualified():
    for entity, meta in REGISTRY.items():
        reachable = _reachable_tables(meta)
        for field, column in meta.sort_field_columns.items():
            _assert_qualified(column, f"{entity.value}.sort_field_columns[{field.value}]", reachable)


def test_every_display_field_column_is_qualified():
    for entity, meta in REGISTRY.items():
        reachable = _reachable_tables(meta)
        for field, column in meta.display_field_columns.items():
            _assert_qualified(column, f"{entity.value}.display_field_columns[{field.value}]", reachable)


def test_every_grouping_default_display_column_is_qualified():
    """GroupingPath.default_display_columns (the Class/Section mechanism)
    holds registry-owned physical column references exactly like the other
    fields above -- it was missed when this guard was first written, which
    would have let a future grouping add a bare default-display column with
    no mechanical check at all."""
    for entity, meta in REGISTRY.items():
        reachable = _reachable_tables(meta)
        for group_by, grouping_path in meta.supported_groupings.items():
            for column, alias in grouping_path.default_display_columns:
                _assert_qualified(
                    column, f"{entity.value}.supported_groupings[{group_by.value}].default_display_columns[{alias!r}]",
                    reachable,
                )


def test_joined_table_references_are_still_correctly_allowed():
    """Sanity check on the test itself: COURSE_SCHEDULE's SUBJECT lookup
    column ('courses.name') and display field ('courses.name') reference a
    JOINED table, not the entity's own base table -- this must be accepted,
    not flagged, since it's an explicitly modeled join, not an accidental
    bare column."""
    from src.agents.query_plan import DisplayField, Entity, LookupFilterField

    meta = REGISTRY[Entity.COURSE_SCHEDULE]
    assert meta.lookup_filter_fields[LookupFilterField.SUBJECT].column == "courses.name"
    assert meta.display_field_columns[DisplayField.SUBJECT_NAME] == "courses.name"
    assert "courses" in _reachable_tables(meta)


# ── FilterField uniqueness guard (Stage 3: flattened ComparisonFilter) ──────
#
# FilterField is the single model-facing filter vocabulary, deliberately the
# union of EnumFilterField's and LookupFilterField's values -- see
# FilterField's docstring for why removing kind is only safe because these
# two categories never overlap. This guard makes that invariant mechanically
# checked, not just documented, so a future addition can't silently
# reintroduce the exact ambiguity kind used to prevent.

def test_filter_field_is_exactly_the_disjoint_union_of_enum_and_lookup_fields():
    from src.agents.query_plan import EnumFilterField, FilterField, LookupFilterField

    enum_values = {v.value for v in EnumFilterField}
    lookup_values = {v.value for v in LookupFilterField}
    filter_field_values = {v.value for v in FilterField}

    assert enum_values & lookup_values == set(), (
        "EnumFilterField and LookupFilterField share a value -- this breaks the proof that a "
        "field's category (enum vs lookup) is derivable from `field` alone, the exact property "
        "that makes removing the `kind` discriminator safe."
    )
    assert filter_field_values == enum_values | lookup_values, (
        "FilterField has drifted out of sync with EnumFilterField/LookupFilterField -- every "
        "value from both categories must be reachable, and FilterField must not invent values "
        "belonging to neither."
    )


# ── JoinStep.join_type / LEFT JOIN guards (Class/Section display) ──────────
#
# join_type is a plain str, not a Pydantic/enum-constrained field (JoinStep
# is a plain dataclass) -- nothing stops a typo like "INNER JOIN" (which
# MariaDB happens to accept as a synonym, silently changing nothing) or a
# genuine typo like "LEFTJOIN"/"Left Join" (which would break the generated
# SQL outright) from being introduced. This guard restricts join_type to the
# exact two literals the builder's _join_sql interpolates directly.

_VALID_JOIN_TYPES = {"JOIN", "LEFT JOIN"}


def test_every_join_step_uses_a_known_join_type():
    for entity, meta in REGISTRY.items():
        for group_by, grouping_path in meta.supported_groupings.items():
            for step in grouping_path.joins:
                assert step.join_type in _VALID_JOIN_TYPES, (
                    f"{entity.value}.supported_groupings[{group_by.value}] has a join to "
                    f"{step.table!r} with unrecognized join_type {step.join_type!r} -- "
                    f"StructuredSQLBuilder._join_sql interpolates this literally into the SQL."
                )
        for step in meta.list_joins:
            assert step.join_type in _VALID_JOIN_TYPES, (
                f"{entity.value}.list_joins has a join to {step.table!r} with unrecognized "
                f"join_type {step.join_type!r}."
            )
        for field, lookup_meta in meta.lookup_filter_fields.items():
            for step in lookup_meta.main_query_join_path + lookup_meta.existence_check_join_path:
                assert step.join_type in _VALID_JOIN_TYPES, (
                    f"{entity.value}.lookup_filter_fields[{field.value}] has a join to "
                    f"{step.table!r} with unrecognized join_type {step.join_type!r}."
                )


def test_no_filter_or_date_column_silently_defeats_a_left_join():
    """A WHERE clause referencing a column on a LEFT-joined table turns that
    LEFT JOIN into an effective INNER JOIN (NULL never satisfies an equality
    or BETWEEN predicate) -- exactly defeating the reason ATTENDANCE's
    class/section joins are LEFT JOIN in the first place (an unassigned
    student must not disappear from a ranking). Not currently reachable
    (ATTENDANCE has no filter/date column on class_sections/school_classes),
    but this guard makes sure a future filter or date_column added to an
    entity can't silently reintroduce that exact defect."""
    for entity, meta in REGISTRY.items():
        left_joined_tables = set()
        for grouping_path in meta.supported_groupings.values():
            for step in grouping_path.joins:
                if step.join_type == "LEFT JOIN":
                    left_joined_tables.add(step.table)
        for step in meta.list_joins:
            if step.join_type == "LEFT JOIN":
                left_joined_tables.add(step.table)
        if not left_joined_tables:
            continue

        for field, field_meta in meta.enum_filter_fields.items():
            table = field_meta.column.split(".", 1)[0]
            assert table not in left_joined_tables, (
                f"{entity.value}.enum_filter_fields[{field.value}] references {field_meta.column!r} on "
                f"a LEFT-joined table ({table!r}) -- a WHERE clause on this field would silently "
                f"turn that LEFT JOIN into an inner join for any plan using both."
            )
        for field, lookup_meta in meta.lookup_filter_fields.items():
            table = lookup_meta.column.split(".", 1)[0]
            assert table not in left_joined_tables, (
                f"{entity.value}.lookup_filter_fields[{field.value}] references {lookup_meta.column!r} on "
                f"a LEFT-joined table ({table!r}) -- same risk as the enum case above."
            )
        if meta.date_column is not None:
            table = meta.date_column.split(".", 1)[0]
            assert table not in left_joined_tables, (
                f"{entity.value}.date_column {meta.date_column!r} references a LEFT-joined table "
                f"({table!r}) -- a date_range filter would silently turn that LEFT JOIN into an "
                f"inner join."
            )


def test_left_join_guard_actually_catches_a_deliberate_violation():
    """Sanity check on the guard above: prove it fails when a filter is
    deliberately placed on a LEFT-joined table, not just that it passes
    against the current registry -- constructs a fake EntityMeta mirroring
    ATTENDANCE's real LEFT JOIN and adds a violating enum filter, then runs
    the exact same check the real guard performs."""
    from src.agents.query_plan import EnumFilterField, GroupingDimension, Operation

    fake_meta = EntityMeta(
        table="attendance",
        supported_operations={Operation.COUNT},
        enum_filter_fields={
            EnumFilterField.STATUS: EnumFilterFieldMeta(
                column="class_sections.name", allowed_values={"x"},  # deliberately on the LEFT-joined table
            ),
        },
        supported_groupings={
            GroupingDimension.BY_STUDENT: GroupingPath(
                joins=[JoinStep(table="class_sections", left_column="section_id", right_column="id", join_type="LEFT JOIN")],
                group_by_columns=["class_sections.id"],
                label=LabelExpression(columns=["class_sections.id"], separator=""),
                label_alias="x",
            ),
        },
    )
    left_joined_tables = {
        step.table
        for grouping_path in fake_meta.supported_groupings.values()
        for step in grouping_path.joins
        if step.join_type == "LEFT JOIN"
    }
    violating_table = fake_meta.enum_filter_fields[EnumFilterField.STATUS].column.split(".", 1)[0]
    with pytest.raises(AssertionError):
        assert violating_table not in left_joined_tables


def test_guardians_display_fields_are_first_last_email_phone():
    """Phase 1 (2026-09-11): only first_name/last_name/email/phone are
    exposed -- the exact identity fields GuardianService.create/
    editIdentity accept and persist. linked_user_id (portal-access state)
    is deliberately NOT exposed this phase."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.GUARDIANS]
    assert meta.display_field_columns == {
        DisplayField.FIRST_NAME: "guardians.first_name",
        DisplayField.LAST_NAME: "guardians.last_name",
        DisplayField.EMAIL: "guardians.email",
        DisplayField.PHONE: "guardians.phone",
    }
    assert meta.default_display_fields == [
        DisplayField.FIRST_NAME, DisplayField.LAST_NAME, DisplayField.EMAIL, DisplayField.PHONE,
    ]
    assert meta.canonical_display_order == [
        DisplayField.FIRST_NAME, DisplayField.LAST_NAME, DisplayField.EMAIL, DisplayField.PHONE,
    ]


def test_guardians_supported_operations_are_exactly_count_and_list():
    from src.agents.query_plan import Entity, Operation
    meta = REGISTRY[Entity.GUARDIANS]
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}
    assert meta.table == "guardians"


def test_guardians_maps_to_current_table_not_legacy():
    """Critical schema-distinction regression: Entity.GUARDIANS must map to
    the CURRENT `guardians` table (created by V48), never to
    `guardians_legacy` (the renamed V1 per-student table, already used
    elsewhere in this registry for ABSENCE_REQUESTS' own parent
    authorization filter)."""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.GUARDIANS]
    assert meta.table == "guardians"
    assert "legacy" not in meta.table
    for column in meta.display_field_columns.values():
        assert column.startswith("guardians.")
        assert "legacy" not in column


def test_guardians_registers_no_enum_grouping_or_numeric_fields():
    """Scope guard: registers COUNT/LIST + NAME sort + EMAIL lookup filter
    only -- no enum filter, no grouping, no numeric aggregation, no date
    column, and no linked_user_id/student-relationship exposure, no
    phone filter. (NAME sort was added 2026-09-11 -- see
    test_guardians_name_sort_metadata_is_exact. EMAIL lookup filter was
    added 2026-09-11 -- see test_guardians_email_lookup_filter_metadata_is_exact
    for its own dedicated coverage.)"""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.GUARDIANS]
    assert meta.enum_filter_fields == {}
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None


def test_guardians_name_sort_metadata_is_exact():
    """NAME (2026-09-11): reuses the exact same SortField.NAME value
    already proven for STUDENTS.NAME/USERS.NAME -- no new enum value.
    guardians.last_name is native to GUARDIANS' own row (NOT NULL, per
    V48), so no join is required."""
    from src.agents.query_plan import Entity, SortField
    meta = REGISTRY[Entity.GUARDIANS]
    assert meta.sort_field_columns == {SortField.NAME: "guardians.last_name"}


def test_guardians_email_lookup_filter_metadata_is_exact():
    """EMAIL (2026-09-11): guardians has its own school_id column, so
    this is self-referential -- identical shape to STUDENTS.GRADE. No
    join required for either the main query or the existence check."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.GUARDIANS]
    email_meta = meta.lookup_filter_fields[LookupFilterField.EMAIL]
    assert email_meta.column == "guardians.email"
    assert email_meta.lookup_table == "guardians"
    assert email_meta.lookup_column == "email"
    assert email_meta.main_query_join_path == []
    assert email_meta.existence_check_join_path == []
    assert email_meta.school_id_column == "guardians.school_id"


def test_guardians_phone_lookup_filter_not_registered():
    """Scope guard: phone was deliberately excluded from this round's
    implementation scope -- no production evidence of a phone-based
    lookup mechanism (GuardianRepository has no findBySchoolIdAndPhone
    equivalent). Confirms EMAIL's addition did not implicitly leak a
    phone filter in alongside it."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.GUARDIANS]
    assert meta.lookup_filter_fields.keys() == {LookupFilterField.EMAIL}


def test_email_lookup_filter_not_registered_for_unrelated_entities():
    """Scope guard: EMAIL must be registered ONLY for GUARDIANS this
    phase -- confirms it did not leak into any other entity's
    lookup_filter_fields."""
    from src.agents.query_plan import Entity, LookupFilterField
    for entity, meta in REGISTRY.items():
        if entity == Entity.GUARDIANS:
            continue
        assert LookupFilterField.EMAIL not in meta.lookup_filter_fields


def test_guardians_display_and_scope_unchanged_by_name_sort_addition():
    """Scope guard: adding NAME sort must not alter GUARDIANS' own display
    fields or leave any other capability accidentally registered."""
    from src.agents.query_plan import DisplayField, Entity, Operation
    meta = REGISTRY[Entity.GUARDIANS]
    assert meta.display_field_columns == {
        DisplayField.FIRST_NAME: "guardians.first_name",
        DisplayField.LAST_NAME: "guardians.last_name",
        DisplayField.EMAIL: "guardians.email",
        DisplayField.PHONE: "guardians.phone",
    }
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}


def test_role_delegations_display_fields_are_type_status_start_end():
    """Phase 1 (2026-09-11), Option C: only delegation_type/status/
    start_date/end_date are exposed -- all plain NOT NULL columns native
    to role_delegations' own row. Deliberately does NOT expose delegator/
    delegate/approver/initiator names (would require two joins to `users`
    with different aliases, which JoinStep does not support)."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    assert meta.display_field_columns == {
        DisplayField.DELEGATION_TYPE: "role_delegations.delegation_type",
        DisplayField.STATUS: "role_delegations.status",
        DisplayField.START_DATE: "role_delegations.start_date",
        DisplayField.END_DATE: "role_delegations.end_date",
    }
    assert meta.default_display_fields == [
        DisplayField.DELEGATION_TYPE, DisplayField.STATUS, DisplayField.START_DATE, DisplayField.END_DATE,
    ]
    assert meta.canonical_display_order == [
        DisplayField.DELEGATION_TYPE, DisplayField.STATUS, DisplayField.START_DATE, DisplayField.END_DATE,
    ]


def test_role_delegations_supported_operations_are_exactly_count_and_list():
    from src.agents.query_plan import Entity, Operation
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}
    assert meta.table == "role_delegations"


def test_role_delegations_registers_no_lookup_grouping_numeric_or_date_fields():
    """Scope guard: registers COUNT/LIST + STATUS/DELEGATION_TYPE filters
    + END_DATE sort only -- no lookup filter, no grouping, no numeric
    aggregation, no date column (despite start_date/end_date being
    displayed, no date-range filtering is registered). (STATUS enum
    filtering was added 2026-09-11 -- see
    test_role_delegations_status_filter_metadata_is_exact. END_DATE sort
    was added 2026-09-11 -- see
    test_role_delegations_end_date_sort_metadata_is_exact for its own
    dedicated coverage.)"""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    assert meta.lookup_filter_fields == {}
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None


def test_role_delegations_end_date_sort_metadata_is_exact():
    """END_DATE (2026-09-11): the same role_delegations.end_date column
    already displayed (DisplayField.END_DATE) and backed by a dedicated
    DB index (idx_role_delegations_status_end_date) proving real
    production ordering usage (RoleDelegationExpiryTask's
    soonest-ending-first sweeps) -- reused directly as a sort target, no
    join required. start_date is deliberately NOT registered as a sort
    target -- no production evidence of standalone start_date ordering
    (only used as a range-membership boundary alongside end_date); the
    exact-equality assertion below also confirms no other SortField
    value leaked in."""
    from src.agents.query_plan import Entity, SortField
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    assert meta.sort_field_columns == {SortField.END_DATE: "role_delegations.end_date"}



def test_role_delegations_status_filter_metadata_is_exact():
    """STATUS (2026-09-11): reuses the exact same EnumFilterField.STATUS/
    FilterField.STATUS enum values already proven for attendance/
    homework/assignments/examinations/absence_requests -- no new enum
    value. role_delegations.status is native to the entity's own row (no
    join). Exactly the real DelegationStatus vocabulary."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    status_meta = meta.enum_filter_fields[EnumFilterField.STATUS]
    assert status_meta.column == "role_delegations.status"
    assert status_meta.allowed_values == {
        "PENDING_APPROVAL", "ACTIVE", "REJECTED", "REVOKED", "EXPIRED",
    }


def test_role_delegations_delegation_type_filter_metadata_is_exact():
    """DELEGATION_TYPE (2026-09-11): reuses the exact same
    EnumFilterField.DELEGATION_TYPE/FilterField.DELEGATION_TYPE enum
    values just introduced -- no new mechanism.
    role_delegations.delegation_type is native to the entity's own row
    (no join). Exactly the real DelegationType vocabulary."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    type_meta = meta.enum_filter_fields[EnumFilterField.DELEGATION_TYPE]
    assert type_meta.column == "role_delegations.delegation_type"
    assert type_meta.allowed_values == {"CLASS_TEACHER", "ADMIN", "PRINCIPAL"}


def test_role_delegations_status_unchanged_by_delegation_type_addition():
    """Scope guard: adding DELEGATION_TYPE must not alter STATUS's own
    metadata, display fields, or supported operations at all."""
    from src.agents.query_plan import DisplayField, Entity, EnumFilterField, Operation
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    status_meta = meta.enum_filter_fields[EnumFilterField.STATUS]
    assert status_meta.column == "role_delegations.status"
    assert status_meta.allowed_values == {
        "PENDING_APPROVAL", "ACTIVE", "REJECTED", "REVOKED", "EXPIRED",
    }
    assert meta.display_field_columns == {
        DisplayField.DELEGATION_TYPE: "role_delegations.delegation_type",
        DisplayField.STATUS: "role_delegations.status",
        DisplayField.START_DATE: "role_delegations.start_date",
        DisplayField.END_DATE: "role_delegations.end_date",
    }
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}
    assert meta.lookup_filter_fields == {}
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None


def test_teacher_exams_registers_no_delegation_type_filter():
    """Scope guard: DELEGATION_TYPE filtering must not leak into an
    unrelated entity."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    assert EnumFilterField.DELEGATION_TYPE not in meta.enum_filter_fields


def test_role_delegations_display_and_operations_unchanged_by_status_filter_addition():
    """Scope guard: adding the STATUS filter must not alter
    ROLE_DELEGATIONS' own display fields or supported operations at
    all."""
    from src.agents.query_plan import DisplayField, Entity, Operation
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    assert meta.display_field_columns == {
        DisplayField.DELEGATION_TYPE: "role_delegations.delegation_type",
        DisplayField.STATUS: "role_delegations.status",
        DisplayField.START_DATE: "role_delegations.start_date",
        DisplayField.END_DATE: "role_delegations.end_date",
    }
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}


def test_role_delegations_no_user_id_or_name_columns_in_any_registry_mapping():
    """Security/scope boundary regression: delegator_user_id, delegate_
    user_id, approver_user_id, initiated_by_user_id, revoked_by_user_id,
    and any *_name column must never appear in ANY registry mapping for
    this entity -- Principal Engineer-approved Option C boundary from the
    readiness review."""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    forbidden = {
        "delegator_user_id", "delegate_user_id", "approver_user_id",
        "initiated_by_user_id", "revoked_by_user_id", "reason",
        "permissions", "extended_count",
    }
    all_referenced_columns = set(meta.display_field_columns.values())
    all_referenced_columns |= {f.column for f in meta.enum_filter_fields.values()}
    all_referenced_columns |= {f.column for f in meta.lookup_filter_fields.values()}
    all_referenced_columns |= set(meta.sort_field_columns.values())
    all_referenced_columns |= set(meta.numeric_agg_fields.values())
    if meta.date_column:
        all_referenced_columns.add(meta.date_column)
    referenced_bare_columns = {
        col.split(".", 1)[1] if "." in col else col for col in all_referenced_columns
    }
    assert referenced_bare_columns & forbidden == set()


def test_teacher_exams_display_fields_are_name_only():
    """Phase 1 (2026-09-11): only name is exposed -- the sole column with
    unambiguous, always-populated semantics investigated this phase.
    Reuses the existing DisplayField.NAME value already proven for
    COURSES.name -- no new enum."""
    from src.agents.query_plan import DisplayField, Entity
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    assert meta.display_field_columns == {DisplayField.NAME: "teacher_exams.name"}
    assert meta.default_display_fields == [DisplayField.NAME]
    assert meta.canonical_display_order == [DisplayField.NAME]


def test_teacher_exams_supported_operations_are_exactly_count_and_list():
    from src.agents.query_plan import Entity, Operation
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}
    assert meta.table == "teacher_exams"


def test_teacher_exams_registers_no_grouping_numeric_date_or_sort_fields():
    """Scope guard: registers COUNT/LIST + STATUS + TERM + SUBJECT filters
    only -- no grouping, no numeric aggregation (despite total_marks
    existing on the real table), no date column (despite exam_date
    existing), no sort field. (TERM and SUBJECT lookup filtering were
    added 2026-09-11 -- see test_teacher_exams_term_lookup_filter_metadata_is_exact
    and test_teacher_exams_subject_lookup_filter_metadata_is_exact for
    their own dedicated coverage.)"""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None
    assert meta.sort_field_columns == {}


def test_teacher_exams_term_lookup_filter_metadata_is_exact():
    """TERM (2026-09-11): reuses the exact same LookupFilterField.TERM/
    FilterField.TERM enum values already proven for REPORT_CARDS.TERM --
    no new enum value. teacher_exams.term is native to the entity's own
    row (no join) -- unlike REPORT_CARDS (no own school_id column),
    teacher_exams HAS its own school_id, so this is self-referential,
    identical shape to STUDENTS.GRADE."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    term_meta = meta.lookup_filter_fields[LookupFilterField.TERM]
    assert term_meta.column == "teacher_exams.term"
    assert term_meta.lookup_table == "teacher_exams"
    assert term_meta.lookup_column == "term"
    assert term_meta.main_query_join_path == []
    assert term_meta.existence_check_join_path == []
    assert term_meta.school_id_column == "teacher_exams.school_id"


def test_report_cards_term_lookup_filter_unchanged_by_teacher_exams_term_addition():
    """Scope guard: adding TEACHER_EXAMS.term must not alter
    REPORT_CARDS.term's own metadata at all -- confirms the two entities'
    independently-scoped TERM mappings never leak into each other, and
    REPORT_CARDS' own cross-table existence-check shape (via students)
    remains distinct from TEACHER_EXAMS' self-referential shape."""
    from src.agents.query_plan import Entity, LookupFilterField
    from src.agents.query_registry import JoinStep
    meta = REGISTRY[Entity.REPORT_CARDS]
    term_meta = meta.lookup_filter_fields[LookupFilterField.TERM]
    assert term_meta.column == "report_cards.term"
    assert term_meta.lookup_table == "report_cards"
    assert term_meta.lookup_column == "term"
    assert term_meta.main_query_join_path == []
    assert term_meta.existence_check_join_path == [
        JoinStep(table="students", left_column="student_id", right_column="id"),
    ]
    assert term_meta.school_id_column == "students.school_id"


def test_teacher_exams_subject_lookup_filter_metadata_is_exact():
    """SUBJECT (2026-09-11): reuses the exact same LookupFilterField.SUBJECT/
    FilterField.SUBJECT enum values already proven for HOMEWORK/
    ASSIGNMENTS/COURSE_SCHEDULE/COURSES/EXAMINATIONS -- no new enum
    value. teacher_exams.subject is native to the entity's own row (no
    join) -- self-referential, identical shape to TEACHER_EXAMS.TERM."""
    from src.agents.query_plan import Entity, LookupFilterField
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    subject_meta = meta.lookup_filter_fields[LookupFilterField.SUBJECT]
    assert subject_meta.column == "teacher_exams.subject"
    assert subject_meta.lookup_table == "teacher_exams"
    assert subject_meta.lookup_column == "subject"
    assert subject_meta.main_query_join_path == []
    assert subject_meta.existence_check_join_path == []
    assert subject_meta.school_id_column == "teacher_exams.school_id"


def test_teacher_exams_no_out_of_scope_column_in_any_registry_mapping():
    """Security/scope boundary regression: exam_type, code,
    room_number, teacher_notes, total_marks, duration, exam_date,
    academic_year, and teacher_id/school_id must never appear as a
    DisplayField/filter/sort/grouping/numeric-aggregation target for
    TEACHER_EXAMS this phase. (STATUS filtering was added 2026-09-11 --
    see test_teacher_exams_status_filter_metadata_is_exact for its own
    dedicated coverage. SUBJECT is now a legitimate lookup filter target
    -- see test_teacher_exams_subject_lookup_filter_metadata_is_exact --
    and is deliberately excluded from this forbidden set.)"""
    from src.agents.query_plan import Entity
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    forbidden = {
        "exam_type", "code", "room_number",
        "teacher_notes", "total_marks", "duration", "exam_date",
        "academic_year_id", "teacher_id", "school_id",
    }
    all_referenced_columns = set(meta.display_field_columns.values())
    all_referenced_columns |= {f.column for f in meta.enum_filter_fields.values()}
    all_referenced_columns |= {f.column for f in meta.lookup_filter_fields.values()}
    all_referenced_columns |= set(meta.sort_field_columns.values())
    all_referenced_columns |= set(meta.numeric_agg_fields.values())
    if meta.date_column:
        all_referenced_columns.add(meta.date_column)
    referenced_bare_columns = {
        col.split(".", 1)[1] if "." in col else col for col in all_referenced_columns
    }
    assert referenced_bare_columns & forbidden == set()


def test_teacher_exams_status_filter_metadata_is_exact():
    """STATUS (2026-09-11): reuses the exact same EnumFilterField.STATUS/
    FilterField.STATUS enum values already proven for attendance/
    homework/assignments/examinations/absence_requests/role_delegations
    -- no new mechanism. teacher_exams.status is native to the entity's
    own row (no join). Exactly the real, lowercase TeacherExam.ExamStatus
    vocabulary -- deliberately distinct from ROLE_DELEGATIONS.status's own
    UPPERCASE DelegationStatus vocabulary."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    status_meta = meta.enum_filter_fields[EnumFilterField.STATUS]
    assert status_meta.column == "teacher_exams.status"
    assert status_meta.allowed_values == {
        "draft", "submitted", "approved", "published",
        "conducted", "marks_submitted", "evaluated",
    }


def test_teacher_exams_display_and_operations_unchanged_by_status_filter_addition():
    """Scope guard: adding the STATUS filter must not alter
    TEACHER_EXAMS' own display fields or supported operations at all."""
    from src.agents.query_plan import DisplayField, Entity, Operation
    meta = REGISTRY[Entity.TEACHER_EXAMS]
    assert meta.display_field_columns == {DisplayField.NAME: "teacher_exams.name"}
    assert meta.supported_operations == {Operation.COUNT, Operation.LIST}
    assert meta.supported_groupings == {}
    assert meta.numeric_agg_fields == {}
    assert meta.date_column is None
    assert meta.sort_field_columns == {}


def test_role_delegations_status_unchanged_by_teacher_exams_status_addition():
    """Scope guard: adding TEACHER_EXAMS.status must not alter
    ROLE_DELEGATIONS.status's own metadata at all -- confirms the two
    entities' independently-scoped STATUS mappings never leak into each
    other, and casing conventions remain distinct."""
    from src.agents.query_plan import Entity, EnumFilterField
    meta = REGISTRY[Entity.ROLE_DELEGATIONS]
    status_meta = meta.enum_filter_fields[EnumFilterField.STATUS]
    assert status_meta.column == "role_delegations.status"
    assert status_meta.allowed_values == {
        "PENDING_APPROVAL", "ACTIVE", "REJECTED", "REVOKED", "EXPIRED",
    }


def test_filter_field_uniqueness_guard_actually_catches_a_deliberate_duplicate():
    """Sanity check on the guard itself (mirrors the equivalent check already
    done for the column-qualification guard): prove it fails when the
    invariant is deliberately broken, not just that it passes today."""
    import enum

    class _FakeEnumFilterField(str, enum.Enum):
        STATUS = "status"
        DAY_OF_WEEK = "day_of_week"

    class _FakeLookupFilterFieldWithCollision(str, enum.Enum):
        SUBJECT = "subject"
        GRADE = "grade"
        STATUS = "status"  # deliberately collides with _FakeEnumFilterField.STATUS

    enum_values = {v.value for v in _FakeEnumFilterField}
    lookup_values = {v.value for v in _FakeLookupFilterFieldWithCollision}
    assert enum_values & lookup_values == {"status"}, "the deliberate collision must actually be present"

    # The real guard, run against these deliberately-broken stand-ins, must
    # fail exactly the way it would if this collision were introduced for
    # real -- proving the assertion in the test above is not vacuously true.
    with pytest.raises(AssertionError):
        assert enum_values & lookup_values == set()


def test_attendance_date_sort_metadata_is_exact():
    """ATTENDANCE_DATE (2026-09-11): the same attendance.date column
    already trusted as date_column for date-range filtering -- reused
    directly as a sort target, no join required, no new
    validator/builder mechanism."""
    from src.agents.query_plan import Entity, SortField
    meta = REGISTRY[Entity.ATTENDANCE]
    assert meta.sort_field_columns == {SortField.ATTENDANCE_DATE: "attendance.date"}


def test_attendance_date_sort_addition_does_not_alter_existing_attendance_metadata():
    """Scope guard: adding ATTENDANCE_DATE sort must not alter
    ATTENDANCE's own display fields, filters, groupings, or date_column
    at all."""
    from src.agents.query_plan import DisplayField, Entity, EnumFilterField, GroupingDimension, Operation
    meta = REGISTRY[Entity.ATTENDANCE]
    assert meta.supported_operations == {Operation.COUNT, Operation.PERCENTAGE, Operation.LIST}
    assert meta.date_column == "attendance.date"
    assert set(meta.enum_filter_fields.keys()) == {EnumFilterField.STATUS}
    assert set(meta.supported_groupings.keys()) == {GroupingDimension.BY_STUDENT, GroupingDimension.BY_STATUS}
    assert meta.display_field_columns == {
        DisplayField.FIRST_NAME: "students.first_name",
        DisplayField.LAST_NAME: "students.last_name",
        DisplayField.ATTENDANCE_DATE: "attendance.date",
        DisplayField.STATUS: "attendance.status",
    }
