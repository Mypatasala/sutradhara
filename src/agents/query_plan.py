"""
Structured query intent schema for the deterministic query pipeline.

Design context (see docs/architecture/Patasala-OPA-Policy-Status.md "Open
item" and the design-review conversation that produced this): a small local
LLM (llama3.2) proved unreliable at freely generating SQL text -- it
hallucinated identity literals, aliased tables unpredictably, and sometimes
dropped required joins entirely. IdentityFilterGuard and AliasAwareFilter
Injector are deterministic backstops against the first two failure modes,
but they can only detect and reject bad SQL after the fact -- they cannot
make the model choose the *right* joins in the first place.

QueryPlan is the structural fix: the model is constrained (via Ollama/
langchain structured output, i.e. schema-constrained decoding, not just a
prompt instruction) to choose from closed, semantic vocabulary -- WHICH
entity, WHICH operation, WHICH grouping dimension -- and is never given a
field through which it could express a table name, a join condition, a SQL
expression, an alias, or an identity/tenant filter. All of that is owned by
query_registry.py (the relationship metadata) and structured_sql_builder.py
(the deterministic composer). See query_validator.py for the semantic rules
that gate a parsed plan before any SQL is built, and query_normalizer.py for
the canonicalization step that makes "same intent, differently serialized"
plans converge to identical SQL.
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Entity(str, Enum):
    """A Phase-1 allowlist of currently-supported top-level query subjects,
    NOT a permanent claim about what can ever be queried.

    school_classes (grade levels, e.g. "5th Grade") is registered as its own
    subject -- see SCHOOL_CLASSES below. class_sections (subdivisions within
    a class, e.g. "A"/"B") is still deliberately absent: per the product's
    own established UI/API terminology (myPatasala's "Select Class" /
    "Select Section" as two always-distinct controls, `AdminController`'s
    `/classes` vs `/classes/{id}/sections`), a bare "class" means the grade
    level alone, never a class+section combination -- confirmed by product-
    terminology investigation, 2026-09-02. Every current question about
    sections specifically (or the combined "5th Grade - A" grouping) still
    treats them as a dimension of `students` (group_by=by_class), never as a
    subject in their own right. Adding a future Entity.CLASS_SECTIONS (with
    its own registry entry) remains a legitimate, purely additive way to
    support "how many sections does grade 3 have"-style questions later; it
    does not require touching this design's core mechanism.
    """

    STUDENTS = "students"
    ATTENDANCE = "attendance"
    HOMEWORK = "homework"
    REPORT_CARDS = "report_cards"
    COURSE_SCHEDULE = "course_schedule"
    USERS = "users"
    SCHOOL_CLASSES = "school_classes"
    # ASSIGNMENTS (Phase 1, 2026-09-08): deliberately narrow -- count/list/
    # status/by_status ONLY, mirroring HOMEWORK's own original bootstrap
    # scope. Counts/lists ROWS IN THE assignments TABLE, not student-
    # assignment relationships: assignments.student_id is nullable and its
    # write-path population was not fully traced during investigation (a
    # pre-existing application/domain property, not something this
    # registration resolves or needs to) -- so no student-level filtering,
    # no course/subject, no date, and no grade/points querying are exposed
    # this phase. See query_registry.py's ASSIGNMENTS entry for the full
    # investigation citations (OPA policy coverage, grade/points semantic
    # rejection).
    ASSIGNMENTS = "assignments"
    # COURSES (Phase 1, 2026-09-10): deliberately narrow -- count/list ONLY,
    # mirroring SCHOOL_CLASSES' own minimal bootstrap scope. One courses row
    # represents one offered course section-instance. LIST exposes only
    # name/code/credits -- courses.semester, enrollment_count, and
    # max_enrollment were traced against my_patasala's actual services and
    # confirmed dead/unpopulated at every write path (semester is never set
    # by CreateCourseRequestDTO/UpdateCourseRequestDTO; enrollment_count/
    # max_enrollment are never assigned anywhere in the codebase), so none
    # are exposed here. No filters, no groupings, no sort, no numeric
    # aggregation this phase -- courses.section_id/instructor_id are both
    # nullable and deliberately not yet modeled. See query_registry.py's
    # COURSES entry for the full investigation citations (OPA policy
    # coverage, dead-column findings).
    COURSES = "courses"
    # EXAMINATIONS (Phase 1, 2026-09-11): COUNT, LIST, STATUS filter, SUBJECT
    # filter only -- deliberately narrow, mirroring ASSIGNMENTS' own Phase 1
    # bootstrap scope. One examinations row is one student's result for one
    # examination in one course -- unlike ASSIGNMENTS/HOMEWORK,
    # examinations.course_id AND examinations.student_id are BOTH NOT NULL
    # at the DB level, enforced identically at the JPA level
    # (@JoinColumn(nullable = false) on both), and the only real write path
    # (ReportCardGenerationService.generateForStudent) explicitly skips
    # creating a row at all when no course can be resolved -- confirmed no
    # row with a null course or student can exist. No grouping, numeric
    # aggregation (obtained_marks/total_marks deliberately unmodeled this
    # phase), date filtering, or sort is registered. "Subject" and "course"
    # are the same concept here too (courses.name), confirmed directly from
    # ReportCardGenerationService.resolveCourse's own
    # course.getName().equalsIgnoreCase(subjectName) matching logic. See
    # query_registry.py's EXAMINATIONS entry for the full investigation
    # citations (OPA policy coverage per role, join/fanout analysis).
    EXAMINATIONS = "examinations"
    # ABSENCE_REQUESTS (Phase 1, 2026-09-11): COUNT, LIST, STATUS filter
    # only -- mirrors EXAMINATIONS' own Phase 1 bootstrap scope. One row is
    # one student's leave request for a date or date range.
    # absence_requests.student_id is NOT NULL (unlike ASSIGNMENTS/
    # HOMEWORK's own student_id), confirmed the sole authorization anchor
    # and the sole real write path (AttendanceService.submitAbsenceRequest)
    # always sets it via a validated, never-null Student lookup. status is
    # a plain varchar(32) column at the DB level but is enforced as a
    # closed, 4-value vocabulary at the JPA layer
    # (@Enumerated(EnumType.STRING) AbsenceRequest.AbsenceStatus =
    # {pending, forwarded_to_principal, approved, rejected}) -- all four
    # confirmed reachable via real transition methods in AttendanceService
    # (approve/reject/forward), not merely declared. No grouping, numeric
    # aggregation, date filtering, sorting, or lookup filters this phase --
    # absence_requests has no course/subject dimension of any kind. See
    # query_registry.py's ABSENCE_REQUESTS entry for the full investigation
    # citations (OPA policy coverage per role, including the teacher-
    # specific nested-subquery ownership filter -- the first entity in this
    # registry where the teacher authorization shape genuinely differs from
    # admin/principal's own).
    ABSENCE_REQUESTS = "absence_requests"
    # TEACHER_PROFILES (Phase 1, 2026-09-11): COUNT, LIST (designation,
    # department only), EMPLOYMENT_TYPE filter only -- deliberately the
    # narrowest bootstrap of any entity so far. teacher_profiles.user_id ->
    # users.id is UNIQUE (one-to-one, confirmed in V1__baseline.sql), so no
    # fanout risk from the users join used for authorization. employment_type
    # is a plain varchar(20) column at the DB level but is enforced as a
    # closed, 4-value vocabulary at the JPA layer
    # (@Enumerated(EnumType.STRING) TeacherProfile.EmploymentType =
    # {FULL_TIME, PART_TIME, CONTRACT, VISITING}). It was added in a later
    # migration (V26) than the table itself, and the one-time backfill
    # migration (V25, pre-dating V26) that created a profile row for every
    # pre-existing teacher set no columns beyond id/user_id -- so legacy/
    # backfilled rows can have NULL employment_type, and an equality filter
    # on it naturally excludes those rows (no "UNKNOWN" value invented,
    # database behavior unchanged). hire_date and every V26 HR/verification/
    # qualification/registration/experience column (notes, bio, identity
    # verification, qualifications, registrations, experience, subjects)
    # are deliberately NOT exposed this phase -- see query_registry.py's
    # TEACHER_PROFILES entry for the full investigation citations (schema,
    # write-path reliability, per-role OPA coverage including the first
    # entity in this registry with two unconditionally-denied roles at
    # once: parent and student).
    TEACHER_PROFILES = "teacher_profiles"
    # Additional Phase-1-adjacent entities (teacher_exams, guardians,
    # role_delegations) are intentionally NOT yet registered here -- they
    # have OPA coverage but no reviewed registry entry (join paths, display
    # fields, etc.) yet. A question about them correctly falls through to
    # the legacy free-text path via UnresolvedReason.OUT_OF_SCOPE until a
    # registry entry is added for each, following the same pattern as the
    # entities above.


class Operation(str, Enum):
    COUNT = "count"
    LIST = "list"
    PERCENTAGE = "percentage"
    AVERAGE = "average"
    SUM = "sum"


# Operations that produce an aggregate result and are therefore the only
# ones grouping is meaningful with. LIST returns individual rows, so
# LIST + group_by has ambiguous semantics -- QueryPlanValidator rejects that
# combination outright rather than letting the builder guess at it (see the
# Principal Engineer Review finding, 2026-08-30: this exact gap let
# operation=LIST, group_by=BY_CLASS/BY_SUBJECT reach the builder and produce
# invalid GROUP BY SQL). Defined here, not just in the validator, so any
# future aggregate Operation added to the enum is automatically grouping-
# eligible without a second change.
AGGREGATE_OPERATIONS = {Operation.COUNT, Operation.PERCENTAGE, Operation.AVERAGE, Operation.SUM}

# The subset of AGGREGATE_OPERATIONS that require a QueryPlan.aggregate_target
# (WHICH numeric column to aggregate over) -- COUNT/PERCENTAGE need no such
# target (COUNT(*) needs none; PERCENTAGE's target is percentage_of.numerator,
# a categorical filter field, not a numeric column). Defined here, not just
# in the validator, for the same reason AGGREGATE_OPERATIONS is: any future
# numeric-target operation added to Operation is automatically covered by
# QueryPlanValidator's aggregate_target rule without a second change.
NUMERIC_AGGREGATE_OPERATIONS = {Operation.AVERAGE, Operation.SUM}


class GroupingDimension(str, Enum):
    NONE = "none"
    BY_CLASS = "by_class"
    BY_STATUS = "by_status"
    BY_DAY_OF_WEEK = "by_day_of_week"
    BY_SUBJECT = "by_subject"
    BY_TERM = "by_term"
    BY_STUDENT = "by_student"
    BY_ACADEMIC_YEAR = "by_academic_year"


class RelativeDate(str, Enum):
    """Closed vocabulary for date scoping -- deliberately NOT free-form date
    math. The model only ever picks one of these; StructuredSQLBuilder
    resolves it to concrete date bounds using Python's own datetime.now()
    at build time. This removes the entire class of relative-date-
    arithmetic errors the free-text prompt's rules 16/17 could only
    instruct against, not prevent.
    """

    ALL_TIME = "all_time"
    TODAY = "today"
    # A single calendar day, exactly one day before TODAY -- inclusive
    # (start == end == today - 1 day), same shape as TODAY itself. Added
    # 2026-09-05 alongside LAST_7_DAYS below: without a dedicated value,
    # "yesterday" had no correct target in this vocabulary at all.
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    # A true rolling 30-calendar-day window (today plus the preceding 29
    # days -- 30 days total, inclusive of today), NOT a calendar-month
    # approximation. Added 2026-09-02 after a live-traced incident: with no
    # vocabulary entry matching "the last 30 days", the model silently
    # substituted LAST_WEEK, scoping "last 30 days" questions to only 7 days
    # with no error or indication anything was wrong -- a silent
    # correctness bug, not a failure. Deliberately a single named value, not
    # a general LAST_N_DAYS + a model-facing numeric field: only a 30-day
    # need has been demonstrated, and prior structured-output investigations
    # (see intent_agent.py's FilterField docstring) showed adding
    # unnecessary schema surface can itself reduce this model's reliability.
    LAST_30_DAYS = "last_30_days"
    # A true rolling 7-calendar-day window (today plus the preceding 6
    # days -- 7 days total, inclusive of today), deliberately NOT the same
    # window as LAST_WEEK (the previous calendar Monday-Sunday, which does
    # not include today at all). Added 2026-09-05 for the same reason
    # LAST_30_DAYS was: without a dedicated value, "the last 7 days" would
    # have no correct target and risk being silently conflated with
    # LAST_WEEK, exactly the incident LAST_30_DAYS's own docstring records.
    LAST_7_DAYS = "last_7_days"


class EnumFilterField(str, Enum):
    """Filter fields with a small, fixed, known-in-advance value set.
    Registry-internal categorization only -- NOT exposed to the model as its
    own type anymore (see FilterField/ComparisonFilter below); still used as
    query_registry.py's EntityMeta.enum_filter_fields dict key and to decide
    enum-style (allowed-set) validation vs lookup-style (existence-check)
    validation for a given field."""

    STATUS = "status"
    DAY_OF_WEEK = "day_of_week"
    # TEACHER_PROFILES.employment_type (added 2026-09-11): a plain bare
    # column on teacher_profiles' own row (no join needed), closed 4-value
    # vocabulary enforced at the JPA layer -- see Entity.TEACHER_PROFILES'
    # docstring for the legacy-NULL caveat.
    EMPLOYMENT_TYPE = "employment_type"


class LookupFilterField(str, Enum):
    """Filter fields whose values are dynamic, tenant-specific real data
    (e.g. subject names) -- validated by an existence check against the
    caller's own school's data, never against a fixed set. Registry-internal
    categorization only, like EnumFilterField above."""

    SUBJECT = "subject"
    # Deliberately categorized as a lookup, NOT an enum, despite role names
    # LOOKING like a small fixed global set (my_patasala's own RoleEnum:
    # STUDENT, PARENT, TEACHER, ADMIN, PRINCIPAL, SUPERUSER): the enum-filter
    # mechanism (EnumFilterFieldMeta) only ever expresses a bare column on
    # the entity's OWN table, with no join support at all -- but a user's
    # role isn't a column on `users`, it only exists via the users ->
    # user_roles -> roles join (verified against my_patasala's actual
    # V1__baseline.sql migration: users has no role/role_id column of its
    # own). Reaching a joined table's column REQUIRES the lookup mechanism's
    # main_query_join_path, regardless of how fixed the value set feels.
    # The existence check itself mirrors SUBJECT's real-data semantics, not
    # a hardcoded Python set: it confirms the named role is actually
    # assigned to at least one user at the caller's own school (existence-
    # check-join-path roles -> user_roles -> users, scoped by
    # users.school_id) -- see query_registry.py's USERS.lookup_filter_fields
    # entry.
    ROLE = "role"
    # Deliberately categorized as a lookup, NOT an enum, despite grade
    # LOOKING like a small fixed set: the application has a platform-level
    # PlatformGradeConfig system (my_patasala's appadmin/model/
    # PlatformGradeConfig.java) where each school is assigned its own
    # ordered list of grade labels (e.g. ["1".."10"], or ["KG","1".."12"])
    # -- verified in the app's own source, not assumed. A hardcoded Python
    # allowed_values set would either reject a school's real grade ("KG")
    # or accept a grade a specific school doesn't have. Existence-checked
    # against the caller's own school's real students.grade data instead,
    # exactly like SUBJECT.
    GRADE = "grade"
    # TERM (REPORT_CARDS, 2026-09-10): deliberately categorized as a lookup,
    # NOT an enum, despite term labels LOOKING like a small fixed set
    # ("Term 1"/"Term 2"/"Final"): traced against my_patasala's actual
    # ReportCardGenerationService/ReportCardController -- report_cards.term
    # is a plain free-text column with NO server-side enum or fixed-
    # vocabulary validation anywhere; real seeded/test data uses genuinely
    # inconsistent values ("Term 1", "Term 2", "Term 3", "Midterm", "Final").
    # A hardcoded Python allowed_values set would either reject a school's
    # real term label or accept one it doesn't use -- exactly the GRADE
    # rationale above, applied to report_cards.term instead of
    # students.grade. Existence-checked against the caller's own school's
    # real report_cards.term data (via students.school_id, since
    # report_cards has no school_id column of its own -- see
    # query_registry.py's REPORT_CARDS.lookup_filter_fields entry).
    TERM = "term"
    # ACADEMIC_YEAR (REPORT_CARDS, 2026-09-11): same rationale as TERM
    # immediately above, applied to report_cards.academic_year instead --
    # both are set by the exact same ReportCardGenerationService.
    # generateForStudent code path (rc.setAcademicYear(academicYear),
    # unconditional, from the same caller-supplied natural-key parameter
    # pair as term; both columns are part of the same DB-enforced
    # UNIQUE(student_id, term, academic_year) natural key added by V13).
    # Deliberately a lookup, not an enum: academic-year labels ("2025-2026")
    # are per-school free text with no fixed vocabulary anywhere in the
    # application. Existence-checked against the caller's own school's real
    # report_cards.academic_year data (via students.school_id, identical
    # existence-check shape to TERM -- see query_registry.py's
    # REPORT_CARDS.lookup_filter_fields entry).
    ACADEMIC_YEAR = "academic_year"
    # TEACHER_PROFILES.department (2026-09-11): deliberately a lookup, not
    # an enum -- department labels are per-school free text with no
    # master-data/fixed vocabulary anywhere in the application (confirmed
    # against AdminService's own create/update paths, same reasoning
    # already applied to USERS.department's display-only status).
    # teacher_profiles has NO school_id column of its own -- existence-
    # checked via teacher_profiles.user_id -> users.id -> users.school_id,
    # the same authorization anchor already proven for this entity's own
    # row filter -- see query_registry.py's
    # TEACHER_PROFILES.lookup_filter_fields entry.
    DEPARTMENT = "department"
    # TEACHER_PROFILES.designation (2026-09-11): same rationale as
    # DEPARTMENT immediately above, applied to teacher_profiles.designation
    # instead -- confirmed via AdminService's own create (request.get(
    # "designation"), optional) and update (updateDto.getDesignation(),
    # optional) paths: plain free text, no master-data list anywhere in the
    # application, identical reliability profile to DEPARTMENT. Existence-
    # checked via the exact same teacher_profiles.user_id -> users.id ->
    # users.school_id path already proven for DEPARTMENT.
    DESIGNATION = "designation"


class FilterField(str, Enum):
    """The single, MODEL-FACING closed vocabulary for ComparisonFilter.field
    -- deliberately the flat union of every EnumFilterField and
    LookupFilterField value, with NO overlap between the two (verified:
    {"status","day_of_week"} ∩ {"subject","grade"} = ∅ -- see
    tests/test_registry_column_qualification.py's FilterField uniqueness
    guard, which fails loudly if a future addition ever breaks this).

    Because the two source enums' values never collide, which VALIDATION
    STYLE applies (a fixed allowed-values check vs a DB existence check) is
    always fully determined by `field` alone -- there is no case where it's
    ambiguous. The model previously also had to emit a redundant `kind`
    literal ("enum"/"lookup") on every filter to make this same fact explicit
    a second time; live testing proved llama3.2 unreliably omits that literal
    (a discriminated-union decoding failure, not a semantic one), causing a
    hard Pydantic parse failure before QueryPlan even exists -- unrecoverable
    by any validator/normalizer rule, since the object was never parsed.
    Removing `kind` doesn't lose any information the registry didn't already
    have: query_validator.py now derives the same fact by checking which of
    meta.enum_filter_fields / meta.lookup_filter_fields the field belongs to,
    a closed-vocabulary registry lookup, never a natural-language guess."""

    STATUS = "status"
    DAY_OF_WEEK = "day_of_week"
    SUBJECT = "subject"
    GRADE = "grade"
    EMPLOYMENT_TYPE = "employment_type"
    ROLE = "role"
    TERM = "term"
    ACADEMIC_YEAR = "academic_year"
    DEPARTMENT = "department"
    DESIGNATION = "designation"


class ComparisonFilter(BaseModel):
    """Flat filter representation -- no discriminator. `field` alone fully
    determines both which registry sub-dict governs it (enum vs lookup) and,
    transitively, which entities it can even apply to. See FilterField's
    docstring for why this is provably safe, not merely convenient."""

    field: FilterField
    value: str


class PercentageSpec(BaseModel):
    """Explicit numerator/denominator contract for operation=PERCENTAGE.

    Denominator = every row of the entity's table matching the plan's
    date_range and filters, EXCLUDING numerator.field entirely from that
    population (never "AND status != numerator.value" -- simply not
    applied as a population-narrowing condition at all).
    Numerator = the subset of that same population additionally matching
    numerator.field = numerator.value.

    Deliberately ENUM-only for now (not every FilterField) -- every current
    percentage question is a category ratio (e.g. attendance status=present);
    lookup-based percentages aren't a demonstrated need. ComparisonFilter no
    longer encodes this restriction at the type level (both enum- and
    lookup-backed fields now share one flat type), so
    QueryPlanValidator explicitly rejects a lookup-backed numerator instead
    -- see QueryPlanValidator's percentage_of handling.
    """

    numerator: ComparisonFilter


class NumericField(str, Enum):
    """Closed, MODEL-FACING vocabulary for QueryPlan.aggregate_target --
    WHICH numeric column operation=average/sum computes over. Deliberately
    its own enum, not a reuse of DisplayField: DisplayField mixes strings,
    dates, and enums with no type guarantee at all (e.g. STUDENTS.grade or
    REPORT_CARDS.term are both DisplayField values but neither is
    aggregatable), so it cannot itself prove a value is numeric. A field
    only appears here once a human has deliberately reviewed it, confirmed
    it is a real numeric column with genuine aggregate meaning, and added
    it to some EntityMeta.numeric_agg_fields -- registry membership is what
    actually authorizes a target for a given entity (see EntityMeta docs);
    this enum only bounds what the model may even ATTEMPT to name, exactly
    like FilterField bounds ComparisonFilter.field.

    Phase 1 (2026-09-07): scoped to exactly the one target approved after
    investigation -- report_cards.overall_percentage, average only. sum is
    deliberately not enabled anywhere yet (no demonstrated use case even on
    this column -- summing percentages/GPAs across students isn't a
    meaningful school-admin question); gpa is a distinct, deliberately
    deferred follow-on, not included here. Do not add a value here without
    also adding the matching EntityMeta.numeric_agg_fields entry and
    confirming (per the investigation's fan-out analysis) that reaching the
    column requires no join through a table with real one-to-many
    multiplicity relative to the entity's own base row -- a duplicated row
    corrupts SUM/AVERAGE far more insidiously than it would COUNT (see
    query_registry.py's EntityMeta.numeric_agg_fields docstring)."""

    OVERALL_PERCENTAGE = "overall_percentage"


class DisplayField(str, Enum):
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    EMAIL = "email"
    PHONE = "phone"
    DEPARTMENT = "department"
    TERM = "term"
    ACADEMIC_YEAR = "academic_year"
    OVERALL_GRADE = "overall_grade"
    OVERALL_PERCENTAGE = "overall_percentage"
    CLASS_TEACHER_NAME = "class_teacher_name"
    REMARKS = "remarks"
    ISSUE_DATE = "issue_date"
    SUBJECT_NAME = "subject_name"
    START_TIME = "start_time"
    END_TIME = "end_time"
    ROOM = "room"
    DAY_OF_WEEK = "day_of_week"
    # ATTENDANCE list support (added 2026-09-02): the record's own date
    # (attendance.date) and status (attendance.status) -- see query_registry
    # .py's ATTENDANCE entry. Named ATTENDANCE_DATE (not a bare "date") to
    # stay distinct from ISSUE_DATE (report_cards) and any future date-typed
    # display field on a different entity -- DisplayField values are a
    # single flat namespace shared across every entity, so ambiguous generic
    # names are avoided the same way SUBJECT_NAME/CLASS_TEACHER_NAME already
    # are. STATUS is left generic (not ATTENDANCE_STATUS): unlike date,
    # "status" is not ambiguous across today's registered entities, and a
    # future entity with its own status column can safely reuse it -- each
    # EntityMeta.display_field_columns mapping is independently scoped.
    ATTENDANCE_DATE = "attendance_date"
    STATUS = "status"
    # ASSIGNMENTS list support (added 2026-09-08): the row's own title
    # (assignments.title). No entity currently exposes a generic "name" or
    # "title" display field, so this is unambiguous; STATUS above is reused
    # as-is for assignments.status.
    TITLE = "title"
    # COURSES list support (added 2026-09-10): the row's own name/code/
    # credits (courses.name/code/credits). NAME is generic (not
    # COURSE_NAME) -- unambiguous the same way TITLE above is, since no
    # other entity currently exposes a bare "name"/"code"/"credits" display
    # field (STUDENTS/USERS use FIRST_NAME/LAST_NAME, COURSE_SCHEDULE uses
    # SUBJECT_NAME, both already disambiguated). CODE and CREDITS are new,
    # entity-specific concepts with no existing collision risk.
    NAME = "name"
    CODE = "code"
    CREDITS = "credits"
    # HOMEWORK list-shape fix (2026-09-10): the row's own subject
    # (homework.subject) -- a plain, denormalized free-text column NATIVE
    # to homework's own row, the exact same column already used by the
    # existing LookupFilterField.SUBJECT filter and BY_SUBJECT grouping.
    # Deliberately a NEW value, NOT a reuse of SUBJECT_NAME above --
    # SUBJECT_NAME represents a structurally different concept (COURSE_
    # SCHEDULE's joined courses.name), and conflating the two would violate
    # this enum's own established convention of keeping genuinely different
    # data sources distinct (see ATTENDANCE_DATE's docstring above for the
    # same reasoning applied to a date-typed value).
    SUBJECT = "subject"
    # ABSENCE_REQUESTS list support (added 2026-09-11): the row's own
    # reason (absence_requests.reason) -- a plain, user-entered free-text
    # column NATIVE to absence_requests' own row. No other entity currently
    # exposes a "reason" concept, so this is unambiguous.
    REASON = "reason"
    # TEACHER_PROFILES list support (added 2026-09-11): the row's own
    # designation (teacher_profiles.designation) -- a plain, optional
    # free-text position/title column, distinct from the Teacher Role Model
    # concepts (TeacherRoleAssignment, dashboard designation) noted as
    # deferred in the real Java entity's own docstring. DEPARTMENT above is
    # reused as-is (already independently scoped per-entity for USERS'
    # users.department); no new DisplayField needed for it.
    DESIGNATION = "designation"
    # Deliberately never includes "password" or any other identity-guard-
    # blocked column -- the enum itself is the allowlist, a stronger
    # guarantee than a runtime check.


class SortField(str, Enum):
    ISSUE_DATE = "issue_date"
    START_TIME = "start_time"
    NAME = "name"
    # Sentinel, not a physical column: means "the aggregate value this
    # operation itself computed" (the COUNT/PERCENTAGE the builder already
    # aliases as "count"/"percentage"), resolved by StructuredSQLBuilder
    # against its own just-built alias rather than any per-entity registry
    # column mapping. Only meaningful alongside a grouped aggregate -- see
    # QueryPlanValidator's rule for AGGREGATE_VALUE.
    AGGREGATE_VALUE = "aggregate_value"


class SortSpec(BaseModel):
    field: SortField
    direction: Literal["asc", "desc"] = "desc"


class ExtremeSelector(str, Enum):
    """Closed vocabulary for "lowest X" / "highest X" with NO number stated
    in the question -- deliberately distinct from SortSpec+limit, which is
    for an explicit top/bottom N ("5 lowest"). "Lowest" without a stated
    count means every row tied at the minimum aggregate value, not an
    arbitrary single row -- see QueryPlan.extreme's docstring and
    query_lifecycle.py's post-execution extreme-selection step, which is
    where ties are actually resolved (never in SQL -- see structured_sql
    _builder.py's module docstring for why)."""

    LOWEST = "lowest"
    HIGHEST = "highest"


class UnresolvedReason(str, Enum):
    """Required whenever can_answer is False -- distinguishes the two cases
    that MUST be handled differently by the caller (see query_lifecycle.py):

    OUT_OF_SCOPE: the question needs a concept (entity/operation/dimension)
    that the current structured vocabulary genuinely doesn't model at all
    (e.g. fee amounts, notifications). Legacy free-text fallback is
    permitted here -- this is what "registry coverage gap" means
    operationally for a closed-enum schema.

    AMBIGUOUS: the question IS about something within supported structured
    scope, but the model cannot confidently map it to a specific plan (e.g.
    missing a needed detail, or genuinely vague phrasing). This must produce
    a clarification response -- it must NEVER fall back to legacy free-text
    generation, since the vocabulary to answer it correctly already exists;
    falling back here would let the model escape the deterministic
    architecture whenever it merely feels unsure, defeating the purpose of
    this whole redesign.
    """

    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"


class QueryPlan(BaseModel):
    can_answer: bool = True
    unresolved_reason: Optional[UnresolvedReason] = None
    clarification_question: Optional[str] = None

    entity: Optional[Entity] = None
    operation: Optional[Operation] = None
    group_by: GroupingDimension = GroupingDimension.NONE
    display_fields: List[DisplayField] = Field(default_factory=list)
    percentage_of: Optional[PercentageSpec] = None
    # WHICH numeric column operation=average/sum computes over -- required
    # iff operation is average or sum, validated against the target
    # entity's own EntityMeta.numeric_agg_fields (QueryPlanValidator), not
    # inferred from question text. See NumericField's own docstring for the
    # full closed-vocabulary rationale.
    aggregate_target: Optional[NumericField] = None
    filters: List[ComparisonFilter] = Field(default_factory=list)
    date_range: RelativeDate = RelativeDate.ALL_TIME
    # Explicit literal date scoping -- deliberately a SIBLING pair of plain
    # strings, not a native pydantic `date` type and not folded into a
    # discriminated union with date_range. Both reasons come from evidence
    # already in this file: (1) nothing else in QueryPlan uses a native
    # date/datetime type -- every model-facing value is str/Enum/bool/int,
    # so this stays consistent with that pattern; (2) FilterField's own
    # docstring records that a discriminated-union `kind` literal was tried
    # and removed because llama3.2 unreliably omits it, causing a hard
    # Pydantic parse failure before QueryPlan even exists -- unrecoverable
    # by any validator/retry-with-feedback rule. A plain str field that
    # later fails QueryPlanValidator's format check is fully recoverable
    # via that same retry loop; a native-type parse failure would not be.
    # A single explicit day is expressed as start == end (no third field).
    # Mutually exclusive with date_range != ALL_TIME, enforced by
    # QueryPlanValidator, not by the schema itself (the schema-level
    # discriminator approach is exactly what the removed `kind` literal
    # already proved unreliable for this model).
    explicit_start_date: Optional[str] = None
    explicit_end_date: Optional[str] = None
    sort: Optional[SortSpec] = None
    limit: Optional[int] = Field(None, ge=1, le=100)
    distinct: bool = False
    # "Lowest X" / "highest X" with no number stated -- every row tied at
    # the aggregate's min/max, computed in Python AFTER authorization/
    # execution (see query_lifecycle.py), never as a SQL LIMIT (which would
    # arbitrarily cut ties) and never as a second SQL aggregate subquery
    # (which the live-traced authorization pipeline cannot safely scope --
    # see structured_sql_builder.py's module docstring). Mutually exclusive
    # with sort/limit at validation time: an explicit "N lowest" is a
    # different question, answered by sort=aggregate_value + limit=N.
    extreme: Optional[ExtremeSelector] = None


class QueryPlanPatch(BaseModel):
    """A partial plan expressing ONLY the fields a follow-up question
    changes relative to the prior turn's validated-and-normalized QueryPlan.
    Every field is Optional and means "inherit unchanged" when absent.
    `entity` is deliberately not patchable in this minimal design -- a
    follow-up that changes what the question is fundamentally about is
    treated as a fresh QueryPlan, not a patch.

    NOT YET WIRED INTO PRODUCTION: this schema exists per the approved
    design for follow-up handling, but the merge logic (applying a patch
    onto the prior turn's canonical plan) has not been implemented, and
    neither query_lifecycle.py nor intent_agent.py reference this class or
    QueryPlanResponse yet -- resolve_structured() doesn't even accept
    conversation history today. Every question, including follow-ups, is
    currently resolved as a fresh, context-free QueryPlan. Implementing the
    merge is deferred future work, not a bug."""

    operation: Optional[Operation] = None
    group_by: Optional[GroupingDimension] = None
    date_range: Optional[RelativeDate] = None
    filters: Optional[List[ComparisonFilter]] = None
    display_fields: Optional[List[DisplayField]] = None
    sort: Optional[SortSpec] = None
    limit: Optional[int] = Field(None, ge=1, le=100)


class QueryPlanResponse(BaseModel):
    """Top-level structured-output schema INTENDED for a follow-up turn,
    where the model would choose whether a question needs a fresh plan or
    is a refinement of the previous one.

    RESERVED, NOT CURRENTLY USED: no code constructs, parses, or requests
    this schema anywhere today -- resolve_structured() always uses QueryPlan
    directly, regardless of whether conversation history exists. Kept here
    (rather than deleted) because it corresponds to the already-approved
    follow-up design; deleting and later recreating it would be pure churn
    once that work is picked up. If follow-up handling is abandoned instead
    of implemented, this class (and QueryPlanPatch) should be removed at
    that point rather than left as permanent dead code."""

    is_patch: bool = False
    plan: Optional[QueryPlan] = None
    patch: Optional[QueryPlanPatch] = None


# ── Ranking-field coherence: a pure QueryPlan-shape semantic property ───────
#
# Architectural placement note (2026-09-03 Principal Engineer review): this
# lives here, in the schema module itself, deliberately NOT in
# query_validator.py or query_normalizer.py. It is a plain predicate/
# transform over QueryPlan's own fields with zero DB, registry, or
# validator-instance dependency -- the same category as AGGREGATE_OPERATIONS
# just above. query_validator.py depends on query_plan.py already (for
# QueryPlan, AGGREGATE_OPERATIONS, etc.); the reverse is never true, so
# defining this here and having the validator import it introduces no cycle.
# query_normalizer.normalize() was considered and rejected as the home for
# the repair below: normalize()'s own contract ("Canonicalization stage
# BETWEEN validation and SQL building") assumes the plan already passed
# QueryPlanValidator.validate() and takes that call's own return value
# (resolved_lookups) as a required argument -- it cannot run before
# validation without breaking that dependency, and this repair specifically
# MUST run before validation (see clear_incoherent_ranking_fields' docstring
# for why). The two are genuinely different concerns with different data
# dependencies: normalize() converges DIFFERENT VALID SERIALIZATIONS of the
# same already-valid intent (filter order/casing, resolved lookup values);
# this repair instead converges a plan whose OWN internal fields are
# self-contradictory into a coherent one, before validity is even
# established -- a different concern with no DB access needed at all.

def is_ranking_capable(plan: "QueryPlan") -> bool:
    """The one structural precondition a QueryPlan must satisfy for either
    `extreme` or a `sort.field == SortField.AGGREGATE_VALUE` to mean
    anything at all: a grouped aggregate result to rank rows within.
    Neither field is ever meaningful without BOTH a real grouping
    (group_by != NONE) AND an aggregate operation to group -- see
    AGGREGATE_OPERATIONS' own docstring. This is the SINGLE source of truth
    for that rule -- both QueryPlanValidator's own rejection checks and
    clear_incoherent_ranking_fields' repair below apply the exact same
    condition, so a plan is never rejected under one definition of
    "ranking-capable" while being silently repaired under a looser one.
    """
    return plan.group_by != GroupingDimension.NONE and plan.operation in AGGREGATE_OPERATIONS


def clear_incoherent_ranking_fields(plan: "QueryPlan") -> "QueryPlan":
    """Deterministically clears `extreme` and an aggregate-value `sort`
    (+ its `limit`) whenever the plan's OWN group_by/operation choice
    already makes them structurally meaningless. Applied unconditionally to
    every plan resolve_structured() returns (see that method, the single
    canonical point every caller -- direct callers, resolve_structured_
    with_feedback's retry, and the full QueryLifecycleAgent pipeline --
    ultimately goes through) BEFORE QueryPlanValidator ever sees it: this
    guarantees one single QueryPlan contract regardless of which caller
    resolved the plan, rather than a repair that only some higher-level
    caller happens to apply.

    Root-cause investigation (2026-09-02/03): live-traced, non-ranking
    ATTENDANCE questions ("Show attendance for the last 30 days.", "How
    many attendance records are there...", and the pre-existing "What is my
    attendance percentage?" -- confirmed present on the untouched baseline
    via git stash, not introduced by any LAST_30_DAYS/ATTENDANCE-LIST work)
    deterministically produced a spurious `extreme` (or, less often, a
    `sort=aggregate_value` + `limit`) with `group_by=NONE` -- a combination
    QueryPlanValidator already, correctly, rejects. Two targeted prompt-only
    attempts to stop the model emitting the field in the first place each
    showed partial, UNSTABLE improvement (fixing one phrasing while a
    different one regressed) -- exactly the model-specific reliability
    instability this project has previously decided not to chase
    indefinitely with more prompt tuning.

    Safety proof (why this is a repair, not a guess): a GENUINE ranking
    question can only ever be a valid, executable plan if it has BOTH
    group_by set AND an aggregate operation -- that is what "ranking"
    structurally MEANS in this design (there is no way to rank/compare rows
    without grouping them). Live testing across every ranking paraphrase in
    this codebase's own test suite confirms the model reliably sets
    group_by=BY_STUDENT together with extreme/sort whenever ranking
    language is actually present -- this function's condition therefore
    never fires for those plans (proven, not assumed: see
    tests/test_ranking_field_sanitization.py's identity-check tests).
    Whenever the condition DOES fire, the plan was NEVER a valid,
    executable ranking plan to begin with, regardless of what the true user
    intent was -- clearing the field converts an unnecessarily-rejected
    plan into a valid, answerable one, and can never turn a correct ranking
    answer into a wrong non-ranking one (a correct ranking answer requires
    the very precondition this function checks for).

    Explicitly does NOT repair `operation=LIST` combined with a ranking-
    shaped group_by/extreme/sort -- see the dedicated Top-N investigation
    (2026-09-03): unlike clearing extreme/sort (which discards a claim with
    no valid interpretation), rewriting operation=list to percentage would
    require INVENTING percentage_of (never present on a LIST plan, no
    coherent way to recover it from the rest of the plan), and defaulting
    to count instead would silently give a *different, potentially
    misleading* answer (a raw record count is not what "highest/lowest
    attendance" means in this product -- it's the percentage-present rate).
    Neither is an unambiguous transformation of already-present
    information, so neither is performed here -- that failure mode is
    correctly left to the validator's rejection + retry-with-feedback path.

    `sort`/`limit` on a PHYSICAL column (e.g. report_cards.issue_date,
    course_schedule.start_time) are completely untouched -- the aggregate-
    value branch only ever fires for SortField.AGGREGATE_VALUE specifically.
    Never touches filters/percentage_of/date_range/display_fields/entity/
    operation -- scoped exclusively to extreme/sort(aggregate_value)/limit.
    """
    updates = {}
    if plan.extreme is not None and not is_ranking_capable(plan):
        updates["extreme"] = None
        if plan.limit is not None:
            updates["limit"] = None
        if plan.sort is not None:
            updates["sort"] = None
    if plan.sort is not None and plan.sort.field == SortField.AGGREGATE_VALUE and not is_ranking_capable(plan):
        updates["sort"] = None
        updates["limit"] = None
    return plan.model_copy(update=updates) if updates else plan
