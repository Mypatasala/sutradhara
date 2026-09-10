"""
Deterministic relationship/metadata registry for the structured query
pipeline -- the single source of truth mapping a semantic Entity/
GroupingDimension/FilterField choice to real tables, columns, and joins.

The model NEVER sees this file's contents and NEVER supplies a table name,
column name, join condition, or alias. Every join is a plain (table,
left_column, right_column) equi-join pair -- no raw SQL fragments -- and
every label is a structured CONCAT_WS specification, composed by
structured_sql_builder.py, never authored as a string here.

IMPORTANT: code outside this module must never treat an Entity's `.value`
as a table name. The only correct way to get the physical table for an
entity is `REGISTRY[entity].table`. This keeps Entity semantic rather than
physically coupled, even though today's enum values happen to read the same
as the tables they back.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .query_plan import (
    DisplayField,
    Entity,
    EnumFilterField,
    GroupingDimension,
    LookupFilterField,
    NumericField,
    Operation,
    SortField,
)


@dataclass(frozen=True)
class JoinStep:
    """A single equi-join: {join_type} {table} ON {prev_table_or_alias}.{left_column} = {table}.{right_column}

    join_type defaults to "JOIN" (inner) -- every join in this registry was
    inner-only until this field was added, so the default preserves every
    existing join byte-for-byte. Set to "LEFT JOIN" for an OPTIONAL
    relationship where a missing match must not remove the anchor row --
    e.g. a student's class/section (students.section_id is nullable): an
    unassigned student must still appear in an attendance ranking with a
    NULL class/section, never silently disappear because of a display-only
    join (see ATTENDANCE.BY_STUDENT's own joins for the concrete case this
    was added for)."""

    table: str
    left_column: str
    right_column: str
    join_type: str = "JOIN"


@dataclass(frozen=True)
class LabelExpression:
    """CONCAT_WS(separator, columns[0], columns[1], ...) -- structured,
    never a raw SQL string."""

    columns: List[str]
    separator: str = " - "


@dataclass(frozen=True)
class GroupingPath:
    joins: List[JoinStep]
    group_by_columns: List[str]
    label: LabelExpression
    label_alias: str
    default_order_by: List[str] = field(default_factory=list)
    # (qualified_column, alias) pairs ALWAYS included in the SELECT for any
    # query using this grouping, regardless of operation -- entirely
    # registry-owned, never a model-selectable DisplayField. Exists so a
    # grouping can declare supplementary, always-present display columns
    # (e.g. a student's class/section alongside a per-student attendance
    # aggregate) without the model ever being asked to request them.
    default_display_columns: List["tuple[str, str]"] = field(default_factory=list)


@dataclass(frozen=True)
class EnumFilterFieldMeta:
    # Fully qualified as "<owning_table>.<column>", e.g. "attendance.status"
    # -- NEVER a bare column name. This was bare originally (safe only
    # because every entity queried a single table with no joins); BY_STUDENT
    # introducing attendance's first join exposed the gap the moment a
    # joined table (students) turned out to share a column name (status),
    # producing a live "Column 'status' in SELECT is ambiguous" MariaDB
    # error. Every registry-owned physical column reference (this field,
    # EntityMeta.date_column, sort_field_columns values, and the base-table
    # entries of display_field_columns) must be qualified for the same
    # reason -- see structured_sql_builder.py's module docstring and
    # tests/test_registry_column_qualification.py, which enforces this
    # convention mechanically so it can't silently regress.
    column: str
    allowed_values: Set[str]


@dataclass(frozen=True)
class LookupFilterFieldMeta:
    """Dynamic/tenant-specific filter value, validated by a real existence
    check (see query_validator.py) rather than a fixed set.

    Deliberately keeps TWO separate join paths, since they serve different
    purposes and are not generally the same joins:
      - main_query_join_path: how the ENTITY's own query (e.g.
        course_schedule) reaches the lookup table (e.g. courses) so the
        filter's column is available in the main SELECT's WHERE clause.
      - existence_check_join_path: how the VALIDATOR's independent
        existence check reaches a school-scoping column STARTING FROM
        lookup_table (e.g. courses -> class_sections, since courses has no
        direct school_id column) -- this is a completely different join
        than main_query_join_path and must not be conflated with it.
    """

    column: str  # qualified column reference used in the MAIN query's WHERE, e.g. "courses.name"
    lookup_table: str
    lookup_column: str
    main_query_join_path: List[JoinStep] = field(default_factory=list)
    existence_check_join_path: List[JoinStep] = field(default_factory=list)
    school_id_column: str = "school_id"  # column (after existence_check_join_path) used to scope the existence check to the caller's school


@dataclass(frozen=True)
class EntityMeta:
    table: str
    supported_operations: Set[Operation]
    display_field_columns: Dict[DisplayField, str] = field(default_factory=dict)
    default_display_fields: List[DisplayField] = field(default_factory=list)
    canonical_display_order: List[DisplayField] = field(default_factory=list)
    enum_filter_fields: Dict[EnumFilterField, EnumFilterFieldMeta] = field(default_factory=dict)
    lookup_filter_fields: Dict[LookupFilterField, LookupFilterFieldMeta] = field(default_factory=dict)
    date_column: Optional[str] = None
    # Explicit allowlist of numeric columns operation=average/sum may
    # aggregate over -- deliberately NOT derived from display_field_columns
    # (which mixes strings/dates/enums with no type guarantee, and a value
    # landing there for display purposes must never silently become
    # aggregatable). Membership here is what actually authorizes a
    # NumericField as a valid QueryPlan.aggregate_target for this entity
    # (QueryPlanValidator checks this dict, not the enum alone) -- a
    # NumericField value existing in the model-facing enum only bounds what
    # the model may attempt to name, same relationship FilterField has to
    # enum_filter_fields/lookup_filter_fields. A future entry must be added
    # only after confirming (per the 2026-09-07 AVERAGE/SUM investigation)
    # that the column requires no join through a table with real
    # one-to-many multiplicity relative to this entity's own base row --
    # a duplicated row corrupts SUM/AVERAGE far more insidiously than it
    # would COUNT (COUNT(*) is inflated by +1 per duplicate; SUM is
    # inflated by the full duplicated value, and AVERAGE's mean is skewed
    # toward it with extra weight -- a wrong-looking-plausible answer, not
    # an obviously-wrong one).
    numeric_agg_fields: Dict[NumericField, str] = field(default_factory=dict)
    supported_groupings: Dict[GroupingDimension, GroupingPath] = field(default_factory=dict)
    sort_field_columns: Dict[SortField, str] = field(default_factory=dict)
    school_id_column: str = "school_id"  # bare column on `table` itself, used only for the lookup existence check's own scoping; row-level authorization remains entirely OPA's job
    # Joins ALWAYS included for operation=LIST, regardless of group_by --
    # the LIST-equivalent of GroupingPath.joins, but scoped to the LIST
    # operation itself rather than to a grouping dimension. Exists because
    # LIST's display fields can legitimately need a table the base table
    # alone doesn't have (e.g. ATTENDANCE has no student name column of its
    # own) even with no grouping requested at all -- see
    # structured_sql_builder.py's _collect_joins, which is the one and only
    # place this is consumed; entity-agnostic and registry-owned, never a
    # per-entity special case in the builder.
    list_joins: List[JoinStep] = field(default_factory=list)


REGISTRY: Dict[Entity, EntityMeta] = {
    Entity.STUDENTS: EntityMeta(
        table="students",
        supported_operations={Operation.COUNT, Operation.LIST},
        display_field_columns={
            DisplayField.FIRST_NAME: "students.first_name",
            DisplayField.LAST_NAME: "students.last_name",
        },
        default_display_fields=[DisplayField.FIRST_NAME, DisplayField.LAST_NAME],
        canonical_display_order=[DisplayField.FIRST_NAME, DisplayField.LAST_NAME],
        lookup_filter_fields={
            # GRADE, not an EnumFilterField: verified against the app's own
            # source (my_patasala/.../appadmin/model/PlatformGradeConfig.java)
            # that grade labels are a per-school-configurable ordered list
            # (e.g. ["1".."10"] or ["KG","1".."12"]), not one fixed global
            # set -- a hardcoded Python allowed_values set would either
            # reject a school's real grade or accept one it doesn't have.
            # lookup_table=table="students" (self-referential: the existence
            # check runs directly against the same table the main query
            # already targets) -- no join needed either way, matching the
            # empty main_query_join_path/existence_check_join_path below.
            LookupFilterField.GRADE: LookupFilterFieldMeta(
                column="students.grade",
                lookup_table="students",
                lookup_column="grade",
                main_query_join_path=[],
                existence_check_join_path=[],
                school_id_column="students.school_id",
            ),
        },
        supported_groupings={
            GroupingDimension.BY_CLASS: GroupingPath(
                joins=[
                    JoinStep(table="class_sections", left_column="section_id", right_column="id"),
                    JoinStep(table="school_classes", left_column="school_class_id", right_column="id"),
                ],
                group_by_columns=["class_sections.id", "school_classes.name", "class_sections.name"],
                label=LabelExpression(columns=["school_classes.name", "class_sections.name"], separator=" - "),
                label_alias="class_name",
                default_order_by=["school_classes.level", "class_sections.name"],
            ),
        },
        sort_field_columns={SortField.NAME: "students.last_name"},
    ),
    Entity.ATTENDANCE: EntityMeta(
        table="attendance",
        supported_operations={Operation.COUNT, Operation.PERCENTAGE, Operation.LIST},
        # LIST support added 2026-09-02 -- "show me attendance" was
        # previously always structurally unanswerable (no plan could ever
        # satisfy it): ATTENDANCE had no LIST-capable shape at all. Minimal,
        # evidence-based default shape, verified against the real
        # `attendance` table (id, date, notes, status, course_id,
        # student_id, marked_by, version): a record's own date and status
        # come straight from its own columns; student identity is only
        # reachable via student_id -> students, so list_joins below is
        # required even with group_by=NONE. Deliberately does NOT include
        # course/subject (course_id -> courses.name) or class/section
        # (student_id -> students -> class_sections/school_classes) in the
        # default shape -- neither was asked for, both would need an
        # additional join purely for display, and unlike BY_STUDENT
        # grouping (where class/section disambiguates same-named students
        # collapsed into one aggregate row), a flat list has no such
        # disambiguation need: every row is already a distinct record.
        display_field_columns={
            DisplayField.FIRST_NAME: "students.first_name",
            DisplayField.LAST_NAME: "students.last_name",
            DisplayField.ATTENDANCE_DATE: "attendance.date",
            DisplayField.STATUS: "attendance.status",
        },
        default_display_fields=[
            DisplayField.FIRST_NAME, DisplayField.LAST_NAME,
            DisplayField.ATTENDANCE_DATE, DisplayField.STATUS,
        ],
        canonical_display_order=[
            DisplayField.FIRST_NAME, DisplayField.LAST_NAME,
            DisplayField.ATTENDANCE_DATE, DisplayField.STATUS,
        ],
        # Plain INNER JOIN, unlike BY_STUDENT's LEFT JOINs to class_sections/
        # school_classes below: attendance.student_id is NOT NULL (FK
        # constraint), so every attendance row is guaranteed to have a real
        # matching student -- no risk of an inner join silently dropping a
        # record the way it would for the nullable section_id chain.
        list_joins=[JoinStep(table="students", left_column="student_id", right_column="id")],
        enum_filter_fields={
            EnumFilterField.STATUS: EnumFilterFieldMeta(
                column="attendance.status", allowed_values={"present", "absent", "late", "excused"}
            ),
        },
        date_column="attendance.date",
        supported_groupings={
            # Verified against the real production schema (my_patasala's
            # V1__baseline.sql MariaDB migration, NOT the stale demo
            # school_dump.sql SQLite fixture): attendance.student_id ->
            # students.id directly; students itself owns first_name/
            # last_name (no students -> users join needed). group_by_columns
            # leads with students.id (a stable identifier) -- the name
            # columns are label-only and never the thing rows are grouped by,
            # since names are not guaranteed unique -- confirmed live: real
            # seeded data has multiple distinct students sharing the same
            # first+last name (different students.id, different section),
            # which is exactly why default_display_columns below adds
            # class/section, disambiguating same-named students in any
            # ranking result rather than letting them look like duplicate
            # rows of one person.
            #
            # class_sections/school_classes are joined with LEFT JOIN, not
            # the default inner JOIN: students.section_id is nullable, and a
            # student with no section assigned must still appear in an
            # attendance ranking (with a NULL class/section) -- an inner
            # join here would silently drop them from the ranking entirely
            # merely because of a display-only addition, potentially hiding
            # the true lowest/highest-attendance student.
            GroupingDimension.BY_STUDENT: GroupingPath(
                joins=[
                    JoinStep(table="students", left_column="student_id", right_column="id"),
                    JoinStep(table="class_sections", left_column="section_id", right_column="id", join_type="LEFT JOIN"),
                    JoinStep(table="school_classes", left_column="school_class_id", right_column="id", join_type="LEFT JOIN"),
                ],
                group_by_columns=[
                    "students.id", "students.first_name", "students.last_name",
                    "class_sections.id", "class_sections.name",
                    "school_classes.id", "school_classes.name",
                ],
                label=LabelExpression(columns=["students.first_name", "students.last_name"], separator=" "),
                label_alias="student_name",
                default_display_columns=[
                    ("school_classes.name", "class_name"),
                    ("class_sections.name", "section_name"),
                ],
            ),
            # BY_STATUS (P0-2, 2026-09-05): groups by the exact same column
            # already used by EnumFilterField.STATUS above (attendance.status)
            # -- no join needed, since it's a column on ATTENDANCE's own base
            # table, unlike BY_STUDENT above which crosses into students/
            # class_sections/school_classes.
            GroupingDimension.BY_STATUS: GroupingPath(
                joins=[],
                group_by_columns=["attendance.status"],
                label=LabelExpression(columns=["attendance.status"], separator=""),
                label_alias="status",
            ),
        },
    ),
    Entity.HOMEWORK: EntityMeta(
        table="homework",
        supported_operations={Operation.COUNT, Operation.LIST},
        enum_filter_fields={
            EnumFilterField.STATUS: EnumFilterFieldMeta(
                column="homework.status", allowed_values={"pending", "submitted", "graded", "late"}
            ),
        },
        lookup_filter_fields={
            # SUBJECT (2026-09-07): unlike COURSE_SCHEDULE.SUBJECT below,
            # homework.subject is a plain varchar column NATIVE to
            # homework's own row (verified against my_patasala's actual
            # V1__baseline.sql -- homework has its own `subject` and
            # `school_id` columns; it does NOT join through `courses`/
            # `subjects` to reach either). Both join paths are therefore
            # deliberately EMPTY -- do not "fix" this into a join through
            # courses; there is no join to make, and adding one would be
            # both unnecessary and wrong (homework.subject is denormalized
            # free text, not FK-backed).
            LookupFilterField.SUBJECT: LookupFilterFieldMeta(
                column="homework.subject",
                lookup_table="homework",
                lookup_column="subject",
                main_query_join_path=[],
                existence_check_join_path=[],
                school_id_column="homework.school_id",
            ),
        },
        supported_groupings={
            # BY_STATUS (P0-2): same pattern as ATTENDANCE.BY_STATUS above --
            # groups by the exact same column already used by
            # EnumFilterField.STATUS (homework.status), no join needed.
            GroupingDimension.BY_STATUS: GroupingPath(
                joins=[],
                group_by_columns=["homework.status"],
                label=LabelExpression(columns=["homework.status"], separator=""),
                label_alias="status",
            ),
            # BY_SUBJECT (2026-09-08): same no-join pattern as BY_STATUS
            # above -- groups by the exact same column already used by
            # LookupFilterField.SUBJECT (homework.subject), a plain native
            # column on homework's own row (see that filter's own comment
            # for why no join exists or is needed).
            GroupingDimension.BY_SUBJECT: GroupingPath(
                joins=[],
                group_by_columns=["homework.subject"],
                label=LabelExpression(columns=["homework.subject"], separator=""),
                label_alias="subject",
            ),
        },
    ),
    # ASSIGNMENTS (Phase 1, 2026-09-08): deliberately narrow scope -- COUNT,
    # LIST, status filter, BY_STATUS grouping only. Counts/lists ROWS IN THE
    # assignments TABLE, not student-assignment relationships:
    # assignments.student_id is nullable and its write-path population was
    # not fully traced (a pre-existing application/domain property, not
    # something this registration resolves) -- so deliberately NO
    # lookup_filter_fields (no course/subject filter), NO date_column, NO
    # numeric_agg_fields (grade is nullable-until-graded and a raw score on
    # a per-assignment scale; points varies per assignment; AVG(grade) has
    # no consistent unit), and NO student/course grouping this phase.
    # Authorization: OPA's admin/teacher/student/parent.rego each already
    # apply an identical OR-shaped filter to {"homework", "assignments"}
    # ("student_id IN (...) OR course_id IN (...)"), so no new authorization
    # or filter-injector code is needed -- AliasAwareFilterInjector is
    # already fully generic. assignments has no school_id column of its own
    # (confirmed against my_patasala's V1__baseline.sql/V77 migration), so
    # (unlike homework) no lookup_filter_fields/school_id_column-dependent
    # feature is registered here.
    Entity.ASSIGNMENTS: EntityMeta(
        table="assignments",
        supported_operations={Operation.COUNT, Operation.LIST},
        display_field_columns={
            DisplayField.TITLE: "assignments.title",
            DisplayField.STATUS: "assignments.status",
        },
        default_display_fields=[DisplayField.TITLE, DisplayField.STATUS],
        canonical_display_order=[DisplayField.TITLE, DisplayField.STATUS],
        enum_filter_fields={
            EnumFilterField.STATUS: EnumFilterFieldMeta(
                column="assignments.status",
                allowed_values={"not_started", "in_progress", "submitted", "graded", "overdue"},
            ),
        },
        supported_groupings={
            # BY_STATUS: same no-join pattern as HOMEWORK.BY_STATUS above --
            # groups by the exact same column already used by
            # EnumFilterField.STATUS (assignments.status), no join needed.
            GroupingDimension.BY_STATUS: GroupingPath(
                joins=[],
                group_by_columns=["assignments.status"],
                label=LabelExpression(columns=["assignments.status"], separator=""),
                label_alias="status",
            ),
        },
    ),
    Entity.REPORT_CARDS: EntityMeta(
        table="report_cards",
        # date_column wired (P1, 2026-09-05): report_cards.issue_date was
        # already a registered DisplayField/SortField column but had never
        # been wired for date_range filtering -- this is a pure registry
        # addition, reusing the exact same column already in use elsewhere
        # in this EntityMeta, unlocking date_range (e.g. LAST_MONTH,
        # YESTERDAY) for "report cards issued last month"-style questions.
        date_column="report_cards.issue_date",
        # COUNT added alongside BY_TERM below (P0-2, 2026-09-05): grouping
        # is only ever reachable through an aggregate operation (see
        # QueryPlanValidator's group_by/AGGREGATE_OPERATIONS rule) -- adding
        # a GroupingPath with no aggregate operation registered would leave
        # BY_TERM exactly as unreachable/orphaned as the defect this
        # workstream fixes (see COURSE_SCHEDULE.supported_operations' own
        # comment below for the same reasoning, applied there too). Uses the
        # exact same generic COUNT(*) path every other entity's COUNT
        # already goes through -- no new authorization path, no special-
        # casing. LIST's own behavior is completely untouched.
        # AVERAGE added (Phase 1, 2026-09-07): the sole approved target
        # after investigation -- report_cards.overall_percentage is a plain
        # column on report_cards' own base row (no join), and
        # report_cards' own DB-enforced UNIQUE(student_id, term,
        # academic_year) natural key structurally rules out row
        # duplication via any grouping join (report_cards is the "one"
        # side of every plausible grouping relationship, e.g.
        # report_cards -> students -> class_sections is strictly N:1
        # outward, never the reverse). SUM is deliberately NOT enabled --
        # no demonstrated use case even on this column (summing
        # percentages/GPAs across students isn't a meaningful question).
        # SQL-builder support (Phase 2, 2026-09-07) is implemented --
        # structured_sql_builder.py emits AVG(report_cards.overall_percentage)
        # AS average for this operation, fully tested. SUM remains
        # unsupported: no entity registers it, and the builder still raises
        # NotImplementedError if it is ever reached.
        supported_operations={Operation.COUNT, Operation.LIST, Operation.AVERAGE},
        numeric_agg_fields={
            NumericField.OVERALL_PERCENTAGE: "report_cards.overall_percentage",
        },
        lookup_filter_fields={
            # TERM (Phase 1, 2026-09-10): report_cards.term is a plain
            # varchar column NATIVE to report_cards' own row (verified
            # against my_patasala's actual V1__baseline.sql/V13 natural-key
            # migration) -- main_query_join_path is therefore empty, exactly
            # like HOMEWORK.SUBJECT. report_cards has NO school_id column of
            # its own, so the existence check reaches school scoping via
            # students (report_cards.student_id -> students.id ->
            # students.school_id), one join shorter than COURSE_SCHEDULE.
            # SUBJECT's own courses -> class_sections chain but the same
            # underlying pattern (reach a school_id column that doesn't
            # exist on lookup_table itself). Reuses the SAME
            # "report_cards.term" column already registered below as
            # DisplayField.TERM and already used by BY_TERM grouping --
            # both are left completely unmodified by this addition.
            LookupFilterField.TERM: LookupFilterFieldMeta(
                column="report_cards.term",
                lookup_table="report_cards",
                lookup_column="term",
                main_query_join_path=[],
                existence_check_join_path=[
                    JoinStep(table="students", left_column="student_id", right_column="id"),
                ],
                school_id_column="students.school_id",
            ),
        },
        display_field_columns={
            DisplayField.TERM: "report_cards.term",
            DisplayField.ACADEMIC_YEAR: "report_cards.academic_year",
            DisplayField.OVERALL_GRADE: "report_cards.overall_grade",
            DisplayField.OVERALL_PERCENTAGE: "report_cards.overall_percentage",
            DisplayField.CLASS_TEACHER_NAME: "report_cards.class_teacher_name",
            DisplayField.REMARKS: "report_cards.remarks",
            DisplayField.ISSUE_DATE: "report_cards.issue_date",
        },
        default_display_fields=[DisplayField.TERM, DisplayField.OVERALL_GRADE, DisplayField.OVERALL_PERCENTAGE],
        canonical_display_order=[
            DisplayField.TERM, DisplayField.ACADEMIC_YEAR, DisplayField.OVERALL_GRADE,
            DisplayField.OVERALL_PERCENTAGE, DisplayField.CLASS_TEACHER_NAME, DisplayField.REMARKS,
            DisplayField.ISSUE_DATE,
        ],
        sort_field_columns={SortField.ISSUE_DATE: "report_cards.issue_date"},
        supported_groupings={
            # BY_TERM (P0-2): groups by report_cards.term -- the SAME plain
            # varchar column already exposed as DisplayField.TERM above, NOT
            # my_patasala's vestigial `terms` table. No join needed, since
            # it's a column on REPORT_CARDS' own base table.
            GroupingDimension.BY_TERM: GroupingPath(
                joins=[],
                group_by_columns=["report_cards.term"],
                label=LabelExpression(columns=["report_cards.term"], separator=""),
                label_alias="term",
            ),
        },
    ),
    Entity.COURSE_SCHEDULE: EntityMeta(
        table="course_schedule",
        # COUNT added alongside BY_DAY_OF_WEEK below (P0-2, 2026-09-05): this
        # entity's pre-existing BY_SUBJECT grouping was ALREADY unreachable
        # via the validator (LIST is the only supported operation, and
        # grouping requires an aggregate one -- see
        # test_list_with_by_subject_rejected in test_query_validator.py,
        # which predates this change and is left intact) -- exactly the
        # "orphaned grouping dimension" defect this workstream targets.
        # Adding BY_DAY_OF_WEEK with no aggregate operation available would
        # reproduce that same defect immediately. Uses the exact same
        # generic COUNT(*) path every other entity's COUNT already goes
        # through -- no new authorization path, no special-casing. LIST's
        # own behavior (including BY_SUBJECT remaining rejected under LIST)
        # is completely untouched.
        supported_operations={Operation.COUNT, Operation.LIST},
        enum_filter_fields={
            EnumFilterField.DAY_OF_WEEK: EnumFilterFieldMeta(
                column="course_schedule.day_of_week",
                allowed_values={"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"},
            ),
        },
        lookup_filter_fields={
            LookupFilterField.SUBJECT: LookupFilterFieldMeta(
                column="courses.name",
                lookup_table="courses",
                lookup_column="name",
                # course_schedule -> courses, for the MAIN query's own join.
                main_query_join_path=[JoinStep(table="courses", left_column="course_id", right_column="id")],
                # courses -> class_sections, for the EXISTENCE CHECK's own
                # school-scoping -- courses has no direct school_id column.
                existence_check_join_path=[JoinStep(table="class_sections", left_column="section_id", right_column="id")],
                school_id_column="class_sections.school_id",
            ),
        },
        display_field_columns={
            DisplayField.SUBJECT_NAME: "courses.name",
            DisplayField.START_TIME: "course_schedule.start_time",
            DisplayField.END_TIME: "course_schedule.end_time",
            DisplayField.ROOM: "course_schedule.room",
            DisplayField.DAY_OF_WEEK: "course_schedule.day_of_week",
        },
        default_display_fields=[DisplayField.SUBJECT_NAME, DisplayField.START_TIME, DisplayField.END_TIME, DisplayField.ROOM],
        canonical_display_order=[
            DisplayField.DAY_OF_WEEK, DisplayField.SUBJECT_NAME, DisplayField.START_TIME,
            DisplayField.END_TIME, DisplayField.ROOM,
        ],
        supported_groupings={
            GroupingDimension.BY_SUBJECT: GroupingPath(
                joins=[JoinStep(table="courses", left_column="course_id", right_column="id")],
                group_by_columns=["courses.id", "courses.name"],
                label=LabelExpression(columns=["courses.name"], separator=""),
                label_alias="subject",
                default_order_by=["courses.name"],
            ),
            # BY_DAY_OF_WEEK (P0-2): groups by the exact same column already
            # used by EnumFilterField.DAY_OF_WEEK above
            # (course_schedule.day_of_week) -- no join needed, since it's a
            # column on COURSE_SCHEDULE's own base table.
            GroupingDimension.BY_DAY_OF_WEEK: GroupingPath(
                joins=[],
                group_by_columns=["course_schedule.day_of_week"],
                label=LabelExpression(columns=["course_schedule.day_of_week"], separator=""),
                label_alias="day_of_week",
            ),
        },
        sort_field_columns={SortField.START_TIME: "course_schedule.start_time"},
    ),
    Entity.USERS: EntityMeta(
        table="users",
        # COUNT added (P0-1, 2026-09-04/05): "how many teachers" was
        # previously structurally unanswerable -- USERS had no aggregate
        # shape at all. Uses the exact same generic COUNT(*) path every
        # other entity's COUNT already goes through in
        # structured_sql_builder.py (no special-casing); the ROLE lookup
        # filter below is what actually narrows it to "teachers" -- see that
        # filter's own comment for why. display_field_columns/LIST are
        # completely untouched by this addition.
        supported_operations={Operation.COUNT, Operation.LIST},
        display_field_columns={
            DisplayField.FIRST_NAME: "users.first_name",
            DisplayField.LAST_NAME: "users.last_name",
            DisplayField.EMAIL: "users.email",
            DisplayField.PHONE: "users.phone",
            DisplayField.DEPARTMENT: "users.department",
        },
        default_display_fields=[DisplayField.FIRST_NAME, DisplayField.LAST_NAME, DisplayField.EMAIL, DisplayField.PHONE, DisplayField.DEPARTMENT],
        canonical_display_order=[
            DisplayField.FIRST_NAME, DisplayField.LAST_NAME, DisplayField.EMAIL,
            DisplayField.PHONE, DisplayField.DEPARTMENT,
        ],
        # Deliberately mirrors USER_COLUMNS from my_patasala/policy/opa/*.rego
        # -- "password" cannot appear here because DisplayField never defines
        # it at all, a stronger guarantee than a runtime allowlist check.
        lookup_filter_fields={
            # ROLE (P0-1): verified against my_patasala's actual production
            # schema (V1__baseline.sql) -- users has NO role/role_id column
            # of its own; role membership only exists via the join table
            # `user_roles` (user_id, role_id) to `roles` (id, name), where
            # roles.name is a MariaDB ENUM('ADMIN','PARENT','PRINCIPAL',
            # 'STUDENT','SUPERUSER','TEACHER') -- matching my_patasala's own
            # RoleEnum (dto/RoleEnum.java) exactly. Deliberately does NOT
            # reference teacher_profiles (a different, narrower concept --
            # not every TEACHER-role user need have one, and role itself is
            # the thing being asked about here).
            #
            # Categorized as a LOOKUP, not an enum, even though the value
            # set is effectively fixed and global -- see FilterField.ROLE's
            # docstring in query_plan.py for why: EnumFilterFieldMeta has no
            # join support, and reaching roles.name requires one. The
            # existence check mirrors SUBJECT's real-data semantics (roles
            # has no school_id of its own -- it's shared, unscoped reference
            # data across every school -- so the check instead verifies the
            # named role is actually assigned to at least one user at the
            # caller's own school): existence_check_join_path starts at
            # `roles` and joins back through user_roles -> users, scoped by
            # users.school_id, exactly analogous to how COURSE_SCHEDULE's
            # SUBJECT existence check reaches class_sections.school_id
            # because `courses` itself has no direct school_id column.
            LookupFilterField.ROLE: LookupFilterFieldMeta(
                column="roles.name",
                lookup_table="roles",
                lookup_column="name",
                # users -> user_roles -> roles, for the MAIN query's own join.
                main_query_join_path=[
                    JoinStep(table="user_roles", left_column="id", right_column="user_id"),
                    JoinStep(table="roles", left_column="role_id", right_column="id"),
                ],
                # roles -> user_roles -> users, for the EXISTENCE CHECK's own
                # school-scoping -- roles has no direct school_id column (it
                # is global reference data, not tenant data).
                existence_check_join_path=[
                    JoinStep(table="user_roles", left_column="id", right_column="role_id"),
                    JoinStep(table="users", left_column="user_id", right_column="id"),
                ],
                school_id_column="users.school_id",
            ),
        },
    ),
    Entity.SCHOOL_CLASSES: EntityMeta(
        table="school_classes",
        # COUNT only, deliberately -- there is no display_field_columns
        # entry (no LIST support) and no supported_groupings: a "class" is
        # the grade-level subject itself here, not something to further
        # group or list fields from yet. school_id is a direct column (no
        # join needed for the plain COUNT this supports today), matching
        # OPA's own row_filter shape for this table (admin.rego etc.:
        # "id IN (SELECT id FROM school_classes WHERE school_id = %v)").
        supported_operations={Operation.COUNT},
    ),
    # COURSES (Phase 1, 2026-09-10): deliberately narrow scope -- COUNT and
    # LIST only, mirroring SCHOOL_CLASSES' own minimal bootstrap above. One
    # courses row represents one offered course section-instance. courses
    # has no school_id column of its own (confirmed against my_patasala's
    # actual V1__baseline.sql) -- tenant scoping is entirely OPA's job via
    # its existing row_filter for this table (admin/teacher/student/
    # parent.rego, identical across all four: "section_id IN (SELECT id
    # FROM class_sections WHERE school_id = %v)"), a bare-column subquery
    # filter requiring no join in the application-level SQL itself, exactly
    # like SCHOOL_CLASSES' own filter shape above.
    #
    # display_field_columns limited to name/code/credits -- the only
    # columns confirmed write-path-authoritative against my_patasala's own
    # CourseService/CreateCourseRequestDTO/UpdateCourseRequestDTO:
    # `semester` is never set by either DTO (dead at creation time, only
    # ever read); `enrollment_count`/`max_enrollment` are never assigned
    # anywhere in the entire my_patasala codebase (traced exhaustively via
    # grep across src/main/java -- both are permanently NULL in practice).
    # Deliberately NO lookup_filter_fields (no name/code/section/instructor
    # filter), NO enum_filter_fields, NO supported_groupings (no
    # term/section grouping -- both would need the courses -> class_sections
    # join this phase avoids entirely), NO numeric_agg_fields (credits is
    # real but no aggregation need has been demonstrated -- see the
    # NumericField/numeric_agg_fields docstrings for why a numeric column
    # existing is never sufficient justification on its own), NO
    # date_column (courses has no date-typed column at all), and NO
    # sort_field_columns this phase.
    Entity.COURSES: EntityMeta(
        table="courses",
        supported_operations={Operation.COUNT, Operation.LIST},
        display_field_columns={
            DisplayField.NAME: "courses.name",
            DisplayField.CODE: "courses.code",
            DisplayField.CREDITS: "courses.credits",
        },
        default_display_fields=[DisplayField.NAME, DisplayField.CODE, DisplayField.CREDITS],
        canonical_display_order=[DisplayField.NAME, DisplayField.CODE, DisplayField.CREDITS],
    ),
}


def get_entity_meta(entity: Entity) -> EntityMeta:
    return REGISTRY[entity]
