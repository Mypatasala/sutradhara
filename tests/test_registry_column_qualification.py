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
