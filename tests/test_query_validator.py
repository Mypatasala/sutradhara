import pytest

from src.agents.query_plan import (
    ComparisonFilter,
    DisplayField,
    Entity,
    ExtremeSelector,
    FilterField,
    GroupingDimension,
    Operation,
    PercentageSpec,
    QueryPlan,
    RelativeDate,
    SortField,
    SortSpec,
)
from src.agents.query_registry import REGISTRY
from src.agents.query_validator import QueryPlanValidator, QueryPlanValidationError
from src.agents.query_normalizer import normalize
from src.retrieval.structured_sql_builder import StructuredSQLBuilder


class FakeDB:
    """Returns a match only for a fixed set of (lowercased) values, to
    exercise both the found and not-found lookup-filter paths."""

    KNOWN = {"mathematics": "Mathematics", "5": "5", "10": "10", "teacher": "TEACHER"}

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


def test_homework_by_status_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.HOMEWORK, operation=Operation.COUNT, group_by=GroupingDimension.BY_STATUS)
    validator.validate(plan, school_id=56)  # must not raise


def test_course_schedule_by_day_of_week_grouping_passes(validator):
    plan = QueryPlan(entity=Entity.COURSE_SCHEDULE, operation=Operation.COUNT, group_by=GroupingDimension.BY_DAY_OF_WEEK)
    validator.validate(plan, school_id=56)  # must not raise


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

            plan = QueryPlan(**kwargs)

            expected_valid = (
                operation in meta.supported_operations
                and (group_by == GroupingDimension.NONE or group_by in meta.supported_groupings)
                and (group_by == GroupingDimension.NONE or operation in _ORACLE_AGGREGATE_OPERATIONS)
            )

            if expected_valid:
                resolved = validator.validate(plan, school_id=56)  # must not raise
                canonical = normalize(plan, resolved)
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
