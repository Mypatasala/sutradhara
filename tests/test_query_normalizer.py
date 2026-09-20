from src.agents.query_plan import (
    ComparisonFilter,
    DisplayField,
    Entity,
    FilterField,
    Operation,
    QueryPlan,
)
from src.agents.query_normalizer import normalize


def test_filter_order_normalized():
    plan_a = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="pending")],
    )
    plan_b = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[ComparisonFilter(field=FilterField.STATUS, value="PENDING")],
    )
    canon_a = normalize(plan_a, {})
    canon_b = normalize(plan_b, {})
    assert canon_a == canon_b


def test_duplicate_filters_deduplicated():
    plan = QueryPlan(
        entity=Entity.HOMEWORK, operation=Operation.COUNT,
        filters=[
            ComparisonFilter(field=FilterField.STATUS, value="pending"),
            ComparisonFilter(field=FilterField.STATUS, value="PENDING"),
        ],
    )
    canon = normalize(plan, {})
    assert len(canon.filters) == 1


def test_lookup_filter_uses_resolved_value_not_raw_text():
    plan = QueryPlan(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="mathematics")],
    )
    canon = normalize(plan, {FilterField.SUBJECT: "Mathematics"})
    assert canon.filters[0].value == "Mathematics"


def test_display_fields_reordered_to_canonical_order():
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        display_fields=[DisplayField.ISSUE_DATE, DisplayField.TERM, DisplayField.OVERALL_GRADE],
    )
    canon = normalize(plan, {})
    assert canon.display_fields == [DisplayField.TERM, DisplayField.OVERALL_GRADE, DisplayField.ISSUE_DATE]


def test_canonical_plan_equal_regardless_of_original_field_order():
    plan_a = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        display_fields=[DisplayField.OVERALL_GRADE, DisplayField.TERM],
    )
    plan_b = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST,
        display_fields=[DisplayField.TERM, DisplayField.OVERALL_GRADE],
    )
    assert normalize(plan_a, {}) == normalize(plan_b, {})


def test_explicit_dates_pass_through_normalize_unchanged():
    """Phase 2: explicit_start_date/explicit_end_date follow the exact same
    pass-through as RelativeDate -- normalize() has no special handling for
    either, and neither is listed in normalize()'s update= dict, so
    model_copy() alone must carry both through byte-for-byte."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT,
        explicit_start_date="2026-08-01", explicit_end_date="2026-08-15",
    )
    canon = normalize(plan, {})
    assert canon.explicit_start_date == "2026-08-01"
    assert canon.explicit_end_date == "2026-08-15"
