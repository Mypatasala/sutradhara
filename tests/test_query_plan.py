import pytest
from pydantic import ValidationError

from src.agents.query_plan import (
    ComparisonFilter,
    Entity,
    EnumFilterField,
    FilterField,
    LookupFilterField,
    Operation,
    PercentageSpec,
    QueryPlan,
    QueryPlanPatch,
)


def test_minimal_plan_defaults():
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT)
    assert plan.can_answer is True
    assert plan.group_by.value == "none"
    assert plan.filters == []
    assert plan.date_range.value == "all_time"
    assert plan.distinct is False


def test_unknown_entity_rejected_at_schema_level():
    with pytest.raises(ValidationError):
        QueryPlan(entity="class_sections", operation=Operation.COUNT)


def test_comparison_filter_has_no_kind_discriminator():
    """Flattened representation: ComparisonFilter is field+value only, no
    discriminator -- see FilterField's docstring for why this is provably
    safe (EnumFilterField and LookupFilterField values never overlap)."""
    assert "kind" not in ComparisonFilter.model_fields
    f = ComparisonFilter(field=FilterField.SUBJECT, value="Mathematics")
    assert f.field == FilterField.SUBJECT
    assert f.value == "Mathematics"


def test_filter_field_accepts_every_registered_field_value():
    """Any of the four currently-known filter fields (from either the
    former enum-style or lookup-style category) parses as a plain
    FilterField -- which category applies is decided later, by the
    registry, in QueryPlanValidator, not at the schema level."""
    for value in ("status", "day_of_week", "subject", "grade"):
        assert ComparisonFilter(field=value, value="x").field.value == value


def test_filter_field_values_have_no_overlap_with_enum_and_lookup_categories():
    """Mechanical proof that the two former categories partition FilterField
    with no overlap -- the exact property that makes deriving 'enum vs
    lookup' from `field` alone unambiguous. See
    tests/test_registry_column_qualification.py for the broader registry-
    level version of this guard."""
    enum_values = {v.value for v in EnumFilterField}
    lookup_values = {v.value for v in LookupFilterField}
    all_filter_field_values = {v.value for v in FilterField}
    assert enum_values & lookup_values == set()
    assert enum_values | lookup_values == all_filter_field_values


def test_limit_bounds_enforced():
    with pytest.raises(ValidationError):
        QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, limit=0)
    with pytest.raises(ValidationError):
        QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, limit=101)
    QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, limit=100)  # boundary ok


def test_percentage_spec_accepts_any_filter_field_at_schema_level():
    """The former ENUM-only restriction on PercentageSpec.numerator was a
    type-level guarantee (numerator: EnumComparisonFilter); flattening
    ComparisonFilter removes that type distinction, so this is now enforced
    by QueryPlanValidator instead (see
    test_query_validator.py::test_percentage_of_with_lookup_backed_field_rejected)
    -- this test documents that the SCHEMA itself no longer blocks it (by
    design), only the validator does."""
    spec = PercentageSpec(numerator=ComparisonFilter(field=FilterField.SUBJECT, value="Mathematics"))
    assert spec.numerator.field == FilterField.SUBJECT


def test_query_plan_patch_all_fields_optional():
    patch = QueryPlanPatch()
    assert patch.operation is None
    assert patch.filters is None


def test_query_plan_patch_has_no_entity_field():
    assert "entity" not in QueryPlanPatch.model_fields
