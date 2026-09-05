"""
Canonicalization stage between validation and SQL building.

A plan that already passed QueryPlanValidator is semantically valid, but two
plans representing the identical intent can still be serialized differently
by the model (filters in a different order, different casing on an enum
value, display_fields listed in a different order). normalize() maps any
such plan onto a single canonical form so the determinism claim becomes:
canonical-plan -> byte-identical SQL, not merely
identically-shaped-plan -> byte-identical SQL.

RelativeDate is deliberately NOT resolved to concrete dates here -- it stays
declarative through normalization and is resolved to literal date bounds
only inside StructuredSQLBuilder, immediately before SQL text is produced
(a pure function of the enum value and the current server time).

explicit_start_date/explicit_end_date (Phase 2, 2026-09-05) follow the
exact same pass-through as RelativeDate, for the exact same reason: a
validator-enforced strict YYYY-MM-DD string already has exactly one valid
textual representation per calendar date (see QueryPlanValidator's
_parse_strict_iso_date, which rejects non-zero-padded input specifically to
guarantee this), so there is no "same intent, differently serialized"
variance for normalize() to canonicalize here -- unlike enum-filter values
(needing case-folding) or lookup values (needing DB-confirmed casing).
Neither field is listed in the `update=` dict below; model_copy() already
carries every field not explicitly overridden through unchanged, which is
how date_range itself has always been "preserved" here too.
"""

from typing import Dict, List

from .query_plan import ComparisonFilter, DisplayField, FilterField, QueryPlan
from .query_registry import REGISTRY


def _filter_sort_key(f: ComparisonFilter):
    return (f.field.value, f.value.lower())


def normalize(plan: QueryPlan, resolved_lookups: Dict[FilterField, str]) -> QueryPlan:
    """Returns a new QueryPlan in canonical form. Assumes `plan` already
    passed QueryPlanValidator.validate(), and that `resolved_lookups` is
    exactly the dict validate() returned for this same plan (mapping each
    lookup-backed FilterField present to its real, DB-confirmed value) --
    normalize() performs no DB access of its own."""
    meta = REGISTRY[plan.entity]

    canonical_filters: List[ComparisonFilter] = []
    for f in plan.filters:
        enum_meta = meta.enum_filter_fields.get(f.field.value)
        if enum_meta is not None:
            canonical_value = next(v for v in enum_meta.allowed_values if v.lower() == f.value.lower())
            canonical_filters.append(ComparisonFilter(field=f.field, value=canonical_value))
        else:
            # Lookup-backed -- validator already confirmed f.field is a
            # registered lookup field and populated resolved_lookups for it.
            canonical_filters.append(ComparisonFilter(field=f.field, value=resolved_lookups[f.field]))
    canonical_filters.sort(key=_filter_sort_key)

    deduped: List[ComparisonFilter] = []
    seen = set()
    for f in canonical_filters:
        key = _filter_sort_key(f)
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    canonical_percentage_of = plan.percentage_of
    if canonical_percentage_of is not None:
        enum_meta = meta.enum_filter_fields[canonical_percentage_of.numerator.field.value]
        canonical_value = next(
            v for v in enum_meta.allowed_values if v.lower() == canonical_percentage_of.numerator.value.lower()
        )
        canonical_percentage_of = canonical_percentage_of.model_copy(
            update={"numerator": ComparisonFilter(field=canonical_percentage_of.numerator.field, value=canonical_value)}
        )

    if plan.display_fields:
        order = meta.canonical_display_order or list(plan.display_fields)
        canonical_display_fields: List[DisplayField] = [d for d in order if d in plan.display_fields]
    else:
        canonical_display_fields = []

    return plan.model_copy(
        update={
            "filters": deduped,
            "percentage_of": canonical_percentage_of,
            "display_fields": canonical_display_fields,
        }
    )
