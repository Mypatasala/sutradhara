"""
Deterministic semantic validation for a parsed QueryPlan, run BEFORE any SQL
is built. Every rule here fails closed: a plan that doesn't cleanly satisfy
every applicable rule is rejected, never guessed at or partially honored.

This is the layer that turns "the model produced *some* schema-conformant
JSON" into "the model produced a plan we can trust to build correct,
authorized SQL from" -- schema conformance (guaranteed by Ollama/langchain's
structured output) and semantic validity (guaranteed here) are deliberately
separate concerns.
"""

import re
from datetime import datetime
from typing import Dict, List

from .query_plan import (
    AGGREGATE_OPERATIONS,
    ComparisonFilter,
    FilterField,
    GroupingDimension,
    Operation,
    QueryPlan,
    SortField,
    is_ranking_capable,
)
from .query_registry import REGISTRY, EntityMeta

# Exactly 4-2-2 zero-padded digits -- see _parse_strict_iso_date's docstring
# for why this shape check exists alongside strptime's calendar-validity
# check, not as a redundant duplicate of it.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# is_ranking_capable's canonical home is query_plan.py (a pure QueryPlan-
# shape predicate, no validator-instance or DB dependency -- see that
# module for the full architectural placement rationale, 2026-09-03
# Principal Engineer review). Imported here for use in this class's own
# extreme/sort checks below, so both this validator and query_plan.py's
# clear_incoherent_ranking_fields apply the exact same condition.


class QueryPlanValidationError(Exception):
    """Raised when a QueryPlan fails one or more semantic validation rules.
    Callers must treat this as a rejected plan -- retry structured
    generation once with `str(exc)` as corrective feedback, then fail
    closed to a clarification if the retry also fails. NEVER fall back to
    legacy free-text SQL generation for this exception -- see
    query_lifecycle.py's fallback decision tree."""

    def __init__(self, reasons: List[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class QueryPlanValidator:
    def __init__(self, db_client):
        self.db_client = db_client

    def validate(self, plan: QueryPlan, school_id) -> Dict[FilterField, str]:
        """Raises QueryPlanValidationError with every failed rule's reason
        (not just the first) if the plan is invalid. On success, returns a
        dict of {FilterField: actual_stored_value} for every lookup-backed
        filter present (including inside percentage_of, though that's
        enum-only today) -- callers pass this straight into
        query_normalizer.normalize() so the lookup value is never re-queried
        or re-trusted from the model's raw text. Assumes plan.can_answer is
        True -- callers must handle can_answer=False before ever calling
        this."""
        reasons: List[str] = []
        resolved_lookups: Dict[FilterField, str] = {}

        if plan.entity is None or plan.operation is None:
            raise QueryPlanValidationError(["Both entity and operation must be set."])

        meta = REGISTRY.get(plan.entity)
        if meta is None:
            raise QueryPlanValidationError([f"Entity {plan.entity!r} has no registry entry."])

        if plan.operation not in meta.supported_operations:
            reasons.append(
                f"Entity {plan.entity.value!r} does not support operation {plan.operation.value!r}; "
                f"supported operations are {sorted(o.value for o in meta.supported_operations)}."
            )

        if plan.group_by != GroupingDimension.NONE and plan.group_by not in meta.supported_groupings:
            reasons.append(
                f"Entity {plan.entity.value!r} does not support grouping by {plan.group_by.value!r}; "
                f"supported groupings are {sorted(g.value for g in meta.supported_groupings)}."
            )

        if plan.group_by != GroupingDimension.NONE and plan.operation not in AGGREGATE_OPERATIONS:
            reasons.append(
                f"operation {plan.operation.value!r} cannot be combined with grouping; grouping requires "
                f"an aggregate operation ({sorted(o.value for o in AGGREGATE_OPERATIONS)})."
            )

        for f in plan.filters:
            self._validate_filter(f, meta, plan.entity.value, school_id, reasons, resolved_lookups)

        if plan.operation == Operation.PERCENTAGE:
            if plan.percentage_of is None:
                reasons.append("operation=percentage requires percentage_of to be set.")
            else:
                num = plan.percentage_of.numerator
                if num.field.value not in meta.enum_filter_fields:
                    reasons.append(
                        f"percentage_of.numerator.field {num.field.value!r} must be an enum-backed "
                        f"filter field (one of {sorted(v.value for v in meta.enum_filter_fields)}), "
                        f"never a lookup-backed one -- a percentage's numerator is always a fixed "
                        f"category, not dynamic per-school data."
                    )
                else:
                    self._validate_filter(num, meta, plan.entity.value, school_id, reasons, resolved_lookups)
                    if any(f.field.value == num.field.value for f in plan.filters):
                        reasons.append(
                            f"percentage_of.field {num.field.value!r} must not also appear in filters "
                            f"(redundant/contradictory with the percentage's own population scope)."
                        )

        if plan.date_range.value != "all_time" and meta.date_column is None:
            reasons.append(f"Entity {plan.entity.value!r} has no date column to scope date_range by.")

        self._validate_explicit_date_range(plan, meta, reasons)

        for d in plan.display_fields:
            if d not in meta.display_field_columns:
                reasons.append(f"display_field {d.value!r} is not valid for entity {plan.entity.value!r}.")

        if plan.sort is not None:
            if plan.sort.field == SortField.AGGREGATE_VALUE:
                # Not a physical column, so meta.sort_field_columns (an
                # entity-specific mapping) doesn't apply -- it's only
                # meaningful when there's a grouped aggregate result to rank.
                if not is_ranking_capable(plan):
                    reasons.append(
                        "sort field 'aggregate_value' requires a grouped aggregate result "
                        "(group_by set and operation one of "
                        f"{sorted(o.value for o in AGGREGATE_OPERATIONS)})."
                    )
            elif plan.sort.field not in meta.sort_field_columns:
                reasons.append(f"sort field {plan.sort.field.value!r} is not valid for entity {plan.entity.value!r}.")

        if plan.extreme is not None:
            if plan.sort is not None or plan.limit is not None:
                reasons.append(
                    "extreme cannot be combined with sort or limit -- 'lowest/highest' (ties included) "
                    "and an explicit top/bottom N are different questions; use sort=aggregate_value + "
                    "limit=N for an explicit count instead."
                )
            if not is_ranking_capable(plan):
                reasons.append(
                    "extreme requires a grouped aggregate result (group_by set and operation one of "
                    f"{sorted(o.value for o in AGGREGATE_OPERATIONS)})."
                )

        if reasons:
            raise QueryPlanValidationError(reasons)

        return resolved_lookups

    def _validate_explicit_date_range(self, plan: QueryPlan, meta: EntityMeta, reasons: List[str]) -> None:
        """Phase 1 of explicit date/date-range support (2026-09-05): a
        strict, fail-closed gate on explicit_start_date/explicit_end_date --
        SQL-builder and normalizer support are a separate, not-yet-started
        follow-up, so a plan that passes these checks today still cannot
        reach the builder with these fields set (they aren't read there
        yet); this rule exists so the schema fields can land safely ahead
        of that, never silently accepting an unusable/ambiguous plan.

        Both-or-neither, strict YYYY-MM-DD (via datetime.strptime, which
        already rejects impossible calendar dates like Feb 30 or month 13
        with no separate check needed), start<=end, mutual exclusivity with
        date_range, and the same "entity has no date column" gate
        date_range itself is already subject to -- all fail closed, never
        guessed at or silently reinterpreted, matching every other rule in
        this validator."""
        start, end = plan.explicit_start_date, plan.explicit_end_date

        if (start is None) != (end is None):
            reasons.append(
                "explicit_start_date and explicit_end_date must both be set or both be omitted."
            )
            return

        if start is None:
            return  # neither set -- nothing further to validate

        if plan.date_range.value != "all_time":
            reasons.append(
                "explicit_start_date/explicit_end_date cannot be combined with a non-all_time "
                "date_range -- these are two different ways of scoping dates; use exactly one."
            )

        parsed_start = self._parse_strict_iso_date(start, "explicit_start_date", reasons)
        parsed_end = self._parse_strict_iso_date(end, "explicit_end_date", reasons)

        if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
            reasons.append(
                f"explicit_start_date {start!r} must not be after explicit_end_date {end!r}."
            )

        if meta.date_column is None:
            reasons.append(
                f"Entity {plan.entity.value!r} has no date column to scope explicit_start_date/"
                f"explicit_end_date by."
            )

    @staticmethod
    def _parse_strict_iso_date(value: str, field_name: str, reasons: List[str]):
        """Strict, exactly-4-2-2-digit YYYY-MM-DD parse. The regex pre-check
        matters on top of strptime alone: strptime's "%Y-%m-%d" happily
        accepts non-zero-padded input like "2026-9-2", which would let two
        different literal strings denote the identical calendar date with
        no canonical form -- exactly the "same intent, differently
        serialized" hazard query_normalizer.py exists to eliminate
        elsewhere in this pipeline. Rejecting anything but the exact
        zero-padded form here keeps a single valid string per calendar date
        by construction, so no further canonicalization is needed later.
        The regex only constrains shape; strptime (via .date()) still does
        the real calendar-validity check (rejects 2026-02-30, 2026-13-01,
        etc.)."""
        if not _ISO_DATE_RE.match(value):
            reasons.append(f"{field_name} {value!r} is not a valid YYYY-MM-DD calendar date.")
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            reasons.append(f"{field_name} {value!r} is not a valid YYYY-MM-DD calendar date.")
            return None

    def _validate_filter(self, f: ComparisonFilter, meta: EntityMeta, entity_name: str, school_id,
                          reasons: List[str], resolved_lookups: Dict[FilterField, str]) -> None:
        """Dispatches on registry membership, not a model-supplied
        discriminator: `f.field` belongs to at most one of
        meta.enum_filter_fields / meta.lookup_filter_fields for any given
        entity (and, globally, FilterField's own values never overlap
        between the two categories -- see FilterField's docstring), so this
        is a deterministic, closed-vocabulary lookup, never a guess."""
        enum_meta = meta.enum_filter_fields.get(f.field.value)
        if enum_meta is not None:
            if f.value.lower() not in {v.lower() for v in enum_meta.allowed_values}:
                reasons.append(
                    f"filter value {f.value!r} is not a recognized {f.field.value} for {entity_name!r}; "
                    f"allowed values are {sorted(enum_meta.allowed_values)}."
                )
            return

        lookup_meta = meta.lookup_filter_fields.get(f.field.value)
        if lookup_meta is not None:
            actual_value = self._resolve_lookup_value(lookup_meta, f.value, school_id)
            if actual_value is None:
                reasons.append(f"Could not find a {f.field.value} matching {f.value!r} for this school.")
            else:
                resolved_lookups[f.field] = actual_value
            return

        reasons.append(f"filter field {f.field.value!r} does not apply to entity {entity_name!r}.")

    def _resolve_lookup_value(self, field_meta, value: str, school_id):
        """Safe, parameterized-in-spirit existence check: table/column names
        come exclusively from the registry (never the model); only the
        bound value is model-influenced, and it's a plain equality
        comparison, not concatenated SQL structure. Runs via the same
        read-only DBClient already used for real query execution. Uses
        `existence_check_join_path` (starting FROM lookup_table) -- NOT
        `main_query_join_path`, which serves a different purpose (see
        LookupFilterFieldMeta's docstring).

        Returns the ACTUAL stored value (real casing/whitespace) if a match
        exists, else None. The caller (and, downstream, the normalizer)
        uses this real value rather than trusting the model's raw text."""
        joins_sql = ""
        prev = field_meta.lookup_table
        for step in field_meta.existence_check_join_path:
            joins_sql += f" JOIN {step.table} ON {prev}.{step.left_column} = {step.table}.{step.right_column}"
            prev = step.table
        safe_value = value.replace("'", "''")
        sql = (
            f"SELECT {field_meta.lookup_table}.{field_meta.lookup_column} AS matched_value "
            f"FROM {field_meta.lookup_table}{joins_sql} "
            f"WHERE LOWER({field_meta.lookup_table}.{field_meta.lookup_column}) = LOWER('{safe_value}')"
        )
        if school_id is not None:
            # SUPERUSER (school_id=None) is cross-tenant, mirroring how OPA
            # itself treats it -- checked across all schools in that case.
            sql += f" AND {field_meta.school_id_column} = {school_id}"
        sql += " LIMIT 1"
        result = self.db_client.execute(sql)
        if not result or "error" in result[0]:
            return None
        return result[0].get("matched_value")
