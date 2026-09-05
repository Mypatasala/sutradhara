"""
Deterministic MariaDB SQL composer for a validated, normalized QueryPlan.

Precondition (by contract, never re-checked here): `plan` has already passed
QueryPlanValidator.validate() and query_normalizer.normalize(). Every table
name, column name, join condition, and label expression comes exclusively
from query_registry.REGISTRY -- the model never supplies any of it. This
function does not raise for a validated plan; if it ever does, that is
itself a bug in the validator (a gap between what it allows and what this
builder can compose), not a runtime condition to handle gracefully.

Deliberately does NOT build any SQL for `plan.extreme` ("lowest X"/"highest
X" with no number stated): a live trace through the real authorization
pipeline (IdentityFilterGuard -> AliasAwareFilterInjector -> SQLSanitizer)
proved that a second, nested aggregate subquery computing the global MIN/MAX
would only ever be authorized in its OUTER occurrence -- AliasAwareFilter
Injector resolves the target table's alias from the outer FROM/JOIN scope
only, by design, and never qualifies columns inside a nested Subquery/Select.
A row_filter would reach the outer grouped query but never the inner MIN/MAX
scope, letting it scan every tenant's data and potentially compare an
authorized caller's rows against an extreme value belonging to a different
school entirely. Rather than change the injector's intentionally
conservative single-scope behavior, `plan.extreme` produces the EXACT SAME
flat, single-base-table-reference SQL as an ordinary grouped aggregate query
(no ORDER BY, no LIMIT) -- ties are resolved in Python, strictly after
authorization and execution, in query_lifecycle.py. See this module's
compare-in-Python approach there for why: MariaDB returns the PERCENTAGE
expression as an exact `decimal.Decimal` (verified live), so direct `==`
comparison there is exact, not floating-point-approximate -- no rounding or
quantization is used or needed.
"""

from datetime import date, datetime, timedelta
from typing import List

from ..agents.query_plan import (
    GroupingDimension,
    Operation,
    QueryPlan,
    RelativeDate,
    SortField,
)
from ..agents.query_registry import REGISTRY, EntityMeta, JoinStep


def _resolve_relative_date(date_range: RelativeDate, today: date = None) -> "tuple[date, date]":
    """Pure function of the enum value and the current date -- zero LLM
    involvement, zero ambiguity. Returns an inclusive (start, end) range."""
    today = today or date.today()
    if date_range == RelativeDate.TODAY:
        return today, today
    if date_range == RelativeDate.YESTERDAY:
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday
    if date_range == RelativeDate.THIS_WEEK:
        start = today - timedelta(days=today.weekday())
        return start, start + timedelta(days=6)
    if date_range == RelativeDate.LAST_WEEK:
        this_week_start = today - timedelta(days=today.weekday())
        start = this_week_start - timedelta(days=7)
        return start, start + timedelta(days=6)
    if date_range == RelativeDate.THIS_MONTH:
        start = today.replace(day=1)
        next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        return start, next_month - timedelta(days=1)
    if date_range == RelativeDate.LAST_MONTH:
        first_of_this_month = today.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        start = end.replace(day=1)
        return start, end
    if date_range == RelativeDate.THIS_YEAR:
        return today.replace(month=1, day=1), today.replace(month=12, day=31)
    if date_range == RelativeDate.LAST_YEAR:
        return today.replace(year=today.year - 1, month=1, day=1), today.replace(year=today.year - 1, month=12, day=31)
    if date_range == RelativeDate.LAST_30_DAYS:
        # A true rolling 30-calendar-day window: today plus the preceding
        # 29 days = 30 days total, inclusive of today -- NOT a
        # calendar-month approximation (THIS_MONTH/LAST_MONTH above are
        # deliberately left as-is; this is a distinct, separately-chosen
        # semantic, not a replacement for them). Matches the inclusive-of-
        # today convention every other "current window" member here already
        # uses (TODAY, THIS_WEEK, THIS_MONTH, THIS_YEAR all include today).
        # attendance.date (the only date_column this is exercised against
        # today) is a DATE column with no time component, so "rolling
        # 30x24 hours" and "30 calendar days" are the same window here --
        # there is no separate hours-based semantic to choose between for a
        # DATE-typed column; this returns exactly 30 distinct calendar dates.
        return today - timedelta(days=29), today
    if date_range == RelativeDate.LAST_7_DAYS:
        # A true rolling 7-calendar-day window: today plus the preceding 6
        # days = 7 days total, inclusive of today -- deliberately NOT the
        # same window as LAST_WEEK (the previous calendar Monday-Sunday,
        # which excludes today entirely). Same rolling-window convention as
        # LAST_30_DAYS above, at N=7 instead of N=30.
        return today - timedelta(days=6), today
    raise ValueError(f"Unhandled RelativeDate: {date_range!r}")  # ALL_TIME is filtered out by the caller before this is reached


def _join_sql(joins: "List[tuple[str, JoinStep]]") -> str:
    """`joins` is a list of (anchor_table, step) pairs -- the anchor is the
    table this SPECIFIC step's ON clause is relative to, as resolved by
    _collect_joins. Deliberately does NOT re-derive the anchor by walking
    the list sequentially: two independent join chains that both start
    from the same base table (e.g. two different relationships that
    happen to target the same table) must each keep their OWN true anchor,
    not silently inherit whatever table a prior, unrelated join happened to
    land on last."""
    sql = ""
    for anchor, step in joins:
        sql += f" {step.join_type} {step.table} ON {anchor}.{step.left_column} = {step.table}.{step.right_column}"
    return sql


def _collect_joins(plan: QueryPlan, meta: "EntityMeta", base_table: str) -> "List[tuple[str, JoinStep]]":
    """Deterministically composes ONE ordered join sequence from every
    registry path this plan needs -- the grouping dimension's joins (if
    any), then the entity's list_joins (if operation=LIST), then each
    lookup filter's own main_query_join_path (if any) --
    deduplicating a join only when its FULL identity matches exactly: same
    anchor table it's joined FROM, same target table, same left/right
    columns. Never deduplicates by target table name alone, so two
    genuinely different joins to the same table (e.g. a hypothetical
    primary_teacher_id vs secondary_teacher_id join, or two independent
    relationships that both happen to target the same table -- not
    supported by any current registry entry, but proven correct by
    tests/test_join_deduplication.py) are never incorrectly collapsed or
    misattributed to the wrong anchor.

    Returns (anchor, step) pairs rather than a bare step list specifically
    so each step's true anchor survives even when it isn't the
    immediately-preceding step in the returned sequence (e.g. two
    independent single-join chains both anchored at base_table).
    """
    ordered: "List[tuple[str, JoinStep]]" = []
    seen = set()

    def add_chain(steps: List[JoinStep], anchor: str) -> None:
        current_anchor = anchor
        for step in steps:
            key = (current_anchor, step.table, step.left_column, step.right_column)
            if key not in seen:
                seen.add(key)
                ordered.append((current_anchor, step))
            current_anchor = step.table

    if plan.group_by != GroupingDimension.NONE:
        add_chain(meta.supported_groupings[plan.group_by].joins, base_table)

    if plan.operation == Operation.LIST:
        add_chain(meta.list_joins, base_table)

    for f in plan.filters:
        lookup_meta = meta.lookup_filter_fields.get(f.field.value)
        if lookup_meta is not None:
            add_chain(lookup_meta.main_query_join_path, base_table)

    return ordered


class StructuredSQLBuilder:
    @staticmethod
    def build(plan: QueryPlan) -> str:
        meta: EntityMeta = REGISTRY[plan.entity]
        base_table = meta.table

        select_exprs: List[str] = []
        group_by_cols: List[str] = []
        order_by_cols: List[str] = []

        # Single deterministic, deduplicated join sequence for the whole
        # query -- see _collect_joins' docstring for why this replaces two
        # separate inline joins_sql += sites that previously could each
        # independently emit a JOIN to the same table (e.g. a grouping
        # dimension and a lookup filter both needing `courses`), producing
        # invalid duplicate-JOIN SQL.
        joins_sql = _join_sql(_collect_joins(plan, meta, base_table))

        if plan.group_by != GroupingDimension.NONE:
            path = meta.supported_groupings[plan.group_by]
            group_by_cols.extend(path.group_by_columns)
            if len(path.label.columns) > 1:
                cols_sql = ", ".join(path.label.columns)
                select_exprs.append(f"CONCAT_WS('{path.label.separator}', {cols_sql}) AS {path.label_alias}")
            else:
                select_exprs.append(f"{path.label.columns[0]} AS {path.label_alias}")
            # Registry-owned, always-present columns for this grouping (e.g.
            # a student's class/section alongside their attendance
            # aggregate) -- generic to any grouping that declares them, not
            # entity-specific logic. The model never selects these; see
            # GroupingPath.default_display_columns' docstring.
            for column, alias in path.default_display_columns:
                select_exprs.append(f"{column} AS {alias}")
            order_by_cols.extend(path.default_order_by)

        aggregate_alias = None
        if plan.operation == Operation.COUNT:
            aggregate_alias = "count"
            select_exprs.append(f"COUNT(*) AS {aggregate_alias}")
        elif plan.operation == Operation.PERCENTAGE:
            aggregate_alias = "percentage"
            num = plan.percentage_of.numerator
            num_col = meta.enum_filter_fields[num.field.value].column
            safe_val = num.value.replace("'", "''")
            select_exprs.append(
                f"(COUNT(CASE WHEN {num_col} = '{safe_val}' THEN 1 END) * 100.0 / COUNT(*)) AS {aggregate_alias}"
            )
        elif plan.operation in (Operation.AVERAGE, Operation.SUM):
            # No current REGISTRY entry declares AVERAGE/SUM in
            # supported_operations, so QueryPlanValidator already rejects
            # any plan reaching here for every entity registered today --
            # this is unreachable in practice. Left as an explicit failure
            # (not a guessed/placeholder SQL shape) so that adding a future
            # entity that DOES support one of these without also adding its
            # target-column metadata here fails loudly during development,
            # rather than silently emitting wrong SQL.
            raise NotImplementedError(
                f"Operation {plan.operation.value!r} has no builder support yet -- "
                f"add target-column metadata to EntityMeta before enabling it for any entity."
            )
        elif plan.operation == Operation.LIST:
            fields = plan.display_fields or meta.default_display_fields
            select_exprs = [meta.display_field_columns[d] for d in fields]

        where_clauses: List[str] = []
        if plan.explicit_start_date is not None:
            # explicit_start_date/explicit_end_date take precedence here
            # ONLY because QueryPlanValidator already guarantees mutual
            # exclusivity with date_range != ALL_TIME (a plan with both set
            # is rejected before it ever reaches the builder) -- this is
            # not a silent builder-invented preference between two
            # simultaneously-valid date specs; by the time this code runs,
            # at most one of the two can actually be present. A single
            # explicit day is start == end, producing a same-day BETWEEN,
            # same as every other single-day case in this file (e.g. TODAY,
            # YESTERDAY above). Escaped the same way every other model-
            # supplied string value in this file already is (see the filter
            # loop below) -- defense-in-depth on top of the validator's own
            # strict YYYY-MM-DD + real-calendar-date check, not a
            # substitute for it.
            safe_start = plan.explicit_start_date.replace("'", "''")
            safe_end = plan.explicit_end_date.replace("'", "''")
            where_clauses.append(f"{meta.date_column} BETWEEN '{safe_start}' AND '{safe_end}'")
        elif plan.date_range != RelativeDate.ALL_TIME:
            start, end = _resolve_relative_date(plan.date_range)
            where_clauses.append(f"{meta.date_column} BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'")

        for f in plan.filters:
            enum_meta = meta.enum_filter_fields.get(f.field.value)
            col = enum_meta.column if enum_meta is not None else meta.lookup_filter_fields[f.field.value].column
            safe_val = f.value.replace("'", "''")
            where_clauses.append(f"{col} = '{safe_val}'")

        distinct_kw = "DISTINCT " if plan.distinct else ""
        sql = f"SELECT {distinct_kw}{', '.join(select_exprs)} FROM {base_table}{joins_sql}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        if group_by_cols:
            sql += " GROUP BY " + ", ".join(group_by_cols)
        if plan.sort:
            if plan.sort.field == SortField.AGGREGATE_VALUE:
                # Sentinel: sort by the aggregate expression this very build
                # just aliased above (validator guarantees group_by is set
                # and operation produced one), not a registry-mapped column.
                col = aggregate_alias
            else:
                col = meta.sort_field_columns[plan.sort.field]
            sql += f" ORDER BY {col} {plan.sort.direction.upper()}"
        elif order_by_cols:
            sql += " ORDER BY " + ", ".join(order_by_cols)
        if plan.limit:
            sql += f" LIMIT {plan.limit}"
        return sql
