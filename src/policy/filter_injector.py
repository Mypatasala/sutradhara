import logging
from typing import Optional

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


class FilterInjectionRejected(Exception):
    """Raised when OPA's trusted row_filter cannot be bound to the LLM's
    generated SQL with certainty -- callers must treat this as a rejected
    query, never execute the SQL with an unqualified or guessed filter.
    """


class AliasAwareFilterInjector:
    """
    A second, independent authorization-pipeline layer, separate from
    IdentityFilterGuard: where that guard strips a self-invented identity
    filter out of the LLM's OWN SQL, this class solves a different problem --
    correctly QUALIFYING OPA's trusted `row_filter` string with whatever
    alias (if any) the LLM actually gave the target table in its generated
    SQL, before SQLSanitizer.apply_constraints blindly string-concatenates
    it in.

    Why this exists: apply_constraints has zero awareness of table names or
    aliases -- it just appends "AND (row_filter)". A bare column reference in
    row_filter (e.g. "school_id = %v", exactly how every existing
    my_patasala/policy/opa/*.rego filter is written) is only unambiguous SQL
    when the target table isn't joined against another table that happens to
    share that column name, AND the LLM doesn't alias the target table away
    from the name the filter assumes. Both assumptions broke in practice:
    "how many students are in each class" joins class_sections (which has
    its own school_id, same as students) and llama3.2 reliably aliases
    class_sections despite explicit few-shot instruction not to -- see
    docs/architecture/Patasala-OPA-Policy-Status.md "Open item".

    This class fixes it generally, for any table and any alias the LLM
    chooses, rather than depending on prompt compliance: it resolves the
    target table's actual alias from the LLM's own parsed SQL (outer
    FROM/JOIN scope only -- never guessing from a nested subquery), then
    qualifies every unqualified top-level column reference in row_filter
    with that alias, leaving nested subquery scopes inside row_filter itself
    completely untouched (they're already self-contained and unambiguous).

    Fails closed -- raises FilterInjectionRejected, never guesses or falls
    back to the original unqualified filter -- whenever:
      - the SQL or the row_filter fails to parse,
      - the target table isn't found in the outer FROM/JOIN scope at all, or
      - the target table appears more than once there (e.g. a self-join) --
        there is no way to know which occurrence the filter should bind to.
    """

    @classmethod
    def inject(cls, sql: str, row_filter: str, target_table: str, dialect: str = "mysql") -> str:
        """Returns `sql` with `row_filter` appended via
        SQLSanitizer-equivalent semantics are NOT applied here -- this
        method only returns the ALIAS-QUALIFIED row_filter string, ready to
        be passed into SQLSanitizer.apply_constraints exactly as before.
        Returns `row_filter` unchanged if it's empty (nothing to qualify --
        mirrors apply_constraints' own `if row_filter:` guard, e.g.
        SUPERUSER's unrestricted case).
        """
        if not row_filter:
            return row_filter

        try:
            tree = sqlglot.parse_one(sql, dialect=dialect)
        except Exception as exc:
            logger.warning(
                "AliasAwareFilterInjector: SQL failed to parse, rejecting. sql=%r error=%s", sql, exc,
            )
            raise FilterInjectionRejected(f"SQL failed to parse: {exc}") from exc

        # A CTE ("WITH ... AS (...) SELECT ...") still parses as a plain
        # exp.Select with an extra `with_` arg -- isinstance alone doesn't
        # exclude it, and this class makes no attempt to reason about a
        # table name that only exists as a CTE's own alias rather than a
        # real table (see the identical issue already fixed once in
        # IdentityFilterGuard for the same underlying sqlglot behavior).
        is_plain_select = isinstance(tree, exp.Select) and tree.args.get("with_") is None
        if not is_plain_select:
            logger.warning(
                "AliasAwareFilterInjector: unsupported top-level SQL shape (%s), rejecting. sql=%r",
                type(tree).__name__ if tree is not None else "None", sql,
            )
            raise FilterInjectionRejected(
                "Cannot resolve a table alias in an unsupported top-level SQL shape (not a plain SELECT)."
            )

        alias = cls._resolve_target_alias(tree, target_table)
        if alias is None:
            logger.warning(
                "AliasAwareFilterInjector: target table %r not found exactly once in the "
                "outer FROM/JOIN scope, rejecting. sql=%r", target_table, sql,
            )
            raise FilterInjectionRejected(
                f"Table {target_table!r} is not referenced exactly once in the query's "
                f"outer FROM/JOIN scope -- cannot determine which occurrence to authorize."
            )

        try:
            condition = sqlglot.parse_one(row_filter, into=exp.Condition, dialect=dialect)
        except Exception as exc:
            logger.warning(
                "AliasAwareFilterInjector: row_filter failed to parse, rejecting. "
                "row_filter=%r error=%s", row_filter, exc,
            )
            raise FilterInjectionRejected(f"row_filter failed to parse: {exc}") from exc

        cls._qualify(condition, alias)
        qualified = condition.sql(dialect=dialect)
        if qualified != row_filter:
            logger.info(
                "AliasAwareFilterInjector: qualified row_filter for target_table=%r alias=%r: "
                "%r -> %r", target_table, alias, row_filter, qualified,
            )
        return qualified

    @classmethod
    def _resolve_target_alias(cls, tree: exp.Select, target_table: str) -> Optional[str]:
        """Returns the alias (or canonical name if unaliased) `target_table`
        is given in the query's own OUTER FROM/JOIN sources -- deliberately
        NOT a recursive find_all, which would incorrectly match a table name
        that only appears inside a nested subquery. Returns None if the
        table isn't found there exactly once (zero matches, or more than one
        e.g. a self-join) -- callers must fail closed either way.
        """
        sources = []
        from_ = tree.args.get("from_")
        if from_ is not None and isinstance(from_.this, exp.Expression):
            sources.append(from_.this)
        for join in tree.args.get("joins") or []:
            if isinstance(join.this, exp.Expression):
                sources.append(join.this)

        matches = [
            source.alias_or_name for source in sources
            if isinstance(source, exp.Table) and source.name.lower() == target_table.lower()
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    @classmethod
    def _qualify(cls, node: exp.Expression, alias: str, in_subquery: bool = False) -> None:
        """Recursively sets `.table` on every unqualified Column reachable
        from `node`, EXCEPT any that live inside a nested Subquery/Select --
        those are already self-contained and must never be touched. Mutates
        `node` in place (matches the existing style/tradeoffs of
        IdentityFilterGuard._strip_and_only, which does the same).
        """
        if isinstance(node, (exp.Subquery, exp.Select)):
            in_subquery = True

        if isinstance(node, exp.Column) and not in_subquery and not node.table:
            node.set("table", exp.to_identifier(alias))
            return

        for child in node.args.values():
            children = child if isinstance(child, list) else [child]
            for c in children:
                if isinstance(c, exp.Expression):
                    cls._qualify(c, alias, in_subquery)
