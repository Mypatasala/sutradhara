import logging
from typing import Optional, Set

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


class IdentityFilterRejected(Exception):
    """Raised when LLM-generated SQL contains a self-invented identity/tenant
    literal predicate that cannot be safely and unambiguously removed.
    Callers must treat this as a rejected query -- never execute the SQL as
    generated, and never fall back to a partial/best-effort string rewrite.
    """


class IdentityFilterGuard:
    """
    Narrow, deterministic backstop enforcing intent_agent.py's own system
    prompt rule 11: LLM-generated SQL must never introduce its own identity
    or tenant literal filter -- that scoping belongs exclusively to OPA's
    trusted `row_filter`, injected separately by SQLSanitizer.apply_constraints
    and never touched by this class. This exists because a small local model
    (observed: Ollama llama3.2) can silently violate that rule by inventing a
    plausible-looking but nonexistent literal (e.g. a fabricated UUID for
    `id`), which then zeroes out the query's real result set once ANDed with
    OPA's own correct filter -- see docs/architecture
    /Patasala-OPA-Policy-Status.md "Open item" for the incident this closes.

    This is NOT a general SQL rewriter. It recognizes exactly one
    transformation:

        WHERE (business_condition AND blocked_identity_condition)
            -> WHERE business_condition

    (or drops the WHERE clause entirely if the blocked predicate was the
    only condition present) -- and fails closed (rejects the whole query,
    strips nothing, raises IdentityFilterRejected) for every other shape a
    blocked predicate could appear in: beneath an OR, inside a subquery,
    inside HAVING, inside a JOIN...ON, inside CASE/NOT, in any non-SELECT
    top-level statement shape, or if the SQL fails to parse at all. When in
    doubt, this class always rejects rather than guesses.
    """

    # Evidence-based: every column any current my_patasala/policy/opa/*.rego
    # file's `filter` string uses to scope by caller identity or tenant.
    BLOCKED_COLUMNS: Set[str] = {
        "id", "school_id", "user_id", "student_id", "teacher_id",
        "delegator_user_id", "delegate_user_id", "approver_user_id",
        "initiated_by_user_id", "revoked_by_user_id", "email",
    }

    @classmethod
    def strip(cls, sql: str, dialect: str = "mysql") -> str:
        """Returns `sql` with any blocked identity/tenant literal predicate
        removed from its top-level WHERE clause, provided that predicate is
        reachable from the WHERE root exclusively through AND (optionally
        wrapped in parentheses). Returns `sql` completely unchanged if no
        blocked predicate is present anywhere. Raises IdentityFilterRejected
        for every case that can't be proven safe to rewrite this narrowly.
        """
        try:
            tree = sqlglot.parse_one(sql, dialect=dialect)
        except Exception as exc:
            logger.warning("IdentityFilterGuard: SQL failed to parse, rejecting. sql=%r error=%s", sql, exc)
            raise IdentityFilterRejected(f"SQL failed to parse: {exc}") from exc

        if tree is None:
            return sql

        # A CTE (`WITH ... AS (...) SELECT ...`) still parses as a plain
        # exp.Select with an extra `with_` arg -- the isinstance check alone
        # doesn't exclude it, and this guard makes no attempt to reason about
        # predicates that depend on a CTE body's own semantics. Treated the
        # same as any other unsupported top-level shape below.
        is_plain_select = isinstance(tree, exp.Select) and tree.args.get("with_") is None

        if not is_plain_select:
            # WITH/CTE, UNION, or any other top-level shape: only a plain
            # SELECT's top-level WHERE is ever eligible for stripping. If a
            # blocked predicate exists anywhere in an unsupported shape like
            # this, reject outright rather than guess at where it's safe to
            # remove; if there is none, there is nothing to do.
            if cls._contains_blocked_predicate(tree):
                logger.warning(
                    "IdentityFilterGuard: blocked predicate found in unsupported "
                    "top-level SQL shape (%s), rejecting. sql=%r",
                    type(tree).__name__, sql,
                )
                raise IdentityFilterRejected(
                    "Blocked identity/tenant predicate found in an unsupported "
                    "top-level SQL shape (not a plain SELECT)."
                )
            return sql

        where = tree.args.get("where")

        # Anything outside the top-level WHERE -- JOIN...ON, HAVING, the
        # SELECT list, GROUP BY/ORDER BY/LIMIT expressions -- is never
        # touched. If any of it contains a blocked predicate, reject rather
        # than attempt to rewrite a join or a HAVING clause.
        for key, node in tree.args.items():
            if key == "where" or node is None:
                continue
            nodes = node if isinstance(node, list) else [node]
            for n in nodes:
                if isinstance(n, exp.Expression) and cls._contains_blocked_predicate(n):
                    logger.warning(
                        "IdentityFilterGuard: blocked predicate found outside "
                        "top-level WHERE (in %s), rejecting. sql=%r", key, sql,
                    )
                    raise IdentityFilterRejected(
                        f"Blocked identity/tenant predicate found outside the "
                        f"top-level WHERE clause (in {key})."
                    )

        if where is None:
            return sql

        new_condition, changed = cls._strip_and_only(where.this)

        if not changed:
            # Nothing changed -- no blocked predicate reachable purely
            # through AND was found in the top-level WHERE at all (any
            # blocked predicate under OR/NOT/CASE/subquery within this same
            # WHERE would already have raised inside _strip_and_only).
            return sql

        if new_condition is None:
            tree.set("where", None)
        else:
            where.set("this", new_condition)

        rewritten = tree.sql(dialect=dialect)
        logger.warning(
            "IdentityFilterGuard: stripped a self-invented identity/tenant "
            "predicate from LLM-generated SQL. original=%r rewritten=%r",
            sql, rewritten,
        )
        return rewritten

    @classmethod
    def _is_blocked_leaf(cls, node: exp.Expression) -> bool:
        if not isinstance(node, exp.EQ):
            return False
        left, right = node.this, node.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            return left.name.lower() in cls.BLOCKED_COLUMNS
        if isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            return right.name.lower() in cls.BLOCKED_COLUMNS
        return False

    @classmethod
    def _contains_blocked_predicate(cls, node: exp.Expression) -> bool:
        if cls._is_blocked_leaf(node):
            return True
        return any(cls._is_blocked_leaf(n) for n in node.find_all(exp.EQ))

    @classmethod
    def _strip_and_only(cls, node: exp.Expression) -> "tuple[Optional[exp.Expression], bool]":
        """Recursively strips blocked leaves reachable from `node` purely
        through AND (optionally parenthesized). Returns (rewritten_node,
        changed) -- rewritten_node is None if `node` was fully removed;
        `changed` is an explicit flag (NOT identity comparison, since
        sqlglot's `.set()` mutates nodes in place rather than replacing them,
        which would make an `is`-based "did anything change" check silently
        wrong). Raises IdentityFilterRejected if a blocked predicate is found
        anywhere beneath an OR, NOT, CASE, or subquery within this subtree --
        those are never partially rewritten, only ever cause a full
        rejection.
        """
        if cls._is_blocked_leaf(node):
            return None, True

        if isinstance(node, exp.And):
            left, left_changed = cls._strip_and_only(node.this)
            right, right_changed = cls._strip_and_only(node.expression)
            if not left_changed and not right_changed:
                return node, False
            if left is None and right is None:
                return None, True
            if left is None:
                return right, True
            if right is None:
                return left, True
            node.set("this", left)
            node.set("expression", right)
            return node, True

        if isinstance(node, exp.Paren):
            inner, inner_changed = cls._strip_and_only(node.this)
            if not inner_changed:
                return node, False
            if inner is None:
                return None, True
            node.set("this", inner)
            return node, True

        # OR, NOT, CASE, subquery (IN/EXISTS/scalar), or any other shape is
        # opaque to this guard. If a blocked predicate is anywhere inside it,
        # removing it could change which rows match in a way we can't prove
        # is safe -- fail closed for the WHOLE query rather than leaving it
        # in place silently or guessing at a rewrite.
        if cls._contains_blocked_predicate(node):
            raise IdentityFilterRejected(
                "Blocked identity/tenant predicate found in an unsupported "
                "boolean context (OR / NOT / CASE / subquery) within WHERE."
            )
        return node, False
