import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from .intent_agent import IntentResolutionAgent
from ..policy.engine import PolicyEngine
from ..retrieval.db_client import DBClient
from ..retrieval.schema_provider import SchemaProvider
from ..policy.sanitizer import SQLSanitizer
from ..policy.identity_guard import IdentityFilterGuard, IdentityFilterRejected
from ..policy.filter_injector import AliasAwareFilterInjector, FilterInjectionRejected
from ..semantics.mapper import SemanticMapper
from .query_plan import GroupingDimension, Operation, UnresolvedReason
from .query_registry import REGISTRY
from .query_validator import QueryPlanValidator, QueryPlanValidationError
from .query_normalizer import normalize as normalize_query_plan
from ..retrieval.structured_sql_builder import StructuredSQLBuilder

logger = logging.getLogger(__name__)

# Deliberately narrow: matches "my profile"/"my details"/"about me"/"who am
# i"-shaped phrasing, NOT a bare "my" (which also appears in legitimate
# broader questions like "students in my school" or "my school's timetable"
# that must NOT be narrowed to the caller's own row).
_SELF_REFERENCE_RE = re.compile(
    r"\bmy\s+(profile|details|information|info|account|record)\b|\babout\s+me\b|\bwho\s+am\s+i\b",
    re.IGNORECASE,
)


class AgentState(TypedDict):
    query: str
    schema: Optional[str]
    context: Optional[dict]
    history: Optional[List[dict]]
    intent: Optional[dict]
    authorized: bool
    sql: Optional[str]
    data: Optional[List[dict]]
    answer: Optional[str]
    clarification: Optional[dict]
    # Minimal extreme-value metadata carried from the validated QueryPlan
    # (structured path only -- always absent/None for the legacy free-text
    # path) through to _execute_sql's post-authorization selection step.
    # `extreme` is the raw ExtremeSelector value ("lowest"/"highest");
    # `extreme_field` is the aggregate column alias ("count"/"percentage")
    # the builder itself just produced -- see _apply_extreme_selection.
    extreme: Optional[str]
    extreme_field: Optional[str]
    # Structured-path-only result-shape metadata (see _try_structured_resolution)
    # -- always absent/None for the legacy free-text path, exactly like extreme
    # above. Consumed by _summarize's deterministic grounding step; see
    # _compute_deterministic_aggregate's docstring for the four result_kind
    # values and why only two of them carry a groundable number.
    result_kind: Optional[str]
    aggregate_alias: Optional[str]


def _apply_extreme_selection(data: List[dict], extreme: Optional[str], extreme_field: Optional[str]) -> List[dict]:
    """Deterministic, post-authorization, post-execution tie resolution for
    plan.extreme -- see structured_sql_builder.py's module docstring for why
    this is done here in Python rather than as a second SQL aggregate scope.

    Operates ONLY on `data`, the rows the DB already returned for the fully
    authorized, already-filtered query -- there is no broader dataset for
    this function to accidentally see, by construction. Never calls min()/
    max() on an empty collection: an empty or error-carrying `data` is
    returned unchanged (nothing to select an extreme from).

    Comparison is a direct `==`/`<`/`>` on whatever Python value the DB
    driver returned for `extreme_field` -- verified live that MariaDB
    returns the PERCENTAGE expression as an exact `decimal.Decimal` (and
    COUNT as a plain int), so this is exact arithmetic comparison, never a
    rounded/quantized approximation. Every row whose value equals the
    resulting min/max is kept -- ties are a feature (see ExtremeSelector's
    docstring), never collapsed to one arbitrary row.
    """
    if not extreme or not data or "error" in data[0]:
        return data
    values = [row[extreme_field] for row in data if extreme_field in row and row[extreme_field] is not None]
    if not values:
        return data
    target = min(values) if extreme == "lowest" else max(values)
    return [row for row in data if row.get(extreme_field) == target]


_BOLD_NUMBER_RE = re.compile(r"\*\*(-?[\d,]*\.?\d+)\*\*")


def _compute_deterministic_aggregate(data: List[dict], result_kind: Optional[str], aggregate_alias: Optional[str]):
    """Derives the one groundable "headline number" for a result set, driven
    ENTIRELY by QueryPlan-derived metadata (result_kind, aggregate_alias,
    already computed by _try_structured_resolution from plan.operation/
    plan.group_by) -- never by inspecting the LLM's answer text, column
    names, or summing arbitrary numeric fields.

    Principal-engineer review note (2026-09-02): an earlier version of this
    function also tried to backstop "grouped_aggregate" results by counting
    bolded numbers in the LLM's own answer as a proxy for "did it collapse
    the breakdown into a summary". That was itself a free-text semantic
    heuristic -- fragile for the exact reasons flagged in review (numbers
    also appear in grade labels, dates, percentages, ids) -- and BROKE the
    one thing it was meant to protect: a real live regression showed it
    replacing a fully correct 9-row breakdown with a bare "**45**". Removed
    entirely. The desired architecture is:

      QueryPlan semantics (operation, group_by) decide scalar vs grouped ->
      deterministic execution shapes the actual data accordingly ->
      the LLM may explain/format that data, but the ONE case this function
      grounds is the one case a single authoritative number is structurally
      well-defined by the plan itself: an ungrouped aggregate, where the SQL
      already returns exactly one row with the aggregate as its own column.

    A "grouped_aggregate" plan means the question explicitly asked for a
    per-group breakdown (group_by is only ever set for that reason -- see
    the structured prompt's contrastive worked examples for both the
    student-total-vs-by_class and classes-vs-students distinctions). The
    correct answer IS the breakdown; there is no separate "total" the plan
    asked for, so there is nothing here for this function to compute or
    enforce a number against -- returns None, unconditionally, letting the
    LLM narrate the (already fully authorized and correct) rows however it
    likes. This is also why the original incident cannot recur through this
    path anymore: the plan-classification fix (A1) already guarantees a
    plain total question resolves to group_by=NONE/"scalar_aggregate" in
    the first place, so the scalar branch below is reached, and grounds it.

    "list" and any other combination: no single number is well-defined;
    returns None.
    """
    if not data or "error" in data[0] or not aggregate_alias:
        return None
    if result_kind == "scalar_aggregate" and len(data) == 1 and aggregate_alias in data[0]:
        return data[0][aggregate_alias]
    return None


def _numeric_token_matches(token: str, expected) -> bool:
    """True if a bolded token in the LLM's answer states the same value as
    `expected`, tolerating the token's OWN rounding precision -- never more.

    Principal-engineer review finding (2026-09-02): COUNT aggregates are
    always exact integers (verified live: MariaDB/the DB driver returns
    COUNT as a plain int), but PERCENTAGE aggregates come back as
    high-precision Decimals (verified live: `50.47619`, not a clean `50.5`)
    -- reachable today via a plain, ungrouped "what percentage..." question
    (operation=percentage, group_by unset is a valid plan; see
    QueryPlanValidator, which does not require grouping for PERCENTAGE). An
    earlier version of this function did a bare string-equality check,
    which would reject a perfectly correct, reasonably-rounded LLM answer
    like "50.48%" as "wrong" (since "50.48" != "50.47619") and corrupt it
    into the ugly raw value -- a real defect, not just a formatting nit.

    Fix: compare numerically, rounding `expected` to exactly the number of
    decimal places the token itself states (never fewer, never more) --
    "50.48" matches 50.47619 (round(50.47619, 2) == 50.48); "50" matches
    too (round(50.47619, 0) == 50); "44" does NOT match 45 at any
    precision, so this cannot make a genuinely wrong number look right --
    rounding only ever removes precision the token never claimed to have,
    it never changes which whole/rounded value `expected` actually is.
    """
    cleaned = token.replace(",", "")
    try:
        token_value = Decimal(cleaned)
        expected_value = Decimal(str(expected))
    except InvalidOperation:
        return cleaned == str(expected)
    exponent = token_value.as_tuple().exponent
    decimals = -exponent if exponent < 0 else 0
    return round(expected_value, decimals) == token_value


def _ground_numeric_answer(answer: str, computed_value) -> str:
    """Enforces that the one number _compute_deterministic_aggregate
    identified as authoritative actually appears, correctly, in the LLM's
    answer -- a deterministic post-check, not a stronger prompt instruction
    (a prompt alone was proven live to not be sufficient: the same model,
    given the exact correct grouped rows, deterministically stated a total
    that matched none of them -- see the investigation this fixes).

    If the LLM's answer already states the correct number (wrapped in
    **bold**, per its own formatting instructions -- tolerating the
    number's own rounding precision, see _numeric_token_matches), it's left
    untouched -- this only overrides when the number is actually wrong.
    If the answer has exactly one bolded number and it's wrong, that number
    is surgically replaced in place, preserving the LLM's own sentence/
    formatting rather than discarding it.
    Otherwise (zero or multiple bolded numbers -- can't unambiguously tell
    which one was meant to be the answer), falls back to a minimal,
    unambiguously correct statement rather than risk leaving a wrong number
    standing anywhere in the response.
    """
    bold_numbers = _BOLD_NUMBER_RE.findall(answer)
    if any(_numeric_token_matches(tok, computed_value) for tok in bold_numbers):
        return answer
    if len(bold_numbers) == 1:
        return _BOLD_NUMBER_RE.sub(f"**{computed_value}**", answer, count=1)
    return f"**{computed_value}**"


class QueryLifecycleAgent:
    def __init__(self):
        self.intent_agent = IntentResolutionAgent()
        self.policy_engine = PolicyEngine()
        self.db_client = DBClient()
        self.schema_provider = SchemaProvider()
        self.semantic_mapper = SemanticMapper()
        self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("fetch_schema", self._fetch_schema)
        workflow.add_node("resolve_intent", self._resolve_intent)
        workflow.add_node("enforce_policy", self._enforce_policy)
        workflow.add_node("execute_sql", self._execute_sql)
        workflow.add_node("summarize", self._summarize)

        workflow.set_entry_point("fetch_schema")
        workflow.add_edge("fetch_schema", "resolve_intent")
        workflow.add_edge("resolve_intent", "enforce_policy")
        workflow.add_conditional_edges(
            "enforce_policy",
            self._is_authorized,
            {
                "authorized": "execute_sql",
                "denied": END,
                "clarify": END,  # clarification is returned immediately; no summarize needed
            },
        )
        workflow.add_edge("execute_sql", "summarize")
        workflow.add_edge("summarize", END)

        self.app = workflow.compile()

    async def _fetch_schema(self, state: AgentState):
        context = state.get("context") or {}
        tenant = context.get("tenant")

        if tenant:
            # Real tenant: scope the schema to what this tenant's policies
            # actually cover (see PolicyEngine.get_covered_tables), and skip
            # semantic_mapper — its concept/join-path map is hand-curated for
            # this repo's own bundled demo schema and would misdirect the LLM
            # against a different tenant's real tables.
            covered_tables = await self.policy_engine.get_covered_tables(tenant)
            return {"schema": self.schema_provider.get_schema_summary(covered_tables)}

        raw_schema = self.schema_provider.get_schema_summary()
        # Enrich with semantic concept hints so the LLM maps business terms correctly
        enriched = self.semantic_mapper.enrich_schema_summary(raw_schema)
        return {"schema": enriched}

    async def _resolve_intent(self, state: AgentState):
        structured_result = await self._try_structured_resolution(state)
        if structured_result is not None:
            return structured_result

        # Fell through to the legacy free-text path -- either a technical
        # structured-output failure exhausted its retry, or the model
        # itself declared the question out_of_scope for the structured
        # vocabulary. See _try_structured_resolution's docstring for the
        # full fallback decision tree; a semantic validation failure on a
        # can_answer=True plan NEVER reaches this line.
        result = await self.intent_agent.resolve(
            state["query"], state["schema"], state.get("context"), state.get("history")
        )
        if "error" in result:
            print(f"ERROR in _resolve_intent: {result['error']}")
            return {
                "answer": f"Error: {result['error']}",
                "authorized": False,
                "data": [{"error": result["error"]}],
            }
        if "clarification" in result:
            # Preserve the full clarification payload (question + options)
            return {"clarification": result["clarification"], "authorized": False}
        return {"intent": result, "sql": result.get("sql")}

    async def _try_structured_resolution(self, state: AgentState) -> Optional[dict]:
        """Implements the approved fallback decision tree:

        - Technical structured-output failure (provider error, timeout, or
          the output can't be decoded against the QueryPlan schema at all)
          -> retry once -> still failing -> return None (caller falls back
          to legacy free-text resolve() -- a mechanism failure isn't a
          semantic error, there's nothing to have gotten wrong yet).
        - Parsed plan with can_answer=False, unresolved_reason=OUT_OF_SCOPE
          -> return None (legacy fallback permitted -- this is the only
          thing "registry coverage gap" can mean for a closed-enum schema).
        - Parsed plan with can_answer=False, unresolved_reason=AMBIGUOUS
          -> return a clarification response. NEVER legacy fallback: the
          vocabulary to answer this already exists, so falling back here
          would let the model escape the deterministic architecture
          whenever it merely feels unsure.
        - Parsed plan with can_answer=True that fails semantic validation
          -> retry structured generation ONCE with the validator's specific
          failure reasons as corrective feedback -> if the retry also fails
          validation (or itself comes back can_answer=False) -> fail closed
          to a clarification. NEVER legacy fallback -- this is a model
          error on a capability the registry supports, not a coverage gap.
        - Parsed plan with can_answer=True that passes validation -> build
          SQL deterministically and return the intent/sql the rest of the
          pipeline expects (identical shape to the legacy path's result).
          If normalize/build themselves raise (a validator/builder mismatch
          bug, not a user-facing concern) -> fail closed to a clarification,
          logging the exception internally only. NEVER legacy fallback here
          either -- the plan already passed semantic validation.

        Returns None to signal "fall back to legacy free-text resolve()";
        otherwise returns the dict _resolve_intent should return directly.
        """
        context = state.get("context") or {}
        query = state["query"]

        plan = None
        for attempt in range(2):
            try:
                plan = await self.intent_agent.resolve_structured(query, context)
                break
            except Exception as e:
                logger.warning("Structured resolution technical failure (attempt %d): %s", attempt + 1, e)
                plan = None
        if plan is None:
            return None  # technical failure exhausted its retry -> legacy fallback permitted

        if not plan.can_answer:
            if plan.unresolved_reason == UnresolvedReason.OUT_OF_SCOPE:
                return None  # genuine registry-coverage gap -> legacy fallback permitted
            return self._clarification_response(plan.clarification_question)

        # No ranking-field coherence repair needed here -- intent_agent.
        # resolve_structured() (and, transitively, resolve_structured_with_
        # feedback's retry below) already guarantees it on every plan it
        # returns. See that method's docstring for the single-canonical-
        # boundary rationale (2026-09-03 Principal Engineer review).
        school_id = context.get("school_id")
        validator = QueryPlanValidator(self.db_client)
        try:
            resolved_lookups = validator.validate(plan, school_id)
        except QueryPlanValidationError as exc:
            logger.warning("Structured plan failed validation, retrying once with feedback: %s", exc)
            try:
                plan = await self.intent_agent.resolve_structured_with_feedback(query, context, str(exc))
            except Exception as e:
                logger.warning("Structured retry hit a technical failure: %s", e)
                return self._clarification_response(
                    "I couldn't build a valid query for that — could you rephrase?"
                )
            if not plan.can_answer:
                return self._clarification_response(plan.clarification_question)
            try:
                resolved_lookups = validator.validate(plan, school_id)
            except QueryPlanValidationError as exc2:
                logger.warning("Structured plan failed validation again after retry, failing closed: %s", exc2)
                return self._clarification_response(
                    "I couldn't build a valid query for that — could you rephrase?"
                )

        try:
            canonical_plan = normalize_query_plan(plan, resolved_lookups)
            sql = StructuredSQLBuilder.build(canonical_plan)
        except Exception:
            # A plan that already passed semantic validation should never
            # fail to normalize/build -- if it does, that's a validator/
            # builder mismatch bug (see structured_sql_builder.py's
            # docstring), not a user-facing SQL syntax problem to explain.
            # Per the approved fallback tree: once validation has passed,
            # NEVER fall back to legacy free-text generation -- return a
            # controlled clarification instead, and keep the internal
            # exception (which may reference table/column internals) out of
            # the response the caller sees.
            logger.error(
                "Validated QueryPlan failed to normalize/build deterministically -- "
                "this indicates a validator/builder mismatch bug: query=%r plan=%r",
                query, plan, exc_info=True,
            )
            return self._clarification_response(
                "I couldn't safely build a query for that — could you rephrase?"
            )

        action = "select" if canonical_plan.operation == Operation.LIST else "aggregate"
        table = REGISTRY[canonical_plan.entity].table
        # aggregate_alias mirrors StructuredSQLBuilder.build's own naming --
        # only COUNT/PERCENTAGE are buildable aggregate operations today.
        # AVERAGE/SUM aren't registered for any entity's supported_operations
        # yet, so they can't reach here at all (see StructuredSQLBuilder.build,
        # which would raise NotImplementedError first if they somehow did).
        aggregate_alias = (
            "count" if canonical_plan.operation == Operation.COUNT
            else "percentage" if canonical_plan.operation == Operation.PERCENTAGE
            else None
        )
        extreme_value = canonical_plan.extreme.value if canonical_plan.extreme else None
        extreme_field = aggregate_alias if canonical_plan.extreme else None
        # result_kind captures, from plan metadata alone (never column-name
        # guessing), which of the four shapes _summarize's deterministic
        # grounding step (see _compute_deterministic_aggregate) needs to
        # handle differently -- see that function's docstring for why only
        # "scalar_aggregate" and "grouped_aggregate" carry a groundable
        # number at all.
        if canonical_plan.operation == Operation.LIST:
            result_kind = "list"
        elif canonical_plan.group_by != GroupingDimension.NONE:
            result_kind = "grouped_aggregate"
        else:
            result_kind = "scalar_aggregate"
        return {
            "intent": {"table": table, "action": action, "sql": sql},
            "sql": sql,
            "extreme": extreme_value,
            "extreme_field": extreme_field,
            "result_kind": result_kind,
            "aggregate_alias": aggregate_alias,
        }

    @staticmethod
    def _clarification_response(question: Optional[str]) -> dict:
        return {
            "clarification": {"question": question or "Could you rephrase that?", "options": []},
            "authorized": False,
        }

    async def _enforce_policy(self, state: AgentState):
        # Short-circuit if upstream already set a clarification or hard error
        if state.get("clarification") or (
            state.get("data") and "error" in state["data"][0]
        ):
            return {"authorized": False}

        decision = await self.policy_engine.evaluate(
            state.get("intent", {}), state.get("context", {})
        )

        if not decision["authorized"]:
            error_msg = decision.get("error", "Access Denied by Policy.")
            return {
                "authorized": False,
                "answer": f"Forbidden: {error_msg}",
                "data": [{"error": error_msg}],
            }

        return {
            "authorized": True,
            "intent": {**state["intent"], **decision},
        }

    def _is_authorized(self, state: AgentState) -> str:
        if state.get("clarification"):
            return "clarify"
        return "authorized" if state["authorized"] else "denied"

    async def _execute_sql(self, state: AgentState):
        sql = state.get("sql")
        if not sql:
            return {"data": [{"error": "No SQL generated"}]}

        intent = state.get("intent", {})
        allowed_cols = intent.get("columns", [])
        row_filter = intent.get("filter", "")

        # Deterministic backstop against a small LLM inventing its own
        # identity/tenant literal filter (e.g. a fabricated users.id UUID) --
        # see IdentityFilterGuard's docstring and docs/architecture
        # /Patasala-OPA-Policy-Status.md "Open item". Operates ONLY on the
        # LLM's own `sql`; row_filter (OPA's trusted, separately-sourced
        # string) is never inspected, parsed, or modified by this call, and
        # is still applied by apply_constraints below exactly as before.
        try:
            guarded_sql = IdentityFilterGuard.strip(sql)
        except IdentityFilterRejected as exc:
            logger.warning(
                "Rejected LLM-generated SQL with an unsafe self-invented "
                "identity/tenant predicate: query=%r sql=%r reason=%s",
                state.get("query"), sql, exc,
            )
            return {
                "data": [{"error": "Could not safely determine the authorized scope for this query."}],
                "sql": sql,
            }

        # Second, independent authorization-pipeline layer -- separate from
        # IdentityFilterGuard above, which never touches OPA's row_filter at
        # all. This one qualifies row_filter's own column references with
        # whatever alias the LLM actually gave target_table in guarded_sql,
        # since apply_constraints below has no awareness of aliases and a
        # bare column filter is ambiguous SQL the moment the target table is
        # joined against another table sharing that column name (e.g.
        # class_sections + students both have school_id) -- see
        # AliasAwareFilterInjector's docstring and docs/architecture
        # /Patasala-OPA-Policy-Status.md "Open item".
        target_table = intent.get("table", "")
        try:
            qualified_filter = AliasAwareFilterInjector.inject(guarded_sql, row_filter, target_table)
        except FilterInjectionRejected as exc:
            logger.warning(
                "Rejected LLM-generated SQL because OPA's row_filter could not be "
                "unambiguously bound to a table alias: query=%r sql=%r target_table=%r "
                "row_filter=%r reason=%s",
                state.get("query"), guarded_sql, target_table, row_filter, exc,
            )
            return {
                "data": [{"error": "Could not safely determine the authorized scope for this query."}],
                "sql": guarded_sql,
            }

        sanitized_sql = SQLSanitizer.apply_constraints(guarded_sql, allowed_cols, qualified_filter)
        data = self.db_client.execute(sanitized_sql)

        # Extreme-value tie resolution happens here, strictly AFTER
        # authorization and execution -- `data` is already the fully
        # authorized result set, so this cannot see (and therefore cannot
        # leak) anything outside the caller's own authorized scope. See
        # _apply_extreme_selection's docstring and structured_sql_builder
        # .py's module docstring for why this isn't done in SQL.
        data = _apply_extreme_selection(data, state.get("extreme"), state.get("extreme_field"))

        return {"data": data, "sql": sanitized_sql}

    async def _summarize(self, state: AgentState):
        data = state.get("data") or []
        sql = state.get("sql") or ""
        context = state.get("context") or {}
        query = state.get("query", "")

        # Deterministically narrow to the caller's own row BEFORE
        # intent_agent.summarize()'s data[:50] preview truncation -- a policy
        # filter that's broader than self (e.g. ADMIN's same-school "users"
        # access) can legitimately return far more than 50 rows, so the
        # caller's own row is not guaranteed to survive that truncation for
        # the LLM to find. Only narrows on an explicit "my profile"-shaped
        # question (see _SELF_REFERENCE_RE) AND only when a real match
        # exists -- never fabricates, never touches a genuine list/aggregate
        # question ("students in my school" does not match), and the
        # underlying `data` was already fully authorized by policy either way
        # (this only picks a subset of it, never expands access).
        if len(data) > 1 and context and _SELF_REFERENCE_RE.search(query):
            caller_id = context.get("user_id")
            caller_email = context.get("email")
            matches = [
                row for row in data
                if (caller_id is not None and str(row.get("id")) == str(caller_id))
                or (caller_id is not None and str(row.get("user_id")) == str(caller_id))
                or (caller_email is not None and row.get("email") == caller_email)
            ]
            if matches:
                data = matches

        # Use LLM to produce a natural language answer from the retrieved rows.
        # context is also passed to summarize() itself as a secondary backstop
        # for cases the regex above doesn't catch (see its docstring).
        answer = await self.intent_agent.summarize(state["query"], sql, data, context)

        # Deterministic grounding: if plan metadata identifies a single
        # authoritative number for this result (see
        # _compute_deterministic_aggregate), enforce that the LLM's answer
        # actually states it -- structured-path-only (result_kind/
        # aggregate_alias are always None on the legacy free-text path, so
        # this is a no-op there, unchanged from before).
        computed_value = _compute_deterministic_aggregate(
            data, state.get("result_kind"), state.get("aggregate_alias")
        )
        if computed_value is not None:
            answer = _ground_numeric_answer(answer, computed_value)

        return {"answer": answer}

    async def run(self, query: str, context: Optional[dict] = None, history: Optional[List[dict]] = None):
        initial_state: AgentState = {
            "query": query,
            "schema": None,
            "context": context,
            "history": history,
            "intent": None,
            "authorized": False,
            "sql": None,
            "data": None,
            "answer": None,
            "clarification": None,
        }
        result = await self.app.ainvoke(initial_state)
        return result
