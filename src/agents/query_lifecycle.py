import logging
import re
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
from .query_plan import Operation, UnresolvedReason
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
        # only COUNT/PERCENTAGE are buildable aggregate operations today, and
        # extreme is only reachable when operation is one of those (validator
        # rule). AVERAGE/SUM aren't registered for any entity's
        # supported_operations yet, so they can't reach here at all.
        extreme_value = canonical_plan.extreme.value if canonical_plan.extreme else None
        extreme_field = (
            ("count" if canonical_plan.operation == Operation.COUNT else "percentage")
            if canonical_plan.extreme else None
        )
        return {
            "intent": {"table": table, "action": action, "sql": sql},
            "sql": sql,
            "extreme": extreme_value,
            "extreme_field": extreme_field,
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
