from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from .intent_agent import IntentResolutionAgent
from ..policy.engine import PolicyEngine
from ..retrieval.db_client import DBClient
from ..retrieval.schema_provider import SchemaProvider
from ..policy.sanitizer import SQLSanitizer
from ..semantics.mapper import SemanticMapper


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

        sanitized_sql = SQLSanitizer.apply_constraints(sql, allowed_cols, row_filter)
        data = self.db_client.execute(sanitized_sql)
        return {"data": data, "sql": sanitized_sql}

    async def _summarize(self, state: AgentState):
        data = state.get("data") or []
        sql = state.get("sql") or ""

        # Use LLM to produce a natural language answer from the retrieved rows
        answer = await self.intent_agent.summarize(state["query"], sql, data)
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
