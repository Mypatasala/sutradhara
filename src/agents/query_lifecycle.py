from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from .intent_agent import IntentResolutionAgent
from ..policy.engine import PolicyEngine
from ..retrieval.db_client import DBClient
from ..retrieval.schema_provider import SchemaProvider

class AgentState(TypedDict):
    query: str
    schema: Optional[str]
    context: Optional[dict]
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
        self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("fetch_schema", self._fetch_schema)
        workflow.add_node("resolve_intent", self._resolve_intent)
        workflow.add_node("enforce_policy", self._enforce_policy)
        workflow.add_node("execute_sql", self._execute_sql)
        workflow.add_node("summarize", self._summarize)
        
        # Define edges
        workflow.set_entry_point("fetch_schema")
        workflow.add_edge("fetch_schema", "resolve_intent")
        workflow.add_edge("resolve_intent", "enforce_policy")
        workflow.add_conditional_edges(
            "enforce_policy",
            self._is_authorized,
            {
                "authorized": "execute_sql",
                "denied": END,
                "clarify": "summarize"
            }
        )
        workflow.add_edge("execute_sql", "summarize")
        workflow.add_edge("summarize", END)
        
        self.app = workflow.compile()

    async def _fetch_schema(self, state: AgentState):
        schema = self.schema_provider.get_schema_summary()
        return {"schema": schema}

    async def _resolve_intent(self, state: AgentState):
        result = await self.intent_agent.resolve(state["query"], state["schema"])
        if "error" in result:
            print(f"ERROR in _resolve_intent: {result['error']}")
            return {"answer": f"Error: {result['error']}", "authorized": False, "data": [{"error": result['error']}]}
        if "clarification" in result:
            return {"clarification": result, "authorized": False}
        return {"intent": result, "sql": result.get("sql")}

    async def _enforce_policy(self, state: AgentState):
        # If already failed or clarification needed, don't override
        if state.get("clarification") or state.get("data") and "error" in state["data"][0]:
            return {"authorized": False}
        
        # Simplified policy for now
        authorized = await self.policy_engine.evaluate(state.get("intent", {}), state.get("context", {}))
        return {"authorized": authorized}

    def _is_authorized(self, state: AgentState):
        if state.get("clarification"):
            return "clarify"
        return "authorized" if state["authorized"] else "denied"

    async def _execute_sql(self, state: AgentState):
        if not state.get("sql"):
            return {"data": [{"error": "No SQL generated"}]}
        data = self.db_client.execute(state["sql"])
        return {"data": data}

    async def _summarize(self, state: AgentState):
        if state.get("clarification"):
            return {"answer": state["clarification"]["clarification"], "type": "clarification"}
            
        # Format the data results as a table if possible
        data_summary = ""
        if state["data"]:
            if isinstance(state["data"], list) and len(state["data"]) > 0:
                if "error" in state["data"][0]:
                    data_summary = f"\nError: {state['data'][0]['error']}"
                else:
                    # Implement simple Markdown table formatting
                    headers = state["data"][0].keys()
                    header_str = " | ".join(headers)
                    sep_str = " | ".join(["---"] * len(headers))
                    rows = []
                    for row in state["data"]:
                        rows.append(" | ".join([str(v) for v in row.values()]))
                    data_summary = f"\n\n| {header_str} |\n| {sep_str} |\n| " + " |\n| ".join(rows) + " |"
            else:
                data_summary = "\nNo records found."
        else:
            data_summary = "\nNo data returned."
            
        answer = f"**SQL used:**\n```sql\n{state['sql']}\n```\n\n**Results:**{data_summary}"
        return {"answer": answer}

    async def run(self, query: str, context: Optional[dict] = None):
        initial_state = {
            "query": query,
            "schema": None,
            "context": context,
            "intent": None,
            "authorized": False,
            "sql": None,
            "data": None,
            "answer": None,
            "clarification": None
        }
        result = await self.app.ainvoke(initial_state)
        return result
