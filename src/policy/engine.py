import os
import httpx
from typing import Dict, Any, List, Optional

class PolicyEngine:
    """
    Enforces RBAC/ABAC policies via Open Policy Agent (OPA).
    Uses modular role-based policies for Students, Teachers, Admins, and Principals.
    """

    def __init__(self, opa_url: str = "http://localhost:8181/v1/data/sutradhara/main/decision"):
        self.opa_url = opa_url
        # Which tenant's policy bundle to dispatch into (see main.rego). A
        # single-tenant deployment (e.g. one sutradhara instance dedicated to
        # myPatasala) can set this once via env instead of every caller
        # passing context["tenant"]. Left unset, main.rego falls back to the
        # generic ./roles/ demo policies.
        self.default_tenant = os.getenv("OPA_TENANT")
        # Every tenant's decision lives at .../<tenant>/main/decision (see
        # main.rego); any other data document defined in that same "main"
        # package (e.g. covered_tables) lives at .../<tenant>/main/<doc>.
        self._opa_base = self.opa_url.rsplit("/main/decision", 1)[0]
        self._covered_tables_cache: Dict[str, Optional[set]] = {}

    async def evaluate(self, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls OPA to evaluate the user intent against role-based policies.
        Mandatory role check is enforced at the entry point of the agent.
        """
        # Mandatory Role Check
        if not context or not context.get("role"):
            return {
                "authorized": False,
                "error": "Access Denied: Mandatory user role missing from session context."
            }

        payload = {
            "input": {
                "tenant": context.get("tenant", self.default_tenant),
                "user": {
                    "id": context.get("user_id"),
                    "role": context.get("role"),
                    "email": context.get("email"),
                    "school_id": context.get("school_id"),
                    "department_id": context.get("department_id")
                },
                "action": intent.get("action", "select"),
                "target_table": intent.get("table")
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.opa_url, json=payload, timeout=2.0)
                response.raise_for_status()
                
                result = response.json().get("result", {})
                
                return {
                    "authorized": result.get("authorized", False),
                    "columns": result.get("columns", []),
                    "filter": result.get("filter", ""),
                    "error": result.get("error") if not result.get("authorized") else None
                }
        except Exception as e:
            return {
                "authorized": False,
                "error": f"Policy Enforcement Error: {str(e)}"
            }

    async def get_covered_tables(self, tenant: str) -> Optional[set]:
        """
        Fetches the tenant's own `covered_tables` data document from OPA (a
        plain data lookup, not a policy decision) — the set of tables this
        tenant's schema summary should be scoped to before it's handed to
        the LLM. A tenant that defines no such document (or any lookup
        error) yields None, meaning "don't scope, use the full schema" —
        this is additive hardening, never a hard dependency.

        Cached for the process lifetime: OPA doesn't hot-reload a plain
        directory load without a server restart, so this can't go stale
        without a restart happening anyway.
        """
        if tenant in self._covered_tables_cache:
            return self._covered_tables_cache[tenant]

        url = f"{self._opa_base}/tenants/{tenant}/main/covered_tables"
        tables: Optional[set] = None
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=2.0)
                response.raise_for_status()
                result = response.json().get("result")
                if isinstance(result, list):
                    tables = set(result)
        except Exception:
            tables = None

        self._covered_tables_cache[tenant] = tables
        return tables
