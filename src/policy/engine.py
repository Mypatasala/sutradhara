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
                "user": {
                    "id": context.get("user_id"),
                    "role": context.get("role"),
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
