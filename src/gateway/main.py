import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()  # must run before any os.getenv() calls in imported modules

from fastapi import FastAPI, Depends
from .auth import get_current_user
from .models import AskRequest, AskResponse, ResponseType
from ..agents.query_lifecycle import QueryLifecycleAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sutradhara API",
    description="Generic, Domain-Agnostic AI Data Access Platform",
    version="0.1.0",
)

orchestrator = QueryLifecycleAgent()


def _audit(event: str, context: Dict[str, Any], query: str, result: Dict[str, Any], conversation_id: str = None) -> None:
    # `context` is the merged context actually evaluated by policy — for a
    # service caller (see the merge logic above) that's the real end-user's
    # identity, not the service account's own. Logging the raw auth token's
    # identity here would make every call routed through a trusted backend
    # show up in the audit trail as one indistinguishable service account.
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "tenant": context.get("tenant"),
        "user_id": context.get("user_id"),
        "role": context.get("role"),
        "auth_type": context.get("auth_type"),
        "conversation_id": conversation_id,
        "query": query,
        "table": (result.get("intent") or {}).get("table"),
        "authorized": result.get("authorized", False),
        "response_type": "clarification" if result.get("clarification") else "answer",
    }
    logger.info("AUDIT %s", json.dumps(entry))


@app.get("/")
async def root():
    return {"message": "Welcome to Sutradhara API"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    # Auth-derived context overrides any caller-supplied fields of the same
    # name — this is what stops a normal caller from just putting a
    # different role/user_id in `context` to escalate themselves.
    #
    # A "service" role is the one deliberate exception: a trusted backend
    # (e.g. myPatasala's AskController) authenticates to sutradhara as
    # itself, having already resolved the real end-user's role/school/email
    # through its own auth system — that's the identity that must reach
    # policy evaluation, not the service account's own (otherwise
    # meaningless) token identity. So for a service caller, `context` wins.
    if user.get("role") == "service":
        merged_context = {**user, **(request.context or {})}
    else:
        merged_context = {**(request.context or {}), **user}

    history = [h.model_dump() for h in request.history] if request.history else None
    result = await orchestrator.run(request.query, merged_context, history)
    _audit("ask", merged_context, request.query, result, request.conversation_id)

    if result.get("clarification"):
        return AskResponse(
            type=ResponseType.CLARIFICATION,
            clarification=result["clarification"],
            conversation_id=request.conversation_id,
        )

    return AskResponse(
        type=ResponseType.ANSWER,
        answer=result.get("answer", "No answer generated."),
        conversation_id=request.conversation_id,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
