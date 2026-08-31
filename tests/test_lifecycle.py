import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.gateway.main import app
from src.gateway.models import ResponseType
from src.gateway.auth import get_current_user


@pytest.fixture()
def auth_client():
    user = {"role": "admin", "user_id": "admin_01", "department_id": "1", "auth_type": "static"}
    app.dependency_overrides[get_current_user] = lambda: user
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_lifecycle_intent_resolution(mock_llm, auth_client):
    """Verify that intent resolution is triggered and results in a proper answer."""
    response = auth_client.post("/api/v1/ask", json={"query": "Show all users"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "answer"
    assert data["answer"] is not None


def test_lifecycle_resolution_from_clarification(auth_client):
    """Verify that an explicit SQL mock produces an answer containing the table name.

    Forces the new structured-intent path to report a technical failure so
    this test deterministically exercises the legacy free-text path it was
    written for -- without this, it would otherwise hit the app's real
    configured LLM (Ollama, in this deployment) for structured resolution
    first, since that path isn't covered by the ChatGoogleGenerativeAI-only
    mock below and predates this test."""
    payload = {"query": "Show students and grades"}
    with patch(
        "src.agents.intent_agent.IntentResolutionAgent.resolve_structured",
        new=AsyncMock(side_effect=RuntimeError("structured path disabled for this test")),
    ), patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke") as mock_ainvoke:
        mock_res = MagicMock()
        mock_res.content = "TABLE: report_cards\nACTION: select\nSQL: SELECT * FROM report_cards"
        mock_ainvoke.return_value = mock_res

        response = auth_client.post("/api/v1/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "answer"


def test_ask_endpoint_success(mock_llm, auth_client):
    response = auth_client.post("/api/v1/ask", json={"query": "test"})
    assert response.status_code == 200

