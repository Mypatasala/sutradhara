import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.gateway.main import app
from src.gateway.models import ResponseType

client = TestClient(app)

@pytest.mark.asyncio
async def test_lifecycle_intent_resolution():
    """Verify that intent resolution is triggered and results in a proper answer."""
    payload = {"query": "Show all users"}
    # The global mock in conftest.py will handle the LLM call
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "answer"
    assert "SQL used" in data["answer"]
    assert "SELECT * FROM users" in data["answer"]

def test_lifecycle_resolution_from_clarification():
    """Verify that the system can handle clarification context (Phase 2/3)."""
    payload = {
        "query": "Show students and grades",
        "context": {"selection": "online_sales"}
    }
    # Mock specific SQL for this test
    with patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke") as mock_ainvoke:
        mock_res = MagicMock()
        mock_res.content = "SELECT * FROM report_cards"
        mock_ainvoke.return_value = mock_res
        
        response = client.post("/api/v1/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "answer"
        assert "report_cards" in data["answer"]

def test_ask_endpoint_success():
    """Generic successful endpoint test."""
    response = client.post("/api/v1/ask", json={"query": "test"})
    assert response.status_code == 200
