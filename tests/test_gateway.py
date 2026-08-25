import pytest
from fastapi.testclient import TestClient
from src.gateway.main import app
from src.gateway.auth import get_current_user, _STATIC_TOKENS

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Sutradhara API"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_requires_auth():
    """No Authorization header must return 401."""
    response = client.post("/api/v1/ask", json={"query": "Show all users"})
    assert response.status_code == 401


def test_ask_endpoint_invalid_token():
    """A malformed JWT must return 401."""
    response = client.post(
        "/api/v1/ask",
        json={"query": "Show all users"},
        headers={"Authorization": "Bearer not.a.jwt"},
    )
    assert response.status_code == 401


def test_ask_endpoint_static_token(mock_llm, monkeypatch):
    """A recognised static token must authenticate and return an answer."""
    monkeypatch.setitem(_STATIC_TOKENS, "test-static-token", {
        "role": "admin",
        "user_id": "admin_01",
        "department_id": "1",
        "linked_student_id": None,
        "auth_type": "static",
    })
    response = client.post(
        "/api/v1/ask",
        json={"query": "Show users"},
        headers={"Authorization": "Bearer test-static-token"},
    )
    assert response.status_code == 200
    assert response.json()["type"] == "answer"


def test_ask_endpoint_success(mock_llm, auth_client):
    payload = {"query": "Tell me about student attendance"}
    response = auth_client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "answer"


def test_ask_endpoint_invalid_payload(auth_client):
    response = auth_client.post("/api/v1/ask", json={"not_query": "This should fail"})
    assert response.status_code == 422


def test_ask_conversation_id_echoed(mock_llm, auth_client):
    """conversation_id from the request must be echoed back in the response."""
    response = auth_client.post(
        "/api/v1/ask",
        json={"query": "Show users", "conversation_id": "sess-abc-123"},
    )
    assert response.status_code == 200
    assert response.json()["conversation_id"] == "sess-abc-123"

