import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.gateway.main import app
from src.gateway.auth import get_current_user

_DEFAULT_ADMIN = {
    "role": "admin",
    "user_id": "admin_01",
    "department_id": "1",
    "linked_student_id": None,
    "auth_type": "static",
}


@pytest.fixture(autouse=True)
def mock_policy_engine():
    """Globally mock PolicyEngine.evaluate to return authorized by default."""
    with patch("src.policy.engine.PolicyEngine.evaluate") as mock_eval:
        mock_eval.return_value = {
            "authorized": True,
            "columns": [],
            "filter": "",
            "error": None,
        }
        yield mock_eval


@pytest.fixture()
def mock_llm():
    """Mock LLM calls to prevent real API requests during tests."""
    mock_response = MagicMock()
    mock_response.content = "TABLE: users\nACTION: select\nSQL: SELECT * FROM users"

    with patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.ChatOpenAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.os.getenv", side_effect=lambda k, *a: "fake_key" if ("GOOGLE" in k or "OPENAI" in k) else None):
        yield


@pytest.fixture()
def auth_client():
    """TestClient with admin auth dependency overridden (no real JWT needed)."""
    app.dependency_overrides[get_current_user] = lambda: _DEFAULT_ADMIN
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def admin_user():
    return _DEFAULT_ADMIN

