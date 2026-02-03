import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(autouse=True)
def mock_llm():
    """Globally mock LLM calls to prevent real API requests during tests."""
    mock_response = MagicMock()
    mock_response.content = "SELECT * FROM users"
    
    with patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.ChatOpenAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.os.getenv", side_effect=lambda k, *args: "fake_key" if ("GOOGLE" in k or "OPENAI" in k) else None):
        yield
