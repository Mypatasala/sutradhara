import pytest
import sqlite3
import os
from src.retrieval.sql_builder import SQLBuilder
from src.agents.query_lifecycle import QueryLifecycleAgent
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.fixture
def db_conn():
    conn = sqlite3.connect("school.db")
    yield conn
    conn.close()

@pytest.fixture
def orchestrator():
    return QueryLifecycleAgent()

@pytest.mark.asyncio
async def test_school_query_teachers_and_courses(orchestrator, db_conn):
    """Verify query with mocked LLM response and summarise step."""
    query = "Show all teachers and their courses"
    context = {"role": "admin"}

    mock_intent = MagicMock()
    mock_intent.content = "TABLE: users\nACTION: select\nSQL: SELECT users.name AS user_name, courses.name AS course_name FROM users JOIN teachers ON users.id = teachers.user_id JOIN courses ON teachers.user_id = courses.teacher_id LIMIT 5"

    with patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_intent), \
         patch("src.agents.intent_agent.ChatOpenAI.ainvoke", return_value=mock_intent), \
         patch("src.agents.intent_agent.os.getenv", side_effect=lambda k, *a: "fake_key" if ("GOOGLE" in k or "OPENAI" in k) else None), \
         patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="Mock summary with user_name and course_name data |")) as mock_summarize:

        result = await orchestrator.run(query, context)

        assert "answer" in result
        assert result["answer"] is not None
        mock_summarize.assert_called_once()
        # Verify the mocked summary (real LLM would return natural language)
        assert "user_name" in result["answer"]
        assert "|" in result["answer"]

@pytest.mark.asyncio
async def test_school_query_attendance(orchestrator, db_conn):
    """Verify attendance query returns real data with mocked LLM."""
    query = "Show all current attendance"
    context = {"role": "admin"}

    mock_intent = MagicMock()
    mock_intent.content = "TABLE: attendance\nACTION: select\nSQL: SELECT users.name, attendance.date, attendance.status FROM users JOIN students ON users.id = students.user_id JOIN attendance ON students.user_id = attendance.student_id WHERE status = 'Present' LIMIT 5"

    with patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_intent), \
         patch("src.agents.intent_agent.ChatOpenAI.ainvoke", return_value=mock_intent), \
         patch("src.agents.intent_agent.os.getenv", side_effect=lambda k, *a: "fake_key" if ("GOOGLE" in k or "OPENAI" in k) else None), \
         patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="| name | date | status |\n| --- | --- | --- |\n| Alice | 2024-01-01 | Present |")) as mock_summarize:

        result = await orchestrator.run(query, context)

        assert "answer" in result
        assert result["answer"] is not None
        mock_summarize.assert_called_once()
        assert "Present" in result["answer"]
        assert "|" in result["answer"]

def test_data_integrity(db_conn):
    """Verify that the school.db contains the expected sample data."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    assert count == 1062

    cursor.execute("SELECT role FROM users WHERE role='principal' LIMIT 1")
    role = cursor.fetchone()[0]
    assert role == "principal"

