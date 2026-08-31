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
    """Verify query with mocked LLM response and summarise step.

    Forces the structured-intent path to report a technical failure so this
    test deterministically exercises the legacy free-text path it was
    written for (same isolation pattern as test_gateway.py/test_lifecycle.py
    and this file's own test_school_query_attendance). "Show all teachers
    and their courses" was previously OUT_OF_SCOPE for the structured
    registry every time (no TEACHERS entity), which happened to fall back to
    legacy on its own -- but that's the live model's current first-attempt
    output, not a guarantee, and a later prompt change (unrelated to this
    test's own intent) shifted it to instead pick entity=users with an
    invalid filter/sort, correctly failing closed to a clarification per the
    fallback design rather than reaching this test's mock. That's correct
    production behavior, not a defect; this test's dependency on live model
    output was the actual gap, exactly as previously found and fixed for
    test_school_query_attendance."""
    query = "Show all teachers and their courses"
    context = {"role": "admin"}

    mock_intent = MagicMock()
    mock_intent.content = "TABLE: users\nACTION: select\nSQL: SELECT users.name AS user_name, courses.name AS course_name FROM users JOIN teachers ON users.id = teachers.user_id JOIN courses ON teachers.user_id = courses.teacher_id LIMIT 5"

    with patch(
        "src.agents.intent_agent.IntentResolutionAgent.resolve_structured",
        new=AsyncMock(side_effect=RuntimeError("structured path disabled for this test")),
    ), patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_intent), \
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
    """Verify attendance query returns real data with mocked LLM.

    Forces the structured-intent path to report a technical failure so this
    test deterministically exercises the legacy free-text path it was
    written for (same isolation pattern as test_gateway.py/test_lifecycle.py).
    "Show all current attendance" IS in the structured registry's scope
    (entity=attendance exists), so without this, the test's outcome depends
    on the real configured LLM's (Ollama, in this deployment) exact
    first-attempt output for this question -- previously masked by that
    output happening to hit a parse failure on both attempts (falling back
    to legacy by luck); a prompt change unrelated to this test's own
    intent (structured-path prompt tuning for grade-filter/ranking
    reliability) made the first attempt succeed with a plan structurally
    invalid for ATTENDANCE (operation=list, which ATTENDANCE doesn't
    support), correctly triggering the retry-with-feedback path instead --
    which is deliberately never allowed to fall back to legacy on a
    validation failure (see query_lifecycle.py's fallback decision tree),
    so it failed closed to a clarification instead of reaching this test's
    mock at all. That is correct production behavior, not a defect; this
    test's own dependency on live model output was the actual gap."""
    query = "Show all current attendance"
    context = {"role": "admin"}

    mock_intent = MagicMock()
    mock_intent.content = "TABLE: attendance\nACTION: select\nSQL: SELECT users.name, attendance.date, attendance.status FROM users JOIN students ON users.id = students.user_id JOIN attendance ON students.user_id = attendance.student_id WHERE status = 'Present' LIMIT 5"

    with patch(
        "src.agents.intent_agent.IntentResolutionAgent.resolve_structured",
        new=AsyncMock(side_effect=RuntimeError("structured path disabled for this test")),
    ), patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_intent), \
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

