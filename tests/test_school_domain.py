import pytest
import sqlite3
import os
from src.retrieval.sql_builder import SQLBuilder
from src.agents.query_lifecycle import QueryLifecycleAgent
from unittest.mock import patch, MagicMock

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
    """Verify query with mocked LLM response."""
    query = "Show all teachers and their courses"
    
    # Use aliases to avoid column name collisions
    mock_response = MagicMock()
    mock_response.content = "SELECT users.name AS user_name, courses.name AS course_name FROM users JOIN teachers ON users.id = teachers.user_id JOIN courses ON teachers.user_id = courses.teacher_id"
    
    with patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.ChatOpenAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.os.getenv", side_effect=lambda k, *args: "fake_key" if ("GOOGLE" in k or "OPENAI" in k) else None):
        
        result = await orchestrator.run(query)
        
        assert "answer" in result
        assert "SQL used" in result["answer"]
        assert "Charlie Brown" in result["answer"]
        assert "Physics 101" in result["answer"]
        assert "|" in result["answer"]

@pytest.mark.asyncio
async def test_school_query_attendance(orchestrator, db_conn):
    """Verify attendance query returns real data with mocked LLM."""
    query = "Show all current attendance"
    
    # Join with users to get the name
    mock_response = MagicMock()
    mock_response.content = "SELECT users.name, attendance.date, attendance.status FROM users JOIN students ON users.id = students.user_id JOIN attendance ON students.user_id = attendance.student_id WHERE status = 'present'"
    
    with patch("src.agents.intent_agent.ChatGoogleGenerativeAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.ChatOpenAI.ainvoke", return_value=mock_response), \
         patch("src.agents.intent_agent.os.getenv", side_effect=lambda k, *args: "fake_key" if ("GOOGLE" in k or "OPENAI" in k) else None):
         
        result = await orchestrator.run(query)
        
        assert "answer" in result
        assert "present" in result["answer"]
        assert "Alice Smith" in result["answer"]

def test_data_integrity(db_conn):
    """Verify that the school.db contains the expected sample data."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    assert count == 7
    
    cursor.execute("SELECT name FROM users WHERE role='principal'")
    principal = cursor.fetchone()[0]
    assert principal == "Edward Miller"
