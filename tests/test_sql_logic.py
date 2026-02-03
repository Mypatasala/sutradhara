import pytest
from src.retrieval.sql_builder import SQLBuilder

@pytest.fixture
def generator():
    return SQLBuilder()

def test_sql_join_students_grades(generator):
    """Requirement: Join multiple tables based on semantic relationships."""
    intent = {
        "action": "list",
        "entities": ["student.name", "grade.score"],
        "joins": [{"from": "students", "to": "grades", "on": "students.id = grades.student_id"}]
    }
    sql = generator.generate(intent)
    assert "JOIN grades" in sql
    assert "name" in sql
    assert "score" in sql

def test_sql_aggregate_avg_grade_per_class(generator):
    """Requirement: Aggregate functions and grouping."""
    intent = {
        "action": "aggregate",
        "function": "AVG",
        "target": "grade.score",
        "group_by": "class.id"
    }
    sql = generator.generate(intent)
    assert "AVG" in sql
    assert "GROUP BY" in sql

def test_sql_window_function_ranking(generator):
    """Requirement: Window functions like RANK(), ROW_NUMBER()."""
    intent = {
        "action": "rank",
        "target": "total_score",
        "partition_by": "school_id",
        "order_by": "total_score DESC"
    }
    sql = generator.generate(intent)
    assert "RANK() OVER" in sql
    assert "PARTITION BY school_id" in sql
    assert "ORDER BY total_score DESC" in sql

def test_sql_filtering_with_temporal_constraints(generator):
    """Requirement: Advanced filtering (BETWEEN, conditional logic)."""
    intent = {
        "entity": "invoices",
        "filters": [
            {"column": "created_at", "operator": "BETWEEN", "value": ["2023-01-01", "2023-06-30"]},
            {"column": "status", "operator": "==", "value": "paid"}
        ]
    }
    sql = generator.generate(intent)
    assert "BETWEEN '2023-01-01' AND '2023-06-30'" in sql
    assert "status = 'paid'" in sql


def test_sql_complex_selection_subset():
    """Requirement: Selecting specific sets of columns from different tables."""
    # Intent: "Show teacher names, their department, and their assigned classroom number"
    intent = {
        "entities": ["teachers.name", "departments.name", "classrooms.room_number"],
        "join_path": ["teachers", "departments", "classrooms"]
    }
    # Expected SQL should handle multiple joins and specific column aliasing
    pass
