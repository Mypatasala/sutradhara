import pytest
from fastapi.testclient import TestClient
from src.gateway.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Sutradhara API"}

def test_ask_endpoint_success():
    payload = {"query": "Tell me about student attendance"}
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "answer"
    assert "SQL used" in data["answer"]

def test_ask_endpoint_invalid_payload():
    payload = {"not_query": "This should fail"}
    response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
