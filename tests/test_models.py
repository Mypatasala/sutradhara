import pytest
from pydantic import ValidationError
from src.gateway.models import AskRequest, AskResponse, ResponseType, ClarificationPayload, ClarificationOption

def test_ask_request_valid():
    request = AskRequest(query="What is the sales for last month?")
    assert request.query == "What is the sales for last month?"
    assert request.context is None

def test_ask_request_with_context():
    request = AskRequest(query="Show sales", context={"table": "online_sales"})
    assert request.context == {"table": "online_sales"}

def test_ask_response_answer():
    response = AskResponse(
        type=ResponseType.ANSWER,
        answer="The total sales is $5000."
    )
    assert response.type == ResponseType.ANSWER
    assert response.answer == "The total sales is $5000."
    assert response.clarification is None

def test_ask_response_clarification():
    options = [
        ClarificationOption(label="Online Sales", value="online_sales"),
        ClarificationOption(label="Retail Sales", value="retail_sales")
    ]
    payload = ClarificationPayload(question="Which sales are you referring to?", options=options)
    response = AskResponse(
        type=ResponseType.CLARIFICATION,
        clarification=payload
    )
    assert response.type == ResponseType.CLARIFICATION
    assert response.clarification.question == "Which sales are you referring to?"
    assert len(response.clarification.options) == 2

def test_invalid_response_type():
    with pytest.raises(ValidationError):
        AskResponse(type="invalid_type", answer="test")
