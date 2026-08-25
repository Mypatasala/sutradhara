from typing import List, Optional
from pydantic import BaseModel
from enum import Enum


class ResponseType(str, Enum):
    ANSWER = "answer"
    CLARIFICATION = "clarification"


class ClarificationOption(BaseModel):
    label: str
    value: str


class ClarificationPayload(BaseModel):
    question: str
    options: List[ClarificationOption]


class HistoryTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AskRequest(BaseModel):
    query: str
    context: Optional[dict] = None
    conversation_id: Optional[str] = None  # multi-turn clarification session ID
    history: Optional[List[HistoryTurn]] = None  # full prior turns of this Ask session


class AskResponse(BaseModel):
    type: ResponseType
    answer: Optional[str] = None
    clarification: Optional[ClarificationPayload] = None
    conversation_id: Optional[str] = None
