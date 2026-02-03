from typing import List, Optional, Union
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

class AskRequest(BaseModel):
    query: str
    context: Optional[dict] = None

class AskResponse(BaseModel):
    type: ResponseType
    answer: Optional[str] = None
    clarification: Optional[ClarificationPayload] = None
