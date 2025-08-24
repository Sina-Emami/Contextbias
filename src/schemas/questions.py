from typing import List, Optional, Literal
from pydantic import BaseModel


# Your original parallel-arrays shape (kept for compatibility with OSS 20B output)
class QuestionSet(BaseModel):
    questions_list: List[str]
    reason_of_question: List[str]


# Pipeline-friendly shapes
class CheckQuestion(BaseModel):
    question: str
    reason: str


class FactResult(BaseModel):
    question: str
    answer: str  # "42%", "YES"/"NO", one word, or "NOT FOUND"
    source: Optional[str] = None
    status: Literal["FOUND", "NOT_FOUND"] = "FOUND"
