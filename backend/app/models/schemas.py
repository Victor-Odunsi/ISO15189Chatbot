from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(
        description="The user's question, rewritten to stand alone without needing the chat history"
    )
    intent: Literal["general", "checklist", "sop"] = Field(
        description=(
            "'general' for a direct question/explanation, 'checklist' if the user "
            "wants an audit/compliance checklist, 'sop' if they want a Standard "
            "Operating Procedure document"
        )
    )


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
