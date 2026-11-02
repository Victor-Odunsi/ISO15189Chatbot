from typing import Literal, Optional

from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(
        description="The user's question, rewritten to stand alone without needing the chat history"
    )
    intent: Literal["chitchat", "general", "checklist", "sop"] = Field(
        description=(
            "'chitchat' for greetings, thanks, farewells, or questions about what "
            "the assistant can do -- anything that is not actually a question "
            "about ISO 15189 content; 'general' for a direct question/explanation, "
            "'checklist' if the user wants an audit/compliance checklist, 'sop' if "
            "they want a Standard Operating Procedure document"
        )
    )


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
