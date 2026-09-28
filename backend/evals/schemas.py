from pydantic import BaseModel, Field


class GeneratedQuestion(BaseModel):
    question: str = Field(description="A question that this passage directly and fully answers")


class Paraphrases(BaseModel):
    paraphrases: list[str] = Field(
        description="Alternative phrasings of the question, same meaning and intent, different wording"
    )


class Claims(BaseModel):
    claims: list[str] = Field(
        description="Atomic, independently-checkable factual claims made in the answer"
    )


class ClaimVerdict(BaseModel):
    supported: bool = Field(description="Whether the claim is directly supported by the given context")


class BacktranslatedQuestions(BaseModel):
    questions: list[str] = Field(
        description="Questions that this answer would be a good, direct response to"
    )
