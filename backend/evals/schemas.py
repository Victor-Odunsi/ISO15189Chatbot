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


class RelevanceVerdict(BaseModel):
    relevant: bool = Field(description="Whether this passage is relevant to answering the question")


class AbstentionVerdict(BaseModel):
    abstained: bool = Field(
        description=(
            "Whether the answer appropriately declined or hedged (said it lacks "
            "sufficient information / the topic is not covered), rather than "
            "confidently asserting specific facts as if it had a good answer"
        )
    )


class StandaloneQuestionVerdict(BaseModel):
    is_self_contained: bool = Field(
        description=(
            "Whether the standalone question can be understood on its own, without "
            "the prior conversation, and preserves the same topic and intent as what "
            "the user actually meant by their follow-up"
        )
    )
