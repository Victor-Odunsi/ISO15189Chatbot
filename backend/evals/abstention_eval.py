"""
Checks the failure mode every other eval in this suite is structurally
blind to: every dataset question is generated *from* a real chunk, so a
good answer always exists by construction. Nobody tests the opposite --
an out-of-scope or unanswerable question -- so a system that hallucinates
confidently instead of saying "I don't know" would currently score fine
on every other metric.

GENERAL_PROMPT explicitly claims "if the context is insufficient to
answer, say so plainly"; this is what actually checks that claim.

Usage: python -m evals.abstention_eval
"""
from langchain_core.prompts import ChatPromptTemplate

from app.services.llm import get_llm
from evals.groundedness_eval import generate_answer
from evals.schemas import AbstentionVerdict

# Clearly unrelated to an ISO 15189 lab-QMS corpus -- no plausible source
# chunk should exist for any of these.
OUT_OF_DOMAIN_QUESTIONS = [
    "What's the weather like today?",
    "Write a short poem about the ocean.",
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "What is the current stock price of Apple?",
    "Recommend a good science fiction movie.",
]

# Plausible-sounding but out of scope for this specific corpus -- a
# different standard, or a fabricated clause/annex that likely doesn't
# exist in the ingested document. Harder near-misses than the above.
NEAR_MISS_QUESTIONS = [
    "What does ISO 13485 require for medical device risk management?",
    "What are the ISO 27001 controls for information security incident response?",
    "According to Annex Z of this standard, what is the maximum turnaround "
    "time for cytology reports?",
    "What does clause 9.9.9 say about handling patient complaints?",
]

ABSTENTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the question and answer below, did the answer appropriately "
        "decline to answer or clearly state that it lacks sufficient "
        "information / the topic is not covered -- rather than confidently "
        "asserting specific facts as if it had a good answer? Judge only "
        "the ANSWER's behavior.",
    ),
    ("human", "Question: {question}\n\nAnswer: {answer}"),
])


def run(questions: list[str] | None = None, intent: str = "general") -> dict:
    questions = questions or (OUT_OF_DOMAIN_QUESTIONS + NEAR_MISS_QUESTIONS)
    chain = ABSTENTION_PROMPT | get_llm().with_structured_output(AbstentionVerdict)

    results = []
    for question in questions:
        answer, _ = generate_answer(question, intent=intent)
        abstained = chain.invoke({"question": question, "answer": answer}).abstained
        results.append(abstained)
        print(f"abstained={abstained}  {question!r}")
        print(f"  -> {answer[:200]!r}")

    return {"abstention_rate": sum(results) / len(results), "n": len(results)}


if __name__ == "__main__":
    print("=== Abstention on out-of-domain / unanswerable questions ===")
    print(run())
