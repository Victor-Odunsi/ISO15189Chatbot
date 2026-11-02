"""
Checks whether generated answers are grounded in retrieved knowledge:

- Faithfulness: the answer is decomposed into atomic claims, and each claim
  is checked against the actual retrieved context (LLM-as-judge, entailed or
  not). Score = fraction of claims supported. Directly catches hallucination.
- Answer relevancy (backtranslation): the LLM generates questions that the
  answer would be a good response to, embeds them, and compares them to the
  original question by cosine similarity. Catches a different failure mode
  than faithfulness -- an answer can be fully grounded yet evasive or
  off-topic, which faithfulness alone would still score as "supported".

Usage: python -m evals.groundedness_eval
Requires evals/dataset.jsonl -- run `python -m evals.dataset` first.
"""
import json
from pathlib import Path

import numpy as np
from langchain_core.prompts import ChatPromptTemplate

from app.services.chain import format_context, get_generation_chain, retrieve_context
from app.services.llm import get_llm
from app.services.retrieval import get_embeddings
from evals.schemas import BacktranslatedQuestions, Claims, ClaimVerdict

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

CLAIMS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Break the answer below into a list of atomic, independently "
        "checkable factual claims. Split compound sentences into separate "
        "claims. Omit hedges or caveats that are not themselves factual claims.",
    ),
    ("human", "{answer}"),
])

VERDICT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the context below, is the claim directly supported by it? "
        "Judge only against this context, not outside knowledge.\n\nContext:\n{context}",
    ),
    ("human", "Claim: {claim}"),
])

BACKTRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the answer below, write {n} questions that this answer would "
        "be a good, direct response to.",
    ),
    ("human", "{answer}"),
])


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise RuntimeError(f"No dataset at {DATASET_PATH} -- run `python -m evals.dataset` first.")
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


def generate_answer(question: str, intent: str = "general") -> tuple[str, str]:
    documents = retrieve_context(question)
    context = format_context(documents)
    answer = get_generation_chain(intent).invoke({
        "context": context,
        "chat_history": [],
        "standalone_question": question,
    })
    return answer, context


def faithfulness(answer: str, context: str) -> float:
    claims_chain = CLAIMS_PROMPT | get_llm().with_structured_output(Claims)
    claims = claims_chain.invoke({"answer": answer}).claims
    if not claims:
        return 1.0

    verdict_chain = VERDICT_PROMPT | get_llm().with_structured_output(ClaimVerdict)
    supported = sum(
        verdict_chain.invoke({"context": context, "claim": claim}).supported for claim in claims
    )
    return supported / len(claims)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def answer_relevancy(question: str, answer: str, n_questions: int = 3) -> float:
    backtranslate_chain = BACKTRANSLATE_PROMPT | get_llm().with_structured_output(BacktranslatedQuestions)
    generated = backtranslate_chain.invoke({"answer": answer, "n": n_questions}).questions
    if not generated:
        return 0.0

    embeddings = get_embeddings()
    original_vec = embeddings.embed_query(question)
    generated_vecs = embeddings.embed_documents(generated)

    return sum(cosine_similarity(original_vec, vec) for vec in generated_vecs) / len(generated_vecs)


def run(dataset: list[dict], intent: str = "general") -> dict:
    faithfulness_scores = []
    relevancy_scores = []

    for record in dataset:
        question = record["question"]
        answer, context = generate_answer(question, intent=intent)

        f_score = faithfulness(answer, context)
        r_score = answer_relevancy(question, answer)

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)
        print(f"[{intent}] faithfulness={f_score:.2f}  relevancy={r_score:.2f}  {question!r}")

    n = len(dataset)
    return {
        "mean_faithfulness": sum(faithfulness_scores) / n,
        "mean_answer_relevancy": sum(relevancy_scores) / n,
        "n": n,
    }


def run_all_intents(dataset: list[dict]) -> dict:
    # CHECKLIST_PROMPT/SOP_PROMPT reshape the same retrieved context into a
    # different document structure -- faithfulness/relevancy generalize to
    # them unchanged, but only "general" was ever actually being checked.
    return {intent: run(dataset, intent=intent) for intent in ("general", "checklist", "sop")}


if __name__ == "__main__":
    dataset = load_dataset()
    print("=== Groundedness (faithfulness, answer relevancy) -- all intents ===")
    print(run_all_intents(dataset))
