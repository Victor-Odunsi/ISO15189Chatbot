"""
Checks whether a query fetches the right context:

- Hit Rate@k / MRR: for each synthetic (question, source_chunk_id) pair, does
  hybrid_retrieve() actually surface that chunk, and how highly ranked is it.
- Query agreement: retrieval should be robust to how a question is phrased.
  For each question, an LLM generates paraphrases; we retrieve for each and
  measure the overlap (Jaccard) of retrieved chunk ids against the original
  phrasing. Low agreement flags topics where retrieval is fragile to wording,
  which Hit Rate@k alone (only ever tested against the original phrasing)
  cannot catch.

Usage: python -m evals.retrieval_eval [--k N] [--n-paraphrases N]
Requires evals/dataset.jsonl -- run `python -m evals.dataset` first.
"""
import argparse
import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from app.services.llm import get_llm
from app.services.retrieval import hybrid_retrieve
from evals.schemas import Paraphrases

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

PARAPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Reword the question below in {n} different ways. Keep the exact "
        "same meaning and intent -- only change the phrasing and wording.",
    ),
    ("human", "{question}"),
])


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise RuntimeError(f"No dataset at {DATASET_PATH} -- run `python -m evals.dataset` first.")
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


def retrieved_ids(query: str, k: int) -> list[str]:
    return [doc.metadata.get("chunk_id") for doc in hybrid_retrieve(query, top_k=k)]


def hit_rate_and_mrr(dataset: list[dict], k: int = 5) -> dict:
    hits = 0
    reciprocal_ranks = []

    for record in dataset:
        ids = retrieved_ids(record["question"], k=k)
        if record["chunk_id"] in ids:
            hits += 1
            reciprocal_ranks.append(1.0 / (ids.index(record["chunk_id"]) + 1))
        else:
            reciprocal_ranks.append(0.0)

    n = len(dataset)
    return {"hit_rate@k": hits / n, "mrr": sum(reciprocal_ranks) / n, "k": k, "n": n}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def query_agreement(dataset: list[dict], n_paraphrases: int = 3, k: int = 5) -> dict:
    chain = PARAPHRASE_PROMPT | get_llm().with_structured_output(Paraphrases)
    per_question_scores = []

    for record in dataset:
        question = record["question"]
        original_ids = set(retrieved_ids(question, k=k))

        result = chain.invoke({"question": question, "n": n_paraphrases})
        agreement_scores = [jaccard(original_ids, set(retrieved_ids(p, k=k))) for p in result.paraphrases]

        avg = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 1.0
        per_question_scores.append(avg)
        print(f"agreement={avg:.2f}  {question!r}")

    return {
        "mean_query_agreement": sum(per_question_scores) / len(per_question_scores),
        "n_paraphrases": n_paraphrases,
        "k": k,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-paraphrases", type=int, default=3)
    args = parser.parse_args()

    dataset = load_dataset()

    print("=== Retrieval quality (Hit Rate@k, MRR) ===")
    print(hit_rate_and_mrr(dataset, k=args.k))

    print("\n=== Query agreement (paraphrase robustness) ===")
    print(query_agreement(dataset, n_paraphrases=args.n_paraphrases, k=args.k))
