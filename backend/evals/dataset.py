"""
Bootstraps a synthetic (question, source_chunk_id) eval set from whatever is
already ingested in Postgres -- there is no hand-labeled query/relevance
dataset for this project, so we generate one: for a sample of chunks, ask
the LLM for a question that chunk answers. Retrieval quality can then be
checked by confirming that chunk comes back when that question is asked.

Usage: python -m evals.dataset [--sample-size N]
"""
import argparse
import json
import random
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select

from app.db.models import Chunk
from app.db.session import SyncSessionLocal
from app.services.llm import get_llm
from evals.schemas import GeneratedQuestion

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the passage below from an ISO 15189 lab quality management "
        "document, write ONE specific question that this passage directly "
        "and fully answers. Do not refer to \"the passage\" or \"the "
        "document\" in the question itself.",
    ),
    ("human", "{passage}"),
])


def generate_dataset(sample_size: int = 20, seed: int = 42) -> list[dict]:
    with SyncSessionLocal() as session:
        chunks = session.execute(select(Chunk)).scalars().all()

    if not chunks:
        raise RuntimeError("No chunks found in the database -- run ingestion first.")

    random.seed(seed)
    sample = random.sample(chunks, min(sample_size, len(chunks)))

    chain = QUESTION_PROMPT | get_llm().with_structured_output(GeneratedQuestion)

    records = []
    for chunk in sample:
        result = chain.invoke({"passage": chunk.content})
        records.append({"question": result.question, "chunk_id": chunk.id, "source": chunk.source})
        print(f"[{len(records)}/{len(sample)}] {result.question!r} <- {chunk.source} ({chunk.id[:8]})")

    with open(DATASET_PATH, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nWrote {len(records)} synthetic QA pairs to {DATASET_PATH}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    generate_dataset(sample_size=args.sample_size)
