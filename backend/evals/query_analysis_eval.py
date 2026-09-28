"""
Checks the query-analysis step that runs before retrieval ever happens --
analyze_query() rewrites the question standalone against chat history and
classifies intent (general/checklist/sop). Every other eval in this suite
implicitly assumes that step got it right; if it doesn't, retrieval gets
the wrong query and the failure looks like a retrieval or groundedness
problem downstream instead of what it actually is.

- Intent classification accuracy: hand-crafted phrasings of the same topic
  ("explain X" / "give me a checklist for X" / "write an SOP for X") checked
  against the intent analyze_query() is supposed to assign each one.
- Standalone question quality (coreference resolution): a follow-up that
  relies on the prior turn ("what about that instead?") is only ever tested
  implicitly by dataset questions, which are all single-turn by construction.
  An LLM generates a context-dependent follow-up, analyze_query() resolves
  it against a short history, and a judge checks whether the resolved
  question is actually self-contained and on-topic.

Usage: python -m evals.query_analysis_eval [--sample-size N]
Requires evals/dataset.jsonl -- run `python -m evals.dataset` first.
"""
import argparse
import asyncio
import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from app.services.chain import analyze_query
from app.services.llm import get_llm
from evals.schemas import StandaloneQuestionVerdict

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"

INTENT_TEMPLATES = [
    ("Can you explain {topic}", "general"),
    ("Give me a compliance checklist for {topic}", "checklist"),
    ("Write me an SOP covering {topic}", "sop"),
]

FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given this question, write a natural follow-up question a user "
        "might ask next in the same conversation. Use a pronoun or omit "
        "the subject entirely (e.g. \"what about that\", \"is it required "
        "there too\") so it only makes sense together with the original question.",
    ),
    ("human", "{question}"),
])

STANDALONE_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "A user asked an initial question, then a follow-up that relies on "
        "that context. A rewriting system produced a standalone version of "
        "the follow-up. Judge whether that standalone version can be "
        "understood on its own and preserves the same topic and intent as "
        "the user's actual follow-up.",
    ),
    (
        "human",
        "Initial question: {initial}\nFollow-up: {follow_up}\n"
        "Standalone rewrite: {standalone}",
    ),
])


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise RuntimeError(f"No dataset at {DATASET_PATH} -- run `python -m evals.dataset` first.")
    with open(DATASET_PATH) as f:
        return [json.loads(line) for line in f]


def intent_accuracy(dataset: list[dict], sample_size: int = 10) -> dict:
    sample = dataset[:sample_size]
    correct = 0
    total = 0

    for record in sample:
        topic = record["question"].rstrip("?").lower()
        for template, expected_intent in INTENT_TEMPLATES:
            phrased = template.format(topic=topic)
            result = asyncio.run(analyze_query(phrased, chat_history=[]))
            is_correct = result.intent == expected_intent
            correct += is_correct
            total += 1
            print(f"expected={expected_intent}  got={result.intent}  {phrased!r}")

    return {"intent_accuracy": correct / total, "n": total}


def standalone_question_quality(dataset: list[dict], sample_size: int = 10) -> dict:
    follow_up_chain = FOLLOW_UP_PROMPT | get_llm()
    judge_chain = STANDALONE_JUDGE_PROMPT | get_llm().with_structured_output(StandaloneQuestionVerdict)

    sample = dataset[:sample_size]
    results = []

    for record in sample:
        initial = record["question"]
        follow_up = follow_up_chain.invoke({"question": initial}).content

        history = [
            {"role": "user", "content": initial},
            {"role": "assistant", "content": "Here is information about that."},
        ]
        analysis = asyncio.run(analyze_query(follow_up, chat_history=history))

        verdict = judge_chain.invoke({
            "initial": initial,
            "follow_up": follow_up,
            "standalone": analysis.standalone_question,
        }).is_self_contained
        results.append(verdict)
        print(f"self_contained={verdict}  {follow_up!r} -> {analysis.standalone_question!r}")

    return {"standalone_question_quality": sum(results) / len(results), "n": len(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    dataset = load_dataset()

    print("=== Intent classification accuracy ===")
    print(intent_accuracy(dataset, sample_size=args.sample_size))

    print("\n=== Standalone question quality (coreference resolution) ===")
    print(standalone_question_quality(dataset, sample_size=args.sample_size))
