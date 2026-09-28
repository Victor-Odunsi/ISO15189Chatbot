"""
Runs the full eval suite against evals/dataset.jsonl and prints one summary.
Usage: python -m evals.run_all [--sample-size N]
"""
import argparse

from evals import abstention_eval, groundedness_eval, query_analysis_eval, retrieval_eval


def main(sample_size: int) -> None:
    dataset = retrieval_eval.load_dataset()

    print("=== Retrieval quality (Hit Rate@k, MRR) ===")
    retrieval_metrics = retrieval_eval.hit_rate_and_mrr(dataset)
    print(retrieval_metrics)

    print("\n=== Query agreement (paraphrase robustness) ===")
    agreement_metrics = retrieval_eval.query_agreement(dataset)
    print(agreement_metrics)

    print("\n=== Precision@k ===")
    precision_metrics = retrieval_eval.precision_at_k(dataset)
    print(precision_metrics)

    print("\n=== Stage breakdown (dense / sparse / fused / reranked) ===")
    stage_metrics = retrieval_eval.stage_breakdown(dataset)
    print(stage_metrics)

    print("\n=== Groundedness (faithfulness, answer relevancy) -- all intents ===")
    groundedness_metrics = groundedness_eval.run_all_intents(dataset)
    print(groundedness_metrics)

    print("\n=== Abstention on out-of-domain / unanswerable questions ===")
    abstention_metrics = abstention_eval.run()
    print(abstention_metrics)

    print("\n=== Intent classification accuracy ===")
    intent_metrics = query_analysis_eval.intent_accuracy(dataset, sample_size=sample_size)
    print(intent_metrics)

    print("\n=== Standalone question quality (coreference resolution) ===")
    standalone_metrics = query_analysis_eval.standalone_question_quality(dataset, sample_size=sample_size)
    print(standalone_metrics)

    print("\n=== Summary ===")
    print({
        **retrieval_metrics,
        **agreement_metrics,
        **precision_metrics,
        "stages": stage_metrics,
        "groundedness_by_intent": groundedness_metrics,
        **abstention_metrics,
        **intent_metrics,
        **standalone_metrics,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()
    main(sample_size=args.sample_size)
