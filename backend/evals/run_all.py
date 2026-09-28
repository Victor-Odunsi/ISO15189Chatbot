"""
Runs the full eval suite against evals/dataset.jsonl and prints one summary.
Usage: python -m evals.run_all
"""
from evals import groundedness_eval, retrieval_eval


def main() -> None:
    dataset = retrieval_eval.load_dataset()

    print("=== Retrieval quality (Hit Rate@k, MRR) ===")
    retrieval_metrics = retrieval_eval.hit_rate_and_mrr(dataset)
    print(retrieval_metrics)

    print("\n=== Query agreement (paraphrase robustness) ===")
    agreement_metrics = retrieval_eval.query_agreement(dataset)
    print(agreement_metrics)

    print("\n=== Groundedness (faithfulness, answer relevancy) ===")
    groundedness_metrics = groundedness_eval.run(dataset)
    print(groundedness_metrics)

    print("\n=== Summary ===")
    print({**retrieval_metrics, **agreement_metrics, **groundedness_metrics})


if __name__ == "__main__":
    main()
