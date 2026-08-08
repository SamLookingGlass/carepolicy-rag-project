"""Quick pass/fail check of the golden set against the live pipeline.

Unlike src/eval/run_eval.py (which writes full JSON reports), this just
prints a summary, lists any failures, and exits non-zero if the success
rate drops below 80% — handy as a smoke test after re-ingesting the corpus.
"""

from __future__ import annotations

import json
import sys

from src.eval.metrics import evaluate_result, load_golden, summarize
from src.pipeline import RAGPipeline


def main() -> int:
    pipeline = RAGPipeline()
    items = load_golden()
    results = []
    for item in items:
        result = pipeline.query(item["question"])
        results.append(evaluate_result(item, result))

    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for row in failures:
            if row["expected_refusal"] != row["refused"]:
                flag = "refusal_mismatch"
            elif not row["domain_match"]:
                flag = "domain"
            else:
                flag = "citation"
            print(
                f"  {row['id']} [{flag}] conf={row['confidence']:.2f} "
                f"{row['question'][:60]}"
            )
    return 0 if summary.get("success_rate", 0) >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
