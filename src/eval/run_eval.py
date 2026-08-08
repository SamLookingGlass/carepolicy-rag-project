"""Run golden-set evaluation and retrieval ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import EVAL_DIR
from src.eval.metrics import evaluate_result, load_golden, summarize
from src.pipeline import RAGPipeline


def run_eval(
    golden_path: Path | None = None,
    retrieval_mode: str = "hybrid",
    output_path: Path | None = None,
) -> dict:
    """Run every golden question through the pipeline and write data/eval/report_<mode>.json."""
    items = load_golden(golden_path)
    pipeline = RAGPipeline()
    results = []

    for item in items:
        result = pipeline.query(item["question"], retrieval_mode=retrieval_mode)
        results.append(evaluate_result(item, result))

    summary = summarize(results)
    summary["retrieval_mode"] = retrieval_mode

    report = {"summary": summary, "results": results}
    out = output_path or (EVAL_DIR / f"report_{retrieval_mode}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_ablation(golden_path: Path | None = None) -> dict:
    """Run the same questions in all three retrieval modes to compare them side by side.

    An ablation holds everything else constant and swaps one component (here,
    the retrieval mode) to show what each part contributes.
    """
    modes = ["dense", "bm25", "hybrid"]
    ablation = {}
    for mode in modes:
        report = run_eval(golden_path, retrieval_mode=mode)
        ablation[mode] = report["summary"]

    ablation_path = EVAL_DIR / "ablation_report.json"
    ablation_path.write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    return ablation


def main():
    parser = argparse.ArgumentParser(description="Run CarePolicy RAG evaluation")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25", "ablation"])
    parser.add_argument("--golden", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "ablation":
        report = run_ablation(args.golden)
        print(json.dumps(report, indent=2))
    else:
        report = run_eval(args.golden, retrieval_mode=args.mode)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
