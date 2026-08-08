"""Score pipeline answers against the golden question set.

Plain-language definitions of every metric live in the README under
"Reading the reports".
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import EVAL_DIR
from src.generation.citations import verify_citations
from src.pipeline import QueryResult
from src.retrieval.embeddings import embed_texts


def load_golden(path: Path | None = None) -> list[dict]:
    golden_path = path or (EVAL_DIR / "golden.jsonl")
    items: list[dict] = []
    with golden_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def answer_similarity(answer: str, reference_answer: str) -> float | None:
    """Cosine similarity between generated and reference answer embeddings.

    A proxy for semantic correctness — unlike citation checks, this compares
    the answer's content against the hand-written reference. None when there
    is no reference (refusal cases).
    """
    if not reference_answer:
        return None
    vec_answer, vec_ref = embed_texts([answer, reference_answer])
    return _cosine(vec_answer, vec_ref)


def domain_match(result: QueryResult, expected_domains: list[str]) -> bool:
    if not expected_domains:
        return True
    cited_domains = {c.get("domain", "") for c in result.citations}
    return any(any(exp in d for d in cited_domains) for exp in expected_domains)


def evaluate_result(item: dict, result: QueryResult) -> dict:
    """Score one answer against its golden question: refusal correct, citations valid, right domain cited, similarity to reference."""
    citation_check = verify_citations(result.answer, result.retrieved_chunks)
    expected_refusal = item.get("expect_refusal", False)

    if expected_refusal:
        success = result.refused
    else:
        success = not result.refused and citation_check["valid"]

    domain_ok = domain_match(result, item.get("expected_source_domains", []))

    return {
        "id": item["id"],
        "question": item["question"],
        "category": item.get("category", ""),
        "success": success and domain_ok,
        "refused": result.refused,
        "expected_refusal": expected_refusal,
        "citation_accuracy": citation_check["citation_accuracy"],
        "answer_similarity": answer_similarity(
            result.answer, item.get("reference_answer", "")
        ),
        "domain_match": domain_ok,
        "latency_ms": result.latency_ms,
        "confidence": result.confidence,
    }


def summarize(results: list[dict]) -> dict:
    """Roll per-question scores up into the aggregate numbers reported in the README."""
    if not results:
        return {}
    n = len(results)
    similarities = [
        r["answer_similarity"] for r in results if r.get("answer_similarity") is not None
    ]
    return {
        "total": n,
        "success_rate": sum(1 for r in results if r["success"]) / n,
        "refusal_accuracy": sum(
            1 for r in results if r["refused"] == r["expected_refusal"]
        )
        / n,
        "avg_citation_accuracy": sum(r["citation_accuracy"] for r in results) / n,
        "avg_answer_similarity": (
            sum(similarities) / len(similarities) if similarities else None
        ),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / n,
        "p95_latency_ms": sorted(r["latency_ms"] for r in results)[int(0.95 * (n - 1))],
    }
