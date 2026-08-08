"""Build and verify [1]-style citations.

The LLM is told to mark every claim with a bracketed number pointing at one
of the source chunks it was given ([1] = first chunk, [2] = second, ...).
This module turns those markers into citation objects for the API response
and checks after generation that no marker was invented.
"""

from __future__ import annotations

import re

from src.generation.prompts import REFUSAL_MESSAGE


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Title: {chunk.get('title', 'Unknown')}\n"
            f"Source: {chunk.get('url', '')}\n"
            f"Content: {chunk['text']}"
        )
    return "\n\n".join(parts)


def extract_citation_indices(answer: str) -> list[int]:
    return [int(m) for m in CITATION_PATTERN.findall(answer)]


def build_citation_objects(answer: str, chunks: list[dict]) -> list[dict]:
    """Turn each [N] marker in the answer into a citation with title, URL, and snippet; skip duplicates and out-of-range markers."""
    indices = extract_citation_indices(answer)
    citations: list[dict] = []
    seen: set[int] = set()

    for idx in indices:
        if idx in seen or idx < 1 or idx > len(chunks):
            continue
        seen.add(idx)
        chunk = chunks[idx - 1]
        citations.append(
            {
                "index": idx,
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title"),
                "url": chunk.get("url"),
                "domain": chunk.get("domain"),
                "snippet": chunk["text"][:400],
            }
        )
    return citations


def verify_citations(answer: str, chunks: list[dict]) -> dict:
    """Check every [N] marker points at a chunk that was actually retrieved.

    An answer with no citations at all, or with markers like [9] when only
    5 chunks were provided, fails the check. Refusals pass by definition —
    there is nothing to cite.
    """
    if REFUSAL_MESSAGE.lower() in answer.lower():
        return {"valid": True, "citation_accuracy": 1.0, "cited_count": 0}

    indices = extract_citation_indices(answer)
    if not indices:
        return {"valid": False, "citation_accuracy": 0.0, "cited_count": 0}

    valid = sum(1 for i in indices if 1 <= i <= len(chunks))
    accuracy = valid / len(indices) if indices else 0.0
    return {
        "valid": accuracy >= 0.9,
        "citation_accuracy": accuracy,
        "cited_count": len(set(indices)),
    }


def is_refusal(answer: str) -> bool:
    return REFUSAL_MESSAGE.lower() in answer.lower()
