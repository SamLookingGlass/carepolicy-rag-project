"""Cross-encoder reranking: a second, more careful pass over the retrieved chunks.

Retrieval compares the question and chunks separately (fast but rough). A
cross-encoder reads the question and one chunk *together* and scores how well
they match — slower but much more accurate, so we only run it on the ~20
candidates retrieval already picked.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import get_settings


@lru_cache
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    get_settings()  # ensure HF_TOKEN from .env is exported to os.environ
    # Stay on CPU. On a ZeroGPU Space, leaving the device automatic marks this
    # small model as a CUDA tensor, and the host then unloads it from the
    # process that actually answers questions.
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """Rescore candidates with the cross-encoder and keep the best top_k.

    Scores are raw model outputs (logits), not 0-1 probabilities: a relevant
    pair usually scores above 0, an irrelevant one well below. The pipeline's
    refusal gate compares the top score against a threshold.
    """
    settings = get_settings()
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    reranked: list[dict] = []
    for candidate, score in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        reranked.append(item)

    return reranked[: top_k or settings.top_k_rerank]
