"""Hybrid retrieval: run keyword search and semantic search, then merge the results.

Keyword search (BM25) is good at exact terms like "CHAS"; semantic search
(vectors) is good at paraphrases like "help paying the doctor". Reciprocal
Rank Fusion (RRF) merges the two ranked lists into one, so a chunk that
scores well in either list still surfaces.
"""

from __future__ import annotations

from src.config import INDEX_DIR, get_settings
from src.retrieval.bm25_index import BM25Index
from src.retrieval.vector_store import dense_search


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge several ranked lists into one.

    Each list "votes" for its chunks: rank 1 earns 1/(k+1) points, rank 2
    earns 1/(k+2), and so on, summed across lists. Chunks near the top of
    any list win. k=60 is the standard constant; it softens the gap between
    neighbouring ranks so one list can't dominate.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused: list[dict] = []
    for chunk_id, score in ranked:
        item = dict(chunks[chunk_id])
        item["score"] = score
        item["retriever"] = "hybrid"
        fused.append(item)
    return fused


class HybridRetriever:
    """Front door for retrieval. Loads the BM25 index once and answers queries in any of three modes."""

    def __init__(self, bm25_index: BM25Index | None = None):
        self.settings = get_settings()
        if bm25_index is not None:
            self.bm25 = bm25_index
        else:
            index_path = INDEX_DIR / "bm25.pkl"
            self.bm25 = BM25Index.load() if index_path.exists() else BM25Index.build()

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
    ) -> list[dict]:
        """Fetch the top_k most relevant chunks. Mode is "dense" (meaning), "bm25" (keywords), or "hybrid" (both, fused)."""
        k = top_k or self.settings.top_k_retrieve

        if mode == "dense":
            return dense_search(query, top_k=k)
        if mode == "bm25":
            return self.bm25.search(query, top_k=k)

        dense_results = dense_search(query, top_k=k)
        bm25_results = self.bm25.search(query, top_k=k)
        return reciprocal_rank_fusion([dense_results, bm25_results])[:k]

    def get_chunk(self, chunk_id: str) -> dict | None:
        """Return the full stored chunk for a chunk_id, or None if it is unknown."""
        return self.bm25._id_to_chunk.get(chunk_id)
