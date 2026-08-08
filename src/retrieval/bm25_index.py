"""BM25 keyword search index.

BM25 is the classic search-engine ranking method: it scores each chunk by how
many of the query's words it contains, giving rare words more weight than
common ones. It finds exact word matches; semantic (vector) search finds
matches by meaning. The hybrid retriever combines both.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import INDEX_DIR, PROCESSED_DIR


class BM25Index:
    """Keyword search over all corpus chunks; built once, saved as a pickle, loaded at query time."""

    def __init__(self, chunks: list[dict], bm25: BM25Okapi, chunk_ids: list[str]):
        self.chunks = chunks
        self.bm25 = bm25
        self.chunk_ids = chunk_ids
        self._id_to_chunk = {c["chunk_id"]: c for c in chunks}

    @classmethod
    def build(cls, chunks_path: Path | None = None) -> "BM25Index":
        path = chunks_path or (PROCESSED_DIR / "chunks.jsonl")
        chunks: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        tokenized = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        chunk_ids = [c["chunk_id"] for c in chunks]
        return cls(chunks, bm25, chunk_ids)

    def save(self, index_dir: Path = INDEX_DIR) -> Path:
        index_dir.mkdir(parents=True, exist_ok=True)
        path = index_dir / "bm25.pkl"
        with path.open("wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "bm25": self.bm25, "chunk_ids": self.chunk_ids},
                f,
            )
        return path

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "BM25Index":
        path = index_dir / "bm25.pkl"
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(data["chunks"], data["bm25"], data["chunk_ids"])

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return the top_k chunks whose words best match the query; chunks with no overlap are dropped."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: list[dict] = []
        for chunk_id, score in ranked:
            if score <= 0:
                continue
            chunk = self._id_to_chunk[chunk_id]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(score),
                    "url": chunk["url"],
                    "source": chunk["source"],
                    "category": chunk["category"],
                    "title": chunk["title"],
                    "domain": chunk["domain"],
                    "doc_id": chunk["doc_id"],
                    "retriever": "bm25",
                }
            )
        return results


def build_and_save_bm25() -> BM25Index:
    index = BM25Index.build()
    index.save()
    return index
