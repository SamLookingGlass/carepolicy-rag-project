"""Tools the CarePolicy agent can call: search, then optionally read a full chunk."""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.rerank import rerank

ALLOWED_DOMAINS = ("healthhub.sg", "moh.gov.sg", "hpb.gov.sg", "pdpc.gov.sg")
SNIPPET_CHARS = 400

RerankFn = Callable[[str, list[dict]], list[dict]]


class SearchPolicyArgs(BaseModel):
    query: str = Field(..., description="Search query for Singapore healthcare policy documents")
    domain: str | None = Field(
        default=None,
        description=(
            "Optional official domain filter: healthhub.sg, moh.gov.sg, "
            "hpb.gov.sg, or pdpc.gov.sg"
        ),
    )


class ReadChunkArgs(BaseModel):
    chunk_id: str = Field(..., description="chunk_id from a previous search_policy result")


def passthrough_rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """Keep retrieval order. Used in unit tests so the cross-encoder is not loaded."""
    del query
    k = top_k or 5
    out: list[dict] = []
    for candidate in candidates[:k]:
        item = dict(candidate)
        item.setdefault("rerank_score", float(candidate.get("score", 0.0)))
        out.append(item)
    return out


def normalize_domain(domain: str | None) -> str | None:
    """Map a caller domain to one allowed host, or None if empty / unknown."""
    if not domain or not str(domain).strip():
        return None
    raw = str(domain).lower().strip()
    for allowed in ALLOWED_DOMAINS:
        if raw == allowed or raw in allowed or allowed in raw:
            return allowed
    return None


class PolicyTools:
    """Stateful tool implementations. Citation indices are assigned in first-seen order."""

    def __init__(
        self,
        retriever: HybridRetriever,
        rerank_fn: RerankFn | None = None,
        snippet_chars: int = SNIPPET_CHARS,
    ):
        self.retriever = retriever
        self.rerank_fn = rerank_fn or rerank
        self.snippet_chars = snippet_chars
        self.chunks: list[dict] = []
        self._index_by_id: dict[str, int] = {}
        self.best_score: float = 0.0

    def _register(self, chunk: dict) -> int:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in self._index_by_id:
            self.chunks.append(chunk)
            self._index_by_id[chunk_id] = len(self.chunks)
        score = float(chunk.get("rerank_score", chunk.get("score", 0.0)))
        if score > self.best_score:
            self.best_score = score
        return self._index_by_id[chunk_id]

    def search_policy(self, query: str, domain: str | None = None) -> dict:
        """Hybrid retrieve + rerank. Optional domain filter for comparison / two-topic questions."""
        if not query or not str(query).strip():
            return {"error": "query is required"}

        resolved = None
        if domain not in (None, ""):
            resolved = normalize_domain(domain)
            if resolved is None:
                return {
                    "error": (
                        f"Unknown domain '{domain}'. "
                        f"Allowed: {', '.join(ALLOWED_DOMAINS)}"
                    )
                }

        candidates = self.retriever.retrieve(str(query).strip(), mode="hybrid")
        if resolved:
            candidates = [
                c for c in candidates if resolved in (c.get("domain") or "").lower()
            ]

        top = self.rerank_fn(str(query), candidates) if candidates else []
        results = []
        for chunk in top:
            citation_index = self._register(chunk)
            text = chunk.get("text") or ""
            results.append(
                {
                    "citation_index": citation_index,
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk.get("title"),
                    "url": chunk.get("url"),
                    "domain": chunk.get("domain"),
                    "score": float(chunk.get("rerank_score", chunk.get("score", 0.0))),
                    "snippet": text[: self.snippet_chars],
                }
            )
        return {"results": results}

    def read_chunk(self, chunk_id: str) -> dict:
        """Return the full text of one corpus chunk."""
        if not chunk_id:
            return {"error": "chunk_id is required"}

        if chunk_id in self._index_by_id:
            chunk = self.chunks[self._index_by_id[chunk_id] - 1]
        else:
            getter = getattr(self.retriever, "get_chunk", None)
            chunk = getter(chunk_id) if getter else None
            if chunk is None:
                return {
                    "error": (
                        f"Unknown chunk_id: {chunk_id}. "
                        "Search first, then pass a chunk_id from the results."
                    )
                }
            self._register(chunk)

        return {
            "citation_index": self._index_by_id[chunk_id],
            "chunk_id": chunk_id,
            "title": chunk.get("title"),
            "url": chunk.get("url"),
            "domain": chunk.get("domain"),
            "text": chunk.get("text") or "",
        }

    def execute(self, name: str, args: dict | None) -> dict:
        payload = dict(args or {})
        if name == "search_policy":
            return self.search_policy(
                query=payload.get("query", ""),
                domain=payload.get("domain"),
            )
        if name == "read_chunk":
            return self.read_chunk(chunk_id=payload.get("chunk_id", ""))
        return {"error": f"Unknown tool: {name}"}

    def citation_map(self) -> str:
        if not self.chunks:
            return "(no chunks retrieved)"
        lines = []
        for i, chunk in enumerate(self.chunks, start=1):
            lines.append(
                f"[{i}] {chunk.get('title', 'Untitled')} — "
                f"{chunk.get('url', '')} ({chunk.get('domain', '')})"
            )
        return "\n".join(lines)
