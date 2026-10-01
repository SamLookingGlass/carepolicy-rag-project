"""End-to-end RAG pipeline: from user question to cited answer.

Steps, in order:
1. Rewrite the query (expand acronyms like CHAS)
2. Retrieve the ~20 most relevant chunks (hybrid keyword + semantic search)
3. Rerank them with a cross-encoder and keep the top 5
4. Refuse if even the best chunk scores below the confidence threshold
5. Generate an answer grounded in those chunks, with [1]-style citations
6. Verify every citation points at a chunk that was actually retrieved
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from src.config import get_settings
from src.generation.answer import generate_answer
from src.generation.citations import (
    build_citation_objects,
    is_refusal,
    verify_citations,
)
from src.generation.prompts import REFUSAL_MESSAGE
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.rerank import rerank

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator


@dataclass
class QueryResult:
    answer: str
    citations: list[dict]
    confidence: float
    latency_ms: float
    retrieval_mode: str
    chunks_used: int
    refused: bool
    citation_verification: dict
    retrieved_chunks: list[dict]
    steps: list[dict] = field(default_factory=list)
    model_turns: int = 0
    tool_call_count: int = 0
    intent: str = ""
    trace_text: str = ""
    tokens: int = 0
    estimated_cost_usd: float = 0.0
    retry_count: int = 0
    supported_claim_rate: float | None = None


def _configure_langsmith() -> None:
    settings = get_settings()
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


class RAGPipeline:
    """Runs the whole question-to-answer flow. Build once, then call query() per question."""

    def __init__(self, retriever: HybridRetriever | None = None):
        _configure_langsmith()
        self.settings = get_settings()
        self.retriever = retriever or HybridRetriever()

    @staticmethod
    def _refusal(start: float, retrieval_mode: str, confidence: float, chunks: list[dict]) -> QueryResult:
        return QueryResult(
            answer=REFUSAL_MESSAGE,
            citations=[],
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            retrieval_mode=retrieval_mode,
            chunks_used=0,
            refused=True,
            citation_verification={"valid": True, "citation_accuracy": 1.0},
            retrieved_chunks=chunks,
        )

    @traceable(name="carepolicy_rag_query", run_type="chain")
    def query(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        skip_rerank: bool = False,
        skip_rewrite: bool = False,
    ) -> QueryResult:
        """Answer one question, or refuse if the sources don't cover it.

        The refusal gate exists because a wrong answer about healthcare
        subsidies is worse than no answer: when retrieval confidence is low,
        the system says it doesn't know instead of guessing.
        """
        start = time.perf_counter()

        search_query = question if skip_rewrite else rewrite_query(question)
        candidates = self.retriever.retrieve(search_query, mode=retrieval_mode)
        if not candidates:
            return self._refusal(start, retrieval_mode, confidence=0.0, chunks=[])

        if skip_rerank:
            top_chunks = candidates[: self.settings.top_k_rerank]
            top_score = candidates[0].get("score", 0.0)
        else:
            top_chunks = rerank(question, candidates)
            top_score = top_chunks[0].get("rerank_score", 0.0) if top_chunks else 0.0

        # Cross-encoder scores are logits; relevant pairs are typically > 0
        if not skip_rerank and top_score < self.settings.rerank_score_threshold:
            return self._refusal(start, retrieval_mode, confidence=float(top_score), chunks=top_chunks)

        answer = generate_answer(question, top_chunks)
        citations = build_citation_objects(answer, top_chunks)
        verification = verify_citations(answer, top_chunks)
        refused = is_refusal(answer)

        confidence = float(top_score) if not skip_rerank else float(candidates[0].get("score", 0.5))
        if refused:
            confidence = min(confidence, 0.3)

        return QueryResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            retrieval_mode=retrieval_mode,
            chunks_used=len(top_chunks),
            refused=refused,
            citation_verification=verification,
            retrieved_chunks=top_chunks,
        )
