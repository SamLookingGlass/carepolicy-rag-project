"""FastAPI app exposing the pipeline: GET /health (open) and POST /query (API key + rate limit)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.config import get_settings
from src.llm_provider import active_provider
from src.pipeline import RAGPipeline

app = FastAPI(
    title="CarePolicy RAG",
    description="Healthcare policy RAG with hybrid retrieval and verified citations",
    version="0.1.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_pipeline: RAGPipeline | None = None
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def verify_api_key(api_key: str | None = Security(api_key_header)) -> None:
    settings = get_settings()
    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def make_rate_limiter(limit: int = 30, window: int = 60):
    """Build a rate-limit dependency with fixed limit/window baked in via closure.

    limit/window must NOT be plain function parameters on the dependency itself —
    FastAPI would expose them as client-controllable query params (e.g.
    ?limit=999999), letting any caller bypass their own rate limit.
    """

    def rate_limit(api_key: str | None = Security(api_key_header)) -> None:
        key = api_key or "anonymous"
        now = time.time()
        timestamps = _rate_limit_store[key]
        _rate_limit_store[key] = [t for t in timestamps if now - t < window]
        if len(_rate_limit_store[key]) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        _rate_limit_store[key].append(now)

    return rate_limit


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    skip_rerank: bool = False


class CitationResponse(BaseModel):
    index: int
    chunk_id: str | None = None
    title: str | None = None
    url: str | None = None
    domain: str | None = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    confidence: float
    latency_ms: float
    retrieval_mode: str
    chunks_used: int
    refused: bool
    citation_verification: dict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "carepolicy-rag",
        "llm_provider": active_provider(),
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key), Depends(make_rate_limiter(limit=30, window=60))],
)
def query(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    result = pipeline.query(
        question=request.question,
        retrieval_mode=request.retrieval_mode,
        skip_rerank=request.skip_rerank,
    )
    return QueryResponse(
        answer=result.answer,
        citations=[CitationResponse(**c) for c in result.citations],
        confidence=result.confidence,
        latency_ms=round(result.latency_ms, 2),
        retrieval_mode=result.retrieval_mode,
        chunks_used=result.chunks_used,
        refused=result.refused,
        citation_verification=result.citation_verification,
    )


@app.get("/")
def root():
    return {
        "service": "CarePolicy RAG",
        "docs": "/docs",
        "endpoints": ["/health", "/query"],
    }
