"""Tests for CarePolicy RAG."""

from src.generation.citations import build_citation_objects, format_context, verify_citations
from src.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_merges_results():
    dense = [{"chunk_id": "a", "text": "a", "score": 0.9}]
    bm25 = [{"chunk_id": "b", "text": "b", "score": 1.2}]
    fused = reciprocal_rank_fusion([dense, bm25])
    ids = [r["chunk_id"] for r in fused]
    assert "a" in ids and "b" in ids


def test_citation_verification_valid():
    chunks = [{"chunk_id": "c1", "text": "CHAS provides subsidies.", "title": "CHAS", "url": "http://x", "domain": "x"}]
    answer = "CHAS provides subsidies for citizens [1]."
    result = verify_citations(answer, chunks)
    assert result["valid"] is True
    assert result["citation_accuracy"] == 1.0


def test_citation_verification_missing():
    chunks = [{"chunk_id": "c1", "text": "text", "title": "t", "url": "http://x", "domain": "x"}]
    answer = "Some claim without citation."
    result = verify_citations(answer, chunks)
    assert result["valid"] is False


def test_build_citations():
    chunks = [
        {"chunk_id": "c1", "text": "MediShield Life covers hospital bills.", "title": "MSL", "url": "http://a", "domain": "a"},
        {"chunk_id": "c2", "text": "MediSave helps pay premiums.", "title": "MS", "url": "http://b", "domain": "b"},
    ]
    answer = "MediShield covers hospitals [1] and MediSave pays premiums [2]."
    citations = build_citation_objects(answer, chunks)
    assert len(citations) == 2
    assert citations[0]["index"] == 1


def test_format_context():
    chunks = [{"title": "T", "url": "http://x", "text": "Body text"}]
    ctx = format_context(chunks)
    assert "[1]" in ctx
    assert "Body text" in ctx
