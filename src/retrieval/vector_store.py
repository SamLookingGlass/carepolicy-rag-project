"""Qdrant vector store: index and search chunks by meaning.

Each chunk is stored as an embedding — a list of numbers where texts with
similar meaning end up close together. Searching means embedding the question
and asking Qdrant for the nearest stored vectors, so "help paying the doctor"
can find a page about subsidies even with no words in common.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import INDEX_DIR, PROCESSED_DIR, PROJECT_ROOT, get_settings
from src.ingestion.chunk import Chunk
from src.retrieval.embeddings import embed_texts, embedding_dimension

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_mode == "server":
        return QdrantClient(url=settings.qdrant_url)
    path = PROJECT_ROOT / settings.qdrant_path
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def ensure_collection(client: QdrantClient | None = None) -> None:
    settings = get_settings()
    client = client or get_qdrant_client()
    dim = embedding_dimension()

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s (dim=%d)", settings.qdrant_collection, dim)


def load_chunks(chunks_path: Path | None = None) -> list[Chunk]:
    path = chunks_path or (PROCESSED_DIR / "chunks.jsonl")
    chunks: list[Chunk] = []
    if not path.exists():
        return chunks
    with path.open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))
    return chunks


def index_chunks(chunks: list[Chunk] | None = None, batch_size: int = 32) -> int:
    """Embed every chunk and upsert it into the Qdrant collection. Returns the count indexed."""
    settings = get_settings()
    client = get_qdrant_client()
    ensure_collection(client)

    chunks = chunks or load_chunks()
    if not chunks:
        logger.warning("No chunks to index")
        return 0

    points: list[PointStruct] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        for chunk, vector in zip(batch, vectors):
            payload = asdict(chunk)
            point_id = abs(hash(chunk.chunk_id)) % (2**63 - 1)
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("Indexed %d chunks into Qdrant", len(points))
    return len(points)


def dense_search(query: str, top_k: int | None = None) -> list[dict]:
    """Embed the query and return the top_k chunks with the closest meaning."""
    settings = get_settings()
    client = get_qdrant_client()
    from src.retrieval.embeddings import embed_query

    vector = embed_query(query)
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        limit=top_k or settings.top_k_retrieve,
    )
    return [
        {
            "chunk_id": hit.payload["chunk_id"],
            "text": hit.payload["text"],
            "score": float(hit.score),
            "url": hit.payload["url"],
            "source": hit.payload["source"],
            "category": hit.payload["category"],
            "title": hit.payload["title"],
            "domain": hit.payload["domain"],
            "doc_id": hit.payload["doc_id"],
            "retriever": "dense",
        }
        for hit in response.points
    ]
