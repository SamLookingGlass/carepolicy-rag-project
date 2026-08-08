"""Turn text into embeddings — vectors of numbers where similar meanings sit close together.

Provider order: OpenAI, then Azure, then a small local MiniLM model so the
project still works with no API key.
"""

from __future__ import annotations

from functools import lru_cache

from src.llm_provider import active_provider, get_embedding_model


@lru_cache
def _local_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    if model:
        return model.embed_documents(texts)

    encoder = _local_model()
    vectors = encoder.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    if model:
        return model.embed_query(query)

    encoder = _local_model()
    vector = encoder.encode([query], normalize_embeddings=True)[0]
    return vector.tolist()


def embedding_dimension() -> int:
    if active_provider() == "local":
        return 384

    model = get_embedding_model()
    if model:
        probe = model.embed_query("dimension probe")
        return len(probe)
    return 384
