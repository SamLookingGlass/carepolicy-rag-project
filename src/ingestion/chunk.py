"""Split parsed documents into small overlapping chunks for indexing.

Whole pages are too long to embed or feed to an LLM, so each document is cut
into ~512-token pieces. Neighbouring chunks share 64 tokens of overlap so a
sentence cut at a boundary still appears whole in at least one chunk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken

from src.config import PROCESSED_DIR, get_settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    url: str
    source: str
    category: str
    title: str
    domain: str
    chunk_index: int
    token_count: int


def _get_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _count_tokens(text: str, encoder) -> int:
    if encoder:
        return len(encoder.encode(text))
    return len(text.split())


def _windows(items: list, size: int, overlap: int):
    """Yield sliding windows of `size` items with `overlap` items shared between neighbours."""
    start = 0
    while start < len(items):
        end = min(start + size, len(items))
        yield items[start:end]
        if end >= len(items):
            break
        start += size - overlap


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    encoder = _get_encoder()

    if encoder:
        tokens = encoder.encode(text)
        return [encoder.decode(w) for w in _windows(tokens, size, overlap)]

    words = text.split()
    return [" ".join(w) for w in _windows(words, size, overlap)]


def chunk_document(doc: dict, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]:
    encoder = _get_encoder()
    text_chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for idx, text in enumerate(text_chunks):
        chunk_id = f"{doc['doc_id']}_{idx:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc["doc_id"],
                text=text,
                url=doc["url"],
                source=doc["source"],
                category=doc["category"],
                title=doc["title"],
                domain=doc["domain"],
                chunk_index=idx,
                token_count=_count_tokens(text, encoder),
            )
        )
    return chunks


def chunk_all(processed_dir: Path = PROCESSED_DIR) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for json_path in sorted(processed_dir.glob("*.json")):
        if json_path.name == "index.json":
            continue
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        all_chunks.extend(chunk_document(doc))

    chunks_path = processed_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")

    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all()
    print(f"Created {len(chunks)} chunks")
