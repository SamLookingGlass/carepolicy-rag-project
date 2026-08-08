#!/usr/bin/env python3
"""Full ingestion pipeline: fetch -> parse -> chunk -> index.

Default behaviour is live-only: scraped pages from data/sources/urls.json.
Offline seed summaries are opt-in via --seed-only (for offline dev without network).
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_LIVE_DOCS = 5


def main():
    parser = argparse.ArgumentParser(description="Ingest healthcare policy corpus")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip web fetch, use existing raw data")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Offline dev only: load bundled seed documents (no live scrape)",
    )
    parser.add_argument("--reindex-only", action="store_true", help="Rebuild BM25 + Qdrant from existing chunks.jsonl")
    parser.add_argument(
        "--keep-seeds",
        action="store_true",
        help="Do not remove seed_* files from data/raw (default: purge seeds for live corpus)",
    )
    args = parser.parse_args()

    if args.reindex_only:
        from src.retrieval.bm25_index import build_and_save_bm25
        from src.retrieval.vector_store import index_chunks, load_chunks

        chunks = load_chunks()
        if not chunks:
            logger.error("No chunks found; run full ingestion first")
            sys.exit(1)
        build_and_save_bm25()
        logger.info("Built BM25 index")
        count = index_chunks(chunks)
        logger.info("Indexed %d chunks in Qdrant", count)
        return

    from src.ingestion.corpus_cleanup import purge_live_docs, purge_seed_docs

    if args.seed_only:
        purge_live_docs()
        from src.ingestion.seed_loader import load_seed_documents

        docs = load_seed_documents()
        logger.info("Loaded %d seed documents (offline mode)", len(docs))
    else:
        if not args.keep_seeds:
            purge_seed_docs()
        if not args.skip_fetch:
            from src.ingestion.fetch import fetch_all

            docs = fetch_all()
            logger.info("Fetched %d live documents", len(docs))
            if len(docs) < MIN_LIVE_DOCS:
                logger.error(
                    "Only %d live documents fetched (minimum %d). "
                    "Run scripts/check_urls.py and scripts/scrape_corpus.py, or use --seed-only for offline dev.",
                    len(docs),
                    MIN_LIVE_DOCS,
                )
                sys.exit(1)

    from src.ingestion.parse import parse_all
    from src.ingestion.chunk import chunk_all
    from src.retrieval.bm25_index import build_and_save_bm25
    from src.retrieval.vector_store import index_chunks

    parsed = parse_all()
    logger.info("Parsed %d documents", len(parsed))

    if not args.seed_only and len(parsed) < MIN_LIVE_DOCS:
        logger.error(
            "Only %d documents in corpus after parse (minimum %d). "
            "Scrape live pages first: python scripts/scrape_corpus.py --purge-live",
            len(parsed),
            MIN_LIVE_DOCS,
        )
        sys.exit(1)

    chunks = chunk_all()
    logger.info("Created %d chunks", len(chunks))

    build_and_save_bm25()
    logger.info("Built BM25 index")

    try:
        count = index_chunks(chunks)
        logger.info("Indexed %d chunks in Qdrant", count)
    except Exception as exc:
        logger.error("Qdrant indexing failed: %s", exc)
        logger.error("Stop any running uvicorn process, then retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()