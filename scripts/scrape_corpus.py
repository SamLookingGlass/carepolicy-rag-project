#!/usr/bin/env python3
"""Scrape live pages from the URL catalog into data/raw/.

Usage:
    python scripts/scrape_corpus.py [--purge-live] [--min-success 20] [--urls PATH]

--purge-live removes previously scraped live HTML (16-char hash filenames)
before fetching. Seed documents are not kept in the default live corpus; run
ingest with --seed-only for offline dev.

After scraping, re-index with:
    Remove-Item -Recurse -Force data\\index\\qdrant
    python scripts/ingest_all.py --skip-fetch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR
from src.ingestion.corpus_cleanup import purge_live_docs
from src.ingestion.fetch import fetch_all, load_url_catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def discover_pdpc(entries: list[dict]) -> None:
    """Render PDPC catalog pages and write linked subpage candidates for review."""
    from src.ingestion.browser_fetch import BrowserFetcher, is_playwright_available
    from src.ingestion.pdpc_discover import DISCOVERED_PATH, extract_candidates, write_discoveries

    if not is_playwright_available():
        logger.error('Playwright required for --discover-pdpc. Run: pip install -e ".[browser]" && playwright install chromium')
        sys.exit(1)

    pdpc_entries = [e for e in entries if e["source"] == "pdpc"]
    if not pdpc_entries:
        logger.error("No pdpc entries in catalog")
        sys.exit(1)

    known = {e["url"] for e in entries}
    candidates: list[str] = []
    text_lengths: dict[str, int] = {}

    with BrowserFetcher() as fetcher:
        for entry in pdpc_entries:
            html = fetcher.fetch(entry["url"])
            if not html:
                continue
            for url in extract_candidates(html, entry["url"], known):
                if url not in candidates:
                    candidates.append(url)

        # Probe rendered text length so reviewers can filter thin pages
        from src.ingestion.parse import html_to_text

        for url in candidates:
            html = fetcher.fetch(url)
            if html:
                text_lengths[url] = len(html_to_text(html))

    write_discoveries(candidates, text_lengths)
    logger.info("Review %s, then merge approved entries into data/sources/urls.json", DISCOVERED_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape live corpus pages")
    parser.add_argument("--purge-live", action="store_true", help="Remove old live docs first (keeps seeds)")
    parser.add_argument("--min-success", type=int, default=20, help="Fail if fewer pages fetched")
    parser.add_argument("--urls", type=Path, default=None, help="Override catalog path")
    parser.add_argument("--discover-pdpc", action="store_true", help="Discover PDPC subpage links into pdpc_discovered.json")
    args = parser.parse_args()

    entries = load_url_catalog(args.urls)
    if not entries:
        logger.error("No URLs in catalog — populate data/sources/urls.json first")
        sys.exit(1)

    if args.discover_pdpc:
        discover_pdpc(entries)
        return

    if args.purge_live:
        purge_live_docs()

    logger.info("Fetching %d catalog URLs...", len(entries))
    documents = fetch_all(entries)

    logger.info("Fetched %d/%d pages", len(documents), len(entries))
    if len(documents) < args.min_success:
        logger.error(
            "Only %d pages fetched (minimum %d). Run scripts/check_urls.py to diagnose.",
            len(documents),
            args.min_success,
        )
        sys.exit(1)

    logger.info("Done. Next: Remove-Item -Recurse -Force data\\index\\qdrant ; python scripts/ingest_all.py --skip-fetch")


if __name__ == "__main__":
    main()
