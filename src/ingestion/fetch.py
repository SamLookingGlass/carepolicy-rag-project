"""Fetch public healthcare policy pages from Singapore government sources."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.config import RAW_DIR, URL_CATALOG_PATH

logger = logging.getLogger(__name__)

USER_AGENT = "CarePolicyRAG/0.1 (portfolio project; educational use)"


def load_url_catalog(path: Path | None = None) -> list[dict[str, str]]:
    """Load the curated live URL catalog from data/sources/urls.json."""
    catalog_path = path or URL_CATALOG_PATH
    if not catalog_path.exists():
        logger.warning("URL catalog not found: %s", catalog_path)
        return []
    entries = json.loads(catalog_path.read_text(encoding="utf-8"))
    return [
        {
            "url": e["url"].rstrip("/"),
            "source": e["source"],
            "category": e["category"],
            "topic": e.get("topic", e["url"].rstrip("/").rsplit("/", 1)[-1]),
            "fetch": e.get("fetch", "http"),
        }
        for e in entries
    ]


@dataclass
class FetchedDocument:
    doc_id: str
    url: str
    source: str
    category: str
    title: str
    html: str
    fetched_at: str


def _doc_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _document_from_html(entry: dict[str, str], html: str) -> FetchedDocument:
    url = entry["url"]
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    return FetchedDocument(
        doc_id=_doc_id(url),
        url=url,
        source=entry["source"],
        category=entry["category"],
        title=title,
        html=html,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def fetch_url(client: httpx.Client, entry: dict[str, str]) -> FetchedDocument | None:
    url = entry["url"]
    try:
        response = client.get(url, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    return _document_from_html(entry, response.text)


def save_document(doc: FetchedDocument, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    html_path = raw_dir / f"{doc.doc_id}.html"
    html_path.write_text(doc.html, encoding="utf-8")

    meta_path = raw_dir / f"{doc.doc_id}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "doc_id": doc.doc_id,
                "url": doc.url,
                "source": doc.source,
                "category": doc.category,
                "title": doc.title,
                "fetched_at": doc.fetched_at,
                "domain": urlparse(doc.url).netloc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return html_path


def fetch_all(urls: list[dict[str, str]] | None = None, raw_dir: Path = RAW_DIR) -> list[FetchedDocument]:
    """Download every catalog page and save it to data/raw/.

    Plain pages go through httpx; pages marked "fetch": "browser" (the
    JavaScript-rendered PDPC site) go through headless Chromium instead.
    """
    entries = urls or load_url_catalog()
    http_entries = [e for e in entries if e.get("fetch", "http") != "browser"]
    browser_entries = [e for e in entries if e.get("fetch", "http") == "browser"]
    documents: list[FetchedDocument] = []

    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for entry in http_entries:
            doc = fetch_url(client, entry)
            if doc:
                save_document(doc, raw_dir)
                documents.append(doc)
                logger.info("Fetched: %s", doc.title[:80])

    if browser_entries:
        from src.ingestion.browser_fetch import BrowserFetcher, is_playwright_available

        if not is_playwright_available():
            logger.warning(
                "Skipping %d browser-fetch URLs — Playwright not installed. "
                'Run: pip install -e ".[browser]" && playwright install chromium',
                len(browser_entries),
            )
        else:
            with BrowserFetcher() as fetcher:
                for entry in browser_entries:
                    html = fetcher.fetch(entry["url"])
                    if html:
                        doc = _document_from_html(entry, html)
                        save_document(doc, raw_dir)
                        documents.append(doc)
                        logger.info("Fetched (browser): %s", doc.title[:80])

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(d) for d in documents], indent=2, default=str),
        encoding="utf-8",
    )
    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = fetch_all()
    print(f"Fetched {len(docs)} documents")
