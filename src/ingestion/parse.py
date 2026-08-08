"""Turn raw scraped HTML into clean text.

Strips scripts, menus, headers, and footers, then keeps headings, paragraphs,
and list items as lightly formatted markdown-style text ready for chunking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from src.config import PROCESSED_DIR, RAW_DIR


@dataclass
class ParsedDocument:
    doc_id: str
    url: str
    source: str
    category: str
    title: str
    text: str
    domain: str


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Extract readable text from an HTML page, preferring the main content area and skipping fragments under 20 chars."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|article|main", re.I))
    root = main if main else soup.body if soup.body else soup

    parts: list[str] = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = _clean_text(element.get_text(" ", strip=True))
        if len(text) < 20:
            continue
        if element.name.startswith("h"):
            level = element.name[1]
            parts.append(f"{'#' * int(level)} {text}")
        elif element.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)

    if not parts:
        parts = [_clean_text(root.get_text(" ", strip=True))]

    return "\n\n".join(parts)


def load_metadata(raw_dir: Path, doc_id: str) -> dict:
    meta_path = raw_dir / f"{doc_id}.meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def parse_document(html_path: Path, raw_dir: Path = RAW_DIR) -> ParsedDocument | None:
    doc_id = html_path.stem
    meta = load_metadata(raw_dir, doc_id)
    if not meta:
        return None

    html = html_path.read_text(encoding="utf-8", errors="replace")
    text = html_to_text(html)
    if len(text) < 100:
        return None

    return ParsedDocument(
        doc_id=doc_id,
        url=meta.get("url", ""),
        source=meta.get("source", "unknown"),
        category=meta.get("category", "general"),
        title=meta.get("title", doc_id),
        text=text,
        domain=meta.get("domain", ""),
    )


def parse_all(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> list[ParsedDocument]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    documents: list[ParsedDocument] = []

    for html_path in sorted(raw_dir.glob("*.html")):
        parsed = parse_document(html_path, raw_dir)
        if parsed:
            out_path = processed_dir / f"{parsed.doc_id}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "doc_id": parsed.doc_id,
                        "url": parsed.url,
                        "source": parsed.source,
                        "category": parsed.category,
                        "title": parsed.title,
                        "text": parsed.text,
                        "domain": parsed.domain,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            documents.append(parsed)

    index_path = processed_dir / "index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": d.doc_id,
                    "url": d.url,
                    "source": d.source,
                    "category": d.category,
                    "title": d.title,
                    "domain": d.domain,
                    "char_count": len(d.text),
                }
                for d in documents
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return documents


if __name__ == "__main__":
    docs = parse_all()
    print(f"Parsed {len(docs)} documents")
