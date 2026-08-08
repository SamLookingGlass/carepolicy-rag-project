"""Discover PDPA subpage links from rendered PDPC hub pages.

Writes candidates to data/sources/pdpc_discovered.json for human review;
approved entries are then merged into data/sources/urls.json by hand.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.config import SOURCES_DIR

logger = logging.getLogger(__name__)

DISCOVERED_PATH = SOURCES_DIR / "pdpc_discovered.json"

PDPC_DOMAIN = "www.pdpc.gov.sg"

ALLOW_PREFIXES = (
    "/about/the-legislation/",
    "/overview-of-pdpa/",
    "/help-and-resources/",
    "/organisations/",
)

EXCLUDE_SUBSTRINGS = (
    "/contact",
    "/news",
    "/media",
    "login",
    "facebook",
    "linkedin",
    "whatsapp",
    "/e-services/",  # transactional portals, no policy text
)

MAX_DISCOVERIES = 10


def extract_candidates(html: str, base_url: str, known_urls: set[str]) -> list[str]:
    """Extract same-domain PDPA content links not already in the catalog."""
    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = urljoin(base_url, anchor["href"]).split("#")[0].split("?")[0].rstrip("/")
        parsed = urlparse(href)
        if parsed.netloc != PDPC_DOMAIN:
            continue
        if not any(parsed.path.startswith(p) for p in ALLOW_PREFIXES):
            continue
        if any(s in href.lower() for s in EXCLUDE_SUBSTRINGS):
            continue
        if href.lower().endswith(".pdf"):
            continue
        if href in seen or href in known_urls:
            continue
        seen.add(href)
        out.append(href)

    return out[:MAX_DISCOVERIES]


def write_discoveries(urls: list[str], text_lengths: dict[str, int], path: Path = DISCOVERED_PATH) -> Path:
    entries = [
        {
            "url": url,
            "source": "pdpc",
            "category": "compliance",
            "topic": url.rsplit("/", 1)[-1],
            "fetch": "browser",
            "rendered_text_length": text_lengths.get(url, 0),
        }
        for url in urls
    ]
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    logger.info("Wrote %d discovered PDPC URLs to %s", len(entries), path)
    return path
