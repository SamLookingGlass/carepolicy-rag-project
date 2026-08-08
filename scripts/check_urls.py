#!/usr/bin/env python3
"""Validate the live URL catalog and write data/sources/url_report.json.

Usage:
    python scripts/check_urls.py [--min-ok 10] [--urls PATH]

Exit code 1 if fewer than --min-ok URLs are healthy, so this can gate CI
or a scheduled corpus-freshness job.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import URL_REPORT_PATH
from src.ingestion.fetch import USER_AGENT, load_url_catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 400


def _row_from_html(row: dict, html: str) -> None:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    row["title"] = title_tag.get_text(strip=True)[:100] if title_tag else row.get("final_url", row["url"])
    row["text_length"] = len(soup.get_text(" ", strip=True))
    row["ok"] = row["text_length"] >= MIN_TEXT_LENGTH


def check_urls(entries: list[dict[str, str]]) -> dict:
    results: list[dict] = []
    http_entries = [e for e in entries if e.get("fetch", "http") != "browser"]
    browser_entries = [e for e in entries if e.get("fetch", "http") == "browser"]

    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=30) as client:
        for entry in http_entries:
            url = entry["url"]
            row: dict = {**entry, "fetch_mode": "http", "ok": False}
            try:
                response = client.get(url)
                row["final_url"] = str(response.url).split("?")[0].rstrip("/")
                row["status_code"] = response.status_code
                if response.status_code == 200:
                    _row_from_html(row, response.text)
            except httpx.HTTPError as exc:
                row["error"] = str(exc)
            status = "OK  " if row["ok"] else "FAIL"
            logger.info("%s %s %s", status, row.get("status_code", "---"), url)
            results.append(row)

    if browser_entries:
        from src.ingestion.browser_fetch import BrowserFetcher, is_playwright_available

        if not is_playwright_available():
            logger.warning(
                "Playwright not installed — %d browser-fetch URLs marked FAIL. "
                'Run: pip install -e ".[browser]" && playwright install chromium',
                len(browser_entries),
            )
            for entry in browser_entries:
                results.append({**entry, "fetch_mode": "browser", "ok": False, "error": "playwright not installed"})
        else:
            with BrowserFetcher() as fetcher:
                for entry in browser_entries:
                    url = entry["url"]
                    row = {**entry, "fetch_mode": "browser", "ok": False}
                    html = fetcher.fetch(url)
                    if html:
                        row["final_url"] = url
                        row["status_code"] = 200
                        _row_from_html(row, html)
                    else:
                        row["error"] = "browser fetch failed"
                    status = "OK  " if row["ok"] else "FAIL"
                    logger.info("%s %s %s (browser)", status, row.get("status_code", "---"), url)
                    results.append(row)

    ok = sum(1 for r in results if r["ok"])
    return {"total": len(results), "ok": ok, "failed": len(results) - ok, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check live URL catalog health")
    parser.add_argument("--min-ok", type=int, default=10, help="Fail if fewer URLs are healthy")
    parser.add_argument("--urls", type=Path, default=None, help="Override catalog path")
    args = parser.parse_args()

    entries = load_url_catalog(args.urls)
    if not entries:
        logger.error("No URLs in catalog")
        sys.exit(1)

    report = check_urls(entries)

    URL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    URL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report: %s", URL_REPORT_PATH)
    logger.info("Healthy: %d/%d", report["ok"], report["total"])

    if report["ok"] < args.min_ok:
        logger.error("Only %d healthy URLs (minimum %d) — fix data/sources/urls.json", report["ok"], args.min_ok)
        sys.exit(1)


if __name__ == "__main__":
    main()
