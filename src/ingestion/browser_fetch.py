"""Headless-browser fetching for JS-rendered pages (e.g. pdpc.gov.sg).

Requires the optional browser extra:
    pip install -e ".[browser]"
    playwright install chromium
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

USER_AGENT = "CarePolicyRAG/0.1 (portfolio project; educational use)"

_INSTALL_HINT = (
    'Playwright is not installed. Run: pip install -e ".[browser]" '
    "then: playwright install chromium"
)


class BrowserFetcher:
    """Reusable headless Chromium session. Use as a context manager."""

    def __init__(self, timeout_ms: int = 45000):
        self.timeout_ms = timeout_ms
        self._pw = None
        self._browser = None
        self._page = None

    def __enter__(self) -> "BrowserFetcher":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page(user_agent=USER_AGENT)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()
        self._pw = self._browser = self._page = None

    def fetch(self, url: str) -> str | None:
        """Navigate and return fully rendered HTML, or None on failure."""
        try:
            self._page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
            try:
                self._page.wait_for_selector("main", state="visible", timeout=15000)
            except Exception:
                logger.debug("<main> not visible on %s; continuing", url)
            # Next.js hydration can lag behind DOM-ready
            self._page.wait_for_timeout(2000)
            return self._page.content()
        except Exception as exc:
            logger.warning("Browser fetch failed for %s: %s", url, exc)
            return None


def is_playwright_available() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False
