# URL Source Catalog

`urls.json` is the single source of truth for live web scraping. Each entry:

```json
{
  "url": "https://www.moh.gov.sg/managing-expenses/schemes-and-subsidies/chas",
  "source": "moh",           // moh | healthhub | hpb | pdpc
  "category": "subsidies",   // subsidies | screening | prevention | policy | compliance
  "topic": "moh-chas",       // short unique slug, used for readability in reports
  "fetch": "http"            // optional: "http" (default) or "browser" (headless Chromium)
}
```

## Workflow

```powershell
# 1. Validate all URLs (writes url_report.json, fails if too few OK)
python scripts/check_urls.py

# 2. Scrape validated pages into data/raw/
python scripts/scrape_corpus.py --purge-live

# 3. Re-index (stop uvicorn first — local Qdrant holds a file lock)
Remove-Item -Recurse -Force data\index\qdrant
python scripts/ingest_all.py --skip-fetch
```

## Adding URLs

1. Add the entry to `urls.json`
2. Run `python scripts/check_urls.py` — the page must return HTTP 200 with >400 chars of text
3. Re-scrape and re-index

## Browser fetch (JS-rendered pages)

Entries with `"fetch": "browser"` are rendered with headless Chromium via Playwright.
Install once:

```powershell
pip install -e ".[browser]"
playwright install chromium
```

To discover new PDPA subpages linked from the PDPC catalog pages:

```powershell
python scripts/scrape_corpus.py --discover-pdpc
```

This writes candidates (with rendered text lengths) to `pdpc_discovered.json`.
Review them, then merge approved entries into `urls.json` with `"fetch": "browser"`.

## Notes

- Government sites restructure URLs frequently (HealthHub moved `/programmes/61/chas`
  to `/support-and-tools/costs-and-financing/chas`). Run `check_urls.py` monthly
  or before any demo, and prune/replace dead entries.
- **PDPC pages** (`pdpc.gov.sg`) are Next.js client-rendered — plain httpx returns shell
  HTML only (~100–250 chars). They are marked `"fetch": "browser"` in the catalog and
  scraped with Playwright; `check_urls.py` also routes them through the browser so the
  400-char health gate applies to rendered text. Eight PDPA pages are indexed, and the
  golden set includes answerable compliance questions (consent, DPO, enforcement).
- `url_report.json` is generated — do not edit by hand.
- Seed documents (`src/ingestion/seed_loader.py`) remain for **offline dev only** via `--seed-only`; the default index is live-scraped content only.

Last full validation: 2026-07-04.
