"""Remove live-scraped or offline seed artifacts from the corpus directories."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.config import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

_LIVE_STEM = re.compile(r"^[a-f0-9]{16}$")


def purge_live_docs(raw_dir: Path = RAW_DIR) -> int:
    """Delete hash-named live HTML/meta files; keep seed_* documents."""
    removed = 0
    if not raw_dir.exists():
        return 0
    for path in raw_dir.iterdir():
        stem = path.name.split(".")[0]
        if _LIVE_STEM.match(stem):
            path.unlink()
            removed += 1
    if removed:
        logger.info("Purged %d live-scraped files from %s", removed, raw_dir)
    return removed


def purge_seed_docs(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> int:
    """Delete offline seed HTML/meta and processed JSON."""
    removed = 0
    if raw_dir.exists():
        for path in raw_dir.glob("seed_*"):
            path.unlink()
            removed += 1
    if processed_dir.exists():
        for path in processed_dir.glob("seed_*.json"):
            path.unlink()
            removed += 1
    if removed:
        logger.info("Purged %d seed files from raw/processed", removed)
    return removed
