#!/usr/bin/env python3
"""Run all event scrapers and save to events.db."""
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from scraper.db import init_db
from scraper.sources import luma, cerebral_valley

SOURCES = [
    ("luma", luma.scrape),
    ("cerebral_valley", cerebral_valley.scrape),
]


def main():
    init_db()
    total = 0
    for name, scrape_fn in SOURCES:
        try:
            count = scrape_fn()
            print(f"[{name}] {count} events upserted")
            total += count
        except Exception as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
    print(f"\nDone. Total upserted: {total}")


if __name__ == "__main__":
    main()
