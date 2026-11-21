#!/usr/bin/env python3
"""Run all event scrapers and save to events.db."""
from paths import ENV_FILE
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    pass

from scraper.db import init_db, log_scrape_run
from scraper.sources import luma, luma_sf, luma_my_events, luma_keyword_search, cerebral_valley
import sync_calendars

SOURCES = [
    ("luma", luma.scrape),
    ("luma:sf", luma_sf.scrape),
    ("luma:my-events", luma_my_events.scrape),
    ("luma:search", luma_keyword_search.scrape),
    ("cerebral_valley", cerebral_valley.scrape),
]


def run_source(name, scrape_fn):
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    error = None
    count = 0
    try:
        count = scrape_fn()
    except Exception as e:
        error = str(e)
        print(f"[{name}] FAILED: {e}", file=sys.stderr)
    duration_ms = int((time.monotonic() - t0) * 1000)
    log_scrape_run(name, started_at, duration_ms, count, error)
    return name, count, duration_ms, error


def main():
    init_db()

    try:
        sync_calendars.main()
    except Exception as e:
        print(f"[sync_calendars] FAILED: {e}", file=sys.stderr)

    total = 0
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        futures = {pool.submit(run_source, name, fn): name for name, fn in SOURCES}
        for fut in as_completed(futures):
            name, count, duration_ms, error = fut.result()
            status = "FAILED" if error else f"{count} events upserted"
            print(f"[{name}] {status}  ({duration_ms / 1000:.1f}s)")
            total += count

    print(f"\nDone. Total upserted: {total}")


if __name__ == "__main__":
    main()
