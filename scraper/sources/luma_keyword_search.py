import json
import os
import time
from pathlib import Path

import requests

SOURCE = "luma:search"
KEYWORDS_FILE = Path(__file__).parent.parent.parent / "search_keywords.json"
SF_CITIES = {"San Francisco", "Oakland", "Berkeley", "San Jose", "Menlo Park",
             "Palo Alto", "Mountain View", "Sunnyvale", "Redwood City", "San Mateo"}


def load_keywords() -> list[str]:
    if not KEYWORDS_FILE.exists():
        return []
    return json.loads(KEYWORDS_FILE.read_text())


def search(query: str, cookie: str) -> list[dict]:
    headers = {
        "accept": "*/*",
        "x-luma-client-type": "luma-web",
        "x-luma-timezone": "America/Los_Angeles",
        "x-luma-web-url": "https://luma.com/home",
        "Referer": "https://luma.com/",
    }
    if cookie:
        headers["cookie"] = cookie

    resp = requests.get(
        "https://api.luma.com/search/get-results",
        params={"query": query},
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("events", [])


def is_sf(entry: dict) -> bool:
    event = entry.get("event", {})
    geo = event.get("geo_address_info") or {}
    city = geo.get("city") or ""
    location = (event.get("location") or "")
    if isinstance(location, dict):
        location = location.get("full_address") or ""
    return city in SF_CITIES or "San Francisco" in location


def normalize(entry: dict) -> tuple[str, dict, dict]:
    event = entry.get("event", {})
    external_id = event.get("api_id") or entry.get("api_id")

    geo = event.get("geo_address_info") or {}
    location = geo.get("full_address") or geo.get("city") or ""
    venue = event.get("location", {}).get("name") if isinstance(event.get("location"), dict) else ""
    city = geo.get("city") or ""
    location_type = event.get("location_type") or ""

    gi = entry.get("guest_info")
    reg_status = gi.get("approval_status") if isinstance(gi, dict) else None

    fields = {
        "name": event.get("name", ""),
        "description": event.get("description") or event.get("description_short"),
        "start_datetime": event.get("start_at"),
        "end_datetime": event.get("end_at"),
        "url": f"https://lu.ma/{event.get('url') or external_id}",
        "location": location,
        "venue": venue,
        "event_type": event.get("event_type"),
        "image_url": event.get("cover_url"),
        "registration_status": reg_status,
        "city": city,
        "location_type": location_type,
    }
    return external_id, fields, entry


def scrape() -> int:
    from scraper.db import upsert_event
    cookie = os.environ.get("LUMA_COOKIE", "")
    keywords = load_keywords()
    if not keywords:
        print(f"  [{SOURCE}] no keywords found in {KEYWORDS_FILE.name}")
        return 0

    seen_ids: set[str] = set()
    total = 0

    for keyword in keywords:
        try:
            entries = search(keyword, cookie)
            sf_entries = [e for e in entries if is_sf(e)]
            count = 0
            for entry in sf_entries:
                try:
                    external_id, fields, raw = normalize(entry)
                    if not external_id or external_id in seen_ids:
                        continue
                    seen_ids.add(external_id)
                    upsert_event(SOURCE, external_id, fields, raw)
                    count += 1
                except Exception as e:
                    print(f"  [{SOURCE}] error on entry: {e}")
            print(f"  [{SOURCE}:{keyword}] {count} SF events upserted")
            total += count
            time.sleep(0.3)  # be polite
        except Exception as e:
            print(f"  [{SOURCE}:{keyword}] FAILED: {e}")

    return total
