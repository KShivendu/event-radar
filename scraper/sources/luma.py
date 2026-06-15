import json
import os
from pathlib import Path

import requests

CALENDARS_FILE = Path(__file__).parent.parent.parent / "luma_calendars.json"

# Fallback hardcoded list — used when luma_calendars.json is absent (stale cookie, first run)
FALLBACK_CALENDARS = [
    {"id": "cal-JTdFQadEz0AOxyV", "slug": "genai-sf", "name": "Bond AI - San Francisco and Bay Area"},
    {"id": "cal-F4B0wdJsEABsvTu", "slug": "ai-sf",    "name": "AI SF"},
]


def load_calendars() -> list[dict]:
    if CALENDARS_FILE.exists():
        return json.loads(CALENDARS_FILE.read_text())
    print("[luma] luma_calendars.json not found, using fallback list")
    return FALLBACK_CALENDARS


def fetch_calendar(calendar_id: str, slug: str) -> list[dict]:
    cookie = os.environ.get("LUMA_COOKIE", "")
    url = f"https://api.luma.com/calendar/get-items?calendar_api_id={calendar_id}&pagination_limit=100&period=future"
    headers = {
        "accept": "*/*",
        "accept-language": "en",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "x-luma-client-type": "luma-web",
        "x-luma-timezone": "America/Los_Angeles",
        "x-luma-web-url": f"https://luma.com/{slug}",
        "Referer": "https://luma.com/",
    }
    if cookie:
        headers["cookie"] = cookie

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("entries", [])


def normalize(entry: dict) -> tuple[str, dict, dict]:
    event = entry.get("event", {})
    external_id = event.get("api_id") or entry.get("api_id")

    geo = event.get("geo_address_info") or {}
    location = geo.get("full_address") or geo.get("city") or ""
    venue = event.get("location", {}).get("name") if isinstance(event.get("location"), dict) else ""
    city = geo.get("city") or ""
    location_type = event.get("location_type") or ""  # "offline" | "online" | ""

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
    calendars = load_calendars()
    total = 0
    for cal in calendars:
        source = f"luma:{cal['slug']}" if cal.get("slug") else f"luma:{cal['id']}"
        try:
            entries = fetch_calendar(cal["id"], cal.get("slug", ""))
            count = 0
            for entry in entries:
                try:
                    external_id, fields, raw = normalize(entry)
                    if not external_id:
                        continue
                    upsert_event(source, external_id, fields, raw)
                    count += 1
                except Exception as e:
                    print(f"  [{source}] error on entry: {e}")
            print(f"  [{source}] {count} events upserted")
            total += count
        except Exception as e:
            print(f"  [{source}] FAILED: {e}")
    return total
