import subprocess
import json
import requests
from datetime import datetime, timezone

BASE_URL = "https://api.cerebralvalley.ai/v1/public/event/pull"


def _current_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fetch_events() -> list[dict]:
    headers = {
        "accept": "*/*",
        "origin": "https://cerebralvalley.ai",
        "referer": "https://cerebralvalley.ai/",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    }
    params = {
        "featured": "true",
        "approved": "true",
        "startDateTime": _current_utc(),
    }
    resp = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("events", [])
    return [e for e in events if "San Francisco" in (e.get("location") or "")]


def normalize(event: dict) -> tuple[str, dict, dict]:
    external_id = event.get("id") or event.get("url")

    cv = event.get("CVEvent") or {}
    venue_obj = event.get("venue") or {}
    venue = venue_obj.get("name") if isinstance(venue_obj, dict) else str(venue_obj)

    def to_utc(dt: str | None) -> str | None:
        if not dt:
            return None
        return dt if dt.endswith("Z") or "+" in dt else dt + "Z"

    fields = {
        "name": event.get("name", ""),
        "description": event.get("descriptionSummary") or event.get("description"),
        "start_datetime": to_utc(event.get("startDateTime")),
        "end_datetime": to_utc(event.get("endDateTime")),
        "url": event.get("url"),
        "location": event.get("location"),
        "venue": venue,
        "event_type": event.get("type"),
        "image_url": event.get("imageUrl"),
        "registration_status": None,
    }
    return str(external_id), fields, event


def scrape() -> int:
    from scraper.db import upsert_event
    events = fetch_events()
    count = 0
    for event in events:
        try:
            external_id, fields, raw = normalize(event)
            if not external_id:
                continue
            upsert_event("cerebral_valley", external_id, fields, raw)
            count += 1
        except Exception as e:
            print(f"[cerebral_valley] error on event: {e}")
    return count
