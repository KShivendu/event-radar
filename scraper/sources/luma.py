import os
import requests

CALENDARS = [
    {"id": "cal-JTdFQadEz0AOxyV", "slug": "genai-sf", "source": "luma:genai-sf"},
    {"id": "cal-F4B0wdJsEABsvTu", "slug": "ai-sf",    "source": "luma:ai-sf"},
]


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
    }
    return external_id, fields, entry


def scrape() -> int:
    from scraper.db import upsert_event
    total = 0
    for cal in CALENDARS:
        try:
            entries = fetch_calendar(cal["id"], cal["slug"])
            count = 0
            for entry in entries:
                try:
                    external_id, fields, raw = normalize(entry)
                    if not external_id:
                        continue
                    upsert_event(cal["source"], external_id, fields, raw)
                    count += 1
                except Exception as e:
                    print(f"[{cal['source']}] error on entry: {e}")
            print(f"  [{cal['source']}] {count} events upserted")
            total += count
        except Exception as e:
            print(f"  [{cal['source']}] FAILED: {e}")
    return total
