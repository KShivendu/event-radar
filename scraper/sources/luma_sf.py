import os
import requests

PLACE_API_ID = "discplace-BDj7GNbGlsF7Cka"
SOURCE = "luma:sf"


def fetch_events() -> list[dict]:
    cookie = os.environ.get("LUMA_COOKIE", "")
    headers = {
        "accept": "*/*",
        "x-luma-client-type": "luma-web",
        "x-luma-timezone": "America/Los_Angeles",
        "x-luma-web-url": "https://luma.com/sf",
        "Referer": "https://luma.com/",
    }
    if cookie:
        headers["cookie"] = cookie

    entries = []
    cursor = None
    while True:
        params = {"discover_place_api_id": PLACE_API_ID, "pagination_limit": 100}
        if cursor:
            params["pagination_cursor"] = cursor

        resp = requests.get(
            "https://api.luma.com/discover/get-paginated-events",
            headers=headers, params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        entries.extend(data.get("entries", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    return entries


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
    entries = fetch_events()
    count = 0
    for entry in entries:
        try:
            external_id, fields, raw = normalize(entry)
            if not external_id:
                continue
            upsert_event(SOURCE, external_id, fields, raw)
            count += 1
        except Exception as e:
            print(f"  [{SOURCE}] error on entry: {e}")
    return count
