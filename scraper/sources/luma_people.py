"""Collect the people at a given Luma event — hosts and publicly-visible guests.

Luma's public `event/get` endpoint (no cookie required) returns:
  - `hosts`          — every host, with full profile + social handles
  - `featured_guests`— up to ~10 sampled attendees (the "Going" avatars on the page),
                       each with the same rich profile (LinkedIn / Twitter / bio / website)
  - `guest_count`    — total registered (not the full roster — that needs host auth)

So for any public event we can assemble a "people to talk to" list — hosts plus the
featured guests — already enriched with social handles, without authentication.
"""
import re
from collections import Counter

import requests

EVENT_GET_URL = "https://api.luma.com/event/get"

_API_HEADERS = {
    "accept": "*/*",
    "x-luma-client-type": "luma-web",
    "x-luma-timezone": "America/Los_Angeles",
    "x-luma-web-url": "https://luma.com/",
    "Referer": "https://luma.com/",
}
_PAGE_HEADERS = {"User-Agent": "Mozilla/5.0", "accept": "text/html"}

_EVT_RE = re.compile(r"evt-[A-Za-z0-9]{10,}")


def resolve_event_id(url_or_id: str) -> str:
    """Accept an event api_id (evt-...), a lu.ma short URL, or a bare slug, and
    return the evt- api_id. Slugs are resolved by scraping the public event page,
    which embeds its own evt- id (the most frequently occurring one)."""
    s = url_or_id.strip()
    m = _EVT_RE.fullmatch(s) or _EVT_RE.search(s) if s.startswith("evt-") else None
    if m:
        return m.group(0)

    # Extract the slug from a lu.ma / luma.com URL, or take the bare token.
    slug = s
    url_match = re.search(r"(?:lu\.ma|luma\.com)/([A-Za-z0-9\-]+)", s)
    if url_match:
        slug = url_match.group(1)

    resp = requests.get(f"https://lu.ma/{slug}", headers=_PAGE_HEADERS, timeout=20)
    resp.raise_for_status()
    ids = _EVT_RE.findall(resp.text)
    if not ids:
        raise ValueError(f"Could not find an event id on lu.ma/{slug}")
    return Counter(ids).most_common(1)[0][0]


def fetch_event(event_api_id: str) -> dict:
    resp = requests.get(EVENT_GET_URL, params={"event_api_id": event_api_id},
                        headers=_API_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _person(raw: dict, role: str) -> dict:
    return {
        "person_api_id": raw.get("api_id"),
        "role": role,
        "name": raw.get("name"),
        "first_name": raw.get("first_name"),
        "last_name": raw.get("last_name"),
        "bio_short": raw.get("bio_short") or "",
        "avatar_url": raw.get("avatar_url"),
        "username": raw.get("username"),
        "website": raw.get("website"),
        "twitter_handle": raw.get("twitter_handle"),
        "linkedin_handle": raw.get("linkedin_handle"),
        "instagram_handle": raw.get("instagram_handle"),
        "is_verified": 1 if raw.get("is_verified") else 0,
        "last_online_at": raw.get("last_online_at"),
        "raw": raw,
    }


def extract_people(data: dict) -> list[dict]:
    """Pull hosts + featured guests out of an event/get response, de-duplicated by
    person api_id (a host who is also a featured guest is kept once, as host)."""
    people: dict[str, dict] = {}
    for host in data.get("hosts") or []:
        pid = host.get("api_id")
        if pid:
            people[pid] = _person(host, "host")
    for guest in data.get("featured_guests") or []:
        pid = guest.get("api_id")
        if pid and pid not in people:
            people[pid] = _person(guest, "guest")
    return list(people.values())


def collect(url_or_id: str) -> dict:
    """Resolve, fetch, normalize, and persist the people at one event.
    Returns a summary dict: {event_api_id, event_name, event_url, guest_count, people}."""
    from scraper.db import init_db, upsert_person

    init_db()
    event_api_id = resolve_event_id(url_or_id)
    data = fetch_event(event_api_id)
    event = data.get("event", {})
    event_name = event.get("name", "")
    event_url = f"https://lu.ma/{event.get('url') or event_api_id}"

    people = extract_people(data)
    for p in people:
        if not p["person_api_id"]:
            continue
        upsert_person({
            "event_api_id": event_api_id,
            "event_url": event_url,
            "event_name": event_name,
            "raw_json": __import__("json").dumps(p.pop("raw")),
            **p,
        })

    return {
        "event_api_id": event_api_id,
        "event_name": event_name,
        "event_url": event_url,
        "guest_count": data.get("guest_count"),
        "people": people,
    }
