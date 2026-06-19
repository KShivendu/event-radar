"""Collect the people at a given Luma event.

Two collection modes:

1. Public (no cookie) — `event/get` returns hosts + ~10 featured guests.
2. Full guest list (cookie + ticket_key) — `event/get-guest-list` paginates
   through all attendees. Only works for events you're registered for.
"""
import re
import time
from collections import Counter

import requests

EVENT_GET_URL = "https://api.luma.com/event/get"
GUEST_LIST_URL = "https://api.luma.com/event/get-guest-list"

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


def fetch_guest_list(event_api_id: str, ticket_key: str, cookie: str,
                     page_size: int = 100, sleep_s: float = 1.5) -> list[dict]:
    """Paginate through the full attendee list for an event you're registered for.
    Requires a valid LUMA_COOKIE and the ticket_key from your registration."""
    import json
    headers = {
        **_API_HEADERS,
        "cookie": cookie,
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "origin": "https://luma.com",
        "referer": "https://luma.com/",
        "x-luma-timezone": "America/Los_Angeles",
    }
    attendees = []
    cursor = None
    page = 0
    while True:
        params = {"event_api_id": event_api_id, "pagination_limit": page_size, "ticket_key": ticket_key}
        if cursor:
            params["pagination_cursor"] = cursor
        resp = requests.get(GUEST_LIST_URL, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("entries", [])
        for entry in entries:
            user = entry.get("user", {})
            if user.get("api_id"):
                attendees.append(_person(user, "attendee"))
        page += 1
        print(f"  [guest-list] page {page}: {len(entries)} entries (total so far: {len(attendees)})")
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        time.sleep(sleep_s)
    return attendees


def collect_attendees(url_or_id: str, cookie: str, ticket_key: str | None = None) -> dict:
    """Fetch the full attendee list for an event you're registered for.
    If ticket_key is not provided, it is looked up from the events DB."""
    import json
    from scraper.db import init_db, upsert_person, get_conn

    init_db()
    event_api_id = resolve_event_id(url_or_id)

    if ticket_key is None:
        conn = get_conn()
        row = conn.execute(
            "SELECT raw_json FROM events WHERE external_id = ? AND source LIKE 'luma:%'",
            (event_api_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Event {event_api_id} not in DB — run scraper first or pass ticket_key explicitly.")
        raw = json.loads(row[0])
        ticket_key = (raw.get("guest_info") or {}).get("ticket_key")
        if not ticket_key:
            raise ValueError(f"No ticket_key found for {event_api_id} — are you registered for this event?")

    data = fetch_event(event_api_id)
    event = data.get("event", {})
    event_name = event.get("name", "")
    event_url = f"https://lu.ma/{event.get('url') or event_api_id}"

    attendees = fetch_guest_list(event_api_id, ticket_key, cookie)
    for p in attendees:
        upsert_person({
            "event_api_id": event_api_id,
            "event_url": event_url,
            "event_name": event_name,
            "raw_json": json.dumps(p.pop("raw")),
            **p,
        })

    return {
        "event_api_id": event_api_id,
        "event_name": event_name,
        "event_url": event_url,
        "attendee_count": len(attendees),
        "attendees": attendees,
    }
