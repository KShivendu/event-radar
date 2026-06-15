#!/usr/bin/env python3
"""Fetch the list of Luma calendars the user follows and save to luma_calendars.json."""
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import requests

OUT = Path(__file__).parent / "luma_calendars.json"


def fetch():
    cookie = os.environ.get("LUMA_COOKIE", "")
    headers = {
        "accept": "*/*",
        "x-luma-client-type": "luma-web",
        "x-luma-timezone": "America/Los_Angeles",
        "x-luma-web-url": "https://luma.com/home/calendars",
        "Referer": "https://luma.com/",
    }
    if cookie:
        headers["cookie"] = cookie

    resp = requests.get("https://api.luma.com/home/get-following-calendars", headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("infos", [])


def main():
    infos = fetch()
    if not infos:
        print("No calendars returned — cookie may be expired.")
        return

    calendars = [
        {"id": c["api_id"], "slug": c.get("slug") or "", "name": c["name"]}
        for info in infos
        for c in [info["calendar"]]
    ]
    OUT.write_text(json.dumps(calendars, indent=2))
    print(f"Saved {len(calendars)} calendars to {OUT.name}")
    for cal in calendars:
        print(f"  {cal['id']} | {cal['slug']:30} | {cal['name']}")


if __name__ == "__main__":
    main()
