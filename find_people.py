#!/usr/bin/env python3
"""Given a Luma event, collect the people you could talk to (hosts + featured guests),
enrich them (GitHub / site / current role / face), and rank them against your profile
with Claude — one command.

Usage:
    python3 find_people.py <lu.ma-url | slug | evt-id> [options]

    python3 find_people.py https://lu.ma/yj5uvoei
    python3 find_people.py evt-oiXR0BSLzOsOgtn --contacts-web   # web-search each person
    python3 find_people.py https://lu.ma/yj5uvoei --no-rank     # collect + enrich only

Options:
    --no-contacts    skip GitHub/site/role/face enrichment
    --contacts-web   let Claude web-search each person during enrichment (slower)
    --no-rank        skip LLM ranking
    --web            let Claude web-search people during ranking (needs API key)

Ranking/web use the Anthropic SDK when ANTHROPIC_API_KEY is set, else the local
`claude` CLI (already authenticated).
"""
import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from pipeline import find_people


def main():
    ap = argparse.ArgumentParser(description="Find people to talk to at a Luma event.")
    ap.add_argument("event", help="lu.ma URL, slug, or evt- id")
    ap.add_argument("--no-contacts", action="store_true", help="skip GitHub/site/role/face enrichment")
    ap.add_argument("--contacts-web", action="store_true", help="web-search each person during enrichment")
    ap.add_argument("--no-rank", action="store_true", help="skip LLM ranking")
    ap.add_argument("--web", action="store_true", help="web-search people during ranking")
    args = ap.parse_args()

    print(f"Collecting + enriching people for: {args.event}\n")
    try:
        summary, people = find_people(
            args.event,
            contacts=not args.no_contacts, contacts_web=args.contacts_web,
            rank=not args.no_rank, rank_web=args.web,
        )
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

    hosts = sum(1 for p in people if p["role"] == "host")
    print(f"{summary['event_name']}")
    print(f"{summary['event_url']}  —  {summary['guest_count']} registered")
    print(f"{len(people)} people ({hosts} hosts, {len(people)-hosts} featured guests)\n")

    for p in people:
        links = json.loads(p.get("discovered_links") or "{}")
        score = p.get("rank_score")
        badge = f"[{int(score)}/10] " if score is not None else ""
        tag = "HOST" if p["role"] == "host" else "guest"
        print(f"{badge}{p['name']}  ({tag})")
        role = p.get("current_role") or p.get("bio_short")
        if role:
            print(f"    {role}")
        if p.get("face_url"):
            print(f"    face: {p['face_url']}")
        handles = "  ".join(f"{k}:{links[k]}" for k in ("github", "linkedin", "twitter", "website") if links.get(k))
        if handles:
            print(f"    {handles}")
        if p.get("rank_reason"):
            print(f"    → {p['rank_reason']}")
        if p.get("icebreaker"):
            print(f"    💬 {p['icebreaker']}")
        print()


if __name__ == "__main__":
    main()
