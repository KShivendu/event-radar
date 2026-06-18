#!/usr/bin/env python3
"""Given a Luma event, collect the people you could talk to (hosts + featured guests),
rank them against your profile with Claude, and print who's worth your time.

Usage:
    python3 find_people.py <lu.ma-url | slug | evt-id> [--web] [--no-rank]

    python3 find_people.py https://lu.ma/yj5uvoei
    python3 find_people.py evt-oiXR0BSLzOsOgtn --web

Requires ANTHROPIC_API_KEY (in .env or env) for ranking. --no-rank skips the LLM
and just lists collected people. --web lets Claude web-search people before ranking.
"""
import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from scraper.sources import luma_people


def _fmt_handles(p: dict) -> str:
    bits = []
    if p.get("twitter_handle"):
        bits.append(f"x.com/{p['twitter_handle']}")
    if p.get("linkedin_handle"):
        bits.append("linkedin.com/" + p["linkedin_handle"].lstrip("/"))
    if p.get("website"):
        bits.append(p["website"])
    return "  ".join(bits)


def main():
    ap = argparse.ArgumentParser(description="Find people to talk to at a Luma event.")
    ap.add_argument("event", help="lu.ma URL, slug, or evt- id")
    ap.add_argument("--web", action="store_true", help="let Claude web-search people before ranking")
    ap.add_argument("--no-rank", action="store_true", help="skip LLM ranking, just collect")
    args = ap.parse_args()

    print(f"Resolving and collecting people for: {args.event}")
    summary = luma_people.collect(args.event)
    people = summary["people"]
    print(f"\n{summary['event_name']}")
    print(f"{summary['event_url']}  —  {summary['guest_count']} registered")
    hosts = [p for p in people if p["role"] == "host"]
    guests = [p for p in people if p["role"] == "guest"]
    print(f"Collected {len(people)} people ({len(hosts)} hosts, {len(guests)} featured guests).\n")

    if not args.no_rank and people:
        try:
            from enrich import rank
            from scraper.db import save_ranking
            print("Ranking with Claude" + (" (+ web search)" if args.web else "") + "...\n")
            people = rank(summary, people, use_web=args.web)
            for p in people:
                if p.get("rank_score") is not None:
                    save_ranking(summary["event_api_id"], p["person_api_id"],
                                 p["rank_score"], p["rank_reason"], p["icebreaker"])
        except FileNotFoundError as e:
            print(f"  (skipping ranking: {e})\n", file=sys.stderr)
        except ImportError:
            print("  (skipping ranking: `pip install anthropic` to enable)\n", file=sys.stderr)
        except Exception as e:
            print(f"  (ranking failed, showing unranked list: {e})\n", file=sys.stderr)

    for p in people:
        score = p.get("rank_score")
        badge = f"[{score}/10] " if score is not None else ""
        tag = "HOST" if p["role"] == "host" else "guest"
        print(f"{badge}{p['name']}  ({tag})")
        if p.get("bio_short"):
            print(f"    {p['bio_short']}")
        handles = _fmt_handles(p)
        if handles:
            print(f"    {handles}")
        if p.get("rank_reason"):
            print(f"    → {p['rank_reason']}")
        if p.get("icebreaker"):
            print(f"    💬 {p['icebreaker']}")
        print()


if __name__ == "__main__":
    main()
