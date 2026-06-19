#!/usr/bin/env python3
"""Rank the people at an event by how worth-talking-to they are for you, and draft
an icebreaker for each — using Claude.

Input is whatever Luma exposes publicly (name, role, short bio, social handles).
Your side of the match comes from profile.json (copy profile.example.json).

Two backends, auto-selected by enrich_people():
  - SDK  — used when ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN is set. Supports --web
           (Claude web-searches people before ranking).
  - CLI  — falls back to the local `claude` CLI (Claude Code), which is already
           authenticated, so no API key is needed. No web search in this mode.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
PROFILE_FILE = ROOT / "profile.json"
MODEL = "claude-opus-4-8"
CLI_TIMEOUT = 300


def load_profile() -> dict:
    if PROFILE_FILE.exists():
        return json.loads(PROFILE_FILE.read_text())
    raise FileNotFoundError(
        "profile.json not found — copy profile.example.json to profile.json and edit it."
    )


def _person_card(p: dict) -> dict:
    """A compact, token-light view of one person for the model.
    Includes contact-enrichment fields (current_role, github) when present, so
    ranking benefits from them if contacts ran first."""
    return {
        "id": p["person_api_id"],
        "name": p.get("name"),
        "role": p.get("role"),
        "bio": p.get("bio_short") or None,
        "current_role": p.get("current_role"),
        "github": p.get("github_handle"),
        "linkedin": p.get("linkedin_handle"),
        "twitter": p.get("twitter_handle"),
        "website": p.get("website"),
    }


def _build_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "rankings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "The person's id, copied verbatim."},
                        "score": {"type": "integer", "description": "1-10; how worth your time talking to them is."},
                        "reason": {"type": "string", "description": "One sentence: why they're (not) a fit, grounded in their bio/role."},
                        "icebreaker": {"type": "string", "description": "A specific opener you could actually say to them."},
                    },
                    "required": ["id", "score", "reason", "icebreaker"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rankings"],
        "additionalProperties": False,
    }


SYSTEM = (
    "You help someone decide who to talk to at a tech meetup. Given the user's profile "
    "and a list of people attending (hosts and notable guests, with whatever public info "
    "is available), score each person 1-10 for how worth the user's limited time they are, "
    "give a one-sentence reason grounded in that person's actual bio/role (don't invent "
    "facts), and draft a short, specific icebreaker the user could open with. Reward "
    "relevance to the user's goals and interests; hosts are usually high-value connectors. "
    "Be honest with low scores for clearly-unrelated people. Score every person provided."
)


_JSON_ONLY = (
    '\n\nReturn ONLY a JSON object: {"rankings":[{"id","score","reason","icebreaker"}]}. '
    "Copy each id verbatim. No prose, no markdown fences."
)


def _build_user_content(event: dict, cards: list[dict]) -> str:
    profile = load_profile()
    return (
        f"# Your profile\n{json.dumps(profile, indent=2)}\n\n"
        f"# Event\n{event.get('event_name')} ({event.get('event_url')}) — "
        f"{event.get('guest_count')} registered\n\n"
        f"# People ({len(cards)})\n{json.dumps(cards, indent=2)}"
    )


def _apply_rankings(people: list[dict], rankings: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in rankings}
    for p in people:
        r = by_id.get(p.get("person_api_id"))
        if r:
            p["rank_score"] = r["score"]
            p["rank_reason"] = r["reason"]
            p["icebreaker"] = r["icebreaker"]
    people.sort(key=lambda p: p.get("rank_score") or 0, reverse=True)
    return people


def _extract_json(text: str) -> str:
    """Pull the first {...} object out of a text blob (CLI / web output isn't schema-constrained)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model response: {text[:200]}")
    return text[start : end + 1]


def enrich_people(event: dict, people: list[dict], use_web: bool = False) -> list[dict]:
    """Rank people, auto-selecting a backend: the API SDK when a key is set,
    otherwise the local `claude` CLI. --web requires the SDK backend."""
    if not [p for p in people if p.get("person_api_id")]:
        return people
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return rank(event, people, use_web=use_web)
    if use_web:
        raise RuntimeError("--web needs an API key (ANTHROPIC_API_KEY); the claude CLI backend can't web-search here.")
    if shutil.which("claude"):
        return rank_via_cli(event, people)
    raise RuntimeError("No ranking backend: set ANTHROPIC_API_KEY, or install the `claude` CLI.")


def rank(event: dict, people: list[dict], use_web: bool = False) -> list[dict]:
    """SDK backend. Fills rank_score/rank_reason/icebreaker, sorted best-first.
    Requires ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN."""
    import anthropic

    cards = [_person_card(p) for p in people if p.get("person_api_id")]
    user_content = _build_user_content(event, cards)

    client = anthropic.Anthropic()  # reads credentials from env
    kwargs = dict(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_content}],
    )
    if use_web:
        # With a tool available the response isn't schema-constrained, so ask for raw JSON.
        kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]
        kwargs["system"] = SYSTEM + _JSON_ONLY
        resp = client.messages.create(**kwargs)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        rankings = json.loads(_extract_json(text))["rankings"]
    else:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": _build_schema()}}
        resp = client.messages.create(**kwargs)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        rankings = json.loads(text)["rankings"]

    return _apply_rankings(people, rankings)


def rank_via_cli(event: dict, people: list[dict]) -> list[dict]:
    """Fallback backend using the local `claude` CLI (already authenticated by Claude Code).
    No API key required; no web search."""
    cards = [_person_card(p) for p in people if p.get("person_api_id")]
    prompt = SYSTEM + _JSON_ONLY + "\n\n" + _build_user_content(event, cards)

    proc = subprocess.run(
        ["claude", "-p"], input=prompt,
        capture_output=True, text=True, timeout=CLI_TIMEOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr[:300]}")
    rankings = json.loads(_extract_json(proc.stdout))["rankings"]
    return _apply_rankings(people, rankings)
