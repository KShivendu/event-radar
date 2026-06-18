#!/usr/bin/env python3
"""Rank the people at an event by how worth-talking-to they are for you, and draft
an icebreaker for each — using Claude.

Input is whatever Luma exposes publicly (name, role, short bio, social handles).
Your side of the match comes from profile.json (copy profile.example.json). With
--web, Claude may use web search to pull extra context on notable people before ranking.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
PROFILE_FILE = ROOT / "profile.json"
MODEL = "claude-opus-4-8"


def load_profile() -> dict:
    if PROFILE_FILE.exists():
        return json.loads(PROFILE_FILE.read_text())
    raise FileNotFoundError(
        "profile.json not found — copy profile.example.json to profile.json and edit it."
    )


def _person_card(p: dict) -> dict:
    """A compact, token-light view of one person for the model."""
    return {
        "id": p["person_api_id"],
        "name": p.get("name"),
        "role": p.get("role"),
        "bio": p.get("bio_short") or None,
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


def rank(event: dict, people: list[dict], use_web: bool = False) -> list[dict]:
    """Return the people list with rank_score/rank_reason/icebreaker filled in,
    sorted best-first. Requires ANTHROPIC_API_KEY."""
    import anthropic

    profile = load_profile()
    cards = [_person_card(p) for p in people if p.get("person_api_id")]
    if not cards:
        return people

    user_content = (
        f"# Your profile\n{json.dumps(profile, indent=2)}\n\n"
        f"# Event\n{event.get('event_name')} ({event.get('event_url')}) — "
        f"{event.get('guest_count')} registered\n\n"
        f"# People ({len(cards)})\n{json.dumps(cards, indent=2)}"
    )

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    kwargs = dict(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_content}],
    )
    if use_web:
        # Let Claude look people up before scoring. With a tool available the
        # response isn't constrained to the schema, so we ask for raw JSON instead.
        kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]
        kwargs["system"] = SYSTEM + (
            "\n\nReturn ONLY a JSON object matching: "
            '{"rankings":[{"id","score","reason","icebreaker"}]} with no other text.'
        )
        resp = client.messages.create(**kwargs)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        rankings = json.loads(_extract_json(text))["rankings"]
    else:
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": _build_schema()}}
        resp = client.messages.create(**kwargs)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        rankings = json.loads(text)["rankings"]

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
    """Pull the first {...} object out of a text blob (web mode isn't schema-constrained)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model response: {text[:200]}")
    return text[start : end + 1]
