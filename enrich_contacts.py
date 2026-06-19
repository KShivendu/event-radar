#!/usr/bin/env python3
"""Resolve each person's web presence — GitHub, personal site, and canonical
LinkedIn/Twitter URLs — plus a one-line "current role".

Strategy (cheapest, highest-precision first):
  1. Luma handles      → canonical URLs for LinkedIn / Twitter / Instagram / site (free, already have them)
  2. Their own website → fetch and scrape for a GitHub link + other socials (free)
  3. GitHub API        → search by name, verify the candidate by matching twitter_username
                         or blog against what we already know (free; needs GITHUB_TOKEN here)
  4. Web fallback      → ask Claude (web search) to find/verify the gaps + current role

Backends mirror enrich.py: the Anthropic SDK when an API key is set, otherwise the
local `claude` CLI (already authenticated). The unique anchors we trust for identity
are the Twitter handle and website — everything is matched back to those.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
from urllib.parse import urlparse

import requests

GITHUB_API = "https://api.github.com"
# github.com paths that are never a user login
_GH_RESERVED = {"about", "features", "pricing", "sponsors", "marketplace", "topics",
                "collections", "trending", "orgs", "settings", "login", "join", "search"}
_GH_LINK_RE = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)", re.I)
_UA = {"User-Agent": "Mozilla/5.0", "accept": "text/html"}


# ---------- free tier ----------

def canonical_links(p: dict) -> dict:
    """Turn the handles Luma already gave us into full URLs."""
    out = {}
    if p.get("linkedin_handle"):
        out["linkedin"] = "https://linkedin.com/" + p["linkedin_handle"].lstrip("/")
    if p.get("twitter_handle"):
        out["twitter"] = "https://x.com/" + p["twitter_handle"].lstrip("@")
    if p.get("instagram_handle"):
        out["instagram"] = "https://instagram.com/" + p["instagram_handle"].lstrip("@")
    if p.get("website"):
        out["website"] = p["website"]
    return out


def github_from_url(url: str) -> str | None:
    """Pull a GitHub login out of a URL: github.com/<login> or <login>.github.io."""
    if not url:
        return None
    host = (urlparse(url).netloc or "").lower().removeprefix("www.")
    if host == "github.com":
        parts = [s for s in urlparse(url).path.split("/") if s]
        if parts and parts[0].lower() not in _GH_RESERVED:
            return parts[0]
    if host.endswith(".github.io"):
        return host.split(".")[0]
    return None


def fetch_site(url: str, timeout: int = 15) -> tuple[str, str | None]:
    """Fetch a personal site; return (text, github_login_found_in_links)."""
    try:
        resp = requests.get(url, headers=_UA, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return "", None
    html = resp.text
    for m in _GH_LINK_RE.finditer(html):
        login = m.group(1)
        if login.lower() not in _GH_RESERVED:
            return html, login
    return html, None


def github_api_verify(p: dict, token: str | None) -> dict | None:
    """Search GitHub by name and accept a candidate only if a unique anchor matches
    (twitter_username == our twitter handle, or blog host == our website host)."""
    name = p.get("name")
    if not name:
        return None
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        sr = requests.get(f"{GITHUB_API}/search/users",
                          params={"q": f"{name} in:name", "per_page": 5},
                          headers=headers, timeout=15)
        sr.raise_for_status()
    except Exception:
        return None

    our_tw = (p.get("twitter_handle") or "").lstrip("@").lower()
    our_host = _host(p.get("website"))
    for item in sr.json().get("items", []):
        try:
            u = requests.get(f"{GITHUB_API}/users/{item['login']}", headers=headers, timeout=15).json()
        except Exception:
            continue
        tw = (u.get("twitter_username") or "").lstrip("@").lower()
        if (our_tw and tw == our_tw) or (our_host and _host(u.get("blog")) == our_host):
            return {"github_handle": u["login"], "github_company": u.get("company"),
                    "github_bio": u.get("bio"), "blog": u.get("blog")}
    return None


def _host(url: str | None) -> str | None:
    if not url:
        return None
    if "://" not in url:
        url = "https://" + url
    return (urlparse(url).netloc or "").lower().removeprefix("www.") or None


# ---------- web fallback (Claude) ----------

_WEB_SYSTEM = (
    "You find a specific person's public web presence. You are given their name and the "
    "social handles/site already known (these are TRUSTED, globally-unique anchors). Use web "
    "search to find their GitHub username, personal website, and a one-line current role "
    "(title @ company). ONLY report a link if you are confident it is the SAME person — "
    "cross-check against the known anchors (matching name + handle + employer). When unsure, "
    "leave the field null rather than guessing. "
    'Return ONLY JSON: {"github_handle":null,"website":null,"current_role":null,'
    '"linkedin":null,"twitter":null,"confidence":"low|medium|high","note":""}. '
    "github_handle is the bare username, not a URL. No prose, no markdown fences."
)


def _web_prompt(p: dict, known: dict) -> str:
    return (_WEB_SYSTEM + "\n\n# Person\n" +
            json.dumps({"name": p.get("name"), "bio": p.get("bio_short") or None,
                        "known_links": known}, indent=2))


def web_fallback(p: dict, known: dict) -> dict:
    """Find/verify gaps via web search. SDK when a key is set, else the `claude` CLI."""
    prompt = _web_prompt(p, known)
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-opus-4-8", max_tokens=2000,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
    elif shutil.which("claude"):
        # WebSearch/WebFetch must be explicitly allowed in headless print mode,
        # otherwise the CLI runs with no web access and can only echo known links.
        proc = subprocess.run(
            ["claude", "-p", "--allowedTools", "WebSearch,WebFetch"],
            input=prompt, capture_output=True, text=True, timeout=240,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {proc.stderr[:200]}")
        text = proc.stdout
    else:
        raise RuntimeError("No web backend: set ANTHROPIC_API_KEY or install the `claude` CLI.")
    return json.loads(_extract_json(text))


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return text[start:end + 1]


# ---------- face / avatar ----------

def _url_is_image(url: str, timeout: int = 12) -> bool:
    """True if the URL returns 200 with an image content-type (no full download)."""
    try:
        r = requests.get(url, timeout=timeout, stream=True, allow_redirects=True)
        ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image")
        r.close()
        return ok
    except Exception:
        return False


def gravatar_url(email: str) -> str:
    h = hashlib.sha256(email.strip().lower().encode()).hexdigest()
    return f"https://gravatar.com/avatar/{h}?s=400&d=404"


def resolve_face(p: dict) -> tuple[str | None, str | None]:
    """Best-first face cascade (returns the resolved URL, not a download):
    Luma avatar → GitHub avatar → Gravatar (needs email). Falls back to the Luma
    default placeholder, then None. Each candidate is verified to actually serve an image."""
    av = p.get("avatar_url") or ""
    if av and "avatars-default" not in av:
        return av, "luma"
    if p.get("github_handle"):
        gh = f"https://github.com/{p['github_handle']}.png?size=400"
        if _url_is_image(gh):
            return gh, "github"
    if p.get("email"):  # dormant until we collect emails, but wired per design
        gu = gravatar_url(p["email"])
        if _url_is_image(gu):
            return gu, "gravatar"
    return (av or None), ("luma_default" if av else None)


# ---------- orchestration ----------

def enrich_person(p: dict, use_web: bool = True, token: str | None = None) -> dict:
    """Fill github_handle / github_company / current_role / discovered_links on a person."""
    token = token or os.environ.get("GITHUB_TOKEN")
    links = canonical_links(p)
    github = None
    source = ["luma"]

    # 2. GitHub from the website we already have, then by scraping that site
    if links.get("website"):
        github = github_from_url(links["website"])
        if not github:
            _, github = fetch_site(links["website"])
            if github:
                source.append("site")
    # 3. GitHub API verification
    if not github:
        hit = github_api_verify(p, token)
        if hit:
            github = hit["github_handle"]
            p["github_company"] = hit.get("github_company")
            source.append("github_api")
    # 4. Web fallback for remaining gaps + current role
    if use_web and (not github or not p.get("current_role")):
        try:
            wf = web_fallback(p, links)
            if not github and wf.get("github_handle"):
                github = wf["github_handle"]
            if wf.get("current_role"):
                p["current_role"] = wf["current_role"]
            if not links.get("website") and wf.get("website"):
                links["website"] = wf["website"]
            source.append("web")
        except Exception as e:
            p["_web_error"] = str(e)

    if github:
        p["github_handle"] = github
        links["github"] = f"https://github.com/{github}"
    p["discovered_links"] = json.dumps(links)
    p["contact_source"] = "+".join(source)

    # face cascade (runs after github is known)
    p["face_url"], p["face_source"] = resolve_face(p)
    return p


def save_person(event_api_id: str, p: dict):
    """Persist the contact-enrichment fields of an already-enriched person dict."""
    from scraper.db import save_contact
    links = json.loads(p["discovered_links"]) if p.get("discovered_links") else {}
    save_contact(event_api_id, p["person_api_id"], {
        "github_handle": p.get("github_handle"),
        "github_company": p.get("github_company"),
        "current_role": p.get("current_role"),
        "discovered_links": p.get("discovered_links"),
        "website": links.get("website") or p.get("website"),
        "contact_source": p.get("contact_source"),
        "face_url": p.get("face_url"),
        "face_source": p.get("face_source"),
    })


def enrich_event(event_api_id: str, use_web: bool = True) -> list[dict]:
    from scraper.db import get_event_people
    people = get_event_people(event_api_id)
    for p in people:
        enrich_person(p, use_web=use_web)
        save_person(event_api_id, p)
    return people


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass
    from scraper.sources import luma_people

    ap = argparse.ArgumentParser(description="Enrich event people with GitHub / site / role.")
    ap.add_argument("event", help="lu.ma URL, slug, or evt- id")
    ap.add_argument("--no-web", action="store_true", help="skip the Claude web fallback")
    args = ap.parse_args()

    eid = luma_people.collect(args.event)["event_api_id"]
    print(f"Enriching contacts for {eid}...\n")
    people = enrich_event(eid, use_web=not args.no_web)
    for p in people:
        links = json.loads(p.get("discovered_links") or "{}")
        print(f"{p['name']}  ({p['contact_source']})")
        if p.get("current_role"):
            print(f"    role: {p['current_role']}")
        if p.get("face_url"):
            print(f"    face: {p['face_url']}  [{p['face_source']}]")
        for k in ("github", "website", "linkedin", "twitter"):
            if links.get(k):
                print(f"    {k}: {links[k]}")
        print()
