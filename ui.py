#!/usr/bin/env python3
"""Local web UI for browsing SF events from events.db"""
import json
import threading
from datetime import datetime, timezone
import zoneinfo
from flask import Flask, request, jsonify, render_template_string
from scraper.db import get_conn, get_event_people, init_db

app = Flask(__name__)
init_db()
PT = zoneinfo.ZoneInfo("America/Los_Angeles")

# tracks which events have an active ranking job: event_api_id -> "running" | "done" | "error: ..."
_rank_status: dict[str, str] = {}

# face recognition caches
_face_app = None
_face_indexes: dict = {}  # event_id -> (embeddings np.ndarray, meta list)
_face_prep_status: dict[str, str] = {}  # event_id -> "running" | "done:N" | "error: ..."

def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        import numpy as np
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app

def _get_face_index(event_id: str):
    if event_id not in _face_indexes:
        import numpy as np, pathlib
        path = pathlib.Path("faces") / f"{event_id}.npz"
        if not path.exists():
            return None, None
        data = np.load(path, allow_pickle=True)
        _face_indexes[event_id] = (
            data["embeddings"],
            [json.loads(m) for m in data["meta"]],
        )
    return _face_indexes[event_id]


def to_pt(dt_str: str | None) -> str:
    if not dt_str:
        return ""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        pt = dt.astimezone(PT)
        return pt.strftime("%a %b %-d · %-I:%M %p")
    except Exception:
        return dt_str


def to_pt_date(dt_str: str | None) -> str:
    if not dt_str:
        return ""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.astimezone(PT).strftime("%Y-%m-%d")
    except Exception:
        return dt_str[:10] if dt_str else ""


@app.route("/api/events")
def api_events():
    source = request.args.get("source", "all")
    date_filter = request.args.get("date", "")
    search = request.args.get("q", "").lower()
    location_filter = request.args.get("location", "all")

    BAY_AREA_CITIES = {"San Francisco", "Menlo Park", "San Jose", "Mountain View",
                       "Sunnyvale", "Palo Alto", "Oakland", "Berkeley", "Redwood City",
                       "Santa Clara", "Cupertino", "Fremont", "San Mateo"}

    with get_conn() as conn:
        query = "SELECT * FROM events WHERE start_datetime >= datetime('now') ORDER BY start_datetime"
        rows = [dict(r) for r in conn.execute(query)]

    now_pt_date = datetime.now(PT).strftime("%Y-%m-%d")

    # Pass 1: process Luma events, deduplicate cross-calendar by external_id.
    seen: dict[str, dict] = {}  # external_id -> row
    luma_slugs: set[str] = set()  # slugs of all known Luma events
    enriched = []

    for r in rows:
        if not r["source"].startswith("luma:"):
            continue
        r["start_pt"] = to_pt(r["start_datetime"])
        r["date_pt"] = to_pt_date(r["start_datetime"])
        r.pop("raw_json", None)
        r["is_today"] = r["date_pt"] == now_pt_date

        key = r["external_id"]
        url = r.get("url") or ""
        slug = url.rstrip("/").split("/")[-1]
        if slug:
            luma_slugs.add(slug)

        if key not in seen:
            seen[key] = r
            r["calendars"] = [r["source"]]
            enriched.append(r)
        else:
            existing = seen[key]
            existing["calendars"].append(r["source"])
            if not existing["registration_status"] and r["registration_status"]:
                existing.update({k: v for k, v in r.items() if k != "calendars"})

    # Pass 2: process non-Luma events, skipping CV events that link to a known Luma slug.
    for r in rows:
        if r["source"].startswith("luma:"):
            continue
        r["start_pt"] = to_pt(r["start_datetime"])
        r["date_pt"] = to_pt_date(r["start_datetime"])
        r.pop("raw_json", None)
        r["is_today"] = r["date_pt"] == now_pt_date

        if r["source"] == "cerebral_valley":
            url = r.get("url") or ""
            if "luma.com/" in url:
                slug = url.rstrip("/").split("/")[-1]
                if slug in luma_slugs:
                    continue  # already covered by Luma entry
        r["calendars"] = [r["source"]]
        enriched.append(r)

    result = []
    for r in enriched:
        if source != "all":
            calendars = r.get("calendars", [r["source"]])
            if not any(c == source or c.startswith(source + ":") for c in calendars):
                continue
        if date_filter and r["date_pt"] != date_filter:
            continue
        if location_filter != "all":
            city = r.get("city") or ""
            lt = r.get("location_type") or ""
            if location_filter == "online":
                if lt not in ("online", "zoom", "meet"):
                    continue
            elif location_filter == "sf":
                if city != "San Francisco":
                    continue
            elif location_filter == "bayarea":
                if city not in BAY_AREA_CITIES:
                    continue
        if search and search not in (r["name"] or "").lower() and search not in (r["description"] or "").lower():
            continue
        result.append(r)

    return jsonify(result)


@app.route("/api/luma-search")
def api_luma_search():
    import os, requests as req
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    cookie = os.environ.get("LUMA_COOKIE", "")
    headers = {
        "accept": "*/*",
        "x-luma-client-type": "luma-web",
        "x-luma-timezone": "America/Los_Angeles",
        "x-luma-web-url": "https://luma.com/home",
        "Referer": "https://luma.com/",
    }
    if cookie:
        headers["cookie"] = cookie

    try:
        resp = req.get("https://api.luma.com/search/get-results", params={"query": q}, headers=headers, timeout=10)
        resp.raise_for_status()
        entries = resp.json().get("events", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    results = []
    for entry in entries:
        event = entry.get("event", {})
        geo = event.get("geo_address_info") or {}
        gi = entry.get("guest_info")
        results.append({
            "name": event.get("name", ""),
            "start_pt": to_pt(event.get("start_at")),
            "url": f"https://lu.ma/{event.get('url') or event.get('api_id')}",
            "location": geo.get("city") or geo.get("full_address") or "",
            "venue": event.get("location", {}).get("name") if isinstance(event.get("location"), dict) else "",
            "image_url": event.get("cover_url"),
            "description": event.get("description") or event.get("description_short"),
            "registration_status": gi.get("approval_status") if isinstance(gi, dict) else None,
            "registration_availability": entry.get("registration_availability"),
            "calendar_name": (entry.get("calendar") or {}).get("name", ""),
        })
    return jsonify(results)


def _person_public(p: dict) -> dict:
    from enrich_contacts import canonical_links
    links = json.loads(p["discovered_links"]) if p.get("discovered_links") else canonical_links(p)
    face = p.get("face_url") or p.get("avatar_url")
    if face and "avatars-default" in face:
        face = None
    return {
        "name": p.get("name"),
        "role": p.get("role"),
        "face_url": face,
        "score": p.get("rank_score"),
        "reason": p.get("rank_reason"),
        "icebreaker": p.get("icebreaker"),
        "current_role": p.get("current_role") or p.get("bio_short") or "",
        "github": links.get("github"),
        "linkedin": links.get("linkedin"),
        "twitter": links.get("twitter"),
        "website": links.get("website"),
    }


@app.route("/api/people")
def api_people():
    """People already collected for an event (by evt- id)."""
    eid = request.args.get("event", "")
    rows = get_event_people(eid)
    return jsonify([_person_public(p) for p in rows])


@app.route("/api/people/find", methods=["POST"])
def api_people_find():
    """Collect people synchronously, then rank in the background.
    Returns collected (unranked) people immediately."""
    event = request.args.get("event", "")
    if not event:
        return jsonify({"error": "missing event"}), 400
    try:
        from scraper.sources.luma_people import collect, collect_attendees
        import os
        cookie = os.environ.get("LUMA_COOKIE", "")
        # try full attendee list first (needs ticket_key in DB); fall back to hosts+featured guests
        try:
            summary = collect_attendees(event, cookie)
            people = summary["attendees"]
        except Exception:
            summary = collect(event)
            people = summary["people"]
        eid = summary["event_api_id"]
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    # kick off enrich + rank in background
    if _rank_status.get(eid) != "running":
        _rank_status[eid] = "running"
        def _rank():
            try:
                import enrich, enrich_contacts
                from scraper.db import save_ranking
                all_people = get_event_people(eid)
                for p in all_people:
                    enrich_contacts.enrich_person(p, use_web=False)
                    enrich_contacts.save_person(eid, p)
                ranked = enrich.enrich_people(summary, all_people, use_web=False)
                for p in ranked:
                    if p.get("rank_score") is not None:
                        save_ranking(eid, p["person_api_id"], p["rank_score"],
                                     p["rank_reason"], p["icebreaker"])
                _rank_status[eid] = "done"
            except Exception as e:
                _rank_status[eid] = f"error: {e}"
        threading.Thread(target=_rank, daemon=True).start()

    rows = get_event_people(eid)
    return jsonify([_person_public(p) for p in rows])


@app.route("/api/people/rank-status")
def api_people_rank_status():
    eid = request.args.get("event", "")
    return jsonify({"status": _rank_status.get(eid, "idle")})


@app.route("/api/dates")
def api_dates():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT date(start_datetime) as d FROM events WHERE start_datetime >= datetime('now') ORDER BY d"
        ).fetchall()
    dates = [r["d"] for r in rows]
    return jsonify(dates)


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SF Events</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { background: #0f0f13; color: #e8e8f0; font-family: 'Inter', system-ui, sans-serif; }
  .card { background: #1a1a24; border: 1px solid #2a2a38; }
  .card:hover { border-color: #5b5bff; background: #1e1e2e; }
  .badge { font-size: 0.65rem; padding: 2px 7px; border-radius: 999px; font-weight: 600; letter-spacing: 0.04em; }
  .badge-luma { background: #1a2a4a; color: #60a5fa; }
  .badge-cv { background: #2a1a3a; color: #c084fc; }
  .badge-approved { background: #0f2a1a; color: #4ade80; }
  .badge-waitlist { background: #2a1f0a; color: #fbbf24; }
  .badge-pending { background: #1a1a2a; color: #a78bfa; }
  .badge-invited { background: #0f1f2a; color: #38bdf8; }
  .badge-declined { background: #1a0f0f; color: #ef4444; text-decoration: line-through; opacity: 0.7; }
  input, select { background: #1a1a24; border: 1px solid #2a2a38; color: #e8e8f0; }
  input:focus, select:focus { outline: none; border-color: #5b5bff; }
  .date-btn { background: #1a1a24; border: 1px solid #2a2a38; color: #9090b0; cursor: pointer; }
  .date-btn.active { background: #2a2a4a; border-color: #5b5bff; color: #a0a0ff; }
  .date-btn:hover { border-color: #5b5bff; }
  ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f0f13; }
  ::-webkit-scrollbar-thumb { background: #2a2a38; border-radius: 3px; }
  .group-header { border-left: 3px solid #5b5bff; }
  .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 50; }
  .modal-panel { background: #14141c; border: 1px solid #2a2a38; }
  .ppl-btn { font-size: 0.7rem; color: #9090b0; border: 1px solid #2a2a38; border-radius: 6px; padding: 2px 8px; }
  .ppl-btn:hover { border-color: #5b5bff; color: #a0a0ff; }
  .score { font-size: 0.7rem; font-weight: 700; border-radius: 6px; padding: 2px 7px; }
  .s-hi { background: #0f2a1a; color: #4ade80; } .s-mid { background: #2a2410; color: #fbbf24; } .s-lo { background: #1a1a2a; color: #8888a8; }
  .ice { background: #16161f; border-left: 2px solid #5b5bff; }
</style>
</head>
<body class="min-h-screen">
<div class="max-w-4xl mx-auto px-4 py-8">

  <!-- Header -->
  <div class="flex items-center justify-between mb-8">
    <div>
      <h1 class="text-2xl font-bold text-white">SF Events</h1>
      <p class="text-sm text-gray-500 mt-1" id="subtitle">Loading...</p>
    </div>
    <div class="flex gap-2">
      <a href="/face" class="text-xs px-3 py-1.5 rounded border border-gray-700 text-gray-400 hover:border-indigo-500 hover:text-indigo-400 transition-colors">📷 Face</a>
      <button onclick="loadEvents()" class="text-xs px-3 py-1.5 rounded border border-gray-700 text-gray-400 hover:border-indigo-500 hover:text-indigo-400 transition-colors">↻ Refresh</button>
    </div>
  </div>

  <!-- Filters -->
  <div class="flex flex-wrap gap-3 mb-6">
    <input id="search" type="text" placeholder="Search events..."
      class="flex-1 min-w-48 rounded-lg px-3 py-2 text-sm"
      oninput="if(!document.getElementById('live-search').checked) loadEvents()"
      onkeydown="if(event.key==='Enter') { event.preventDefault(); document.getElementById('live-search').checked ? lumaSearch() : loadEvents(); }" />
    <label class="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none whitespace-nowrap">
      <input id="live-search" type="checkbox" class="accent-indigo-500 w-3.5 h-3.5 cursor-pointer"
        onchange="onLiveModeChange()" />
      Live search
    </label>
    <select id="source" class="rounded-lg px-3 py-2 text-sm" onchange="loadEvents()">
      <option value="all">All sources</option>
      <option value="luma">Luma (all)</option>
      <option value="luma:my-events">Luma · My Events</option>
      <option value="luma:search">Luma · Keyword Search</option>
      <option value="luma:sf">Luma · SF Featured</option>
      <option value="luma:genai-sf">Luma · GenAI SF</option>
      <option value="luma:ai-sf">Luma · AI SF</option>
      <option value="cerebral_valley">Cerebral Valley</option>
    </select>
    <select id="location" class="rounded-lg px-3 py-2 text-sm" onchange="loadEvents()">
      <option value="all">All locations</option>
      <option value="sf">San Francisco</option>
      <option value="bayarea">Bay Area</option>
      <option value="online">Online</option>
    </select>
  </div>

  <!-- Date pills -->
  <div id="date-pills" class="flex gap-2 flex-wrap mb-6"></div>

  <!-- Events (local DB) -->
  <div id="events-container"></div>

  <!-- Luma Live Search results -->
  <div id="luma-search-results" class="hidden"></div>

</div>

<!-- People modal -->
<div id="people-modal" class="modal-overlay hidden flex items-start justify-center p-4 overflow-y-auto" onclick="if(event.target===this) closePeople()">
  <div class="modal-panel rounded-2xl w-full max-w-2xl my-8 p-5">
    <div class="flex items-center justify-between mb-1">
      <h2 class="text-lg font-semibold text-white">People to talk to</h2>
      <button onclick="closePeople()" class="text-gray-500 hover:text-gray-300 text-xl leading-none">✕</button>
    </div>
    <p id="people-subtitle" class="text-xs text-gray-500 mb-4"></p>
    <div id="people-body"></div>
  </div>
</div>

<script>
let allEvents = [];
let selectedDate = '';

async function loadDates() {
  const res = await fetch('/api/dates');
  const dates = await res.json();
  const today = new Date().toLocaleDateString('en-CA', {timeZone: 'America/Los_Angeles'});
  const pills = document.getElementById('date-pills');

  const allBtn = `<button class="date-btn text-xs px-3 py-1.5 rounded-full ${selectedDate===''?'active':''}" onclick="setDate('')">All dates</button>`;
  const dateBtns = dates.map(d => {
    const label = d === today ? 'Today' : new Date(d+'T12:00:00').toLocaleDateString('en-US',{month:'short',day:'numeric'});
    return `<button class="date-btn text-xs px-3 py-1.5 rounded-full ${selectedDate===d?'active':''}" onclick="setDate('${d}')">${label}</button>`;
  }).join('');
  pills.innerHTML = allBtn + dateBtns;
}

function setDate(d) {
  selectedDate = d;
  loadEvents();
}

async function loadEvents() {
  const q = document.getElementById('search').value;
  const source = document.getElementById('source').value;
  const location = document.getElementById('location').value;
  const params = new URLSearchParams({ source, q, location });
  if (selectedDate) params.set('date', selectedDate);

  const res = await fetch('/api/events?' + params);
  allEvents = await res.json();
  renderEvents();
  loadDates(); // refresh pills with active state
}

function renderEvents() {
  const container = document.getElementById('events-container');
  if (!allEvents.length) {
    container.innerHTML = '<p class="text-gray-500 text-center py-12">No events found.</p>';
    document.getElementById('subtitle').textContent = '0 events';
    return;
  }

  document.getElementById('subtitle').textContent = `${allEvents.length} upcoming events · PT`;

  // Group by date
  const groups = {};
  for (const e of allEvents) {
    const d = e.date_pt || 'Unknown';
    if (!groups[d]) groups[d] = [];
    groups[d].push(e);
  }

  const today = new Date().toLocaleDateString('en-CA', {timeZone: 'America/Los_Angeles'});

  let html = '';
  for (const [date, events] of Object.entries(groups)) {
    const label = date === today
      ? 'Today'
      : new Date(date+'T12:00:00').toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'});
    html += `
      <div class="mb-8">
        <h2 class="group-header text-sm font-semibold text-indigo-300 uppercase tracking-widest mb-4 pl-3 py-0.5">${label} <span class="text-gray-600 font-normal normal-case tracking-normal">(${events.length})</span></h2>
        <div class="space-y-3">
          ${events.map(renderCard).join('')}
        </div>
      </div>`;
  }
  container.innerHTML = html;
}

function renderCard(e) {
  const sourceLabels = {'luma:my-events': 'MY EVENTS', 'luma:search': 'KEYWORD', 'luma:sf': 'SF FEATURED','luma:genai-sf': 'GENAI SF', 'luma:ai-sf': 'AI SF', 'cerebral_valley': 'CEREBRAL VALLEY'};
  const calLabel = (e.calendars || [e.source]).map(s => sourceLabels[s] || s).join(' · ');
  const sourceBadge = e.source.startsWith('luma')
    ? `<span class="badge badge-luma">${calLabel}</span>`
    : `<span class="badge badge-cv">CEREBRAL VALLEY</span>`;
  const regMap = {approved: ['badge-approved','✓ Going'], waitlist: ['badge-waitlist','⏳ Waitlisted'], pending_approval: ['badge-pending','⏳ Pending'], invited: ['badge-invited','✉ Invited'], declined: ['badge-declined', '✗ Declined']};
  const regBadge = e.registration_status && regMap[e.registration_status]
    ? `<span class="badge ${regMap[e.registration_status][0]}">${regMap[e.registration_status][1]}</span>` : '';
  const typeBadge = e.event_type ? `<span class="badge" style="background:#1a2a1a;color:#86efac">${e.event_type.toUpperCase()}</span>` : '';
  const venue = e.venue ? `<span class="text-gray-500">·</span> ${e.venue}` : '';
  const desc = e.description ? `<p class="text-gray-500 text-sm mt-2 line-clamp-2">${e.description}</p>` : '';
  const link = e.url ? `<a href="${e.url}" target="_blank" class="text-xs text-indigo-400 hover:text-indigo-300 mt-2 inline-block">Open →</a>` : '';
  const peopleBtn = (e.external_id && e.external_id.startsWith('evt-'))
    ? `<button class="ppl-btn ml-3" onclick="openPeople('${e.external_id}')">👥 People</button>` : '';

  return `
    <div class="card rounded-xl p-4 transition-all duration-150">
      <div class="flex items-start justify-between gap-3">
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 flex-wrap mb-1">
            ${sourceBadge}${typeBadge}${regBadge}
          </div>
          <h3 class="font-medium text-white leading-snug">${e.name}</h3>
          <p class="text-sm text-gray-400 mt-1">${e.start_pt} ${venue}</p>
          ${desc}
          ${link}${peopleBtn}
        </div>
        ${e.image_url ? `<img src="${e.image_url}" loading="lazy" class="w-20 h-20 rounded-lg object-cover flex-shrink-0" onerror="this.style.display='none'">` : ''}
      </div>
    </div>`;
}

function onLiveModeChange() {
  const live = document.getElementById('live-search').checked;
  const input = document.getElementById('search');
  document.getElementById('events-container').classList.toggle('hidden', live);
  document.getElementById('date-pills').classList.toggle('hidden', live);
  document.getElementById('luma-search-results').classList.toggle('hidden', !live);
  input.placeholder = live ? 'Search Luma (press Enter)...' : 'Search events...';
  if (live) { lumaSearch(); } else { document.getElementById('luma-search-results').innerHTML = ''; loadEvents(); }
}


loadEvents();

// ---- People modal ----
let peopleEventId = null;

function esc(s) {
  return (s == null ? '' : String(s)).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function openPeople(eid) {
  peopleEventId = eid;
  const ev = allEvents.find(e => e.external_id === eid);
  document.getElementById('people-modal').classList.remove('hidden');
  document.getElementById('people-subtitle').textContent = ev ? ev.name : eid;
  document.getElementById('people-body').innerHTML = '<p class="text-gray-500 text-sm py-6 text-center">Loading…</p>';
  fetch('/api/people?event=' + encodeURIComponent(eid))
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(renderPeople)
    .catch(err => {
      document.getElementById('people-body').innerHTML =
        `<p class="text-red-400 text-sm py-6 text-center">Error loading people (${err})</p>`;
    });
}

function closePeople() {
  document.getElementById('people-modal').classList.add('hidden');
}

function findPeople() {
  const body = document.getElementById('people-body');
  body.innerHTML = '<p class="text-gray-500 text-sm py-6 text-center">Collecting people…</p>';
  fetch('/api/people/find?event=' + encodeURIComponent(peopleEventId), {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.error) { body.innerHTML = `<p class="text-red-400 text-sm py-6 text-center">${esc(d.error)}</p>`; return; }
      renderPeople(d);
      pollRankStatus();
    });
}

function pollRankStatus() {
  fetch('/api/people/rank-status?event=' + encodeURIComponent(peopleEventId))
    .then(r => r.json())
    .then(d => {
      const body = document.getElementById('people-body');
      if (!body) return;
      if (d.status === 'running') {
        const banner = document.getElementById('rank-banner');
        if (!banner) {
          const b = document.createElement('p');
          b.id = 'rank-banner';
          b.className = 'text-xs text-indigo-400 text-center mb-3';
          b.textContent = 'Ranking with Claude in background…';
          body.prepend(b);
        }
        setTimeout(pollRankStatus, 4000);
      } else if (d.status === 'done') {
        fetch('/api/people?event=' + encodeURIComponent(peopleEventId))
          .then(r => r.json()).then(renderPeople);
      }
    });
}

function profileScore(p) {
  return [p.linkedin, p.twitter, p.website, p.github, p.bio].filter(Boolean).length;
}

function renderPeople(people) {
  const body = document.getElementById('people-body');
  if (!people || !people.length) {
    body.innerHTML = `<div class="text-center py-8">
      <p class="text-gray-500 text-sm mb-4">No people collected yet for this event.</p>
      <button onclick="findPeople()" class="ppl-btn px-4 py-2">✨ Find people</button></div>`;
    return;
  }
  const anyRanked = people.some(p => p.score != null);
  if (!anyRanked) {
    // pre-rank by profile completeness while Claude works
    people = [...people].sort((a, b) => profileScore(b) - profileScore(a));
  }
  const linkIcon = (url, label) => url ? `<a href="${esc(url)}" target="_blank" class="text-indigo-400 hover:text-indigo-300 text-xs mr-2">${label}</a>` : '';
  const rows = people.map(p => {
    const s = p.score;
    const sc = s == null ? '' : `<span class="score ${s>=7?'s-hi':s>=5?'s-mid':'s-lo'}">${s}/10</span>`;
    const tag = `<span class="text-xs text-gray-500">${p.role === 'host' ? 'HOST' : 'guest'}</span>`;
    const face = p.face_url
      ? `<img src="${esc(p.face_url)}" loading="lazy" class="w-12 h-12 rounded-full object-cover flex-shrink-0 bg-gray-800" onerror="this.style.visibility='hidden'">`
      : `<div class="w-12 h-12 rounded-full flex-shrink-0 bg-gray-800 flex items-center justify-center text-gray-500 text-sm">${esc((p.name||'?').slice(0,1))}</div>`;
    return `
      <div class="flex gap-3 py-3 border-b border-gray-800">
        ${face}
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 flex-wrap">
            ${sc}<span class="font-medium text-white">${esc(p.name)}</span>${tag}
          </div>
          ${p.current_role ? `<p class="text-xs text-gray-400 mt-0.5">${esc(p.current_role)}</p>` : ''}
          ${p.reason ? `<p class="text-sm text-gray-300 mt-1">${esc(p.reason)}</p>` : ''}
          ${p.icebreaker ? `<p class="ice text-sm text-gray-300 mt-2 pl-2 py-1 rounded">💬 ${esc(p.icebreaker)}</p>` : ''}
          <div class="mt-2">${linkIcon(p.github,'GitHub')}${linkIcon(p.linkedin,'LinkedIn')}${linkIcon(p.twitter,'Twitter')}${linkIcon(p.website,'Site')}</div>
        </div>
      </div>`;
  }).join('');
  const ranked = people.some(p => p.score != null);
  body.innerHTML = rows +
    `<div class="text-center pt-4 flex gap-2 justify-center flex-wrap">
      <button onclick="findPeople()" class="ppl-btn px-4 py-2">↻ ${ranked ? 'Re-run' : 'Rank with Claude'}</button>
      <button id="face-index-btn" onclick="buildFaceIndex()" class="ppl-btn px-4 py-2">📷 Build face index</button>
    </div>`;
  checkFaceIndexStatus();
}

function checkFaceIndexStatus() {
  if (!peopleEventId) return;
  fetch('/api/face/prep-status?event=' + encodeURIComponent(peopleEventId))
    .then(r => r.json())
    .then(d => {
      const btn = document.getElementById('face-index-btn');
      if (!btn) return;
      if (d.status === 'running') {
        btn.textContent = '⏳ Indexing faces…';
        btn.disabled = true;
        setTimeout(checkFaceIndexStatus, 3000);
      } else if (d.status && d.status.startsWith('done:')) {
        btn.textContent = `✓ ${d.status.replace('done:', '')} faces indexed`;
        btn.disabled = true;
      } else if (d.status && d.status.startsWith('error:')) {
        btn.textContent = '✗ Index failed';
        btn.title = d.status;
      }
    });
}

function buildFaceIndex() {
  const btn = document.getElementById('face-index-btn');
  btn.textContent = '⏳ Indexing faces…';
  btn.disabled = true;
  fetch('/api/face/prep?event=' + encodeURIComponent(peopleEventId), {method: 'POST'})
    .then(() => { setTimeout(checkFaceIndexStatus, 3000); });
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closePeople(); });

async function lumaSearch() {
  const q = document.getElementById('search').value.trim();
  const container = document.getElementById('luma-search-results');
  if (!q) { container.innerHTML = ''; return; }
  container.innerHTML = '<p class="text-gray-500 text-sm py-4">Searching Luma...</p>';
  const res = await fetch('/api/luma-search?q=' + encodeURIComponent(q));
  document.getElementById('subtitle').textContent = `Luma: "${q}"`;
  const data = await res.json();
  if (data.error) { container.innerHTML = `<p class="text-red-400 text-sm">${data.error}</p>`; return; }
  if (!data.length) { container.innerHTML = '<p class="text-gray-500 text-sm">No results.</p>'; return; }

  const regMap = {approved: ['badge-approved','✓ Going'], waitlist: ['badge-waitlist','⏳ Waitlisted'], pending_approval: ['badge-pending','⏳ Pending'], invited: ['badge-invited','✉ Invited'], declined: ['badge-declined', '✗ Declined']};
  const availMap = {'sold-out': 'Sold out', 'waitlist': 'Waitlist open'};

  container.innerHTML = '<div class="space-y-3">' + data.map(e => {
    const regBadge = e.registration_status && regMap[e.registration_status]
      ? `<span class="badge ${regMap[e.registration_status][0]}">${regMap[e.registration_status][1]}</span>` : '';
    const availBadge = !e.registration_status && availMap[e.registration_availability]
      ? `<span class="badge" style="background:#2a1a1a;color:#f87171">${availMap[e.registration_availability]}</span>` : '';
    const cal = e.calendar_name ? `<span class="text-gray-600 text-xs">${e.calendar_name}</span>` : '';
    const venue = e.location ? `· ${e.location}` : '';
    const desc = e.description ? `<p class="text-gray-500 text-sm mt-1 line-clamp-2">${e.description}</p>` : '';
    return `
      <div class="card rounded-xl p-4 transition-all duration-150">
        <div class="flex items-start justify-between gap-3">
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 flex-wrap mb-1">
              <span class="badge badge-luma">LUMA</span>${regBadge}${availBadge}${cal}
            </div>
            <h3 class="font-medium text-white leading-snug">${e.name}</h3>
            <p class="text-sm text-gray-400 mt-1">${e.start_pt} ${venue}</p>
            ${desc}
            ${e.url ? `<a href="${e.url}" target="_blank" class="text-xs text-indigo-400 hover:text-indigo-300 mt-2 inline-block">Open →</a>` : ''}
          </div>
          ${e.image_url ? `<img src="${e.image_url}" loading="lazy" class="w-20 h-20 rounded-lg object-cover flex-shrink-0" onerror="this.style.display='none'">` : ''}
        </div>
      </div>`;
  }).join('') + '</div>';
}
</script>
</body>
</html>"""


@app.route("/api/face/prep", methods=["POST"])
def api_face_prep():
    """Trigger background face indexing for an event."""
    event_id = request.args.get("event", "")
    if not event_id:
        return jsonify({"error": "missing event"}), 400
    if _face_prep_status.get(event_id) == "running":
        return jsonify({"status": "running"})
    _face_prep_status[event_id] = "running"
    # evict cached index so next search reloads the fresh npz
    _face_indexes.pop(event_id, None)
    def _prep():
        try:
            from face_prep import prep_event
            result = prep_event(event_id)
            _face_indexes.pop(event_id, None)  # force reload
            _face_prep_status[event_id] = f"done:{result['embedded']}"
        except Exception as e:
            _face_prep_status[event_id] = f"error:{e}"
    threading.Thread(target=_prep, daemon=True).start()
    return jsonify({"status": "running"})


@app.route("/api/face/prep-status")
def api_face_prep_status():
    event_id = request.args.get("event", "")
    import pathlib, numpy as np
    npz = pathlib.Path("faces") / f"{event_id}.npz"
    status = _face_prep_status.get(event_id)
    if status:
        return jsonify({"status": status})
    if npz.exists():
        try:
            count = len(np.load(npz, allow_pickle=True)["embeddings"])
            return jsonify({"status": f"done:{count}"})
        except Exception:
            pass
    return jsonify({"status": "idle"})


@app.route("/api/face/events")
def api_face_events():
    """Return list of events that have a pre-built face index (.npz)."""
    import pathlib
    faces_dir = pathlib.Path("faces")
    if not faces_dir.exists():
        return jsonify([])
    events = []
    with get_conn() as conn:
        for npz in sorted(faces_dir.glob("*.npz")):
            eid = npz.stem
            row = conn.execute(
                "SELECT name FROM events WHERE external_id = ? LIMIT 1", (eid,)
            ).fetchone()
            count = 0
            try:
                import numpy as np
                data = np.load(npz, allow_pickle=True)
                count = len(data["embeddings"])
            except Exception:
                pass
            events.append({"id": eid, "name": row[0] if row else eid, "count": count})
    return jsonify(events)


@app.route("/api/face/search", methods=["POST"])
def api_face_search():
    """Accept a JPEG image, find the closest face in the event index."""
    import numpy as np, cv2
    event_id = request.args.get("event", "")
    threshold = float(request.args.get("threshold", "0.35"))
    if not event_id:
        return jsonify({"error": "missing event param"}), 400

    embeddings, meta = _get_face_index(event_id)
    if embeddings is None:
        return jsonify({"error": f"No face index for {event_id} — run face_prep.py first"}), 404

    img_bytes = request.get_data()
    if not img_bytes:
        return jsonify({"error": "no image data"}), 400

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    try:
        app_face = _get_face_app()
    except Exception as e:
        return jsonify({"error": f"InsightFace load failed: {e}"}), 500

    faces = app_face.get(frame)
    if not faces:
        return jsonify({"match": None, "sim": 0, "reason": "no face detected"})

    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.normed_embedding
    emb = emb / np.linalg.norm(emb)

    sims = embeddings @ emb
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim < threshold:
        return jsonify({"match": None, "sim": round(best_sim, 3), "reason": "no match above threshold"})

    return jsonify({"match": meta[best_idx], "sim": round(best_sim, 3)})


FACE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>Face Search</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #000; color: #e2e2f0; font-family: system-ui, sans-serif;
         height: 100dvh; overflow: hidden; display: flex; flex-direction: column; }

  /* full-screen camera */
  #cam-wrap { position: relative; flex: 1; overflow: hidden; }
  #video { width: 100%; height: 100%; object-fit: cover; }
  #canvas { display: none; }

  /* top overlay bar */
  #topbar { position: absolute; top: 0; left: 0; right: 0;
            display: flex; align-items: center; gap: 8px; padding: 12px 14px;
            padding-top: max(12px, env(safe-area-inset-top));
            background: linear-gradient(to bottom, rgba(0,0,0,.6), transparent); }
  #topbar a { color: #fff; text-decoration: none; font-size: 14px; white-space: nowrap; }
  #event-sel { flex: 1; background: rgba(0,0,0,.5); color: #fff; border: 1px solid rgba(255,255,255,.25);
               border-radius: 8px; padding: 6px 10px; font-size: 13px; backdrop-filter: blur(6px); }

  /* status pill */
  #status { position: absolute; top: 60px; left: 50%; transform: translateX(-50%);
            background: rgba(0,0,0,.65); color: #e2e2f0; font-size: 13px;
            padding: 4px 14px; border-radius: 99px; white-space: nowrap;
            pointer-events: none; backdrop-filter: blur(4px); }

  /* bottom controls */
  #controls { position: absolute; bottom: 0; left: 0; right: 0;
              display: flex; align-items: center; justify-content: center; gap: 32px;
              padding: 20px 32px; padding-bottom: max(20px, env(safe-area-inset-bottom));
              background: linear-gradient(to top, rgba(0,0,0,.55), transparent); }

  #snap-btn { width: 72px; height: 72px; border-radius: 50%; border: 4px solid #fff;
              background: rgba(255,255,255,.18); backdrop-filter: blur(6px);
              cursor: pointer; transition: background .12s; flex-shrink: 0; }
  #snap-btn:active { background: rgba(255,255,255,.4); }
  #snap-btn::after { content: ''; display: block; width: 56px; height: 56px;
                     border-radius: 50%; background: #fff; margin: 4px auto; }
  #snap-btn:disabled { opacity: .4; }

  #flip-btn { width: 44px; height: 44px; border-radius: 50%; border: 2px solid rgba(255,255,255,.5);
              background: rgba(255,255,255,.12); backdrop-filter: blur(6px);
              cursor: pointer; font-size: 20px; display: flex; align-items: center;
              justify-content: center; transition: background .12s; }
  #flip-btn:active { background: rgba(255,255,255,.3); }

  /* result bottom sheet */
  #sheet { position: absolute; bottom: 0; left: 0; right: 0;
           background: #0f0f1a; border-radius: 20px 20px 0 0;
           border-top: 1px solid #2a2a40;
           transform: translateY(100%); transition: transform .3s ease;
           max-height: 70dvh; overflow-y: auto;
           padding-bottom: max(20px, env(safe-area-inset-bottom)); }
  #sheet.open { transform: translateY(0); }
  #sheet-handle { width: 36px; height: 4px; background: #3a3a55; border-radius: 2px;
                  margin: 10px auto 14px; }
  #sheet-body { padding: 0 16px 8px; }

  .name { font-size: 19px; font-weight: 700; }
  .sim  { font-size: 12px; color: #6b7280; margin-left: 6px; }
  .role { font-size: 11px; color: #6366f1; text-transform: uppercase; letter-spacing: .05em; margin-top: 3px; }
  .bio  { font-size: 14px; color: #a0a0b8; margin-top: 8px; line-height: 1.45; }
  .score { display: inline-block; margin-top: 10px; padding: 3px 10px; border-radius: 99px; font-size: 13px; font-weight: 600; }
  .s-hi  { background: #14532d; color: #4ade80; }
  .s-mid { background: #422006; color: #fbbf24; }
  .s-lo  { background: #450a0a; color: #f87171; }
  .links { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
  .links a { font-size: 13px; color: #818cf8; text-decoration: none;
             background: #1e1e38; padding: 5px 12px; border-radius: 8px; }
  .ice { margin-top: 12px; font-size: 13px; color: #a3e635; font-style: italic;
         border-left: 2px solid #4d7c0f; padding-left: 10px; line-height: 1.5; }
  .unknown { color: #6b7280; font-size: 15px; text-align: center; padding: 24px; }
</style>
</head>
<body>
<div id="cam-wrap">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas"></canvas>

  <div id="topbar">
    <a href="/">←</a>
    <select id="event-sel" onchange="loadEvent()"><option value="">— pick event —</option></select>
  </div>

  <div id="status">Starting camera…</div>

  <div id="controls">
    <div style="width:44px"></div><!-- spacer -->
    <button id="snap-btn" onclick="snap()" disabled></button>
    <button id="flip-btn" onclick="flipCamera()" title="Flip camera">🔄</button>
  </div>

  <div id="sheet" onclick="event.stopPropagation()">
    <div id="sheet-handle" onclick="closeSheet()"></div>
    <div id="sheet-body"></div>
  </div>
</div>

<script>
let currentEvent = null;
let facingMode = 'environment';
let stream = null;
let mirrored = false;

async function init() {
  const r = await fetch('/api/face/events');
  const events = await r.json();
  const sel = document.getElementById('event-sel');
  events.forEach(e => {
    const opt = document.createElement('option');
    opt.value = e.id;
    opt.textContent = `${e.name} (${e.count})`;
    sel.appendChild(opt);
  });
  if (events.length === 1) { sel.value = events[0].id; loadEvent(); }
  startCamera();
}

function loadEvent() {
  currentEvent = document.getElementById('event-sel').value || null;
  closeSheet();
}

async function startCamera(facing) {
  facing = facing || facingMode;
  if (stream) { stream.getTracks().forEach(t => t.stop()); }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    const video = document.getElementById('video');
    video.srcObject = stream;
    await video.play();
    // mirror front cam, don't mirror rear
    mirrored = facing === 'user';
    video.style.transform = mirrored ? 'scaleX(-1)' : '';
    document.getElementById('status').textContent = 'Tap ◉ to identify';
    document.getElementById('snap-btn').disabled = false;
  } catch(e) {
    document.getElementById('status').textContent = 'Camera error: ' + e.message;
  }
}

async function flipCamera() {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  await startCamera(facingMode);
}

function closeSheet() {
  document.getElementById('sheet').classList.remove('open');
}

// tap outside sheet to close
document.getElementById('cam-wrap').addEventListener('click', e => {
  if (!e.target.closest('#sheet') && !e.target.closest('#snap-btn') && !e.target.closest('#flip-btn'))
    closeSheet();
});

async function snap() {
  if (!currentEvent) { alert('Pick an event first'); return; }
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  if (mirrored) { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, 0, 0);

  document.getElementById('status').textContent = 'Identifying…';
  document.getElementById('snap-btn').disabled = true;

  canvas.toBlob(async blob => {
    try {
      const buf = await blob.arrayBuffer();
      const r = await fetch(`/api/face/search?event=${encodeURIComponent(currentEvent)}`, {
        method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: buf
      });
      const d = await r.json();
      renderResult(d);
    } catch(e) {
      document.getElementById('sheet-body').innerHTML = `<p class="unknown">Error: ${e}</p>`;
      document.getElementById('sheet').classList.add('open');
    } finally {
      document.getElementById('status').textContent = 'Tap ◉ to identify';
      document.getElementById('snap-btn').disabled = false;
    }
  }, 'image/jpeg', 0.9);
}

function renderResult(d) {
  const body = document.getElementById('sheet-body');
  const sheet = document.getElementById('sheet');
  if (!d.match) {
    body.innerHTML = `<p class="unknown">No match — ${d.reason || 'try again'}</p>`;
    sheet.classList.add('open');
    return;
  }
  const m = d.match;
  const score = m.score != null ? `<span class="score ${m.score>=7?'s-hi':m.score>=5?'s-mid':'s-lo'}">${m.score}/10</span>` : '';
  const links = [
    m.linkedin ? `<a href="https://linkedin.com${m.linkedin.startsWith('/')?'':'/in/'}${m.linkedin}" target="_blank">LinkedIn</a>` : '',
    m.twitter  ? `<a href="https://x.com/${m.twitter.replace('@','')}" target="_blank">Twitter</a>` : '',
    m.website  ? `<a href="${m.website}" target="_blank">Website</a>` : '',
    m.github   ? `<a href="https://github.com/${m.github}" target="_blank">GitHub</a>` : '',
  ].filter(Boolean).join('');
  const ice = m.icebreaker ? `<div class="ice">${m.icebreaker}</div>` : '';
  body.innerHTML = `
    <div><span class="name">${m.name}</span><span class="sim">${d.sim.toFixed(2)}</span></div>
    <div class="role">${m.role||''}</div>
    <div class="bio">${m.bio||''}</div>
    ${score}
    ${links ? `<div class="links">${links}</div>` : ''}
    ${ice}`;
  sheet.classList.add('open');
}

init();
</script>
</body>
</html>"""


@app.route("/face")
def face_page():
    return FACE_HTML


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(0.8, lambda: webbrowser.open("http://localhost:5050")).start()
    app.run(host="0.0.0.0", port=5050, debug=False)
