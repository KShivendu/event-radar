#!/usr/bin/env python3
"""Local web UI for browsing SF events from events.db"""
import json
from datetime import datetime, timezone
import zoneinfo
from flask import Flask, request, jsonify, render_template_string
from scraper.db import get_conn, get_event_people

app = Flask(__name__)
PT = zoneinfo.ZoneInfo("America/Los_Angeles")


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
    """Run the full pipeline for an event: collect → enrich contacts + faces → rank.
    Web search is left off here to keep the request responsive."""
    event = request.args.get("event", "")
    if not event:
        return jsonify({"error": "missing event"}), 400
    try:
        from pipeline import find_people
        _, people = find_people(event, contacts=True, contacts_web=False, rank=True, rank_web=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify([_person_public(p) for p in people])


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
    <button onclick="loadEvents()" class="text-xs px-3 py-1.5 rounded border border-gray-700 text-gray-400 hover:border-indigo-500 hover:text-indigo-400 transition-colors">
      ↻ Refresh
    </button>
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
    .then(r => r.json())
    .then(renderPeople);
}

function closePeople() {
  document.getElementById('people-modal').classList.add('hidden');
}

function findPeople() {
  const body = document.getElementById('people-body');
  body.innerHTML = '<p class="text-gray-500 text-sm py-6 text-center">Collecting hosts + guests, enriching, and ranking with Claude… (~1 min)</p>';
  fetch('/api/people/find?event=' + encodeURIComponent(peopleEventId), {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      if (d.error) { body.innerHTML = `<p class="text-red-400 text-sm py-6 text-center">${esc(d.error)}</p>`; return; }
      renderPeople(d);
    });
}

function renderPeople(people) {
  const body = document.getElementById('people-body');
  if (!people || !people.length) {
    body.innerHTML = `<div class="text-center py-8">
      <p class="text-gray-500 text-sm mb-4">No people collected yet for this event.</p>
      <button onclick="findPeople()" class="ppl-btn px-4 py-2">✨ Find people</button></div>`;
    return;
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
    `<div class="text-center pt-4"><button onclick="findPeople()" class="ppl-btn px-4 py-2">↻ ${ranked ? 'Re-run' : 'Rank with Claude'}</button></div>`;
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


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(0.8, lambda: webbrowser.open("http://localhost:5050")).start()
    app.run(host="0.0.0.0", port=5050, debug=False)
