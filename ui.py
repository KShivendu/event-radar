#!/usr/bin/env python3
"""Local web UI for browsing SF events from events.db"""
import json
from datetime import datetime, timezone
import zoneinfo
from flask import Flask, request, jsonify, render_template_string
from scraper.db import get_conn

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
  input, select { background: #1a1a24; border: 1px solid #2a2a38; color: #e8e8f0; }
  input:focus, select:focus { outline: none; border-color: #5b5bff; }
  .date-btn { background: #1a1a24; border: 1px solid #2a2a38; color: #9090b0; cursor: pointer; }
  .date-btn.active { background: #2a2a4a; border-color: #5b5bff; color: #a0a0ff; }
  .date-btn:hover { border-color: #5b5bff; }
  ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0f0f13; }
  ::-webkit-scrollbar-thumb { background: #2a2a38; border-radius: 3px; }
  .group-header { border-left: 3px solid #5b5bff; }
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
  const sourceLabels = {'luma:my-events': 'MY EVENTS', 'luma:sf': 'SF FEATURED','luma:genai-sf': 'GENAI SF', 'luma:ai-sf': 'AI SF', 'cerebral_valley': 'CEREBRAL VALLEY'};
  const calLabel = (e.calendars || [e.source]).map(s => sourceLabels[s] || s).join(' · ');
  const sourceBadge = e.source.startsWith('luma')
    ? `<span class="badge badge-luma">${calLabel}</span>`
    : `<span class="badge badge-cv">CEREBRAL VALLEY</span>`;
  const regMap = {approved: ['badge-approved','✓ Going'], waitlist: ['badge-waitlist','⏳ Waitlisted'], pending_approval: ['badge-pending','⏳ Pending'], invited: ['badge-invited','✉ Invited']};
  const regBadge = e.registration_status && regMap[e.registration_status]
    ? `<span class="badge ${regMap[e.registration_status][0]}">${regMap[e.registration_status][1]}</span>` : '';
  const typeBadge = e.event_type ? `<span class="badge" style="background:#1a2a1a;color:#86efac">${e.event_type.toUpperCase()}</span>` : '';
  const venue = e.venue ? `<span class="text-gray-500">·</span> ${e.venue}` : '';
  const desc = e.description ? `<p class="text-gray-500 text-sm mt-2 line-clamp-2">${e.description}</p>` : '';
  const link = e.url ? `<a href="${e.url}" target="_blank" class="text-xs text-indigo-400 hover:text-indigo-300 mt-2 inline-block">Open →</a>` : '';

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
          ${link}
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
  if (!live) { document.getElementById('luma-search-results').innerHTML = ''; loadEvents(); }
}


loadEvents();

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

  const regMap = {approved: ['badge-approved','✓ Going'], waitlist: ['badge-waitlist','⏳ Waitlisted'], pending_approval: ['badge-pending','⏳ Pending'], invited: ['badge-invited','✉ Invited']};
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
    app.run(port=5050, debug=False)
