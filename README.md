# event-radar

A personal event scraper and local web UI for tracking SF tech events. Pulls from multiple Luma calendars, Cerebral Valley, and keyword searches — deduplicates across sources, tracks your registration status, and stores everything in a local SQLite database.

![UI screenshot placeholder](https://placeholder.com)

## Features

- **Multi-source scraping** — Luma followed calendars, Luma SF featured, Luma "my events", Cerebral Valley
- **Registration status** — shows whether you're Going, Pending, Waitlisted, Invited, or Declined (requires Luma cookie)
- **Cross-source deduplication** — same event from multiple calendars or sources shown once
- **Keyword search scraper** — runs configurable keywords against Luma search, stores results
- **Keyword suggestions** — NLP-based script (spaCy + TF-IDF) to surface new search terms from existing data
- **People to talk to** — for any event, collect the hosts + featured guests (with LinkedIn/Twitter), then rank them against your profile with Claude and get an icebreaker for each
- **Local web UI** — dark-themed, filters by date / source / location, live Luma search panel
- **Runs every 6 hours** via cron

## Setup

```bash
git clone https://github.com/KShivendu/event-radar
cd event-radar
pip install -r requirements.txt

# Optional: only needed for suggest_keywords.py
pip install spacy && python3 -m spacy download en_core_web_sm

cp .env.example .env
# Edit .env — add your LUMA_COOKIE (grab from browser devtools on any luma.com request)
```

## Usage

```bash
# Fetch your followed Luma calendars (run once, or whenever you follow/unfollow)
python3 sync_calendars.py

# Scrape all sources into events.db
python3 run_scraper.py

# Launch the UI at http://localhost:5050
python3 ui.py

# Suggest new search keywords from existing event data
python3 suggest_keywords.py
```

## Sources

| Source | What it covers |
|--------|---------------|
| `luma:<calendar-slug>` | All events from each Luma calendar you follow |
| `luma:sf` | Luma SF featured events (discover feed) |
| `luma:my-events` | Events you've RSVP'd / waitlisted / been invited to |
| `luma:search` | Keyword search results (see `search_keywords.json`) |
| `cerebral_valley` | Cerebral Valley SF events |

## People to talk to

For a given meetup, build a ranked list of who's worth your time — hosts and the
featured guests Luma shows publicly — each scored against your profile with an
icebreaker drafted by Claude.

```bash
# One-time: set up your profile
cp profile.example.json profile.json   # edit with your role, goals, interests

# Accepts a lu.ma URL, a slug, or an evt- id
python3 find_people.py https://lu.ma/yj5uvoei
python3 find_people.py evt-oiXR0BSLzOsOgtn --web    # let Claude web-search people first
python3 find_people.py https://lu.ma/yj5uvoei --no-rank   # just collect, no LLM
```

**Ranking backend** is auto-selected:
- If `ANTHROPIC_API_KEY` is set (e.g. in `.env`), it uses the Anthropic SDK — and `--web` lets Claude web-search people before scoring.
- Otherwise it falls back to the local **`claude` CLI** (Claude Code), which is already authenticated — so ranking works with no API key. (`--web` needs the SDK.)

How it works:
- **Collection** uses Luma's public `event/get` endpoint — **no cookie needed**. It
  returns all hosts plus up to ~10 *featured* guests (the "Going" avatars), each with
  name, short bio, and social handles. The full attendee roster is host-only and not
  exposed here.
- **Ranking** sends that list plus your `profile.json` to Claude, which scores each
  person 1–10, explains the fit, and drafts an opener. Results are stored in the
  `people` table of `events.db`.

`profile.json` and the cookie/key live in gitignored files.

## Cron

The setup script adds a cron job to run every 6 hours:

```
0 */6 * * * /usr/bin/python3 /path/to/run_scraper.py >> ~/.logs/event-radar.log 2>&1
```

## Cookie

The Luma cookie is used to:
- Fetch your followed calendars (`sync_calendars.py`)
- Retrieve registration status on events
- Access your personal "my events" feed

Without it, scraping still works for public sources but registration status won't appear. The cookie lives in `.env` (gitignored) and typically expires after a few weeks — refresh it by copying the `Cookie` header from any authenticated `luma.com` network request.

## Customizing keywords

Edit `search_keywords.json` to add/remove search terms. Run `suggest_keywords.py` to get NLP-suggested candidates from your existing event data.
