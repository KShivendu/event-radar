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

`find_people.py` runs the whole flow in one command — **collect → enrich (GitHub /
site / current role / face) → rank** — and stores everything on the `people` table:

```bash
# One-time: set up your profile
cp profile.example.json profile.json   # edit with your role, goals, interests

# Accepts a lu.ma URL, a slug, or an evt- id
python3 find_people.py https://lu.ma/yj5uvoei
python3 find_people.py evt-oiXR0BSLzOsOgtn --contacts-web  # web-search each person while enriching
python3 find_people.py https://lu.ma/yj5uvoei --no-rank    # collect + enrich, no LLM ranking
python3 find_people.py https://lu.ma/yj5uvoei --no-contacts --no-rank  # bare collection
```

Or from the **web UI** (`python3 ui.py`): each Luma event card has a **👥 People**
button that opens a panel of the ranked people — faces, scores, reasons, icebreakers,
and social links — with a button to run the pipeline on demand.

The stages below (`enrich_contacts.py`, ranking) can also be run standalone.

**Ranking backend** is auto-selected:
- If `ANTHROPIC_API_KEY` is set (e.g. in `.env`), it uses the Anthropic SDK — and `--web` lets Claude web-search people before scoring.
- Otherwise it falls back to the local **`claude` CLI** (Claude Code), which is already authenticated — so ranking works with no API key. (`--web` needs the SDK.)

### Contact enrichment (GitHub / site / current role)

Luma already gives us each person's LinkedIn, Twitter, Instagram, and website. The gaps
are **GitHub** and **current role**, which `enrich_contacts.py` fills, cheapest-first:

```bash
python3 enrich_contacts.py https://lu.ma/yj5uvoei      # free tier + Claude web fallback
python3 enrich_contacts.py evt-... --no-web            # free tier only
```

1. **Luma handles** → canonical URLs (free, already have them)
2. **Their own website** → fetched and scraped for a GitHub link (free)
3. **GitHub API** → search by name, accept a match only if `twitter_username` or `blog`
   matches an anchor we already trust (set `GITHUB_TOKEN`; the API needs auth)
4. **Web fallback** → Claude (web search) finds/verifies the gaps + a one-line current role,
   cross-checking against the known handles and leaving fields null when unsure

Identity is always anchored on the globally-unique Twitter handle / website Luma provides,
so matches stay high-precision (it correctly rejects same-name-different-person GitHub
accounts). Results are stored on the `people` row (`github_handle`, `current_role`,
`discovered_links`, `contact_source`).

**Faces** are resolved by a best-first cascade (the URL is stored, not downloaded):
Luma avatar → GitHub avatar (`github.com/<login>.png`, using the handle we resolved) →
Gravatar (when an email is known). Each candidate is verified to actually serve an image,
and Luma's default-placeholder avatars are detected and skipped. Stored as `face_url` /
`face_source`.

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
