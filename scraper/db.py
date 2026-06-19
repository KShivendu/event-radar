import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "events.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                start_datetime TEXT,
                end_datetime TEXT,
                url TEXT,
                location TEXT,
                venue TEXT,
                event_type TEXT,
                image_url TEXT,
                registration_status TEXT,
                raw_json TEXT,
                first_seen_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(source, external_id)
            )
        """)
        # migrations
        for col in ["registration_status TEXT", "city TEXT", "location_type TEXT"]:
            try:
                conn.execute(f"ALTER TABLE events ADD COLUMN {col}")
            except Exception:
                pass
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_start ON events(start_datetime)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON events(source)
        """)
        # People at an event — hosts and publicly-visible (featured) guests.
        # Luma exposes up to ~10 featured guests per event without auth, plus all hosts.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_api_id TEXT NOT NULL,
                event_url TEXT,
                event_name TEXT,
                person_api_id TEXT NOT NULL,
                role TEXT,                    -- 'host' | 'guest'
                name TEXT,
                first_name TEXT,
                last_name TEXT,
                bio_short TEXT,
                avatar_url TEXT,
                username TEXT,
                website TEXT,
                twitter_handle TEXT,
                linkedin_handle TEXT,
                instagram_handle TEXT,
                is_verified INTEGER,
                last_online_at TEXT,
                rank_score REAL,              -- LLM relevance score (filled by enrich.py)
                rank_reason TEXT,
                icebreaker TEXT,
                raw_json TEXT,
                first_seen_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(event_api_id, person_api_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_people_event ON people(event_api_id)
        """)
        # contact-enrichment columns (filled by enrich_contacts.py)
        for col in ["github_handle TEXT", "github_company TEXT", "current_role TEXT",
                    "discovered_links TEXT", "contact_source TEXT", "contact_enriched_at TEXT",
                    "face_url TEXT", "face_source TEXT"]:
            try:
                conn.execute(f"ALTER TABLE people ADD COLUMN {col}")
            except Exception:
                pass


def save_contact(event_api_id: str, person_api_id: str, fields: dict):
    """Update contact-enrichment columns for a person. Only writes keys present in `fields`."""
    allowed = {"github_handle", "github_company", "current_role", "discovered_links",
               "website", "linkedin_handle", "twitter_handle", "contact_source",
               "face_url", "face_source"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    assignments = ", ".join(f"{k} = :{k}" for k in sets)
    sets.update({"eid": event_api_id, "pid": person_api_id})
    with get_conn() as conn:
        conn.execute(
            f"UPDATE people SET {assignments}, contact_enriched_at = datetime('now') "
            f"WHERE event_api_id = :eid AND person_api_id = :pid",
            sets,
        )


def upsert_person(fields: dict):
    """Insert or update a person row. Preserves LLM ranking columns when re-scraping
    (the scraper passes them as None; COALESCE keeps any previously-stored value)."""
    cols = ["event_api_id", "event_url", "event_name", "person_api_id", "role",
            "name", "first_name", "last_name", "bio_short", "avatar_url", "username",
            "website", "twitter_handle", "linkedin_handle", "instagram_handle",
            "is_verified", "last_online_at", "raw_json"]
    row = {c: fields.get(c) for c in cols}
    with get_conn() as conn:
        conn.execute(f"""
            INSERT INTO people ({", ".join(cols)})
            VALUES ({", ".join(":" + c for c in cols)})
            ON CONFLICT(event_api_id, person_api_id) DO UPDATE SET
                role = excluded.role,
                name = excluded.name,
                first_name = excluded.first_name,
                last_name = excluded.last_name,
                bio_short = excluded.bio_short,
                avatar_url = excluded.avatar_url,
                username = excluded.username,
                website = excluded.website,
                twitter_handle = excluded.twitter_handle,
                linkedin_handle = excluded.linkedin_handle,
                instagram_handle = excluded.instagram_handle,
                is_verified = excluded.is_verified,
                last_online_at = excluded.last_online_at,
                raw_json = excluded.raw_json,
                updated_at = datetime('now')
        """, row)


def save_ranking(event_api_id: str, person_api_id: str, score: float, reason: str, icebreaker: str):
    with get_conn() as conn:
        conn.execute("""
            UPDATE people SET rank_score = ?, rank_reason = ?, icebreaker = ?, updated_at = datetime('now')
            WHERE event_api_id = ? AND person_api_id = ?
        """, (score, reason, icebreaker, event_api_id, person_api_id))


def get_event_people(event_api_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM people WHERE event_api_id = ? ORDER BY rank_score DESC NULLS LAST, role, name",
            (event_api_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_event(source: str, external_id: str, fields: dict, raw: dict):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO events (source, external_id, name, description, start_datetime,
                end_datetime, url, location, venue, event_type, image_url,
                registration_status, city, location_type, raw_json)
            VALUES (:source, :external_id, :name, :description, :start_datetime,
                :end_datetime, :url, :location, :venue, :event_type, :image_url,
                :registration_status, :city, :location_type, :raw_json)
            ON CONFLICT(source, external_id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                start_datetime = excluded.start_datetime,
                end_datetime = excluded.end_datetime,
                url = excluded.url,
                location = excluded.location,
                venue = excluded.venue,
                event_type = excluded.event_type,
                image_url = excluded.image_url,
                registration_status = excluded.registration_status,
                city = excluded.city,
                location_type = excluded.location_type,
                raw_json = excluded.raw_json,
                updated_at = datetime('now')
        """, {
            "source": source,
            "external_id": external_id,
            **fields,
            "raw_json": json.dumps(raw),
        })
