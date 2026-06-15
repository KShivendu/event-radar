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
