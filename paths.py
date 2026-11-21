"""Where event-radar keeps its data.

Locally everything sits next to the code (the default). In Docker, set EVENT_RADAR_DATA=/data so the
web and scraper containers share one mounted folder that survives rebuilds and redeploys.
"""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("EVENT_RADAR_DATA", CODE_DIR))

DB_PATH = DATA_DIR / "events.db"
FACES_DIR = DATA_DIR / "faces"
CALENDARS_FILE = DATA_DIR / "luma_calendars.json"
PROFILE_FILE = DATA_DIR / "profile.json"
ENV_FILE = DATA_DIR / ".env" if (DATA_DIR / ".env").exists() else CODE_DIR / ".env"
