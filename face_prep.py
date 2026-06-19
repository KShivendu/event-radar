#!/usr/bin/env python3
"""Pre-compute face embeddings for all people at an event.

Downloads their Luma/GitHub avatars, extracts ArcFace embeddings via InsightFace,
saves to faces_<event_id>.npz for use by face_search.py.

Usage:
    python3 face_prep.py https://luma.com/agentsummit
    python3 face_prep.py evt-vIrxq0fLHJAb5LV
"""
import io
import json
import sys
import time
import pathlib
import sqlite3

import numpy as np
import requests
from PIL import Image

DB_PATH = pathlib.Path("events.db")
FACES_DIR = pathlib.Path("faces")
FACES_DIR.mkdir(exist_ok=True)

LUMA_DEFAULT_AVATAR = "avatars-default"
SKIP_NAMES = {"AWS Builder Loft"}  # venues / orgs, not people

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def resolve_event_id(url_or_id: str) -> str:
    from scraper.sources.luma_people import resolve_event_id as _resolve
    return _resolve(url_or_id)


def load_people(event_api_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM people WHERE event_api_id = ?", (event_api_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def download_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"    download failed: {e}")
        return None


def get_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def extract_embedding(app, img: Image.Image) -> np.ndarray | None:
    import cv2
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = app.get(arr)
    if not faces:
        return None
    # pick the largest face by bounding box area
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = face.normed_embedding
    return emb / np.linalg.norm(emb)


def prep_event(event_id: str) -> dict:
    """Embed all faces for an event and save to faces/<event_id>.npz.
    Returns {"embedded": N, "skipped": N}. Raises on failure."""
    people = load_people(event_id)
    if not people:
        raise RuntimeError(f"No people in DB for {event_id} — run 'Find people' first.")

    app = get_face_app()
    embeddings, ids, meta = [], [], []
    skipped = 0

    for p in people:
        name = p.get("name") or "?"
        face_url = p.get("face_url") or p.get("avatar_url") or ""

        if not face_url or LUMA_DEFAULT_AVATAR in face_url or name in SKIP_NAMES:
            skipped += 1
            continue

        img = download_image(face_url)
        if img is None:
            skipped += 1
            continue

        emb = extract_embedding(app, img)
        if emb is None:
            skipped += 1
            continue

        embeddings.append(emb)
        ids.append(p["person_api_id"])
        meta.append({
            "name": name,
            "role": p.get("role", ""),
            "bio": p.get("bio_short") or p.get("current_role") or "",
            "score": p.get("rank_score"),
            "reason": p.get("rank_reason") or "",
            "icebreaker": p.get("icebreaker") or "",
            "linkedin": p.get("linkedin_handle") or "",
            "twitter": p.get("twitter_handle") or "",
            "website": p.get("website") or "",
        })
        time.sleep(0.1)

    if not embeddings:
        raise RuntimeError("No faces could be embedded (all skipped or no face detected).")

    out = FACES_DIR / f"{event_id}.npz"
    np.savez(out,
             embeddings=np.array(embeddings),
             ids=np.array(ids),
             meta=np.array([json.dumps(m) for m in meta]))
    return {"embedded": len(embeddings), "skipped": skipped}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 face_prep.py <event_url_or_id>")
        sys.exit(1)

    event_id = resolve_event_id(sys.argv[1])
    print(f"Loading InsightFace model (downloads ~200MB on first run)…")
    result = prep_event(event_id)
    print(f"Saved {result['embedded']} embeddings → faces/{event_id}.npz  ({result['skipped']} skipped)")


if __name__ == "__main__":
    main()
