#!/usr/bin/env python3
"""Live face search — press SPACE to identify the person in the webcam frame.

Loads pre-computed embeddings from face_prep.py, opens a webcam window,
and on SPACE captures the frame, finds the closest match, and overlays the result.

Usage:
    python3 face_search.py evt-vIrxq0fLHJAb5LV
    python3 face_search.py https://luma.com/agentsummit
    python3 face_search.py evt-...  --camera 1   # if built-in cam is index 0
"""
import json
import pathlib
import sys

import cv2
import numpy as np

FACES_DIR = pathlib.Path("faces")
MATCH_THRESHOLD = 0.35   # cosine similarity; below this = "unknown"
NO_MATCH_LABEL = "Unknown"

# colours (BGR)
C_GREEN  = (80, 220, 80)
C_YELLOW = (60, 200, 220)
C_RED    = (80, 80, 220)
C_DARK   = (20, 20, 20)
C_WHITE  = (240, 240, 240)


def resolve_event_id(url_or_id: str) -> str:
    from scraper.sources.luma_people import resolve_event_id as _resolve
    return _resolve(url_or_id)


def load_index(event_id: str):
    path = FACES_DIR / f"{event_id}.npz"
    if not path.exists():
        print(f"No embeddings found at {path}")
        print(f"Run:  python3 face_prep.py {event_id}")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]          # (N, 512)
    ids        = data["ids"].tolist()
    meta       = [json.loads(m) for m in data["meta"]]
    return embeddings, ids, meta


def get_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def search(query_emb: np.ndarray, embeddings: np.ndarray, meta: list) -> tuple[dict | None, float]:
    sims = embeddings @ query_emb          # cosine similarity (embeddings are unit-normed)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim < MATCH_THRESHOLD:
        return None, best_sim
    return meta[best_idx], best_sim


def score_color(score):
    if score is None: return C_YELLOW
    if score >= 7: return C_GREEN
    if score >= 5: return C_YELLOW
    return C_RED


def draw_result(frame: np.ndarray, person: dict | None, sim: float) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = 220
    panel_w = min(w, 500)
    overlay = frame.copy()

    # semi-transparent panel bottom-left
    cv2.rectangle(overlay, (0, h - panel_h), (panel_w, h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    x, y = 12, h - panel_h + 24
    lh = 26  # line height

    if person is None:
        cv2.putText(frame, f"Unknown  (sim={sim:.2f})", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_RED, 2)
        return frame

    # name + confidence
    name = person["name"]
    score = person.get("score")
    col = score_color(score)
    label = f"{name}  ({sim:.2f})"
    if score is not None:
        label += f"  [{score}/10]"
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 2)
    y += lh

    # role
    role = person.get("role", "")
    if role:
        cv2.putText(frame, role.upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_YELLOW, 1)
        y += lh

    # bio / current role
    bio = person.get("bio", "")
    if bio:
        bio_short = bio[:70] + ("…" if len(bio) > 70 else "")
        cv2.putText(frame, bio_short, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1)
        y += lh

    # handles
    handles = []
    if person.get("twitter"):  handles.append(f"tw:{person['twitter']}")
    if person.get("linkedin"): handles.append(f"li:{person['linkedin'].lstrip('/in/')}")
    if handles:
        cv2.putText(frame, "  ".join(handles), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_YELLOW, 1)
        y += lh

    # icebreaker
    ice = person.get("icebreaker", "")
    if ice:
        ice_short = ice[:80] + ("…" if len(ice) > 80 else "")
        cv2.putText(frame, f'"{ice_short}"', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREEN, 1)

    return frame


def draw_faces(frame: np.ndarray, faces) -> np.ndarray:
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_GREEN, 2)
    return frame


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 face_search.py <event_url_or_id> [--camera N]")
        sys.exit(1)

    cam_idx = 0
    if "--camera" in sys.argv:
        cam_idx = int(sys.argv[sys.argv.index("--camera") + 1])

    event_id = resolve_event_id(sys.argv[1])
    embeddings, ids, meta = load_index(event_id)
    print(f"Loaded {len(meta)} people for {event_id}")

    print("Loading InsightFace…")
    app = get_face_app()

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_idx}")
        sys.exit(1)

    print("SPACE = identify  |  Q = quit")

    last_result = None   # (person, sim, faces) — persists until next capture
    last_faces  = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()

        # draw face boxes continuously
        faces = app.get(frame)
        draw_faces(display, faces)

        # overlay last result
        if last_result is not None:
            person, sim = last_result
            draw_result(display, person, sim)

        cv2.putText(display, "SPACE: identify   Q: quit",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow("Face Search", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if not faces:
                print("No face detected — try again")
                last_result = (None, 0.0)
                continue
            # use largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            emb = face.normed_embedding
            emb = emb / np.linalg.norm(emb)
            person, sim = search(emb, embeddings, meta)
            last_result = (person, sim)
            name = person["name"] if person else "Unknown"
            print(f"Match: {name}  (sim={sim:.3f})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
