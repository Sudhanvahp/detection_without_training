# Person Face Recognition — No Training Required

Real-time webcam system that identifies known people by name using **ArcFace** embeddings.
No model training needed — add reference photos, register, and run.

## How it works

1. Reference photos for each person live in `faces/<Name>/`.
2. `register_faces.py` computes a 512-dimensional **ArcFace** embedding for every photo and
   stores **all of them per person** (one row-matrix per person) in `data/faces.db` (SQLite).
3. `main.py` opens the webcam and, on a background thread, runs **YuNet** face detection +
   ArcFace recognition. A query face is matched to the **nearest stored photo** of any person
   by cosine distance; if the closest distance is within `MATCH_THRESHOLD` it's that person,
   otherwise `Unknown`.
4. A lightweight **tracker** gives each face a persistent id across frames and confirms a name
   only after it wins `CONFIRM_FRAMES` of the last `TRACK_VOTE_WINDOW` recognition passes —
   eliminating flicker and stabilising labels.

Matching uses **min-distance over per-photo embeddings** (not an averaged template), so adding
more diverse photos per person directly improves robustness.

**No GPU required.** Works on CPU; a GPU speeds up recognition (see Performance).

---

## Quick Start

### 1. Prerequisites
- **Python 3.10 or 3.11** (3.12 has limited TensorFlow support)
- A connected webcam
- ~300 MB free disk space (model weights are cached on first run)

### 2. Install dependencies
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
> The first run downloads the ArcFace weights (~137 MB) and the YuNet detector (~0.2 MB) into
> `~\.deepface\weights\` — once, cached permanently.

### 3. Add face photos
One subfolder per person inside `faces/`:
```
faces\
  ALICE\
    1.jpg
    2.jpg
  BOB\
    1.jpg
```
**Tips for best accuracy:** 3–5 photos per person (frontal + slight left/right turn, varied
lighting), well-lit and sharp, face ≥ 100×100 px, no sunglasses / extreme angles. Blurry photos
are skipped automatically at registration.

### 4. Register faces
```bat
python tools/register_faces.py
```
Or capture and register live from the webcam (best accuracy — same camera/lighting as recognition):
```bat
python tools/register_live.py --name "ALICE"
```

### 5. Start recognition
```bat
python main.py
```

| Control | Action |
|---------|--------|
| `Q` / `ESC` | Quit |
| `S` | Save an annotated snapshot (+ JSON) to `snapshots/` |
| `R` | Hot-reload the DB after registering new faces |
| `C` | Correct a wrong detection — adds the current face to the DB immediately |

---

## CLI Reference

**`register_faces.py`** — build the DB from `faces/` photos
```bat
python tools/register_faces.py                 # register every faces/ subfolder
python tools/register_faces.py --person ALICE  # only ALICE
python tools/register_faces.py --clear         # wipe DB and re-register all
python tools/register_faces.py --list          # list registered people
python tools/register_faces.py --delete BOB    # remove BOB
```

**`register_live.py`** — capture from the webcam and register immediately
```bat
python tools/register_live.py --name "ALICE"              # capture 10 photos
python tools/register_live.py --name "ALICE" --photos 15
python tools/register_live.py --name "ALICE" --camera 0
```

**`main.py`** — real-time recognition
```bat
python main.py                  # defaults from config.py
python main.py --camera 0       # built-in laptop camera
python main.py --threshold 0.35 # stricter matching
python main.py --skip 5         # slower CPU → process fewer frames
python main.py --headless       # service mode: no window, alerts + DB logging only (Ctrl+C to stop)
python main.py --health-port 8686  # + JSON heartbeat at http://127.0.0.1:8686/health
```
Inputs are validated: `--skip ≥ 1`, `--threshold` in `(0, 2]`, `--camera ≥ 0`.

**`calibrate.py`** — auto-tune the match threshold + FAR/FRR report
```bat
python tools/calibrate.py           # F1-optimal threshold + false-accept/false-reject table
python tools/calibrate.py --apply   # write it into config.py
```

**`manage_db.py`** — database tools
```bat
python tools/manage_db.py --stats                   # DB statistics
python tools/manage_db.py --prune-embeddings        # keep the most diverse photos per person
python tools/manage_db.py --max-photos 20           # cap each person to N
python tools/manage_db.py --export embeddings.json  # / --import embeddings.json
```

**`attendance.py`** — reports from the detection log
```bat
python tools/attendance.py --date 2026-07-03   # a specific day
python tools/attendance.py --days 7            # last 7 days
python tools/attendance.py --all --csv out.csv # everything, to CSV
python tools/attendance.py --sessions          # visits: arrival / departure / dwell per person
python tools/attendance.py --sessions --gap 15 # a >15-min gap starts a new visit
```

---

## Project Structure

```
PROJECT_2_PERSON_DETECTION_WITHOUT_TRAINING/
├── main.py             Real-time recognition  (the entry point)
├── start.bat           Double-click launcher for main.py
├── register.bat        Double-click launcher for tools\register_faces.py
├── requirements.txt
│
├── tools/              Command-line tools
│   ├── register_faces.py   Build/update faces.db from photos
│   ├── register_live.py    Capture from webcam and register immediately
│   ├── calibrate.py        Auto-calibrate the match threshold + FAR/FRR report
│   ├── manage_db.py        DB stats / prune / export / import
│   └── attendance.py       Attendance & visit reports from the detection log
│
├── facerec/            Core library package
│   ├── config.py       Central configuration constants
│   ├── logger.py       Rotating file + console logger
│   ├── clihelpers.py   Shared argparse validators
│   ├── embedding.py    Shared embedding extraction (CLAHE + DeepFace lock)
│   ├── database.py     SQLite layer (encrypted embeddings + detection log + meta)
│   ├── recognizer.py   Matching engine (margin check, liveness, thread-safe)
│   ├── tracker.py      Persistent per-face tracking + name confirmation
│   ├── camera.py       OpenCV webcam wrapper (DirectShow, auto-reconnect)
│   └── visualizer.py   Bounding boxes + HUD + prompts + snapshots
│
├── tests/              pytest suite for the pure logic (no camera/model needed)
├── docs/               flowchart.html architecture diagram
│
├── faces/              Reference photos (one subfolder per person)   [gitignored]
├── data/               faces.db + faces.key (auto-created)           [gitignored]
├── logs/               Rotating log files + health.json              [gitignored]
└── snapshots/          Saved frames + JSON sidecars + unknown/       [gitignored]
```
All commands are run from the project root: `python main.py` for recognition,
`python tools/<script>.py` for everything else.

---

## Configuration

Edit `facerec/config.py`. The most-used settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `"ArcFace"` | DeepFace recognition model |
| `DETECTOR_BACKEND` | `"yunet"` | Face detector (fast; `retinaface` optional) |
| `MATCH_THRESHOLD` | `0.38` | Cosine-distance cutoff (lower = stricter) |
| `CAMERA_INDEX` | `1` | Webcam index (0 = built-in, 1 = external) |
| `FRAME_SKIP` | `2` | Frames between recognition passes |
| `CONFIRM_FRAMES` / `TRACK_VOTE_WINDOW` | `3` / `5` | Confirm a name after 3 of the last 5 passes |
| `USE_CLAHE` | `False` | Contrast enhancement — off by default (see note in config.py) |
| `DETECTION_LOG_INTERVAL_S` | `60` | Log a person at most once per N seconds |
| `LOG_DETECTIONS_TO_DB` | `True` | Record detections for attendance |
| `ANTI_SPOOFING` | `True` | Per-face FasNet liveness check — photo/screen replays are labelled `SPOOF` (magenta), alerted, and never logged |
| `ENCRYPT_EMBEDDINGS` | `True` | Encrypt embedding BLOBs at rest (Fernet); key auto-generated at `data/faces.key` — **back it up** |
| `MATCH_MARGIN` | `0.05` | Accept a match only if it beats the closest *other* identity by this gap — ambiguous faces stay `Unknown` |
| `TRACK_EMB_VETO` | `0.60` | Refuse to link a detection to a track with a very different face embedding — no identity swaps when people cross paths |
| `ADMIN_PIN` | `""` | When set, gates the C-key correction and `--clear`/`--delete` behind a PIN |
| `SAVE_UNKNOWN_FACES` | `True` | Save confirmed-Unknown face crops to `snapshots/unknown/` (throttled) for later review |

> **Note on `USE_CLAHE`:** it runs on the whole frame before detection and was measured to make
> YuNet miss some faces on well-lit input, so it defaults off. It's applied identically at
> enrollment and inference; the DB records the setting and warns on mismatch. If you toggle it,
> re-register with `register_faces.py --clear`.

---

## Database Schema

```sql
-- Registered people. `embedding` is an N×512 float32 matrix (N = photo_count).
CREATE TABLE people (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE NOT NULL,
    embedding     BLOB NOT NULL,        -- N × 2048 bytes  (N photos × 512 float32)
    photo_count   INTEGER,
    registered_at TIMESTAMP,
    last_seen     TIMESTAMP
);

-- Detection log (throttled to ~one row per person per DETECTION_LOG_INTERVAL_S).
CREATE TABLE detection_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT,
    confidence REAL,
    timestamp  TIMESTAMP
);

-- Key/value metadata (model_name, detector_backend, use_clahe) for mismatch warnings.
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```
Inspect with [DB Browser for SQLite](https://sqlitebrowser.org/).

---

## Performance

| Hardware | Recognition | Display | Notes |
|----------|-------------|---------|-------|
| CPU (i5/i7) | ~5–15/s | 30 FPS | `FRAME_SKIP=2` (default) |
| NVIDIA GPU (CUDA) | ~15–30/s | 30 FPS | set `FRAME_SKIP=1` |

Recognition runs on a **background thread**, so the display stays at full webcam FPS regardless
of recognition speed. Overlays clear automatically if a person leaves or the worker stalls.

---

## Tests

```bat
venv\Scripts\activate
pytest -q
```
The suite covers the pure logic — tracker association & voting, nearest-identity matching,
the SQLite round-trip, threshold calibration, and embedding pruning — with synthetic data.
It imports no camera or model, so it runs in seconds.

---

## Security

- **Liveness (anti-spoofing)**: `ANTI_SPOOFING = True` runs a per-face FasNet check; a face that
  fails is shown as a magenta `SPOOF?` box, raises a console/log alert, and is never matched or
  written to the attendance log. FasNet needs `torch` (CPU build is fine) and downloads two small
  (~2 MB) weight files on first use. Note this is *software-only* liveness — good against casual
  photo/screen replays, not a substitute for an IR/depth camera in high-security access control.
- **Encryption at rest**: embedding BLOBs in `faces.db` are Fernet-encrypted. The key is created
  at `data/faces.key` on first write — back it up and restrict access; without it embeddings are
  unreadable (photos in `faces/` are unaffected — re-registering rebuilds the DB). Legacy
  unencrypted rows keep working and are re-encrypted the next time a person is registered.
- **Scale**: matching is one vectorised matrix multiply over all stored embeddings, so thousands
  of registered people stay fast on CPU. Re-run `python tools/calibrate.py` after adding many people —
  the optimal threshold tightens as the gallery grows.
- **False-positive hardening**: a match must beat the closest *other* identity by `MATCH_MARGIN`
  (ambiguous faces stay `Unknown`), and every registration run audits the whole gallery for
  cross-person embedding collisions and warns you which pairs are risky.
- **Key protection**: on Windows the encryption key is additionally DPAPI-wrapped, so it only
  decrypts under the enrolling Windows account — copying the `data/` folder elsewhere yields an
  unusable key. (Legacy plain keys keep working; delete `faces.key` and re-register to upgrade.)
- **Operator PIN**: set `ADMIN_PIN` in config.py to gate live corrections (C key) and
  `register_faces.py --clear`/`--delete`. It's a mistake/casual-misuse gate, not cryptography.
- **Monitoring**: a heartbeat JSON is written to `logs/health.json` every few seconds
  (status, FPS, faces on screen); `--health-port N` also serves it over HTTP for supervisors
  (systemd, NSSM, Kubernetes probes, uptime monitors).
- **Unknown evidence**: confirmed unknown faces are cropped to `snapshots/unknown/` (at most one
  per track per `UNKNOWN_SAVE_INTERVAL_S`) so "who was that?" is answerable after the fact.

---

## Privacy

This project stores **biometric data**: reference photos under `faces/` and face embeddings in
`data/faces.db`, both identifying real people.

- `.gitignore` excludes `faces/`, `data/`, `logs/`, `snapshots/`, and `venv/` — do **not** commit them
  (`data/` includes the encryption key `faces.key`).
- Register people only with their consent.
- To remove someone: `python tools/register_faces.py --delete "NAME"`.
- To wipe everything: `python tools/register_faces.py --clear` (also clears the version metadata).
- Detection history is auto-pruned after `LOG_RETENTION_DAYS` (default 30).

---

## Troubleshooting

**`Cannot open camera at index N`** → try `--camera 0` (built-in) or `--camera 1` (external); close other apps using the camera.

**`No face found in photo`** → use a well-lit, sharp, frontal photo, face ≥ 100×100 px. Profile/angled shots are often undetectable.

**`ModuleNotFoundError: deepface`** → `pip install -r requirements.txt` inside the venv.

**TensorFlow errors on Python 3.12** → use Python 3.10 or 3.11.

**All detections show "Unknown"** → `python tools/register_faces.py --list` to confirm registration; run `python tools/calibrate.py`; add more varied photos; or loosen with `--threshold 0.42`.

**Registration mismatch warning** → the DB was built with a different model/detector/CLAHE setting; re-register with `--clear`.

---

## Accuracy

ArcFace achieves **99.40%** on the LFW benchmark; YuNet is a fast, accurate CNN face detector
bundled with OpenCV. Both are pre-trained on millions of faces — no training required for your
use case. Accuracy in practice is dominated by **photo quality and diversity** per person.
