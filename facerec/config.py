import os

# Project root (this file lives in facerec/, one level below it) — all data
# folders (faces/, data/, logs/, snapshots/) stay at the project root.
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── DeepFace / ArcFace ───────────────────────────────────────────────────────
MODEL_NAME       = "ArcFace"      # DeepFace model backend (99.40% LFW accuracy)
DETECTOR_BACKEND = "yunet"        # yunet = neural detector, fast (~10-30ms CPU) + handles glasses/angles well
DISTANCE_METRIC  = "cosine"       # cosine | euclidean | euclidean_l2

# Cosine distance threshold — lower = stricter match.
# ArcFace + cosine: 0.40 is well-validated (~99.4% LFW).
# Tune down to 0.35 for stricter (fewer false positives),
# up to 0.45 for looser (fewer false negatives / "Unknown" labels).
MATCH_THRESHOLD = 0.38  # tightened for 60-person scale (fewer false positives)

# Second-best margin: accept a match only if the best identity beats the closest
# OTHER identity by at least this distance gap. Kills the classic false positive
# where two enrolled people sit near each other in embedding space and a face
# lands between them. 0 disables. Irrelevant with a single registered person.
MATCH_MARGIN = 0.05

# Live-path detection cap: frames whose longest side exceeds this are downscaled
# before YuNet (which degrades badly on very large frames) and the boxes are
# scaled back to original coordinates. Default 1280×720 capture never triggers it;
# this protects 4K camera configs.
MAX_DETECT_SIDE = 1920

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0   # 0 = built-in laptop cam, 1 = external webcam
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
CAMERA_FPS    = 30  # requested capture FPS (driver may pick the nearest supported value)

# Auto-reconnect if the camera is unplugged / stops delivering frames mid-run.
CAMERA_FAILURES_BEFORE_RECONNECT = 3    # consecutive bad reads before a reconnect attempt
CAMERA_RECONNECT_ATTEMPTS        = 5    # how many times to retry opening the device
CAMERA_RECONNECT_DELAY_S         = 2.0  # seconds to wait between reconnect attempts

# ── Frame skipping ───────────────────────────────────────────────────────────
# DeepFace.represent() takes ~200-500 ms per frame on CPU (background thread).
# Display loop always runs at full FPS; recognition fires every FRAME_SKIP frames.
# At 30 FPS display with FRAME_SKIP=3 → ~10 recognition passes/second.
# Increase to 5-10 on slow machines; decrease to 1 if GPU is available.
FRAME_SKIP = 2   # yunet is fast (~10-30ms), every 2nd frame is fine on CPU

# ── Paths ─────────────────────────────────────────────────────────────────────
# Tip: add 3-5 photos per person for better accuracy across lighting/angles.
FACES_DIR    = os.path.join(_BASE, "faces")
DB_PATH      = os.path.join(_BASE, "data", "faces.db")
LOG_DIR      = os.path.join(_BASE, "logs")
SNAPSHOT_DIR = os.path.join(_BASE, "snapshots")

# ── Display ───────────────────────────────────────────────────────────────────
WINDOW_TITLE  = "Face Recognition  |  Q=quit  S=snapshot  R=reload DB  C=correct"
VERBOSE       = True
SHOW_TRACK_ID = True   # prefix each box label with its persistent track id (#3)
FPS_EMA_ALPHA = 0.05   # smoothing for the on-screen FPS counter (0=frozen, 1=no smoothing)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL        = "INFO"
LOG_MAX_BYTES    = 5 * 1024 * 1024  # 5 MB per log file
LOG_BACKUP_COUNT = 3                # keep 3 rotated files

# ── Detection audit log (SQLite) ──────────────────────────────────────────────
LOG_DETECTIONS_TO_DB   = True  # write recognized faces to the detection_log table
DETECTION_LOG_INTERVAL_S = 60  # log a given person at most once per N seconds. Turns the log
                               # into meaningful "visits" (not ~10 rows/sec) and stops DB churn.

# ── Face quality gates ─────────────────────────────────────────────────────────
MIN_FACE_PIXELS    = 60    # faces narrower/shorter than N px are skipped (too small)
MIN_FACE_CONFIDENCE = 0.50 # skip detections below this detector score; guards against the
                           # whole-frame "phantom face" that enforce_detection=False can return
BLUR_THRESHOLD     = 15.0  # Laplacian variance below this = blurry frame, skip recognition.
                           # Purpose: skip MOTION-blurred frames, not to demand a sharp camera —
                           # soft webcams sit at ~20-25 at rest (measured), so 60 rejected every
                           # frame. If you get false matches on a sharp camera, raise it back.
REG_BLUR_THRESHOLD = 25.0  # blur gate at registration (permissive — yunet handles some blur)

# ── Multi-frame confirmation ───────────────────────────────────────────────────
# A name is shown for a tracked face only after it wins CONFIRM_FRAMES of the last
# TRACK_VOTE_WINDOW recognition passes (see tracking below). Smooths single-frame
# flickers and name flip-flops. Confirmation latency ≈ CONFIRM_FRAMES × FRAME_SKIP frames.
CONFIRM_FRAMES = 3

# ── Multi-face tracking ────────────────────────────────────────────────────────
# Each detected face gets a persistent integer id across recognition passes, and
# name confirmation is per-track (not per-name). The tracker runs on the worker
# thread at recognition cadence (every FRAME_SKIP frames) — no extra latency.
TRACK_IOU_THRESHOLD   = 0.30  # min box overlap to link a detection to an existing track (SORT default)
TRACK_CENTROID_FACTOR = 1.50  # fallback link when IoU=0: gap ≤ factor × mean face width (scale-adaptive)
TRACK_MAX_MISSES      = 5     # keep coasting a track this many passes with no match before dropping it
TRACK_VOTE_WINDOW     = 5     # rolling window of recent name votes per track (needs CONFIRM_FRAMES to agree)
TRACK_BBOX_SMOOTHING  = 0.50  # EMA toward the newest box (1.0 = snap/no smoothing, lower = smoother/laggier)
TRACK_EMB_VETO        = 0.60  # refuse to link a detection to a track whose last face embedding is
                              # further than this cosine distance — prevents identity swaps when two
                              # faces cross paths. 0 disables (geometry-only association).

# ── Face-box expiry ────────────────────────────────────────────────────────────
# Safety net for a stalled/crashed worker: if no fresh result arrives within this
# window, overlays are cleared instead of freezing on screen. Keep it well above the
# worst-case single-pass latency (raise it if ANTI_SPOOFING is on or the CPU is slow).
BOX_EXPIRY_MS = 1500

# ── Anti-spoofing ─────────────────────────────────────────────────────────────
# Per-face FasNet liveness check (photo / screen replay attacks). A face that
# fails is labelled SPOOF (magenta box), alerted, and never matched or logged to
# attendance. Adds ~50-150 ms per face on CPU. Enable for any access-control or
# security-sensitive deployment; note FasNet is software-only liveness — for high
# security use an IR/depth camera as well.
ANTI_SPOOFING = True

# A face that passed/failed liveness recently is not re-checked every pass —
# results are cached per screen region for this many seconds (FasNet costs
# ~50-150 ms/face on CPU; a stationary face would otherwise pay it every pass).
LIVENESS_RECHECK_S = 2.0

# ── Embeddings at rest ─────────────────────────────────────────────────────────
# Encrypt embedding BLOBs in faces.db (Fernet/AES-128-CBC+HMAC via `cryptography`).
# The key is auto-generated at KEY_PATH on first write — BACK IT UP and restrict
# file access; without it registered embeddings cannot be read (photos in faces/
# are unaffected; re-registering rebuilds the DB). Existing unencrypted rows stay
# readable; they are re-encrypted the next time each person is (re-)registered.
ENCRYPT_EMBEDDINGS = True
KEY_PATH = os.path.join(_BASE, "data", "faces.key")

# On Windows, additionally wrap the key file with DPAPI (CryptProtectData) so it
# only decrypts under this Windows user account — copying the data/ folder to
# another machine yields an unusable key. No effect on other platforms.
KEY_USE_DPAPI = True

# Operator PIN for destructive / DB-mutating actions: the C-key correction in
# main.py and register_faces.py --clear / --delete. Empty string disables.
# NOT cryptographic security — an honest-mistake and casual-misuse gate.
ADMIN_PIN = ""

# ── Unknown-face evidence ──────────────────────────────────────────────────────
# When a confirmed Unknown is on screen, save their face crop to
# snapshots/unknown/ (at most once per interval) so you can review later WHO the
# unknown person was. Crops are biometric data — covered by the same privacy rules.
SAVE_UNKNOWN_FACES     = True
UNKNOWN_SAVE_INTERVAL_S = 30
UNKNOWN_DIR            = os.path.join(SNAPSHOT_DIR, "unknown")

# ── Health / monitoring (headless service mode) ────────────────────────────────
# main.py writes a heartbeat JSON (timestamp, fps, faces, camera state) every
# HEALTH_INTERVAL_S to HEALTH_FILE; `python main.py --health-port 8686` also
# serves it as JSON on http://127.0.0.1:8686/health for supervisors to probe.
HEALTH_FILE       = os.path.join(LOG_DIR, "health.json")
HEALTH_INTERVAL_S = 5.0

# ── Alerting ──────────────────────────────────────────────────────────────────
ALERT_KNOWN_PERSON   = True  # print to console when a known person is first detected
ALERT_UNKNOWN_PERSON = True  # print to console when an unknown face persists
UNKNOWN_ALERT_FRAMES = 20    # alert after N consecutive unknown-face recognition passes

# ── Detection log retention ────────────────────────────────────────────────────
LOG_RETENTION_DAYS = 30  # auto-delete detection_log rows older than N days (0 = keep all)

# ── Attendance report ──────────────────────────────────────────────────────────
ATTENDANCE_REPORT_ON_EXIT = True  # print session summary table when main.py exits

# ── Embedding management ───────────────────────────────────────────────────────
MAX_PHOTOS_PER_PERSON = 20  # manage_db.py --prune-embeddings caps each person to this

# ── Enrollment preprocessing ───────────────────────────────────────────────────
# YuNet fails to detect faces on very large images (e.g. 4032×3024 phone photos —
# it returns a whole-frame phantom with confidence 0). Photos whose longest side
# exceeds this are downscaled before detection at enrollment. Live webcam frames
# (FRAME_WIDTH×FRAME_HEIGHT) are unaffected.
MAX_ENROLL_SIDE = 1600

# ── Frame preprocessing ────────────────────────────────────────────────────────
# CLAHE (adaptive contrast) is applied IDENTICALLY at enrollment and inference so
# embeddings stay comparable (the DB records the setting and warns on mismatch;
# re-register with `register_faces.py --clear` if you change it).
# Default OFF: CLAHE runs on the WHOLE frame before YuNet detection, and was measured
# to make YuNet miss faces it otherwise detects — cutting enrollment yield and hurting
# recognition on well-lit input. Enable ONLY for consistently dark / backlit cameras.
USE_CLAHE          = False   # CLAHE contrast enhancement (see note above — off by default)
CLAHE_CLIP_LIMIT   = 2.0     # higher = stronger local contrast (and more noise amplification)
CLAHE_TILE_GRID    = (8, 8)  # CLAHE tile grid size
