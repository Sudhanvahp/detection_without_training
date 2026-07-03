"""
main.py — Face Recognition System  |  Entry point

Detects and identifies registered people via webcam in real time.
No model training required — uses ArcFace embeddings stored in SQLite.

Usage:
  python main.py
  python main.py --camera 1         # use second camera
  python main.py --threshold 0.35   # stricter matching
  python main.py --skip 5           # process every 5th frame (slower CPU)

Controls:
  Q / ESC   — quit
  S         — save snapshot to snapshots/
  R         — hot-reload face embeddings from DB (after register_faces.py)
  C         — correct a wrong detection (adds embedding to DB immediately)
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from typing import List, Optional

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from facerec import clihelpers
from facerec import config
from facerec import database
from facerec import embedding
from facerec import visualizer
from facerec.camera import CameraCapture
from facerec.logger import setup_logger
from facerec.recognizer import FaceMatch, FaceRecognizer
from facerec.tracker import FaceTracker

_log = logging.getLogger(__name__)


# ── Background recognition worker ─────────────────────────────────────────────

class _RecogWorker:
    """
    Runs FaceRecognizer.recognize() in a background daemon thread and feeds the
    result through a per-face tracker (persistent ids + per-track name confirmation).

    - submit(frame)        : hand a new frame to the worker (drop-frame strategy)
    - latest_with_age()    : (matches, age_ms) — most recent tracked result + its age
    - request_invalidate() : ask the worker to reset tracker votes (after R / C)
    - stop()               : gracefully shut down the worker thread
    """

    def __init__(self, recognizer: FaceRecognizer) -> None:
        self._recognizer  = recognizer
        self._tracker     = FaceTracker()
        self._lock        = threading.Lock()
        self._has_work    = threading.Event()
        self._invalidate  = threading.Event()
        self._pending     : Optional[np.ndarray] = None
        self._result      : List[FaceMatch] = []
        self._result_time : float = 0.0
        self._running     = True
        self._thread      = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()
        _log.debug("Recognition worker started")

    def submit(self, frame: np.ndarray) -> None:
        """Replace pending frame with newest (drop-frame strategy)."""
        with self._lock:
            self._pending = frame.copy()
        self._has_work.set()

    def latest_with_age(self) -> tuple:
        """Return (matches, age_ms) where age_ms is ms since last recognition pass."""
        with self._lock:
            age = (time.perf_counter() - self._result_time) * 1000 if self._result_time else 99999.0
            return list(self._result), age

    def request_invalidate(self) -> None:
        """Signal the worker to clear tracker votes on its next pass (thread-safe)."""
        self._invalidate.set()

    def stop(self) -> None:
        self._running = False
        self._has_work.set()
        self._thread.join(timeout=3.0)
        _log.debug("Recognition worker stopped")

    def _loop(self) -> None:
        while self._running:
            self._has_work.wait()
            self._has_work.clear()

            if not self._running:
                break

            # Embeddings changed (R reload / C correct) — re-derive tracker labels.
            if self._invalidate.is_set():
                self._invalidate.clear()
                self._tracker.invalidate_votes()

            with self._lock:
                frame = self._pending
                self._pending = None

            if frame is None:
                continue

            try:
                raw    = self._recognizer.recognize(frame)
                result = self._tracker.update(raw)
            except Exception as exc:
                _log.error("Worker recognize() error: %s", exc, exc_info=True)
                result = []

            with self._lock:
                self._result      = result
                self._result_time = time.perf_counter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face recognition via webcam.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--camera",    type=clihelpers.nonneg_int,  default=config.CAMERA_INDEX,    metavar="N")
    parser.add_argument("--threshold", type=clihelpers.threshold,   default=config.MATCH_THRESHOLD, metavar="T")
    parser.add_argument("--skip",      type=clihelpers.positive_int, default=config.FRAME_SKIP,     metavar="N")
    parser.add_argument("--width",     type=clihelpers.positive_int, default=config.FRAME_WIDTH)
    parser.add_argument("--height",    type=clihelpers.positive_int, default=config.FRAME_HEIGHT)
    parser.add_argument("--headless",  action="store_true",
                        help="run as a service: no display window, console/DB logging only "
                             "(stop with Ctrl+C)")
    parser.add_argument("--health-port", type=clihelpers.positive_int, default=None, metavar="PORT",
                        help="serve a JSON health endpoint at http://127.0.0.1:PORT/health")
    return parser.parse_args(argv)


def _configure_tensorflow() -> None:
    try:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:
        _log.debug("TensorFlow GPU configuration skipped: %s", exc)


def _print_banner(n_people: int) -> None:
    print("\n" + "=" * 55)
    print("  Face Recognition System  (No Training Required)")
    print("=" * 55)
    print(f"  Model     : {config.MODEL_NAME} via DeepFace")
    print(f"  Detector  : {config.DETECTOR_BACKEND}")
    print(f"  People    : {n_people} registered")
    print(f"  Threshold : {config.MATCH_THRESHOLD}")
    print(f"  Confirm   : {config.CONFIRM_FRAMES} of {config.TRACK_VOTE_WINDOW} passes")
    print(f"  Liveness  : {'ON (FasNet)' if config.ANTI_SPOOFING else 'off'}")
    print(f"  Encrypted : {'yes' if config.ENCRYPT_EMBEDDINGS else 'no'} (embeddings at rest)")
    print(f"  Frame skip: every {config.FRAME_SKIP} frames")
    print("=" * 55)
    print("  Q / ESC = quit   S = snapshot   R = reload DB   C = correct")
    print("=" * 55 + "\n")


class _CorrectionUI:
    """
    Non-blocking correction flow (C key). Instead of a console input() that froze
    the video, the operator types the name directly into the video window while
    frames keep rendering. Optional config.ADMIN_PIN gates the action.

    States: inactive → (pin →) name → commit/cancel.
    """

    def __init__(self, recognizer: FaceRecognizer, worker: "_RecogWorker") -> None:
        self._recognizer = recognizer
        self._worker     = worker
        self.active  = False
        self._stage  = "name"       # "pin" | "name"
        self._buffer = ""
        self._frame: Optional[np.ndarray] = None

    def begin(self, frame: np.ndarray, matches: List[FaceMatch]) -> None:
        if not matches:
            print("\n[CORRECT] No face detected in current frame — move closer and try again.")
            return
        self.active  = True
        self._stage  = "pin" if config.ADMIN_PIN else "name"
        self._buffer = ""
        self._frame  = frame.copy()

    def prompt(self) -> str:
        """Text to overlay on the video while correction mode is active."""
        if self._stage == "pin":
            return f"CORRECT > enter PIN: {'*' * len(self._buffer)}  (ENTER=ok  ESC=cancel)"
        return f"CORRECT > name: {self._buffer}_  (ENTER=save  ESC=cancel)"

    def handle_key(self, key: int) -> None:
        if key == 27:                       # ESC
            self._cancel("Correction cancelled.")
        elif key in (13, 10):               # ENTER
            self._commit()
        elif key == 8:                      # backspace
            self._buffer = self._buffer[:-1]
        elif 32 <= key <= 126:
            self._buffer += chr(key)

    def _cancel(self, msg: str) -> None:
        self.active = False
        self._frame = None
        print(f"\n[CORRECT] {msg}")

    def _commit(self) -> None:
        if self._stage == "pin":
            if self._buffer == config.ADMIN_PIN:
                self._stage, self._buffer = "name", ""
            else:
                _log.warning("Correction rejected: wrong admin PIN")
                self._cancel("Wrong PIN — correction rejected.")
            return
        name = self._buffer.strip().upper()
        if not name:
            self._cancel("Empty name — cancelled.")
            return
        frame = self._frame
        self.active = False
        self._frame = None
        _apply_correction(frame, name, self._recognizer, self._worker)


def _apply_correction(
    frame: np.ndarray,
    correct_name: str,
    recognizer: FaceRecognizer,
    worker: "_RecogWorker",
) -> None:
    """Extract the stored frame's face embedding and add it to the DB under correct_name."""
    try:
        results = embedding.represent_faces(frame, enforce_detection=False)
        if not results:
            print("  ERROR: Could not extract a face embedding from the frame.")
            return

        new_emb = embedding.normalize(results[0]["embedding"])

        existing = database.load_all_embeddings(recognizer._db_path)
        if correct_name in existing:
            new_matrix = np.vstack([existing[correct_name], new_emb[None, :]])
        else:
            new_matrix = new_emb[None, :]

        database.upsert_person(correct_name, new_matrix, photo_count=new_matrix.shape[0])
        recognizer.reload()
        worker.request_invalidate()
        print(f"\n[CORRECT] Added embedding for '{correct_name}' "
              f"({new_matrix.shape[0]} total photo(s)). Recognition updated immediately.")

    except Exception as exc:
        print(f"\n[CORRECT] ERROR during correction: {exc}")
        _log.error("Correction failed: %s", exc, exc_info=True)


def _save_unknown_crops(frame: np.ndarray, matches: List[FaceMatch], last_saved: dict) -> None:
    """
    Evidence trail for unknowns: save the face crop of each confirmed Unknown to
    UNKNOWN_DIR, at most once per UNKNOWN_SAVE_INTERVAL_S per track id, so a later
    review can see WHO the unrecognised person was.
    """
    if not config.SAVE_UNKNOWN_FACES:
        return
    now = time.monotonic()
    for m in matches:
        if m.name != "Unknown":
            continue
        key = m.track_id
        if now - last_saved.get(key, float("-inf")) < config.UNKNOWN_SAVE_INTERVAL_S:
            continue
        x1, y1, x2, y2 = m.bbox
        h, w = frame.shape[:2]
        crop = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
        if crop.size == 0:
            continue
        os.makedirs(config.UNKNOWN_DIR, exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.UNKNOWN_DIR, f"unknown_{ts}_track{key}.jpg")
        cv2.imwrite(path, crop)
        last_saved[key] = now
        _log.info("Unknown face crop saved: %s", path)


class _HealthMonitor:
    """
    Liveness signal for service deployments: writes a heartbeat JSON to
    config.HEALTH_FILE every HEALTH_INTERVAL_S, and (with --health-port) serves
    the same JSON at http://127.0.0.1:<port>/health so a supervisor can probe
    the process instead of tailing logs.
    """

    def __init__(self, port: Optional[int] = None) -> None:
        self._data: dict = {"status": "starting"}
        self._last_write = 0.0
        if port:
            self._serve(port)

    def update(self, fps: float, faces: int, known: int, people: int) -> None:
        now = time.monotonic()
        if now - self._last_write < config.HEALTH_INTERVAL_S:
            return
        self._last_write = now
        self._data = {
            "status":            "ok",
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fps":               round(fps, 1),
            "faces_on_screen":   faces,
            "known_on_screen":   known,
            "people_registered": people,
        }
        try:
            os.makedirs(os.path.dirname(config.HEALTH_FILE), exist_ok=True)
            with open(config.HEALTH_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except OSError as exc:
            _log.debug("Could not write health file: %s", exc)

    def _serve(self, port: int) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        monitor = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):                              # noqa: N802 (http.server API)
                body = json.dumps(monitor._data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):                     # silence per-request stderr spam
                pass

        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"Health endpoint: http://127.0.0.1:{port}/health")
            _log.info("Health endpoint listening on 127.0.0.1:%d", port)
        except OSError as exc:
            _log.error("Could not start health endpoint on port %d: %s", port, exc)
            print(f"[WARNING] Health endpoint unavailable (port {port}): {exc}")


def _print_session_report() -> None:
    """Print a summary of known-person detections from the last hour."""
    try:
        rows = database.detection_summary(since_hours=1)
        print("\n" + "=" * 52)
        print("  Session Attendance Summary")
        print("=" * 52)
        if not rows:
            print("  No known persons detected this session.")
        else:
            print(f"  {'Name':<20} {'Detections':>10}  Last Seen")
            print("-" * 52)
            for name, hits, last in rows:
                print(f"  {name:<20} {hits:>10}  {last}")
        print("=" * 52 + "\n")
    except Exception as exc:
        _log.warning("Could not generate session report: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = _parse_args(argv)

    config.CAMERA_INDEX    = args.camera
    config.MATCH_THRESHOLD = args.threshold
    config.FRAME_SKIP      = args.skip

    if sys.version_info >= (3, 12):
        print("[WARNING] Python 3.12+ has limited TensorFlow support. Use 3.10 or 3.11.")

    setup_logger()
    _log.info("Starting Face Recognition System")
    _configure_tensorflow()

    database.init_db()
    database.prune_detection_log()   # auto-clean old records on startup

    recognizer = FaceRecognizer()
    n_people   = recognizer.load()

    if n_people == 0:
        print(
            "\n[WARNING] No faces registered in database.\n"
            "  Run:  python register_faces.py\n"
            "  Then place photos under:  faces/PersonName/photo1.jpg\n"
            "  Continuing — all detections will show 'Unknown'.\n"
        )

    _print_banner(n_people)

    camera = CameraCapture(cam_id=args.camera, width=args.width, height=args.height)
    try:
        camera.open()
    except RuntimeError as exc:
        _log.error("Camera error: %s", exc)
        print(f"\nERROR: {exc}\n")
        sys.exit(1)

    worker = _RecogWorker(recognizer)
    worker.start()

    correction = _CorrectionUI(recognizer, worker)
    health     = _HealthMonitor(port=args.health_port)

    # State for alerting and unknown tracking
    alerted_known  : set = set()   # names already alerted this session
    unknown_streak : int = 0       # consecutive recognition passes with an unknown face
    unknown_saved  : dict = {}     # track_id -> monotonic time of last saved crop
    matches        : List[FaceMatch] = []

    fps         = 0.0
    alpha       = config.FPS_EMA_ALPHA
    prev_time   = time.perf_counter()
    frame_count = 0

    _log.info("Webcam loop started")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)   # don't busy-spin while the camera reconnects
                continue

            frame_count += 1

            if frame_count % config.FRAME_SKIP == 0:
                worker.submit(frame)

            # Use the freshest tracked result; clear overlays if the worker has
            # stalled (no fresh result within BOX_EXPIRY_MS) rather than freezing them.
            new_matches, age_ms = worker.latest_with_age()
            if age_ms <= config.BOX_EXPIRY_MS:
                matches = new_matches
            else:
                matches = []

            known_ct = sum(1 for m in matches if m.name not in ("Unknown", "...", "SPOOF"))

            # ── Alerting ──────────────────────────────────────────────────────

            for m in matches:
                if m.name == "SPOOF" and "SPOOF" not in alerted_known:
                    msg = f"[ALERT] Possible spoof attack (photo/screen) — liveness check failed"
                    print(f"\n{msg}")
                    _log.warning(msg)
                    alerted_known.add("SPOOF")
                if m.name not in ("Unknown", "...", "SPOOF") and m.name not in alerted_known:
                    if config.ALERT_KNOWN_PERSON:
                        msg = f"[ALERT] {m.name} detected (confidence={m.confidence * 100:.1f}%)"
                        print(f"\n{msg}")
                        _log.info(msg)
                    alerted_known.add(m.name)

            has_unknown = any(m.name == "Unknown" for m in matches)
            if has_unknown:
                unknown_streak += 1
                if config.ALERT_UNKNOWN_PERSON and unknown_streak == config.UNKNOWN_ALERT_FRAMES:
                    msg = (f"[ALERT] Unknown person has been present for "
                           f"{unknown_streak} consecutive recognition passes")
                    print(f"\n{msg}")
                    _log.warning(msg)
            else:
                unknown_streak = 0

            _save_unknown_crops(frame, matches, unknown_saved)

            # ── Render ────────────────────────────────────────────────────────

            now = time.perf_counter()
            dt  = now - prev_time
            prev_time = now
            if dt > 0:
                fps = alpha * (1.0 / dt) + (1 - alpha) * fps

            health.update(fps, len(matches), known_ct, n_people)

            if args.headless:
                # Service mode: no window / no keyboard. Alerts, DB logging and the
                # health heartbeat above still run; camera.read() paces the loop.
                continue

            annotated = visualizer.render(frame, matches)
            visualizer.draw_hud(annotated, fps, len(matches), known_ct)
            if correction.active:
                visualizer.draw_prompt(annotated, correction.prompt())
            cv2.imshow(config.WINDOW_TITLE, annotated)

            key = cv2.waitKey(1) & 0xFF

            # Correction mode captures the keyboard (typed into the window, video
            # keeps playing) until ENTER/ESC.
            if correction.active:
                if key != 255:
                    correction.handle_key(key)
                continue

            if key in (ord("q"), ord("Q"), 27):
                _log.info("Quit key pressed")
                break
            elif key in (ord("s"), ord("S")):
                visualizer.save_snapshot(annotated, matches)
            elif key in (ord("r"), ord("R")):
                n_people = recognizer.reload()
                worker.request_invalidate()
                print(f"Reloaded — {n_people} person(s) in DB")
            elif key in (ord("c"), ord("C")):
                correction.begin(frame, matches)

    except KeyboardInterrupt:
        _log.info("Interrupted (Ctrl+C)")
        print("\nStopping...")
    finally:
        worker.stop()
        camera.release()
        _log.info("Face Recognition System stopped")
        print("\nSystem stopped.")

        if config.ATTENDANCE_REPORT_ON_EXIT:
            _print_session_report()


if __name__ == "__main__":
    main()
