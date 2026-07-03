import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from facerec import config
from facerec import database
from facerec import embedding

_log = logging.getLogger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@dataclass
class FaceMatch:
    """Result of recognising one face in a frame."""
    bbox: tuple         # (x1, y1, x2, y2) pixel coords
    name: str           # person name, "Unknown", "SPOOF", or "..." (pending confirmation)
    confidence: float   # 0.0–1.0; 1.0 = perfect match
    distance: float     # raw cosine distance; lower = better
    track_id: int = -1  # persistent id from the tracker (-1 = raw, untracked detection)
    # L2-normalised query embedding of this face — used by the tracker to veto
    # geometric associations that would swap identities. Not compared/printed.
    embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


class FaceRecognizer:
    """
    Loads known face embeddings from SQLite at startup.
    Exposes recognize(frame) -> list[FaceMatch].

    Thread-safe: recognize() runs on a background worker thread while load()/reload()
    (the R and C keys) run on the main thread. A lock guards the embedding store so a
    reload can never be observed half-applied, and every DeepFace call is serialised
    by the shared lock in embedding.py.
    """

    def __init__(self, db_path: str = config.DB_PATH) -> None:
        self._db_path = db_path
        self._known: dict = {}
        self._known_names: list = []
        # Vectorised store: all photo embeddings stacked into ONE (M, 512) matrix
        # so _best_match is a single matmul — O(1) numpy calls regardless of how
        # many people are registered (scales to thousands without FAISS).
        self._matrix: Optional[np.ndarray] = None   # (M, 512)
        self._row_names: list = []                  # row index -> person name
        self._row_names_arr: Optional[np.ndarray] = None  # same, as ndarray (margin mask)
        self._lock = threading.RLock()
        self._last_logged: dict = {}   # name -> monotonic time of last DB write (worker thread only)
        self._spoof_model = None       # lazy-loaded FasNet when ANTI_SPOOFING is on
        self._spoof_disabled = False   # set if FasNet fails to load (warn once, keep running)
        self._live_cache: list = []    # recent per-region liveness verdicts (worker thread only)

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self) -> int:
        """Load all embeddings from SQLite. Returns number of people loaded."""
        known = database.load_all_embeddings(self._db_path)
        self._swap_known(known)
        n = len(known)
        if n == 0:
            _log.warning("No faces registered. Run: python register_faces.py")
        else:
            _log.info("Loaded %d person(s): %s", n, list(known.keys()))
        self._check_version_match()
        return n

    def reload(self) -> int:
        """Hot-reload embeddings without restarting."""
        _log.info("Reloading face embeddings from DB...")
        return self.load()

    def recognize(self, frame: np.ndarray) -> list:
        """
        Detect all faces in frame and identify each one.
        Returns list[FaceMatch]. Returns [] on no face or error.
        Runs on the background worker thread.
        """
        # YuNet degrades badly on very large frames — detect on a downscaled copy
        # and map boxes back to original coordinates. No-op at default 720p capture.
        frame_h, frame_w = frame.shape[:2]
        scale = 1.0
        work  = frame
        cap   = getattr(config, "MAX_DETECT_SIDE", 0)
        if cap and max(frame_h, frame_w) > cap:
            scale = cap / max(frame_h, frame_w)
            work  = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        try:
            # CLAHE is applied inside represent_faces (identically to enrollment).
            results = embedding.represent_faces(work, enforce_detection=False)
        except ValueError as exc:
            _log.debug("represent_faces: no face — %s", exc)
            return []
        except Exception as exc:
            _log.error("represent_faces failed: %s", exc, exc_info=True)
            return []

        matches = []
        for face in results:
            emb_raw = face.get("embedding")
            if emb_raw is None:
                continue
            area     = face.get("facial_area", {})
            det_conf = float(face.get("face_confidence", 1.0) or 0.0)

            # work-frame coords (as detected) and original-frame coords (for output)
            wx = int(area.get("x", 0))
            wy = int(area.get("y", 0))
            ww = int(area.get("w", 0))
            wh = int(area.get("h", 0))
            x, y, w, h = (int(v / scale) for v in (wx, wy, ww, wh))

            # Phantom-face guard: enforce_detection=False can return the whole frame
            # as a bogus "face" with confidence 0 when nothing is actually detected.
            if det_conf < config.MIN_FACE_CONFIDENCE:
                _log.debug("Skipping low-confidence detection (%.2f) at (%d,%d)", det_conf, x, y)
                continue
            if w >= frame_w and h >= frame_h:
                _log.debug("Skipping whole-frame phantom detection")
                continue

            # Quality gate: skip faces that are too small
            if w < config.MIN_FACE_PIXELS or h < config.MIN_FACE_PIXELS:
                _log.debug("Skipping small face %dx%d at (%d,%d)", w, h, x, y)
                continue

            # Quality gate: skip blurry face regions. Measured on the WORK frame
            # (the resolution the detector saw) — Laplacian variance is resolution-
            # dependent, so scoring the full-size crop would unfairly fail large frames.
            if not self._is_sharp(work, wx, wy, ww, wh):
                _log.debug("Skipping blurry face at (%d,%d)", x, y)
                continue

            bbox = (x, y, x + w, y + h)

            # Liveness gate (photo/screen replay attacks). Runs per face on the
            # original frame; a failed face is labelled SPOOF and never matched
            # or logged to attendance.
            if config.ANTI_SPOOFING:
                is_real, spoof_score = self._liveness_cached(frame, x, y, w, h)
                if not is_real:
                    _log.warning("Spoof suspected at (%d,%d) score=%.2f", x, y, spoof_score)
                    matches.append(FaceMatch(
                        bbox=bbox, name="SPOOF",
                        confidence=float(spoof_score), distance=float("inf"),
                    ))
                    continue

            query_emb = embedding.normalize(emb_raw)
            name, distance, margin = self._match_with_margin(query_emb)

            # Accept only when within threshold AND clearly closer to this identity
            # than to anyone else (second-best margin) — ambiguous faces stay Unknown.
            if distance <= config.MATCH_THRESHOLD and margin >= config.MATCH_MARGIN:
                person     = name
                confidence = float(np.clip(1.0 - distance, 0.0, 1.0))
                self._record(person, confidence)
            else:
                if distance <= config.MATCH_THRESHOLD:
                    _log.debug("Ambiguous match rejected: %s d=%.3f margin=%.3f", name, distance, margin)
                person     = "Unknown"
                confidence = 0.0

            matches.append(FaceMatch(
                bbox=bbox,
                name=person,
                confidence=confidence,
                distance=float(distance),
                embedding=query_emb,
            ))

        return matches

    # ── Internal ──────────────────────────────────────────────────────────────

    def _swap_known(self, known: dict) -> None:
        """Atomically replace the embedding store (heavy DB load happens outside the lock)."""
        row_names: list = []
        blocks: list = []
        for name, matrix in known.items():
            m = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
            blocks.append(m)
            row_names.extend([name] * m.shape[0])
        stacked = np.vstack(blocks) if blocks else None
        with self._lock:
            self._known = known
            self._known_names = list(known.keys())
            self._matrix = stacked
            self._row_names = row_names
            self._row_names_arr = np.array(row_names) if row_names else None

    def _record(self, name: str, confidence: float) -> None:
        """Throttled DB write: record a person at most once per DETECTION_LOG_INTERVAL_S."""
        now = time.monotonic()
        if now - self._last_logged.get(name, float("-inf")) >= config.DETECTION_LOG_INTERVAL_S:
            database.record_detection(name, confidence, self._db_path)
            self._last_logged[name] = now

    def _best_match(self, query: np.ndarray) -> tuple:
        """Nearest identity → (name, distance). See _match_with_margin."""
        name, dist, _ = self._match_with_margin(query)
        return name, dist

    def _match_with_margin(self, query: np.ndarray) -> tuple:
        """
        Nearest identity across ALL stored photo embeddings — a single matmul over
        the stacked (M, 512) matrix, so cost is one BLAS call however many people
        are registered. Returns (name, distance, margin) where margin is the gap
        to the closest OTHER identity (inf with 0–1 people registered).
        Thread-safe: snapshots the store under the lock, then runs lock-free.
        """
        with self._lock:
            matrix    = self._matrix
            names_arr = self._row_names_arr

        if matrix is None:
            return "Unknown", float("inf"), float("inf")

        dists = 1.0 - matrix @ query          # cosine distance per stored photo
        i     = int(np.argmin(dists))
        name  = str(names_arr[i])
        best  = float(dists[i])

        others = dists[names_arr != name]
        margin = float(np.min(others) - best) if others.size else float("inf")
        return name, best, margin

    def _liveness_cached(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> tuple:
        """
        Region-cached liveness: a face centred near a recently-checked spot reuses
        that verdict for LIVENESS_RECHECK_S seconds instead of re-running FasNet
        every recognition pass. Worker-thread only (no locking needed).
        """
        recheck = getattr(config, "LIVENESS_RECHECK_S", 0)
        now = time.monotonic()
        cx, cy = x + w / 2.0, y + h / 2.0

        self._live_cache = [e for e in self._live_cache if now - e["t"] < recheck]
        for e in self._live_cache:
            # same region if the centre moved less than roughly one face width
            if abs(cx - e["cx"]) < e["w"] and abs(cy - e["cy"]) < e["w"]:
                return e["ok"], e["score"]

        ok, score = self._check_liveness(frame, x, y, w, h)
        self._live_cache.append(
            {"cx": cx, "cy": cy, "w": max(w, 1), "t": now, "ok": ok, "score": score}
        )
        return ok, score

    def _check_liveness(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> tuple:
        """
        Per-face FasNet liveness check → (is_real, score). Loads the model lazily
        on first use; if it can't load, warns once and passes everything through
        (recognition keeps working without the gate).
        """
        if self._spoof_disabled:
            return True, 1.0
        if self._spoof_model is None:
            try:
                from deepface.models.spoofing.FasNet import Fasnet
                with embedding.deepface_lock:
                    self._spoof_model = Fasnet()
                _log.info("Anti-spoofing (FasNet) enabled")
            except Exception as exc:
                self._spoof_disabled = True
                _log.error("Anti-spoofing unavailable (%s) — liveness gate DISABLED", exc)
                return True, 1.0
        try:
            is_real, score = self._spoof_model.analyze(img=frame, facial_area=(x, y, w, h))
            return bool(is_real), float(score)
        except Exception as exc:
            _log.debug("Liveness check failed at (%d,%d): %s — passing face through", x, y, exc)
            return True, 0.0

    @staticmethod
    def _is_sharp(frame: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Return True if the face crop passes the Laplacian blur threshold."""
        if w <= 0 or h <= 0:
            return False
        fh, fw = frame.shape[:2]
        x2, y2 = min(x + w, fw), min(y + h, fh)
        crop = frame[max(y, 0):y2, max(x, 0):x2]
        if crop.size == 0:
            return False
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        return variance >= config.BLUR_THRESHOLD

    def _check_version_match(self) -> None:
        """Warn if the DB was built with a different model/detector/CLAHE setting. Offer auto-migration."""
        stored_model    = database.get_meta("model_name",       self._db_path)
        stored_detector = database.get_meta("detector_backend", self._db_path)
        stored_clahe    = database.get_meta("use_clahe",        self._db_path)

        mismatches = []
        if stored_model and stored_model != config.MODEL_NAME:
            mismatches.append(f"model: DB={stored_model} → config={config.MODEL_NAME}")
        if stored_detector and stored_detector != config.DETECTOR_BACKEND:
            mismatches.append(f"detector: DB={stored_detector} → config={config.DETECTOR_BACKEND}")
        if stored_clahe is not None and stored_clahe != str(config.USE_CLAHE):
            mismatches.append(f"CLAHE: DB={stored_clahe} → config={config.USE_CLAHE}")

        if not mismatches:
            return

        print("\n" + "=" * 60)
        print("  [WARNING] Registration mismatch detected:")
        for m in mismatches:
            print(f"    {m}")
        print("  Embeddings are incompatible — recognition will be poor.")
        print("=" * 60)
        try:
            ans = input("  Re-register all faces now? (Y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans == "y":
            import subprocess
            import sys
            print("  Running register_faces.py --clear ...")
            script = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tools", "register_faces.py",
            )
            subprocess.run(
                [sys.executable, script, "--clear"],
                check=False,
            )
            self._swap_known(database.load_all_embeddings(self._db_path))
            print("  Re-registration complete. Resuming.\n")
        else:
            print("  Skipped. Run:  python register_faces.py --clear\n")
        for m in mismatches:
            _log.warning("Version mismatch — %s", m)
