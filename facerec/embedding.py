"""
embedding.py — Shared face-embedding extraction (single source of truth).

Every path that turns an image into an ArcFace embedding goes through here:
live recognition (recognizer.py), photo/live enrollment (register_faces.py /
register_live.py), and the correction feedback loop (main.py). Centralising it
guarantees three things that used to drift apart when the logic was copy-pasted:

  1. Identical preprocessing (CLAHE) at enrollment AND inference, so stored and
     query embeddings are directly comparable — otherwise cosine matching degrades.
  2. Identical L2 normalisation.
  3. All DeepFace / TensorFlow calls in this process are serialised through ONE
     lock. DeepFace/TF is not thread-safe, and recognition runs on a background
     thread while the C-key correction runs on the main thread.
"""

import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np

from facerec import config

_log = logging.getLogger(__name__)

# The single serialisation point for every DeepFace.represent() call in this
# process (worker-thread recognition, main-thread correction, registration workers).
deepface_lock = threading.Lock()


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement (on the LAB L-channel) when config.USE_CLAHE.
    Applied identically at enrollment and inference so embeddings stay comparable.
    Returns the frame unchanged when CLAHE is disabled.
    """
    if not getattr(config, "USE_CLAHE", False):
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=tuple(config.CLAHE_TILE_GRID),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def represent_faces(
    image,
    *,
    enforce_detection: bool,
    apply_clahe: bool = True,
    anti_spoofing: Optional[bool] = None,
) -> list:
    """
    Run DeepFace.represent under the shared lock and return its result list.

    `image` may be a file path (str) or a BGR ndarray. When apply_clahe is set and
    CLAHE is enabled, the image is enhanced first (a path is read via cv2 so the
    enhancement matches the live path exactly).

    Returns a list of face dicts, each with "embedding", "facial_area" and
    "face_confidence". Raises ValueError when no face is found and
    enforce_detection=True — callers decide how to handle that.
    """
    from deepface import DeepFace

    img = image
    if apply_clahe and getattr(config, "USE_CLAHE", False):
        if isinstance(image, np.ndarray):
            img = enhance_frame(image)
        elif isinstance(image, str):
            loaded = cv2.imread(image)
            if loaded is not None:
                img = enhance_frame(loaded)
            # if unreadable, fall through with the path and let DeepFace report it

    kwargs = dict(
        img_path=img,
        model_name=config.MODEL_NAME,
        detector_backend=config.DETECTOR_BACKEND,
        enforce_detection=enforce_detection,
        align=True,
    )
    # Only when explicitly requested: DeepFace.represent(anti_spoofing=True) raises
    # SpoofDetected for the WHOLE image if any face fails, which is wrong for live
    # multi-face frames. Live liveness runs per face in recognizer._check_liveness
    # instead; config.ANTI_SPOOFING no longer routes through here.
    if anti_spoofing is True:
        kwargs["anti_spoofing"] = True

    with deepface_lock:
        results = DeepFace.represent(**kwargs)

    if isinstance(results, dict):
        results = [results]
    return results


def normalize(vec) -> np.ndarray:
    """L2-normalise a 1-D embedding to float32. Unchanged if the norm is zero."""
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def embed_image(image_path: str, *, apply_clahe: bool = True) -> Optional[np.ndarray]:
    """
    Extract ONE L2-normalised (512,) embedding from an image file, for enrollment.
    Uses enforce_detection=True. Returns None (with a warning) on any failure.
    """
    name = os.path.basename(image_path)

    # YuNet misses faces on very large photos (whole-frame phantom, confidence 0),
    # so oversized enrollment images are downscaled first. Safe here because
    # enrollment only keeps the embedding, never the facial_area coordinates.
    img = cv2.imread(image_path)
    if img is None:
        _log.warning("  Could not read image '%s'", name)
        return None
    longest = max(img.shape[:2])
    cap = getattr(config, "MAX_ENROLL_SIDE", 1600)
    if cap and longest > cap:
        scale = cap / longest
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _log.debug("  Downscaled '%s' by %.2fx for detection", name, scale)

    try:
        results = represent_faces(
            img, enforce_detection=True, apply_clahe=apply_clahe
        )
    except ValueError as exc:
        _log.warning("  No face found in '%s': %s", name, exc)
        return None
    except Exception as exc:
        _log.warning("  Failed to process '%s': %s", name, exc)
        return None

    if not results:
        _log.warning("  No embedding returned for '%s'", name)
        return None
    if len(results) > 1:
        _log.debug("  %d faces in '%s'; using the first for enrollment", len(results), name)

    return normalize(results[0]["embedding"])
