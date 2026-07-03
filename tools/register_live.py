"""
register_live.py — Live webcam registration for face recognition system.

Captures photos directly from the office webcam and registers the person
immediately. This ensures registration and recognition use the same camera,
lighting, and environment — the single biggest accuracy improvement possible.

Usage:
  python register_live.py --name "JOHN DOE"
  python register_live.py --name "JOHN DOE" --photos 15
  python register_live.py --name "JOHN DOE" --camera 1
"""

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Allow running as "python tools/<script>.py" from the project root.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from facerec import clihelpers
from facerec import config
from facerec import database
from facerec.camera import CameraCapture
from facerec.logger import setup_logger

_log = logging.getLogger(__name__)

_CAPTURE_INTERVAL_S = 1.2   # seconds between auto-captures
_INITIAL_COUNTDOWN  = 3     # seconds before first capture


def _draw_guide(frame: np.ndarray, name: str, captured: int, total: int,
                countdown_left: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2

    # Oval face guide
    cv2.ellipse(out, (cx, cy), (110, 140), 0, 0, 360, (0, 200, 255), 2)

    # Header
    cv2.rectangle(out, (0, 0), (w, 50), (20, 20, 20), cv2.FILLED)
    cv2.putText(out, f"Registering: {name}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 210, 60), 2, cv2.LINE_AA)

    if countdown_left > 0:
        txt = f"Starting in {int(countdown_left) + 1}..."
        cv2.putText(out, txt, (cx - 120, cy + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3, cv2.LINE_AA)
    else:
        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 20, h - 50, w - 40, 22
        progress = captured / total
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), cv2.FILLED)
        cv2.rectangle(out, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + bar_h),
                      (0, 210, 60), cv2.FILLED)
        label = f"Photos: {captured}/{total}  — Move head slightly between shots"
        cv2.putText(out, label, (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Tip
    cv2.putText(out, "Q = cancel", (w - 110, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def capture_photos(name: str, n_photos: int, cam_id: int) -> list:
    """
    Open the webcam and auto-capture n_photos at _CAPTURE_INTERVAL_S intervals.
    Returns the list of saved file paths, or [] if cancelled. The camera is ALWAYS
    released — even on error or cancel — via the CameraCapture context manager.
    """
    save_dir = os.path.join(config.FACES_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nCapturing {n_photos} photos for '{name}'.")
    print("Position face in the oval guide. System auto-captures every "
          f"{_CAPTURE_INTERVAL_S:.1f}s.")
    print("Move your head slightly between shots for angle diversity.")
    print("Press Q to cancel.\n")

    captured     = []
    last_capture = 0.0
    start_time   = time.time()

    try:
        with CameraCapture(cam_id=cam_id) as camera:
            while len(captured) < n_photos:
                frame = camera.read()
                if frame is None:
                    continue

                now = time.time()
                elapsed = now - start_time
                countdown_left = max(0.0, _INITIAL_COUNTDOWN - elapsed)

                if countdown_left == 0 and (now - last_capture) >= _CAPTURE_INTERVAL_S:
                    ts   = int(now * 1000)
                    path = os.path.join(save_dir, f"live_{ts}.jpg")
                    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    captured.append(path)
                    last_capture = now
                    print(f"  Photo {len(captured)}/{n_photos} captured")

                annotated = _draw_guide(frame, name, len(captured), n_photos, countdown_left)
                cv2.imshow(f"Live Registration — {name}", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\nCancelled.")
                    return []
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print(f"\nCapture complete — {len(captured)} photos saved to faces/{name}/")
    return captured


def register_from_photos(name: str, photo_paths: list) -> bool:
    """Extract ArcFace embeddings from photos and write to DB. Returns True on success."""
    from register_faces import blur_score, extract_embedding

    print(f"\nExtracting embeddings for '{name}' ...")
    embeddings = []

    for path in photo_paths:
        blur = blur_score(path)
        if blur < config.REG_BLUR_THRESHOLD:
            print(f"  SKIP (blurry {blur:.1f}): {os.path.basename(path)}")
            continue

        emb = extract_embedding(path)
        if emb is not None:
            embeddings.append(emb)
            print(f"  OK: {os.path.basename(path)}")
        else:
            print(f"  FAIL (no face): {os.path.basename(path)}")

    if not embeddings:
        print(f"\nERROR: No valid embeddings extracted for '{name}'.")
        print("Tips: ensure good lighting, face the camera, remove heavy obstructions.")
        return False

    matrix = np.stack(embeddings).astype(np.float32)
    database.upsert_person(name, matrix, photo_count=len(embeddings))
    database.set_meta("model_name",       config.MODEL_NAME)
    database.set_meta("detector_backend", config.DETECTOR_BACKEND)
    database.set_meta("use_clahe",        str(config.USE_CLAHE))

    print(f"\nSUCCESS: '{name}' registered with {len(embeddings)} photo(s).")
    print("Press R in the main recognition window to reload without restarting.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live webcam registration — captures photos and registers immediately.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name",   required=True, metavar="NAME",
                        help="Person's full name, e.g. 'JOHN DOE'")
    parser.add_argument("--photos", type=clihelpers.positive_int, default=10, metavar="N",
                        help="Number of photos to capture (default: 10)")
    parser.add_argument("--camera", type=clihelpers.nonneg_int, default=config.CAMERA_INDEX,
                        help=f"Camera index (default: {config.CAMERA_INDEX})")
    args = parser.parse_args()

    setup_logger()
    database.init_db()

    name = args.name.strip().upper()
    if not name:
        print("ERROR: --name must not be empty.")
        sys.exit(1)

    print(f"\n{'='*52}")
    print(f"  Live Registration")
    print(f"{'='*52}")
    print(f"  Name    : {name}")
    print(f"  Photos  : {args.photos}")
    print(f"  Model   : {config.MODEL_NAME}")
    print(f"  Detector: {config.DETECTOR_BACKEND}")
    print(f"  Camera  : index {args.camera}")
    print(f"{'='*52}\n")

    photos = capture_photos(name, n_photos=args.photos, cam_id=args.camera)
    if not photos:
        sys.exit(1)

    print("\nInitialising DeepFace (may take a moment on first run)...")
    success = register_from_photos(name, photos)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
