import json
import logging
import os
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from facerec import config
from facerec.recognizer import FaceMatch

_log = logging.getLogger(__name__)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
_GREEN   = (0,  210,  60)   # known person
_RED     = (0,   40, 220)   # unknown
_YELLOW  = (0,  200, 220)   # pending confirmation ("...")
_MAGENTA = (200,  0, 200)   # suspected spoof (photo / screen replay)
_WHITE  = (255, 255, 255)
_DARK   = (20,   20,  20)
_GREY   = (190, 190, 190)


def draw_face_box(frame: np.ndarray, match: FaceMatch) -> None:
    """
    Draw a coloured bounding box + name/confidence/distance label for one face.
    Green = known, Red = Unknown, Yellow = pending confirmation.
    """
    x1, y1, x2, y2 = match.bbox

    if match.name == "Unknown":
        colour = _RED
        label  = "Unknown"
    elif match.name == "SPOOF":
        colour = _MAGENTA
        label  = "SPOOF?"
    elif match.name == "...":
        colour = _YELLOW
        label  = "Verifying..."
    else:
        colour = _GREEN
        label  = f"{match.name}  {match.confidence * 100:.1f}%  d={match.distance:.3f}"

    if config.SHOW_TRACK_ID and match.track_id >= 0:
        label = f"#{match.track_id} {label}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.60
    thickness  = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    pad = 4
    bx1 = x1
    by1 = max(y1 - th - 2 * pad, 0)
    bx2 = x1 + tw + 2 * pad
    by2 = y1

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), colour, cv2.FILLED)
    cv2.putText(frame, label, (bx1 + pad, by2 - pad),
                font, font_scale, _WHITE, thickness, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    face_count: int,
    known_count: int,
) -> None:
    """Draw a semi-transparent status panel in the top-left corner."""
    lines = [
        f"FPS:    {fps:5.1f}",
        f"Faces:  {face_count}",
        f"Known:  {known_count}",
        "",
        "S = snapshot",
        "R = reload DB",
        "C = correct name",
        "Q / ESC = quit",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    pad        = 8
    line_h     = 20

    panel_w = 180
    panel_h = len(lines) * line_h + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), _DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, line in enumerate(lines):
        y      = pad + (i + 1) * line_h - 4
        colour = _WHITE if i < 3 else _GREY
        cv2.putText(frame, line, (pad, y), font, font_scale, colour, thickness, cv2.LINE_AA)


def draw_prompt(frame: np.ndarray, text: str) -> None:
    """Bottom-of-frame input bar (correction mode: type directly into the window)."""
    h = frame.shape[0]
    bar_h = 36
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (frame.shape[1], h), _DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, _YELLOW, 2, cv2.LINE_AA)


def render(frame: np.ndarray, matches: List[FaceMatch]) -> np.ndarray:
    """Compose a fully annotated frame copy."""
    out = frame.copy()
    for m in matches:
        draw_face_box(out, m)
    return out


def save_snapshot(frame: np.ndarray, matches: Optional[List[FaceMatch]] = None,
                  save_dir: str = config.SNAPSHOT_DIR) -> str:
    """
    Save the annotated frame to snapshots/snapshot_YYYYMMDD_HHMMSS.jpg
    and write a JSON sidecar file with detection metadata.
    Returns the saved image path.
    """
    os.makedirs(save_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"snapshot_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # JSON sidecar — records who was in the frame at snapshot time
    if matches is not None:
        meta = {
            "timestamp": datetime.now().isoformat(),
            "model":     config.MODEL_NAME,
            "detector":  config.DETECTOR_BACKEND,
            "detections": [
                {
                    "track_id":   m.track_id,
                    "name":       m.name,
                    "confidence": round(m.confidence, 4),
                    "distance":   round(m.distance, 4),
                    "bbox":       list(m.bbox),
                }
                for m in matches
            ],
        }
        json_path = path.replace(".jpg", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    _log.info("Snapshot saved: %s", path)
    print(f"Snapshot saved: {path}")
    return path
