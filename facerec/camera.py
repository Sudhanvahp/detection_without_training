import logging
import time
from typing import Optional

import cv2
import numpy as np

from facerec import config

_log = logging.getLogger(__name__)


class CameraCapture:
    """
    Thin wrapper around cv2.VideoCapture.
    Uses DirectShow backend (CAP_DSHOW) on Windows for fast, reliable open.
    Automatically attempts to reconnect if the camera is unplugged mid-run.
    """

    def __init__(
        self,
        cam_id: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        self._cam_id = cam_id
        self._width  = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0

    def open(self) -> None:
        """
        Open the camera.
        Raises RuntimeError with an actionable message if it fails.
        """
        self._cap = cv2.VideoCapture(self._cam_id, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._cam_id)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self._cam_id}.\n"
                f"  - Make sure a webcam is connected.\n"
                f"  - Try  python main.py --camera 1  for a different camera index.\n"
                f"  - On some systems, close other applications using the camera."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self._consecutive_failures = 0

        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        _log.info("Camera %d opened — %dx%d @ %.0f FPS", self._cam_id, actual_w, actual_h, actual_fps)
        print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")

    def read(self) -> Optional[np.ndarray]:
        """
        Return the next BGR frame, or None on read failure.
        On repeated failures, attempts to reconnect automatically.
        """
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._consecutive_failures = 0
            return frame

        self._consecutive_failures += 1
        _log.warning("Camera read failed (consecutive=%d)", self._consecutive_failures)

        if self._consecutive_failures >= config.CAMERA_FAILURES_BEFORE_RECONNECT:
            _log.warning("Camera appears disconnected — attempting reconnect...")
            print(f"\n[WARNING] Camera disconnected. Attempting to reconnect "
                  f"(up to {config.CAMERA_RECONNECT_ATTEMPTS} times)...")
            self._try_reconnect()

        return None

    def _try_reconnect(self) -> None:
        """Attempt to re-open the camera up to _RECONNECT_ATTEMPTS times."""
        self._cap.release()
        self._cap = None

        attempts = config.CAMERA_RECONNECT_ATTEMPTS
        for attempt in range(1, attempts + 1):
            time.sleep(config.CAMERA_RECONNECT_DELAY_S)
            _log.info("Reconnect attempt %d/%d ...", attempt, attempts)
            print(f"  Reconnect attempt {attempt}/{attempts}...")

            cap = cv2.VideoCapture(self._cam_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._cam_id)

            if cap.isOpened():
                self._cap = cap
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
                self._consecutive_failures = 0
                _log.info("Camera reconnected successfully")
                print("  Camera reconnected.")
                return

        _log.error("Camera reconnect failed after %d attempts", attempts)
        print(f"[ERROR] Could not reconnect to camera after {attempts} attempts. "
              f"Check the USB connection and press Q to quit.")

    def release(self) -> None:
        """Release the capture device and destroy all OpenCV windows."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()
        _log.debug("Camera released")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
