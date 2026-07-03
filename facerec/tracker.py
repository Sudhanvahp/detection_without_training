"""
tracker.py — Lightweight multi-face tracker (SORT-lite, numpy-only).

Gives each detected face a PERSISTENT integer id across recognition passes and
makes name confirmation PER-TRACK instead of per-name. This replaces the old
global name-keyed streak, which conflated two people predicted as the same name,
was confused by movement, and let "Unknown" bypass debouncing.

Association is greedy IoU with a scale-adaptive centroid fallback (a face can move
a lot between passes at the recognition cadence, so IoU alone would spawn spurious
ids). Confirmation is a rolling-window plurality vote with hysteresis, so a name
appears only after winning CONFIRM_FRAMES of the last TRACK_VOTE_WINDOW passes and
is sticky against single-frame flicker.

All state lives on the worker thread (the tracker's only caller), so there is no
new cross-thread shared state — same discipline as the streak dict it replaces.
"""

from collections import Counter, deque
from typing import List, Optional, Tuple

import numpy as np

from facerec import config
from facerec.recognizer import FaceMatch

Bbox = Tuple[int, int, int, int]


# ── Geometry helpers (pure, unit-testable) ─────────────────────────────────────

def iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(b: Bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def centroid_dist(a: Bbox, b: Bbox) -> float:
    ax, ay = centroid(a)
    bx, by = centroid(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def width(b: Bbox) -> float:
    return max(0, b[2] - b[0])


# ── Track ───────────────────────────────────────────────────────────────────

class _Track:
    """One tracked face. Mutable; touched only on the worker thread."""

    def __init__(self, tid: int, det: FaceMatch) -> None:
        self.id = tid
        self.bbox: Bbox = det.bbox
        self.votes: deque = deque(maxlen=config.TRACK_VOTE_WINDOW)  # (name, distance, confidence)
        self.confirmed_name: Optional[str] = None
        self.misses = 0
        self.last_emb: Optional[np.ndarray] = None  # newest face embedding (association veto)
        self.update_with(det)

    def update_with(self, det: FaceMatch) -> None:
        """Matched by a detection this pass: smooth the box, add a vote, re-confirm."""
        self.misses = 0
        a = config.TRACK_BBOX_SMOOTHING
        ox1, oy1, ox2, oy2 = self.bbox
        nx1, ny1, nx2, ny2 = det.bbox
        self.bbox = (
            int(a * nx1 + (1 - a) * ox1),
            int(a * ny1 + (1 - a) * oy1),
            int(a * nx2 + (1 - a) * ox2),
            int(a * ny2 + (1 - a) * oy2),
        )
        if getattr(det, "embedding", None) is not None:
            self.last_emb = det.embedding
        self.votes.append((det.name, det.distance, det.confidence))
        self._recompute()

    def mark_missed(self) -> None:
        """No detection matched this pass: coast (keep box + label), age toward drop."""
        self.misses += 1

    def _recompute(self) -> None:
        """
        Rolling-window plurality vote with hysteresis.

        A label is confirmed only once it wins CONFIRM_FRAMES of the window AND is
        the strict (untied) plurality. Once a label is showing, it stays until a
        DIFFERENT label earns confirmation — never flips on a tie or a single stray
        frame. "Unknown" is just another label, so it is debounced identically.
        """
        names = [v[0] for v in self.votes]
        if not names:
            return
        ranked = Counter(names).most_common()
        top_label, top_count = ranked[0]
        tied = sum(1 for _, c in ranked if c == top_count) > 1
        strong = top_count >= config.CONFIRM_FRAMES and not tied

        if self.confirmed_name is None:
            if strong:
                self.confirmed_name = top_label
        elif top_label != self.confirmed_name and strong:
            self.confirmed_name = top_label

    def _display_metrics(self) -> Tuple[float, float]:
        """(confidence, distance) from the most recent vote matching the confirmed name."""
        for name, dist, conf in reversed(self.votes):
            if name == self.confirmed_name:
                return conf, dist
        last = self.votes[-1]
        return last[2], last[1]

    def to_match(self) -> FaceMatch:
        """Emit a FaceMatch with the stable label + this track's id."""
        if self.confirmed_name is None:
            dist = self.votes[-1][1] if self.votes else float("inf")
            return FaceMatch(bbox=self.bbox, name="...", confidence=0.0,
                             distance=dist, track_id=self.id)
        conf, dist = self._display_metrics()
        if self.confirmed_name == "Unknown":
            conf = 0.0
        return FaceMatch(bbox=self.bbox, name=self.confirmed_name,
                         confidence=conf, distance=dist, track_id=self.id)


# ── Tracker ───────────────────────────────────────────────────────────────────

class FaceTracker:
    """Associates per-pass detections to persistent tracks and confirms names per track."""

    def __init__(self) -> None:
        self._tracks: dict = {}   # id -> _Track
        self._next_id = 1

    def update(self, detections: List[FaceMatch]) -> List[FaceMatch]:
        """
        Feed one recognition pass's raw detections; return the stable, confirmed,
        id-tagged faces to display (includes tracks coasting through a brief miss).
        """
        pairs, unmatched_tracks, unmatched_dets = self._associate(detections)

        for tid, di in pairs:
            self._tracks[tid].update_with(detections[di])

        for di in unmatched_dets:
            self._tracks[self._next_id] = _Track(self._next_id, detections[di])
            self._next_id += 1

        for tid in unmatched_tracks:
            self._tracks[tid].mark_missed()

        for tid in [t for t, trk in self._tracks.items() if trk.misses > config.TRACK_MAX_MISSES]:
            del self._tracks[tid]

        return [t.to_match() for t in self._tracks.values()]

    def invalidate_votes(self) -> None:
        """
        Drop every track's votes + confirmed label (keeps ids/boxes/misses) so labels
        re-derive from scratch. Call after the embedding DB changes (R reload /
        C correct). Worker-thread only.
        """
        for t in self._tracks.values():
            t.votes.clear()
            t.confirmed_name = None

    def _associate(self, dets: List[FaceMatch]):
        """
        Two-stage greedy matching → (pairs, unmatched_track_ids, unmatched_det_idxs).
        Stage 1: IoU ≥ TRACK_IOU_THRESHOLD. Stage 2 (leftovers): centroid gap within
        TRACK_CENTROID_FACTOR × mean face width (scale-adaptive, resolution-free).
        """
        track_ids = list(self._tracks.keys())
        if not track_ids:
            return [], [], list(range(len(dets)))
        if not dets:
            return [], track_ids, []

        unmatched_t = set(track_ids)
        unmatched_d = set(range(len(dets)))
        pairs = []

        # Stage 1 — greedy IoU (candidates failing the appearance veto never pair)
        cand = []
        for tid in track_ids:
            tb = self._tracks[tid].bbox
            for di in range(len(dets)):
                score = iou(tb, dets[di].bbox)
                if score >= config.TRACK_IOU_THRESHOLD and self._emb_compatible(tid, dets[di]):
                    cand.append((score, tid, di))
        for _, tid, di in sorted(cand, key=lambda c: c[0], reverse=True):
            if tid in unmatched_t and di in unmatched_d:
                pairs.append((tid, di))
                unmatched_t.discard(tid)
                unmatched_d.discard(di)

        # Stage 2 — greedy centroid fallback on the leftovers
        cand = []
        for tid in unmatched_t:
            tb = self._tracks[tid].bbox
            for di in unmatched_d:
                db = dets[di].bbox
                gap = centroid_dist(tb, db)
                gate = config.TRACK_CENTROID_FACTOR * 0.5 * (width(tb) + width(db))
                if gap <= gate and self._emb_compatible(tid, dets[di]):
                    cand.append((gap, tid, di))
        for _, tid, di in sorted(cand, key=lambda c: c[0]):
            if tid in unmatched_t and di in unmatched_d:
                pairs.append((tid, di))
                unmatched_t.discard(tid)
                unmatched_d.discard(di)

        return pairs, list(unmatched_t), list(unmatched_d)

    def _emb_compatible(self, tid: int, det: FaceMatch) -> bool:
        """
        Appearance veto: refuse a geometric link when the detection's face embedding
        is far from the track's last one — two people crossing paths keep their own
        ids instead of swapping. Permissive by design: missing embeddings (spoofed
        faces, synthetic tests, veto disabled) always pass.
        """
        veto = getattr(config, "TRACK_EMB_VETO", 0)
        if not veto:
            return True
        track_emb = self._tracks[tid].last_emb
        det_emb   = getattr(det, "embedding", None)
        if track_emb is None or det_emb is None:
            return True
        return (1.0 - float(np.dot(track_emb, det_emb))) <= veto
