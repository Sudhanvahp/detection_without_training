"""Unit tests for the face tracker — geometry, association, coasting, and the
per-track plurality-vote confirmation (including the Unknown-debounce fix)."""

from facerec import config
from facerec.recognizer import FaceMatch
from facerec.tracker import FaceTracker, centroid_dist, iou, width


def det(x1, y1, x2, y2, name="A", dist=0.1, conf=0.9):
    return FaceMatch(bbox=(x1, y1, x2, y2), name=name, confidence=conf, distance=dist)


# ── Geometry helpers ────────────────────────────────────────────────────────

def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # intersection 5x10=50, union 100+100-50=150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_width_and_centroid_dist():
    assert width((10, 0, 40, 0)) == 30
    assert centroid_dist((0, 0, 10, 10), (10, 0, 20, 10)) == 10.0


# ── Association / identity ───────────────────────────────────────────────────

def test_new_detections_get_unique_increasing_ids():
    t = FaceTracker()
    out = t.update([det(0, 0, 50, 50, name="A"), det(300, 300, 360, 360, name="B")])
    assert sorted(m.track_id for m in out) == [1, 2]


def test_two_stationary_faces_keep_distinct_ids():
    t = FaceTracker()
    a, b = (50, 50, 110, 110), (300, 300, 360, 360)
    t.update([det(*a), det(*b)])
    out = t.update([det(*a), det(*b)])
    assert len({m.track_id for m in out}) == 2


def test_centroid_fallback_keeps_id_for_fast_mover():
    """IoU=0 between passes but the centroid gap is within the width-scaled gate → same id."""
    t = FaceTracker()
    t.update([det(100, 100, 180, 180)])          # width 80, centroid (140,140)
    out = t.update([det(200, 100, 280, 180)])    # centroid (240,140), gap 100 <= 1.5*80=120
    assert len(out) == 1
    assert out[0].track_id == 1


# ── Coasting / age-out ───────────────────────────────────────────────────────

def test_track_coasts_then_drops():
    t = FaceTracker()
    box = (100, 100, 200, 200)
    for _ in range(config.CONFIRM_FRAMES):
        t.update([det(*box, name="ALICE")])

    # Missed passes: the track coasts (still emitted) up to TRACK_MAX_MISSES
    for i in range(config.TRACK_MAX_MISSES):
        out = t.update([])
        assert any(m.track_id == 1 for m in out), f"should still coast at miss {i + 1}"

    # One more miss exceeds TRACK_MAX_MISSES → dropped
    assert t.update([]) == []


# ── Per-track confirmation ───────────────────────────────────────────────────

def test_pending_then_confirms_after_confirm_frames():
    t = FaceTracker()
    box = (100, 100, 200, 200)
    first = t.update([det(*box, name="ALICE")])
    assert first[0].name == "..."          # not confirmed on the first pass
    out = first
    for _ in range(config.CONFIRM_FRAMES - 1):
        out = t.update([det(*box, name="ALICE")])
    assert out[0].name == "ALICE"
    assert out[0].track_id == 1


def test_name_flipflop_is_smoothed_to_majority():
    t = FaceTracker()
    box = (100, 100, 200, 200)
    for name in ["A", "B"] + ["A"] * config.CONFIRM_FRAMES:
        out = t.update([det(*box, name=name)])
    assert out[0].name == "A"              # the stray "B" never surfaces


def test_unknown_is_debounced_like_a_name():
    """The old bug: Unknown surfaced instantly. Now it must also win CONFIRM_FRAMES."""
    t = FaceTracker()
    box = (100, 100, 200, 200)
    outs = [t.update([det(*box, name="Unknown", conf=0.0, dist=0.9)])
            for _ in range(config.CONFIRM_FRAMES)]
    assert outs[0][0].name == "..."        # pending, not immediately "Unknown"
    assert outs[-1][0].name == "Unknown"   # confirmed after enough votes


def test_confirmed_label_is_sticky_against_stray_vote():
    t = FaceTracker()
    box = (100, 100, 200, 200)
    for _ in range(config.CONFIRM_FRAMES):
        t.update([det(*box, name="Unknown", conf=0.0, dist=0.9)])
    out = t.update([det(*box, name="ALICE", conf=0.9, dist=0.1)])
    assert out[0].name == "Unknown"        # one stray "ALICE" can't flip a confirmed label


def test_invalidate_votes_resets_labels():
    t = FaceTracker()
    box = (100, 100, 200, 200)
    for _ in range(config.CONFIRM_FRAMES):
        out = t.update([det(*box, name="ALICE")])
    assert out[0].name == "ALICE"
    t.invalidate_votes()
    out = t.update([det(*box, name="ALICE")])
    assert out[0].name == "..."            # votes cleared → pending again, same id
    assert out[0].track_id == 1
