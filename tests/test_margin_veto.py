"""Second-best margin matching and the tracker's embedding association veto."""

import numpy as np

from facerec import config
from facerec.recognizer import FaceMatch, FaceRecognizer
from facerec.tracker import FaceTracker


def _onehot(i, dim=512):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _blend(a, b, w):
    v = w * a + (1 - w) * b
    return (v / np.linalg.norm(v)).astype(np.float32)


# ── Margin ────────────────────────────────────────────────────────────────────

def test_margin_inf_with_single_person():
    r = FaceRecognizer()
    r._swap_known({"ALICE": _onehot(0)[None, :]})
    name, dist, margin = r._match_with_margin(_onehot(0))
    assert name == "ALICE" and margin == float("inf")


def test_margin_is_gap_to_other_identity():
    r = FaceRecognizer()
    alice, bob = _onehot(0), _onehot(1)
    r._swap_known({"ALICE": alice[None, :], "BOB": bob[None, :]})
    # query sits between them but closer to ALICE
    q = _blend(alice, bob, 0.8)
    name, dist, margin = r._match_with_margin(q)
    assert name == "ALICE"
    assert 0 < margin < 1.0
    # exact match → margin is the full gap to BOB
    name2, dist2, margin2 = r._match_with_margin(alice)
    assert margin2 > margin


def test_ambiguous_face_between_two_people_has_tiny_margin():
    """A face equidistant from two enrolled people must have margin ~0 —
    exactly the case the margin gate turns into Unknown."""
    r = FaceRecognizer()
    a, b = _onehot(0), _onehot(1)
    r._swap_known({"A": a[None, :], "B": b[None, :]})
    q = _blend(a, b, 0.5)
    _, _, margin = r._match_with_margin(q)
    assert margin < 1e-6


# ── Tracker embedding veto ────────────────────────────────────────────────────

def det(box, name, emb):
    return FaceMatch(bbox=box, name=name, confidence=0.9, distance=0.1, embedding=emb)


def test_same_face_links_normally_with_embeddings():
    t = FaceTracker()
    e = _onehot(0)
    t.update([det((100, 100, 200, 200), "A", e)])
    out = t.update([det((110, 100, 210, 200), "A", e)])
    assert out[0].track_id == 1


def test_different_face_in_same_spot_gets_new_id():
    """Identity-swap guard: a detection overlapping an old track but with a very
    different face embedding must NOT inherit the track id."""
    t = FaceTracker()
    t.update([det((100, 100, 200, 200), "A", _onehot(0))])
    out = t.update([det((100, 100, 200, 200), "B", _onehot(1))])  # emb dist = 1.0 > veto
    ids = {m.track_id for m in out}
    assert 2 in ids                       # the new face got a fresh id
    assert config.TRACK_EMB_VETO < 1.0    # sanity: veto actually applies here


def test_missing_embeddings_bypass_veto():
    """Synthetic/legacy detections without embeddings associate by geometry only."""
    t = FaceTracker()
    t.update([det((100, 100, 200, 200), "A", None)])
    out = t.update([det((100, 100, 200, 200), "B", None)])
    assert out[0].track_id == 1
