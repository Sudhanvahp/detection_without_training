"""Unit tests for FaceRecognizer._best_match — nearest-identity matching and the
distance-init fix (cosine distance can exceed 1.0)."""

import numpy as np

from facerec.recognizer import FaceRecognizer


def _norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _onehot(i, dim=512):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_empty_db_returns_unknown_and_inf():
    r = FaceRecognizer()
    name, dist = r._best_match(_onehot(0))
    assert name == "Unknown"
    assert dist == float("inf")


def test_picks_nearest_identity():
    r = FaceRecognizer()
    alice = _onehot(0)
    bob   = _onehot(1)
    r._swap_known({"ALICE": alice[None, :], "BOB": bob[None, :]})

    name, dist = r._best_match(alice)
    assert name == "ALICE"
    assert dist < 1e-5


def test_distance_not_clamped_at_one():
    """Opposite vectors → cosine distance 2.0. Before the fix (best_dist init 1.0)
    this match was lost and the distance was pinned at 1.0."""
    r = FaceRecognizer()
    alice = _onehot(0)
    r._swap_known({"ALICE": alice[None, :]})

    name, dist = r._best_match(-alice)
    assert name == "ALICE"
    assert dist > 1.0
    assert abs(dist - 2.0) < 1e-5


def test_multi_photo_uses_closest_photo():
    r = FaceRecognizer()
    # ALICE has two photos; query matches the second exactly
    photos = np.stack([_onehot(0), _onehot(5)])
    r._swap_known({"ALICE": photos})

    name, dist = r._best_match(_onehot(5))
    assert name == "ALICE"
    assert dist < 1e-5
