"""Unit tests for the greedy-diversity embedding selection and end-to-end pruning."""

import numpy as np

from facerec import database
import manage_db


def _norm_rows(m):
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def test_select_returns_all_when_k_ge_n():
    m = _norm_rows(np.random.RandomState(0).randn(3, 8))
    assert manage_db.select_diverse_indices(m, 5) == [0, 1, 2]


def test_select_keeps_distinct_directions_over_near_duplicates():
    base = np.array([
        [1, 0, 0, 0],       # distinct direction
        [0, 1, 0, 0],       # distinct direction
        [0, 0, 1, 0],       # distinct direction
        [1, 0, 0, 0.01],    # near-duplicate of row 0
        [1, 0, 0, 0.02],    # near-duplicate of row 0
    ], dtype=np.float32)
    m = _norm_rows(base)
    sel = manage_db.select_diverse_indices(m, 3)
    assert len(sel) == 3
    assert set(sel) == {0, 1, 2}   # the three orthogonal directions, not the dups


def test_prune_caps_photo_count(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    m = _norm_rows(np.random.RandomState(1).randn(10, 512))
    database.upsert_person("ALICE", m, photo_count=10, db_path=db)

    manage_db.prune_embeddings(max_photos=4, db_path=db)

    assert database.load_all_embeddings(db)["ALICE"].shape == (4, 512)


def test_prune_noop_when_under_cap(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    m = _norm_rows(np.random.RandomState(2).randn(3, 512))
    database.upsert_person("BOB", m, photo_count=3, db_path=db)

    manage_db.prune_embeddings(max_photos=10, db_path=db)

    assert database.load_all_embeddings(db)["BOB"].shape == (3, 512)
