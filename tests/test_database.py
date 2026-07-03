"""Unit tests for the SQLite layer — embedding round-trip, detection recording,
retention pruning, clearing, and the corrupt-shape guard. All run on a tmp DB."""

import sqlite3

import numpy as np

from facerec import database


def _norm_rows(m):
    return (m / np.linalg.norm(m, axis=1, keepdims=True)).astype(np.float32)


def test_embedding_roundtrip(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    m = _norm_rows(np.random.RandomState(0).randn(3, 512))
    database.upsert_person("ALICE", m, photo_count=3, db_path=db)

    loaded = database.load_all_embeddings(db)
    assert set(loaded) == {"ALICE"}
    assert loaded["ALICE"].shape == (3, 512)
    assert np.allclose(loaded["ALICE"], m, atol=1e-6)


def test_upsert_replaces_not_duplicates(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    database.upsert_person("ALICE", _norm_rows(np.random.randn(2, 512)), photo_count=2, db_path=db)
    database.upsert_person("ALICE", _norm_rows(np.random.randn(4, 512)), photo_count=4, db_path=db)
    assert database.count_people(db) == 1
    assert database.load_all_embeddings(db)["ALICE"].shape == (4, 512)


def test_record_detection_and_summary(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    database.upsert_person("ALICE", _norm_rows(np.random.randn(1, 512)), photo_count=1, db_path=db)

    database.record_detection("ALICE", 0.95, db_path=db)
    rows = database.detection_summary(since_hours=1, db_path=db)
    assert len(rows) == 1
    assert rows[0][0] == "ALICE"
    assert rows[0][1] == 1  # one hit


def test_prune_detection_log_removes_old_rows(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO detection_log (name, confidence, timestamp) "
        "VALUES ('X', 0.5, datetime('now','-40 days'))"
    )
    conn.commit()
    conn.close()
    assert database.prune_detection_log(days=30, db_path=db) == 1


def test_clear_people(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    database.upsert_person("ALICE", _norm_rows(np.random.randn(1, 512)), photo_count=1, db_path=db)
    assert database.count_people(db) == 1
    database.clear_people(db)
    assert database.count_people(db) == 0


def test_load_skips_corrupt_shape(tmp_path):
    """A BLOB whose length isn't divisible by photo_count is skipped, not silently reshaped."""
    db = str(tmp_path / "t.db")
    database.init_db(db)
    m = np.zeros((2, 512), dtype=np.float32)     # 1024 floats
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO people (name, embedding, photo_count) VALUES (?, ?, ?)",
        ("BAD", m.tobytes(), 3),                  # claim 3 photos → 1024 % 3 != 0
    )
    conn.commit()
    conn.close()
    assert "BAD" not in database.load_all_embeddings(db)


def test_meta_roundtrip(tmp_path):
    db = str(tmp_path / "t.db")
    database.init_db(db)
    assert database.get_meta("use_clahe", db) is None
    database.set_meta("use_clahe", "True", db)
    assert database.get_meta("use_clahe", db) == "True"
