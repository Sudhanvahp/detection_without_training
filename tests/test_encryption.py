"""Embedding encryption at rest — blobs are ciphertext on disk, transparent on load,
and legacy plaintext rows stay readable."""

import sqlite3

import numpy as np

from facerec import config
from facerec import database


def _matrix(n=2, dim=512, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, dim)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_upsert_encrypts_blob_on_disk(tmp_path):
    db = str(tmp_path / "enc.db")
    database.init_db(db)
    m = _matrix()
    database.upsert_person("ALICE", m, photo_count=2, db_path=db)

    with sqlite3.connect(db) as conn:
        blob = conn.execute("SELECT embedding FROM people WHERE name='ALICE'").fetchone()[0]
    assert bytes(blob[:4]) == b"ENC1"
    assert bytes(blob) != m.tobytes()          # not plaintext

    loaded = database.load_all_embeddings(db)  # transparent decryption
    np.testing.assert_allclose(loaded["ALICE"], m, rtol=1e-6)


def test_legacy_plaintext_rows_still_load(tmp_path):
    db = str(tmp_path / "legacy.db")
    database.init_db(db)
    m = _matrix(n=1, seed=1)
    with sqlite3.connect(db) as conn:  # write a pre-encryption row directly
        conn.execute(
            "INSERT INTO people (name, embedding, photo_count) VALUES (?, ?, ?)",
            ("BOB", m.tobytes(), 1),
        )
    loaded = database.load_all_embeddings(db)
    np.testing.assert_allclose(loaded["BOB"], m, rtol=1e-6)
