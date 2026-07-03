import logging
import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

import numpy as np

from facerec import config

_log = logging.getLogger(__name__)

# ── SQL ───────────────────────────────────────────────────────────────────────

_CREATE_PEOPLE = """
CREATE TABLE IF NOT EXISTS people (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    embedding     BLOB    NOT NULL,
    photo_count   INTEGER DEFAULT 1,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen     TIMESTAMP
);
"""

_CREATE_LOG = """
CREATE TABLE IF NOT EXISTS detection_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT,
    confidence REAL,
    timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# key/value store for system metadata (model version, detector used, etc.)
_CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ── Embedding encryption at rest ──────────────────────────────────────────────
# Fernet-encrypted BLOBs are prefixed with a magic marker so encrypted and legacy
# plaintext rows can coexist; decryption works regardless of the config flag so
# turning ENCRYPT_EMBEDDINGS off never locks you out of existing data.

_ENC_PREFIX   = b"ENC1"
_DPAPI_PREFIX = b"DPAPI1"
_fernet = {"obj": None, "loaded": False}


def _dpapi(data: bytes, protect: bool) -> bytes:
    """
    Windows DPAPI wrap/unwrap (CryptProtectData/CryptUnprotectData, user scope) so
    the key file only decrypts under this Windows account — copying data/ to
    another machine yields an unusable key. Raises OSError off-Windows or on failure.
    """
    import ctypes
    import ctypes.wintypes as wt

    class _BLOB(ctypes.Structure):
        _fields_ = [("cbData", wt.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32  = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    buf_in   = ctypes.create_string_buffer(data, len(data))
    blob_in  = _BLOB(len(data), ctypes.cast(buf_in, ctypes.POINTER(ctypes.c_char)))
    blob_out = _BLOB()

    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out))
    if not ok:
        raise OSError("DPAPI call failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def _get_fernet(for_decrypt: bool = False):
    """Return a cached Fernet, or None when encryption is off / unavailable."""
    if not for_decrypt and not getattr(config, "ENCRYPT_EMBEDDINGS", False):
        return None
    if _fernet["loaded"]:
        return _fernet["obj"]
    _fernet["loaded"] = True
    try:
        from cryptography.fernet import Fernet
    except ImportError:
        _log.warning(
            "cryptography not installed — embeddings will be stored UNENCRYPTED. "
            "Run: pip install cryptography"
        )
        return None
    key_path = getattr(config, "KEY_PATH", None)
    if not key_path:
        return None
    if os.path.exists(key_path):
        with open(key_path, "rb") as f:
            raw = f.read()
        if raw.startswith(_DPAPI_PREFIX):
            try:
                key = _dpapi(raw[len(_DPAPI_PREFIX):], protect=False)
            except OSError as exc:
                _log.error(
                    "Key at %s is DPAPI-protected for a different Windows user/"
                    "machine and cannot be unwrapped (%s). Re-register faces to "
                    "generate a new key.", key_path, exc,
                )
                return None
        else:
            key = raw.strip()
    else:
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        key = Fernet.generate_key()
        stored = key
        if getattr(config, "KEY_USE_DPAPI", False) and os.name == "nt":
            try:
                stored = _DPAPI_PREFIX + _dpapi(key, protect=True)
                _log.info("Encryption key DPAPI-protected (this Windows account only)")
            except OSError as exc:
                _log.warning("DPAPI unavailable (%s) — key stored unwrapped", exc)
        with open(key_path, "wb") as f:
            f.write(stored)
        _log.warning(
            "Generated embedding encryption key at %s — BACK IT UP and restrict "
            "access. Without it the embeddings in faces.db cannot be read.",
            key_path,
        )
    _fernet["obj"] = Fernet(key)
    return _fernet["obj"]


def _encrypt_blob(raw: bytes) -> bytes:
    f = _get_fernet()
    return _ENC_PREFIX + f.encrypt(raw) if f else raw


def _decrypt_blob(blob: bytes) -> bytes:
    """Raises ValueError if the blob is encrypted but the key is missing/wrong."""
    if not blob.startswith(_ENC_PREFIX):
        return blob
    f = _get_fernet(for_decrypt=True)
    if f is None:
        raise ValueError("encrypted blob but no usable key (see KEY_PATH in config.py)")
    from cryptography.fernet import InvalidToken
    try:
        return f.decrypt(blob[len(_ENC_PREFIX):])
    except InvalidToken:
        raise ValueError("wrong encryption key for this blob (KEY_PATH changed?)")


# ── Public API ────────────────────────────────────────────────────────────────

def init_db(db_path: str = config.DB_PATH) -> None:
    """
    Create the database file, all tables, and enable WAL mode.
    Safe to call on every startup — IF NOT EXISTS guards prevent data loss.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with _session(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(_CREATE_PEOPLE)
        conn.execute(_CREATE_LOG)
        conn.execute(_CREATE_META)
    _log.debug("DB initialised at %s", db_path)


def upsert_person(
    name: str,
    embeddings: np.ndarray,
    photo_count: int = 1,
    db_path: str = config.DB_PATH,
) -> None:
    """
    Insert or replace a person's embeddings.
    embeddings: shape (N, 512) — one row per photo, all L2-normalised float32.
    photo_count is stored so load_all_embeddings() can reshape the BLOB correctly.
    """
    matrix = np.atleast_2d(embeddings).astype(np.float32)
    blob = _encrypt_blob(matrix.tobytes())
    with _session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO people (name, embedding, photo_count)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                embedding     = excluded.embedding,
                photo_count   = excluded.photo_count,
                registered_at = CURRENT_TIMESTAMP
            """,
            (name, blob, matrix.shape[0]),
        )
    _log.info("Upserted '%s' (%d photo(s))", name, matrix.shape[0])


def load_all_embeddings(db_path: str = config.DB_PATH) -> dict:
    """
    Return {name: np.ndarray(float32, shape=(N, 512))} for every person in DB.
    N = number of registered photos. Returns {} if DB is empty or missing.

    A person whose stored BLOB length is inconsistent with photo_count is SKIPPED
    with a warning (never silently reshaped) so DB corruption / version drift is
    visible instead of quietly degrading recognition.
    """
    if not os.path.exists(db_path):
        return {}
    result = {}
    with _session(db_path) as conn:
        rows = conn.execute(
            "SELECT name, embedding, photo_count FROM people"
        ).fetchall()
    for name, blob, photo_count in rows:
        try:
            raw = _decrypt_blob(bytes(blob))
        except ValueError as exc:
            _log.warning("Cannot decrypt embedding for '%s': %s — skipping.", name, exc)
            continue
        flat = np.frombuffer(raw, dtype=np.float32).copy()
        n = max(1, photo_count or 1)
        if flat.size == 0 or flat.size % n != 0:
            _log.warning(
                "Embedding for '%s' has %d floats, not divisible by photo_count=%d — "
                "skipping. Re-register this person (register_faces.py --person %s).",
                name, flat.size, n, name,
            )
            continue
        result[name] = flat.reshape(n, -1)
    _log.debug("Loaded %d face embedding(s) from DB", len(result))
    return result


def record_detection(
    name: str,
    confidence: float,
    db_path: str = config.DB_PATH,
) -> None:
    """
    Record one detection of a KNOWN person: bump their last_seen and (when
    LOG_DETECTIONS_TO_DB is set) append a detection_log row — both in a single
    connection/transaction.

    This is on the recognition hot path, so HOW OFTEN it fires per person is
    throttled by the caller (see recognizer.py / DETECTION_LOG_INTERVAL_S) to keep
    the log meaningful ("visits", not one row per frame) and avoid DB churn.
    """
    try:
        with _session(db_path) as conn:
            conn.execute(
                "UPDATE people SET last_seen = CURRENT_TIMESTAMP WHERE name = ?",
                (name,),
            )
            if config.LOG_DETECTIONS_TO_DB:
                conn.execute(
                    "INSERT INTO detection_log (name, confidence) VALUES (?, ?)",
                    (name, round(confidence, 4)),
                )
    except Exception as exc:
        _log.warning("Could not record detection for '%s': %s", name, exc)


def detection_summary(since_hours: float = 1, db_path: str = config.DB_PATH) -> list:
    """
    Return [(name, hits, last_seen)] for known-person detections in the last
    `since_hours` hours, most-detected first. Used for the exit session report.
    """
    if not os.path.exists(db_path):
        return []
    with _session(db_path) as conn:
        return conn.execute(
            "SELECT name, COUNT(*) AS hits, MAX(timestamp) AS last_seen "
            "FROM detection_log "
            "WHERE timestamp >= datetime('now', ?) AND name != 'Unknown' "
            "GROUP BY name ORDER BY hits DESC",
            (f"-{since_hours} hours",),
        ).fetchall()


def list_registered_people(db_path: str = config.DB_PATH) -> list:
    """Return list of rows (name, photo_count, registered_at, last_seen) sorted by name."""
    if not os.path.exists(db_path):
        return []
    with _session(db_path) as conn:
        return conn.execute(
            "SELECT name, photo_count, registered_at, last_seen "
            "FROM people ORDER BY name"
        ).fetchall()


def delete_person(name: str, db_path: str = config.DB_PATH) -> bool:
    """Remove a person from the DB. Returns True if a row was deleted."""
    with _session(db_path) as conn:
        cursor = conn.execute("DELETE FROM people WHERE name = ?", (name,))
        deleted = cursor.rowcount > 0
    if deleted:
        _log.info("Deleted '%s' from DB", name)
    else:
        _log.warning("delete_person: '%s' not found in DB", name)
    return deleted


def clear_people(db_path: str = config.DB_PATH) -> None:
    """Delete every registered person and the version meta (register_faces.py --clear)."""
    with _session(db_path) as conn:
        conn.execute("DELETE FROM people")
        conn.execute("DELETE FROM meta")
    _log.info("Cleared all registrations")


def count_people(db_path: str = config.DB_PATH) -> int:
    """Return the number of registered people."""
    if not os.path.exists(db_path):
        return 0
    with _session(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]


def prune_detection_log(days: int = config.LOG_RETENTION_DAYS, db_path: str = config.DB_PATH) -> int:
    """
    Delete detection_log rows older than `days` days.
    Returns number of rows deleted. No-op if days == 0.
    """
    if days <= 0:
        return 0
    if not os.path.exists(db_path):
        return 0
    try:
        with _session(db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM detection_log WHERE timestamp < datetime('now', ?)",
                (f"-{days} days",),
            )
            n = cursor.rowcount
        if n:
            _log.info("Pruned %d old detection_log rows (>%d days)", n, days)
        return n
    except Exception as exc:
        _log.warning("Could not prune detection_log: %s", exc)
        return 0


def get_meta(key: str, db_path: str = config.DB_PATH) -> Optional[str]:
    """Retrieve a value from the meta table. Returns None if key not found."""
    if not os.path.exists(db_path):
        return None
    try:
        with _session(db_path) as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def set_meta(key: str, value: str, db_path: str = config.DB_PATH) -> None:
    """Insert or update a key/value pair in the meta table."""
    try:
        with _session(db_path) as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
    except Exception as exc:
        _log.warning("Could not write meta[%s]: %s", key, exc)


# ── Internal ──────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    """Open a new SQLite connection with sensible defaults."""
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _session(db_path: str):
    """
    Connection context manager that COMMITS on clean exit and ALWAYS closes.

    Note: `with sqlite3.connect(...) as conn` only manages the transaction — it
    never closes the connection, which previously leaked a connection on every
    call (twice per recognised face, per pass). This closes them.
    """
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
