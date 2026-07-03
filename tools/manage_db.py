"""
manage_db.py — Database management and embedding optimisation tools.

Usage:
  python manage_db.py --stats                  # show DB statistics
  python manage_db.py --prune-embeddings       # remove redundant embeddings
  python manage_db.py --max-photos 20          # cap each person to 20 photos
  python manage_db.py --export embeddings.json # export all embeddings to JSON
  python manage_db.py --import embeddings.json # import embeddings from JSON
  python manage_db.py --list                   # list all registered people
"""

import argparse
import json
import os
import sqlite3
import sys

import numpy as np

# Allow running as "python tools/<script>.py" from the project root.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from facerec import config
from facerec import database


# ── Stats ─────────────────────────────────────────────────────────────────────

def show_stats() -> None:
    """Print a summary of the current database state."""
    known    = database.load_all_embeddings()
    db_size  = os.path.getsize(config.DB_PATH) / 1024

    conn = sqlite3.connect(config.DB_PATH)
    log_count = conn.execute("SELECT COUNT(*) FROM detection_log").fetchone()[0]
    oldest    = conn.execute("SELECT MIN(timestamp) FROM detection_log").fetchone()[0]
    newest    = conn.execute("SELECT MAX(timestamp) FROM detection_log").fetchone()[0]
    conn.close()

    total_emb = sum(m.shape[0] for m in known.values())
    total_ram = sum(m.nbytes   for m in known.values()) / 1024

    print(f"\n{'='*58}")
    print(f"  Database Statistics — {os.path.basename(config.DB_PATH)}")
    print(f"{'='*58}")
    print(f"  File size        : {db_size:.1f} KB")
    print(f"  Registered people: {len(known)}")
    print(f"  Total embeddings : {total_emb}  ({total_ram:.1f} KB in RAM)")
    print(f"  Detection logs   : {log_count} rows")
    if oldest:
        print(f"  Log range        : {oldest[:10]} → {(newest or '')[:10]}")
    print(f"\n  {'Name':<22} {'Photos':>7} {'RAM':>8}")
    print(f"  {'-'*40}")
    for name, matrix in sorted(known.items()):
        ram = matrix.nbytes / 1024
        print(f"  {name:<22} {matrix.shape[0]:>7} {ram:>6.1f} KB")
    print(f"{'='*58}\n")


# ── Prune ─────────────────────────────────────────────────────────────────────

def select_diverse_indices(matrix: np.ndarray, k: int) -> list:
    """
    Greedy max-min-distance selection over L2-normalised rows: return the indices of
    the `k` most diverse embeddings (each new pick is maximally dissimilar to all
    already-picked). Pure — no I/O, unit-testable. Returns all indices if k >= N.
    """
    n = matrix.shape[0]
    if k >= n:
        return list(range(n))
    selected = [0]
    while len(selected) < k:
        sel_mat  = matrix[selected]          # (kk, 512)
        sims     = matrix @ sel_mat.T        # (N, kk) cosine similarities
        min_sims = sims.min(axis=1)          # worst-case similarity per candidate
        min_sims[selected] = 1.0             # never re-pick a selected row
        selected.append(int(np.argmin(min_sims)))
    return selected


def prune_embeddings(max_photos: int, db_path: str = config.DB_PATH) -> None:
    """
    Keep the most diverse `max_photos` embeddings per person (greedy selection).
    """
    known = database.load_all_embeddings(db_path)
    print(f"\nPruning embeddings (max {max_photos} per person) ...")

    for name, matrix in known.items():
        n = matrix.shape[0]
        if n <= max_photos:
            print(f"  {name}: {n} photos — no pruning needed")
            continue

        selected = select_diverse_indices(matrix, max_photos)
        pruned = matrix[selected].astype(np.float32)
        database.upsert_person(name, pruned, photo_count=len(selected), db_path=db_path)
        print(f"  {name}: {n} → {len(selected)} (most diverse kept)")

    print("Pruning complete.\n")


# ── Export / Import ───────────────────────────────────────────────────────────

def export_embeddings(path: str) -> None:
    """Export all embeddings + metadata to a JSON file."""
    known = database.load_all_embeddings()
    payload = {
        "model_name":       config.MODEL_NAME,
        "detector_backend": config.DETECTOR_BACKEND,
        "people": {
            name: matrix.tolist()
            for name, matrix in known.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Exported {len(known)} people to {path}")


def import_embeddings(path: str) -> None:
    """Import embeddings from a JSON file produced by --export."""
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    src_model    = payload.get("model_name",       "unknown")
    src_detector = payload.get("detector_backend", "unknown")
    people       = payload.get("people", {})

    print(f"\nImporting from {path}")
    print(f"  Source model    : {src_model}")
    print(f"  Source detector : {src_detector}")
    print(f"  People          : {len(people)}\n")

    if src_model != config.MODEL_NAME or src_detector != config.DETECTOR_BACKEND:
        print("[WARNING] Source model/detector differs from current config.")
        print("  Embeddings may be incompatible. Re-register recommended.")

    database.init_db()
    for name, emb_list in people.items():
        matrix = np.array(emb_list, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[np.newaxis, :]
        database.upsert_person(name, matrix, photo_count=matrix.shape[0])
        print(f"  Imported '{name}' ({matrix.shape[0]} photo(s))")

    print(f"\nDone. {len(people)} people imported.")


# ── List ──────────────────────────────────────────────────────────────────────

def list_people() -> None:
    rows = database.list_registered_people()
    if not rows:
        print("No people registered.")
        return
    print(f"\n{'Name':<22} {'Photos':>7} {'Registered':<22} {'Last Seen'}")
    print("-" * 72)
    for r in rows:
        last = r["last_seen"] or "never"
        print(f"{r['name']:<22} {r['photo_count']:>7} {r['registered_at']:<22} {last}")
    print(f"\nTotal: {len(rows)} person(s)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Database management for the face recognition system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--stats",            action="store_true",
                        help="Show database statistics")
    parser.add_argument("--prune-embeddings", action="store_true",
                        help="Remove redundant embeddings (keeps most diverse)")
    parser.add_argument("--max-photos",       type=int, metavar="N",
                        help="Cap each person to N photos (implies --prune-embeddings)")
    parser.add_argument("--export",           metavar="FILE",
                        help="Export all embeddings to JSON")
    parser.add_argument("--import",           metavar="FILE", dest="import_file",
                        help="Import embeddings from JSON")
    parser.add_argument("--list",             action="store_true",
                        help="List all registered people")
    args = parser.parse_args()

    if not os.path.exists(config.DB_PATH):
        print(f"ERROR: Database not found at {config.DB_PATH}")
        print("Run  python register_faces.py  first.")
        sys.exit(1)

    acted = False

    if args.list:
        list_people()
        acted = True

    if args.stats:
        show_stats()
        acted = True

    if args.prune_embeddings or args.max_photos:
        cap = args.max_photos or getattr(config, "MAX_PHOTOS_PER_PERSON", 20)
        prune_embeddings(cap)
        acted = True

    if args.export:
        export_embeddings(args.export)
        acted = True

    if args.import_file:
        import_embeddings(args.import_file)
        acted = True

    if not acted:
        parser.print_help()


if __name__ == "__main__":
    main()
