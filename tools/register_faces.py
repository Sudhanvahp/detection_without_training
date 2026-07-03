"""
register_faces.py — Build face embedding database from faces/ folder.

Usage:
  python register_faces.py                   # process all subfolders
  python register_faces.py --person Alice    # process only Alice
  python register_faces.py --clear           # wipe DB and re-register all
  python register_faces.py --list            # list registered people
  python register_faces.py --delete Alice    # remove Alice from DB

Each subfolder under faces/ is treated as one person:
  faces/
    Alice/
      photo1.jpg
      photo2.jpg
    Bob/
      photo1.jpg

Tip: 3-5 diverse photos per person (frontal, slight angles, different lighting)
improve recognition accuracy significantly.
"""

import argparse
import glob
import hashlib
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# Allow running as "python tools/<script>.py" from the project root.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from facerec import clihelpers
from facerec import config
from facerec import database
from facerec import embedding
from facerec.logger import setup_logger

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_log = logging.getLogger(__name__)

_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    """Return MD5 hex digest of a file (used for duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def blur_score(image_path: str) -> float:
    """Return Laplacian variance of the image (lower = more blurry). Shared with register_live.py."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


# ── Core logic ────────────────────────────────────────────────────────────────

def extract_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Extract a single L2-normalised ArcFace embedding from one image file.
    Returns float32 ndarray of shape (512,) or None on failure.

    Delegates to embedding.embed_image so enrollment and live inference share the
    exact same preprocessing (CLAHE) and the single DeepFace lock.
    """
    return embedding.embed_image(image_path)


def register_person(name: str, person_dir: str) -> bool:
    """
    Process all images for one person, write per-photo embeddings to DB.
    Returns True on success.
    """
    image_paths = []
    for ext in _IMAGE_EXTS:
        image_paths.extend(glob.glob(os.path.join(person_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(person_dir, ext.upper())))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        _log.error("No images found in '%s'", person_dir)
        print(f"  ERROR: No images found for {name}. "
              f"Add .jpg/.png files to faces/{name}/")
        return False

    if len(image_paths) == 1:
        _log.warning("'%s' has only 1 photo — add 3-5 for better accuracy", name)
        print(f"  WARNING: {name} has only 1 photo. Add more for better accuracy.")

    # Duplicate detection via file hash
    seen_hashes: set = set()
    unique_paths = []
    for p in image_paths:
        h = _file_hash(p)
        if h in seen_hashes:
            print(f"  SKIP: '{os.path.basename(p)}' is a duplicate of another photo — skipped.")
            _log.warning("Duplicate image skipped: %s", p)
        else:
            seen_hashes.add(h)
            unique_paths.append(p)
    image_paths = unique_paths

    embeddings = []
    skipped_blur = 0
    for path in tqdm(image_paths, desc=f"  Processing {name}", unit="img", leave=False):
        # Blur quality gate at registration time
        blur = blur_score(path)
        if blur < config.REG_BLUR_THRESHOLD:
            print(f"  SKIP: '{os.path.basename(path)}' is too blurry (score={blur:.1f}<{config.REG_BLUR_THRESHOLD}) — skipped.")
            _log.warning("Blurry image skipped: %s (score=%.1f)", path, blur)
            skipped_blur += 1
            continue

        emb = extract_embedding(path)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        _log.error("No valid face found in any photo for '%s'", name)
        print(f"  ERROR: Could not extract a face from any photo for {name}. "
              f"Check image quality.")
        return False

    if skipped_blur:
        print(f"  INFO: {skipped_blur} blurry photo(s) skipped. Using {len(embeddings)} photo(s).")

    # Store all per-photo embeddings as (N, 512) matrix
    matrix = np.stack(embeddings).astype(np.float32)
    database.upsert_person(name, matrix, photo_count=len(embeddings))
    print(f"  OK  '{name}' registered from {len(embeddings)}/{len(image_paths)} photo(s).")
    return True


def register_all(faces_dir: str = config.FACES_DIR, workers: int = 1) -> dict:
    """
    Discover all subfolders of faces_dir, register each one.
    workers > 1 processes multiple people in parallel (I/O overlap; DeepFace calls are serialised).
    Returns {person_name: success_bool}.
    """
    if not os.path.isdir(faces_dir):
        print(f"\nERROR: faces/ directory not found at:\n  {faces_dir}\n"
              f"Create it and add subfolders named after each person.\n")
        sys.exit(1)

    people = sorted(
        d for d in os.listdir(faces_dir)
        if os.path.isdir(os.path.join(faces_dir, d)) and not d.startswith(".")
    )

    if not people:
        print(f"\nNo subfolders found in {faces_dir}\n"
              f"Create one subfolder per person, e.g.  faces/Alice/photo1.jpg\n")
        sys.exit(1)

    print(f"\nFound {len(people)} person folder(s): {', '.join(people)}")
    print(f"Workers: {workers}\n")

    results = {}

    if workers <= 1:
        for name in people:
            print(f"[{name}]")
            results[name] = register_person(name, os.path.join(faces_dir, name))
    else:
        _print_lock = threading.Lock()

        def _register_one(name):
            with _print_lock:
                print(f"[{name}] starting...")
            ok = register_person(name, os.path.join(faces_dir, name))
            with _print_lock:
                print(f"[{name}] {'OK' if ok else 'FAILED'}")
            return name, ok

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_register_one, name): name for name in people}
            for future in as_completed(futures):
                name, ok = future.result()
                results[name] = ok

    return results


def warn_collisions() -> None:
    """
    Cross-person proximity audit: warn when any two people have photos closer
    than MATCH_THRESHOLD + MATCH_MARGIN — those pairs can confuse live matching
    (one bad/mislabeled photo is the usual cause). Run after every registration.
    """
    known = database.load_all_embeddings()
    names = list(known.keys())
    limit = config.MATCH_THRESHOLD + getattr(config, "MATCH_MARGIN", 0.0)
    collisions = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.min(1.0 - known[a] @ known[b].T))
            if d < limit:
                collisions.append((a, b, d))
    if not collisions:
        return
    print("\n" + "!" * 70)
    print("  WARNING: enrolled photos of DIFFERENT people are suspiciously close:")
    for a, b, d in sorted(collisions, key=lambda c: c[2]):
        print(f"    {a} <-> {b}  min distance {d:.3f}  (accept limit {limit:.3f})")
    print("  Live matching may confuse these people. Check for mislabeled or")
    print("  low-quality photos (manage_db.py --prune-embeddings), or re-shoot.")
    print("!" * 70)
    for a, b, d in collisions:
        _log.warning("Embedding collision: %s <-> %s (%.3f)", a, b, d)


def _require_admin_pin(action: str) -> bool:
    """Gate destructive actions behind config.ADMIN_PIN when set. True = allowed."""
    if not config.ADMIN_PIN:
        return True
    import getpass
    try:
        entered = getpass.getpass(f"Admin PIN required to {action}: ")
    except (EOFError, KeyboardInterrupt):
        entered = ""
    if entered == config.ADMIN_PIN:
        return True
    print("Wrong PIN — aborted.")
    _log.warning("Refused '%s': wrong admin PIN", action)
    return False


def print_table() -> None:
    """Print a formatted table of all registered people."""
    rows = database.list_registered_people()
    if not rows:
        print("No people registered in the database.")
        return
    print(f"\n{'Name':<20} {'Photos':>7} {'Registered':<22} {'Last Seen'}")
    print("-" * 70)
    for r in rows:
        last = r["last_seen"] or "never"
        print(f"{r['name']:<20} {r['photo_count']:>7} {r['registered_at']:<22} {last}")
    print(f"\nTotal: {len(rows)} person(s)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build face embedding database from faces/ folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--person",  metavar="NAME", help="Register only this person")
    parser.add_argument("--delete",  metavar="NAME", help="Delete a person from DB")
    parser.add_argument("--clear",   action="store_true", help="Wipe DB before re-registering all")
    parser.add_argument("--list",    action="store_true", help="List registered people and exit")
    parser.add_argument("--workers", type=clihelpers.positive_int, default=1, metavar="N",
                        help="Parallel workers for registration (default: 1)")
    parser.add_argument("--db",      default=config.DB_PATH, help="Override DB path")
    args = parser.parse_args()

    setup_logger()
    database.init_db(args.db)

    if args.list:
        print_table()
        return

    if args.delete:
        if not _require_admin_pin(f"delete '{args.delete}'"):
            return
        if database.delete_person(args.delete, args.db):
            print(f"Deleted '{args.delete}' from database.")
        else:
            print(f"'{args.delete}' not found in database.")
        return

    if args.clear:
        if not _require_admin_pin("clear the database"):
            return
        database.clear_people(args.db)
        print("Database cleared.\n")

    print("\nInitialising DeepFace models (may download ~360 MB on first run)...")
    print("Weights will be cached in ~/.deepface/weights/ for future runs.\n")

    if args.person:
        person_dir = os.path.join(config.FACES_DIR, args.person)
        if not os.path.isdir(person_dir):
            print(f"ERROR: Directory not found: {person_dir}")
            sys.exit(1)
        print(f"[{args.person}]")
        register_person(args.person, person_dir)
    else:
        results = register_all(workers=args.workers)
        ok    = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\nDone. {ok}/{total} person(s) registered successfully.")

    # Store model/detector/preprocessing version so recognizer.py can warn on mismatch
    database.set_meta("model_name",       config.MODEL_NAME,       args.db)
    database.set_meta("detector_backend", config.DETECTOR_BACKEND, args.db)
    database.set_meta("use_clahe",        str(config.USE_CLAHE),   args.db)

    warn_collisions()

    print("\nFinal database state:")
    print_table()
    print("Run  python main.py  to start the webcam recognition.")


if __name__ == "__main__":
    main()
