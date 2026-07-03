"""
calibrate.py — Auto-calibrate the matching threshold for best accuracy.

Loads all registered embeddings, computes pairwise cosine distances between
same-person photo pairs and different-person pairs, then finds the threshold
that maximises the F1 score (best balance of precision and recall).

Usage:
  python calibrate.py           # show recommended threshold
  python calibrate.py --apply   # write it to config.py automatically
"""

import argparse
import os
import re
import sys

import numpy as np

# Allow running as "python tools/<script>.py" from the project root.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from facerec import config
from facerec import database


def compute_distances() -> tuple:
    """
    Returns (same_distances, diff_distances) — lists of cosine distances
    between photo pairs of the same person and different people.
    """
    known = database.load_all_embeddings()
    names = list(known.keys())

    if len(names) < 2:
        print("ERROR: Need at least 2 registered people to calibrate.")
        sys.exit(1)

    same_dists = []
    diff_dists = []

    for i, name_a in enumerate(names):
        mat_a = known[name_a]   # (N, 512)

        # Same-person: distances between each pair of photos
        if mat_a.shape[0] > 1:
            for j in range(mat_a.shape[0]):
                for k in range(j + 1, mat_a.shape[0]):
                    d = float(1.0 - mat_a[j] @ mat_a[k])
                    same_dists.append(d)

        # Different-person: distances across all pairs with every other person
        for name_b in names[i + 1:]:
            mat_b = known[name_b]   # (M, 512)
            cross = 1.0 - mat_a @ mat_b.T   # (N, M)
            diff_dists.extend(cross.flatten().tolist())

    return same_dists, diff_dists


def find_optimal_threshold(same_dists: list, diff_dists: list) -> tuple:
    """
    Sweep candidate thresholds, return (best_threshold, best_f1).
    F1 treats same-person detection as the positive class.
    """
    best_threshold = config.MATCH_THRESHOLD
    best_f1        = 0.0

    for t in np.arange(0.20, 0.65, 0.005):
        tp = sum(1 for d in same_dists if d < t)   # same-person correctly matched
        fp = sum(1 for d in diff_dists if d < t)   # different-person wrongly matched
        fn = sum(1 for d in same_dists if d >= t)  # same-person missed

        if tp == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            continue

        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1        = f1
            best_threshold = round(float(t), 3)

    return best_threshold, best_f1


def error_rates(same_dists: list, diff_dists: list, threshold: float) -> tuple:
    """(FAR, FRR) at a threshold: FAR = impostor pairs accepted, FRR = genuine pairs rejected."""
    far = sum(1 for d in diff_dists if d < threshold) / len(diff_dists) if diff_dists else 0.0
    frr = sum(1 for d in same_dists if d >= threshold) / len(same_dists) if same_dists else 0.0
    return far, frr


def print_error_report(same_dists: list, diff_dists: list) -> None:
    """FAR/FRR table across thresholds — the production accuracy picture."""
    print(f"\n  {'Threshold':>9}  {'FAR (false accept)':>19}  {'FRR (false reject)':>19}")
    print(f"  {'-'*51}")
    for t in (0.30, 0.34, config.MATCH_THRESHOLD, 0.42, 0.46, 0.50):
        far, frr = error_rates(same_dists, diff_dists, t)
        mark = "  <- current" if abs(t - config.MATCH_THRESHOLD) < 1e-9 else ""
        print(f"  {t:>9.3f}  {far:>18.2%}  {frr:>18.2%}{mark}")
    print("\n  NOTE: computed from enrollment photos only — for a true field number,")
    print("  collect probe shots from the live camera and re-run after enrolling them.")


def apply_to_config(threshold: float) -> None:
    """Overwrite MATCH_THRESHOLD in config.py."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facerec", "config.py")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = re.sub(
        r"MATCH_THRESHOLD\s*=\s*[\d.]+[^\n]*",
        f"MATCH_THRESHOLD = {threshold}  # auto-calibrated",
        content,
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"config.py updated: MATCH_THRESHOLD = {threshold}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-calibrate face recognition threshold.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--apply", action="store_true",
                        help="Write optimal threshold to config.py")
    args = parser.parse_args()

    if not os.path.exists(config.DB_PATH):
        print(f"ERROR: Database not found at {config.DB_PATH}. Run register_faces.py first.")
        sys.exit(1)

    print("\nLoading embeddings ...")
    same_dists, diff_dists = compute_distances()

    print(f"  Same-person pairs     : {len(same_dists)}")
    print(f"  Different-person pairs: {len(diff_dists)}")

    if same_dists:
        print(f"\n  Same-person  — min={min(same_dists):.4f}  "
              f"max={max(same_dists):.4f}  avg={np.mean(same_dists):.4f}")
    if diff_dists:
        print(f"  Diff-person  — min={min(diff_dists):.4f}  "
              f"max={max(diff_dists):.4f}  avg={np.mean(diff_dists):.4f}")

    print_error_report(same_dists, diff_dists)

    optimal, f1 = find_optimal_threshold(same_dists, diff_dists)

    print(f"\n{'='*52}")
    print(f"  Current threshold : {config.MATCH_THRESHOLD}")
    print(f"  Optimal threshold : {optimal}  (F1 = {f1:.4f})")
    print(f"{'='*52}")

    if same_dists and diff_dists:
        gap = min(diff_dists) - max(same_dists)
        if gap > 0:
            print(f"\n  Separation gap: +{gap:.4f}  (clean — same/diff do not overlap)")
        else:
            print(f"\n  WARNING: overlap of {gap:.4f} between same/diff distances.")
            print("  Add more diverse photos per person to improve separation.")

    if args.apply:
        apply_to_config(optimal)
        print("\nDone. Restart main.py to use the new threshold.")
    else:
        print(f"\nTo apply: python calibrate.py --apply")


if __name__ == "__main__":
    main()
