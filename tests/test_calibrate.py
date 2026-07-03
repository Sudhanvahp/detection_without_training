"""Unit tests for the pure threshold-optimisation logic in calibrate.py."""

import calibrate


def test_perfectly_separable_gives_f1_one():
    same = [0.10, 0.12, 0.15, 0.20]   # same-person distances (small)
    diff = [0.50, 0.55, 0.60, 0.70]   # different-person distances (large)
    threshold, f1 = calibrate.find_optimal_threshold(same, diff)
    assert f1 == 1.0
    assert 0.20 <= threshold < 0.50   # somewhere in the clean gap between the clusters


def test_overlap_reduces_f1_below_one():
    same = [0.10, 0.30, 0.45]
    diff = [0.35, 0.40, 0.60]         # overlaps the same-person range
    _, f1 = calibrate.find_optimal_threshold(same, diff)
    assert 0.0 < f1 < 1.0


def test_no_same_person_pairs_keeps_default():
    # With no true positives possible, it returns the configured default threshold.
    from facerec import config
    threshold, f1 = calibrate.find_optimal_threshold([], [0.5, 0.6])
    assert threshold == config.MATCH_THRESHOLD
    assert f1 == 0.0
