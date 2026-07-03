"""clihelpers.py — argparse `type=` validators shared by the CLI entry points.

Turning bad values into a clean argparse error (instead of a later crash/hang):
e.g. --skip 0 would raise ZeroDivisionError, --photos 0 would loop forever.
"""

import argparse


def positive_int(value) -> int:
    """int >= 1."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got '{value}'")
    if iv < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {iv})")
    return iv


def nonneg_int(value) -> int:
    """int >= 0."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer, got '{value}'")
    if iv < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0 (got {iv})")
    return iv


def threshold(value) -> float:
    """float in (0, 2] — the valid cosine-distance range."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a number, got '{value}'")
    if not (0.0 < fv <= 2.0):
        raise argparse.ArgumentTypeError(f"must be in (0, 2] (got {fv})")
    return fv
