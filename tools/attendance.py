"""
attendance.py — Generate attendance reports from the detection log.

Usage:
  python attendance.py                      # today's report
  python attendance.py --date 2026-06-24    # specific date
  python attendance.py --days 7             # last 7 days
  python attendance.py --csv report.csv     # export to CSV
  python attendance.py --all                # entire history
"""

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime, timedelta

# Allow running as "python tools/<script>.py" from the project root.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from facerec import config


def _connect():
    if not os.path.exists(config.DB_PATH):
        print(f"ERROR: Database not found at {config.DB_PATH}")
        print("Run  python register_faces.py  first.")
        sys.exit(1)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def report_by_date(date_str: str) -> list:
    """Return per-person detection counts for a given date (YYYY-MM-DD)."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT name,
               COUNT(*)         AS detections,
               MIN(timestamp)   AS first_seen,
               MAX(timestamp)   AS last_seen
        FROM   detection_log
        WHERE  date(timestamp) = ?
          AND  name != 'Unknown'
        GROUP  BY name
        ORDER  BY detections DESC
        """,
        (date_str,),
    ).fetchall()
    conn.close()
    return rows


def report_range(days: int) -> list:
    """Return per-person detection counts for the last N days."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT name,
               date(timestamp)  AS day,
               COUNT(*)         AS detections,
               MIN(timestamp)   AS first_seen,
               MAX(timestamp)   AS last_seen
        FROM   detection_log
        WHERE  timestamp >= datetime('now', ?)
          AND  name != 'Unknown'
        GROUP  BY name, date(timestamp)
        ORDER  BY day DESC, detections DESC
        """,
        (f"-{days} days",),
    ).fetchall()
    conn.close()
    return rows


def report_all() -> list:
    """Return all detection records grouped by person and day."""
    conn = _connect()
    rows = conn.execute(
        """
        SELECT name,
               date(timestamp)  AS day,
               COUNT(*)         AS detections,
               MIN(timestamp)   AS first_seen,
               MAX(timestamp)   AS last_seen
        FROM   detection_log
        WHERE  name != 'Unknown'
        GROUP  BY name, date(timestamp)
        ORDER  BY day DESC, name
        """
    ).fetchall()
    conn.close()
    return rows


def report_sessions(date_str: str, gap_min: int) -> list:
    """
    Derive entry/exit visits from the raw detection log: consecutive detections of
    the same person with gaps under `gap_min` minutes form one visit. Returns
    [(name, arrival, departure, dwell_minutes, detections)] for the given date.
    (Detections are throttled to one per DETECTION_LOG_INTERVAL_S, so dwell has
    that granularity.)
    """
    conn = _connect()
    rows = conn.execute(
        """
        SELECT name, timestamp FROM detection_log
        WHERE  date(timestamp) = ? AND name != 'Unknown'
        ORDER  BY name, timestamp
        """,
        (date_str,),
    ).fetchall()
    conn.close()

    sessions = []
    cur_name, start, last, hits = None, None, None, 0
    fmt = "%Y-%m-%d %H:%M:%S"

    def _close():
        if cur_name is not None:
            dwell = (last - start).total_seconds() / 60.0
            sessions.append((cur_name, start, last, dwell, hits))

    for r in rows:
        ts = datetime.strptime(r["timestamp"][:19], fmt)
        if r["name"] != cur_name or (last and (ts - last).total_seconds() > gap_min * 60):
            _close()
            cur_name, start, hits = r["name"], ts, 0
        last = ts
        hits += 1
    _close()
    return sessions


def print_sessions(sessions: list, date_str: str, gap_min: int) -> None:
    print(f"\n{'='*74}")
    print(f"  Visit Report — {date_str}  (new visit after a gap > {gap_min} min)")
    print(f"{'='*74}")
    if not sessions:
        print("  No visits recorded for this date.")
    else:
        print(f"  {'Name':<20} {'Arrival':<10} {'Departure':<10} {'Dwell':>9}  Detections")
        print(f"  {'-'*70}")
        for name, start, end, dwell, hits in sessions:
            print(f"  {name:<20} {start.strftime('%H:%M:%S'):<10} "
                  f"{end.strftime('%H:%M:%S'):<10} {dwell:>7.1f}m  {hits}")
    print(f"{'='*74}\n")


def export_sessions_csv(sessions: list, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "arrival", "departure", "dwell_minutes", "detections"])
        for name, start, end, dwell, hits in sessions:
            writer.writerow([name, start.isoformat(sep=" "), end.isoformat(sep=" "),
                             round(dwell, 1), hits])
    print(f"Exported {len(sessions)} visit(s) to {path}")


def print_daily(rows: list, date_str: str) -> None:
    print(f"\n{'='*62}")
    print(f"  Attendance Report — {date_str}")
    print(f"{'='*62}")
    if not rows:
        print("  No detections recorded for this date.")
    else:
        print(f"  {'Name':<20} {'Detections':>10}  First Seen   Last Seen")
        print(f"  {'-'*58}")
        for r in rows:
            first = r["first_seen"][11:19] if r["first_seen"] else "—"
            last  = r["last_seen"][11:19]  if r["last_seen"]  else "—"
            print(f"  {r['name']:<20} {r['detections']:>10}  {first}        {last}")
    print(f"{'='*62}\n")


def print_range(rows: list, days: int) -> None:
    print(f"\n{'='*68}")
    print(f"  Attendance Report — Last {days} Day(s)")
    print(f"{'='*68}")
    if not rows:
        print("  No detections recorded in this period.")
    else:
        print(f"  {'Name':<20} {'Date':<12} {'Detections':>10}  First     Last")
        print(f"  {'-'*64}")
        for r in rows:
            first = r["first_seen"][11:19] if r["first_seen"] else "—"
            last  = r["last_seen"][11:19]  if r["last_seen"]  else "—"
            print(f"  {r['name']:<20} {r['day']:<12} {r['detections']:>10}  {first}  {last}")
    print(f"{'='*68}\n")


def export_csv(rows: list, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "day", "detections", "first_seen", "last_seen"])
        for r in rows:
            writer.writerow([r["name"], r.get("day", ""), r["detections"],
                             r["first_seen"], r["last_seen"]])
    print(f"Exported {len(rows)} row(s) to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate attendance reports from the face recognition log.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--date", metavar="YYYY-MM-DD", help="Report for a specific date")
    parser.add_argument("--days", type=int, default=1, help="Report for last N days (default: 1 = today)")
    parser.add_argument("--all",  action="store_true",  help="Report entire detection history")
    parser.add_argument("--csv",  metavar="FILE",       help="Export results to a CSV file")
    parser.add_argument("--sessions", action="store_true",
                        help="Visit report: arrival / departure / dwell per person "
                             "(for --date or today)")
    parser.add_argument("--gap", type=int, default=30, metavar="MIN",
                        help="Minutes without detection that ends a visit (default: 30)")
    args = parser.parse_args()

    if args.sessions:
        date_str = args.date or datetime.now().strftime("%Y-%m-%d")
        sessions = report_sessions(date_str, args.gap)
        print_sessions(sessions, date_str, args.gap)
        if args.csv:
            export_sessions_csv(sessions, args.csv)

    elif args.all:
        rows = report_all()
        print_range(rows, 9999)
        if args.csv:
            export_csv(rows, args.csv)

    elif args.date:
        rows = report_by_date(args.date)
        print_daily(rows, args.date)
        if args.csv:
            export_csv(rows, args.csv)

    else:
        if args.days == 1:
            date_str = datetime.now().strftime("%Y-%m-%d")
            rows = report_by_date(date_str)
            print_daily(rows, date_str)
        else:
            rows = report_range(args.days)
            print_range(rows, args.days)
        if args.csv:
            export_csv(rows, args.csv)


if __name__ == "__main__":
    main()
