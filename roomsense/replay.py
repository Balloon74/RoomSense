"""Print or replay a structured RoomSense JSONL session without a camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from roomsense.recording.session_recorder import read_session


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a RoomSense JSONL session without opening a webcam.")
    parser.add_argument("path", type=Path, help="path to a recorded RoomSense JSONL session")
    parser.add_argument("--realtime", action="store_true", help="wait between records using their recorded timing")
    args = parser.parse_args(argv)

    previous_timestamp: float | None = None
    count = 0
    try:
        for record in read_session(args.path):
            if args.realtime and previous_timestamp is not None:
                time.sleep(max(0.0, record.timestamp - previous_timestamp))
            payload = json.dumps(dict(record.payload), ensure_ascii=False, sort_keys=True, allow_nan=False)
            print(f"{record.timestamp:.3f}  {record.kind.upper():<12} {payload}")
            previous_timestamp = record.timestamp
            count += 1
    except (OSError, ValueError) as exc:
        print(f"could not read session {args.path}: {exc}", file=sys.stderr)
        return 2
    if count == 0:
        print("Session contains no records.")
    else:
        print(f"Read {count} record(s) from {args.path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
