"""외부 HTTPS 시각으로 호스트 시계 오차를 검사하고 원장에 기록한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.clock import (  # noqa: E402
    DEFAULT_MAX_OFFSET_MS,
    DEFAULT_SERVER,
    probe_https_time,
    record_check,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="실험 전 호스트 시계 검증")
    ap.add_argument("--record", action="store_true", help="experiments/clock_checks.tsv 에 기록")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--max-offset-ms", type=float, default=DEFAULT_MAX_OFFSET_MS)
    args = ap.parse_args()

    row = probe_https_time(args.server, max_offset_ms=args.max_offset_ms)
    if args.record:
        record_check(row)
    for key in (
        "status", "server_utc", "local_midpoint_utc", "offset_ms", "rtt_ms",
        "windows_source", "clock_check_sha256", "note",
    ):
        print(f"{key:<22} {row[key]}")
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
