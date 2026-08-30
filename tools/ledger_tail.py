"""원장을 로컬 시각으로 읽는다.

원장의 `ts_utc` 는 **UTC** 다. 기기가 바뀌거나 서머타임이 걸려도 같은 값이
같은 순간을 가리켜야 하기 때문이다 (docs/LEDGER_SCHEMA.md).

그런데 화면에 UTC 만 보이면 한국(KST=UTC+9)에서는 9시간 어긋나 보여서
"기록 시각이 틀렸다"고 오해하기 쉽다. 이 도구가 둘을 나란히 보여준다.
**원장 값은 바꾸지 않는다.** 표시만 바꾼다.

    .conda/python.exe tools/ledger_tail.py
    .conda/python.exe tools/ledger_tail.py --table lm_metrics -n 20
    .conda/python.exe tools/ledger_tail.py --all
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402


def to_local(ts_utc: str) -> str:
    """'2026-08-30T02:26:47Z' -> '2026-08-30 11:26:47' (이 기기의 로컬 시각)."""
    if not ts_utc or ts_utc == ledger.NA:
        return ledger.NA
    try:
        dt = datetime.strptime(ts_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return ts_utc
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def tz_label() -> str:
    off = datetime.now().astimezone().utcoffset()
    total = int(off.total_seconds()) if off else 0
    sign = "+" if total >= 0 else "-"
    return f"UTC{sign}{abs(total) // 3600}"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="원장을 로컬 시각으로 본다")
    ap.add_argument("--table", default="ledger", choices=sorted(ledger.TABLES))
    ap.add_argument("-n", type=int, default=15, help="마지막 몇 행")
    ap.add_argument("--all", action="store_true", help="전부 보여준다")
    args = ap.parse_args(argv)

    rows = ledger.read_rows(args.table)
    if not rows:
        print(f"{args.table}: 행이 없다")
        return 0
    if not args.all:
        rows = rows[-args.n:]

    local = tz_label()
    print(f"{args.table}  ({len(rows)}행)   "
          f"ts_utc 는 UTC, 로컬은 {local}\n")

    if args.table == "ledger":
        print(f"{'로컬 시각':<21}{'run_id':<32}{'phase':<7}{'status':<7}"
              f"{'wall_s':>9}{'raw_bytes':>14}")
        print("-" * 90)
        for r in rows:
            print(f"{to_local(r.get('ts_utc', '')):<21}{r.get('run_id', ''):<32}"
                  f"{r.get('phase', ''):<7}{r.get('status', ''):<7}"
                  f"{r.get('wall_sec', 'NA'):>9}{r.get('raw_bytes_seen', 'NA'):>14}")
    else:
        cols = [c for c in ledger.columns(args.table) if c != "ts_utc"][:6]
        print(f"{'로컬 시각':<21}" + "".join(f"{c:<18}" for c in cols))
        print("-" * (21 + 18 * len(cols)))
        for r in rows:
            print(f"{to_local(r.get('ts_utc', '')):<21}"
                  + "".join(f"{str(r.get(c, 'NA'))[:17]:<18}" for c in cols))

    print(f"\n지금  UTC {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
          f"   로컬 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({local})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
