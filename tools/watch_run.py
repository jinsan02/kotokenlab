"""장시간 GPU 작업 감시 — 문제가 생기면 즉시 빠져나와 알린다.

    .conda/python.exe tools/watch_run.py artifacts/logs/phase4.log --done "PHASE 4 DONE"

세 가지를 본다.
    1. 로그의 치명 키워드 (Traceback / CUDA OOM / RuntimeError / AssertionError)
    2. 원장에 fail·abort 상태 행이 붙었는가
    3. 진척이 멎었는가 — 원장 행도 학습 곡선도 --stall 분 동안 그대로

정지 임계를 넉넉히(기본 30분) 잡는 이유는 **정상인데도 조용한 구간** 이 있기
때문이다. 문서 풀 적재가 5~10분, 모델 저장과 다음 모델 로드가 몇 분, 수술
단계는 학습 곡선을 아예 안 남긴다. 20분으로 잡으면 오탐이 난다.

GPU 를 쓰지 않고 torch 를 import 하지 않는다. 감시가 학습과 경합하면
그 자체가 측정을 오염시킨다 — 실제로 한 번 자기 감시를 의심한 적이 있다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ledger  # noqa: E402

FATAL = ("Traceback (most recent call last)", "out of memory",
         "CUDA error", "RuntimeError", "AssertionError")


def failures() -> set:
    """지금 원장에 있는 실패 행 전부. (run_id, ts, status) 로 식별한다.

    **감시 시작 시점의 집합을 기준선으로 잡고 새로 늘어난 것만 본다.**
    날짜로 거르면 두 가지를 놓친다 — 감시 전에 이미 있던 실패를 잡아 오탐을
    내고(실제로 한 번 그랬다), 실패 뒤에 ok 가 붙어 되살아난 run 도 실패로
    센다. 한 run_id 에 fail 과 ok 가 함께 있는 것은 정상이다: 고쳐서 다시
    돌린 흔적이고 원장은 append-only 라 지우지 않는다.
    """
    return {(r.get("run_id"), r.get("ts_utc"), r.get("status"))
            for r in ledger.read_rows("ledger")
            if r.get("status") in ("fail", "abort")}


# 진척의 증거가 되는 테이블. train_curve 만 보면 CPT 밖의 run 에 눈이 먼다 —
# system_bench run 을 감시하다 "20분 동안 진척 없음" 오탐을 냈다. run 이 실제로
# 무엇에 쓰는지 모르므로 결과가 쌓이는 테이블을 전부 본다.
PROGRESS = ("train_curve", "lm_metrics", "system_bench",
            "tokenizer_metrics", "capability")


def snapshot() -> tuple:
    return (len(ledger.read_rows("ledger")),
            sum(len(ledger.read_rows(t)) for t in PROGRESS))


def report(msg: str) -> None:
    print("!! 감시 발동 —", msg)
    for r in ledger.read_rows("ledger")[-3:]:
        print("  ", r["ts_utc"], r["run_id"], r["status"])


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="장시간 run 감시")
    ap.add_argument("log", help="감시할 로그 파일")
    ap.add_argument("--done", default="DONE", help="정상 종료를 뜻하는 문자열")
    ap.add_argument("--poll", type=int, default=120, help="확인 주기(초)")
    ap.add_argument("--stall", type=int, default=30, help="정지 판정(분)")
    args = ap.parse_args(argv)

    log = Path(args.log)
    base_fail = failures()          # 감시 시작 시점의 실패 집합
    last, last_change = snapshot(), time.time()

    while True:
        time.sleep(args.poll)
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        if args.done in text:
            print("정상 종료 확인")
            return 0
        for kw in FATAL:
            if kw in text:
                report(f"로그에 {kw!r}")
                return 1
        new_fail = failures() - base_fail
        if new_fail:
            report(f"새 실패 행: {sorted({r for r, _, _ in new_fail})}")
            return 1
        n_lg, n_cv = snapshot()
        if (n_lg, n_cv) != last:
            last, last_change = (n_lg, n_cv), time.time()
        elif time.time() - last_change > args.stall * 60:
            report(f"{args.stall}분 동안 진척 없음 (원장 {n_lg}행 / 곡선 {n_cv}행)")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
