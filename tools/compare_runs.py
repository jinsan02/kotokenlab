"""두 run 의 결과를 비교해도 되는지 config 로 확인한다 (GPU 0시간).

    .conda/python.exe tools/compare_runs.py RUN_A RUN_B [...]
    .conda/python.exe tools/compare_runs.py RUN_A RUN_B --allow seed

## 왜 필요한가

2026-09-14 ~ 15 이틀 동안 **같은 실수를 네 번** 했다. 전부 "두 run 의 숫자를
나란히 놓았는데 설정이 달랐다" 다.

    예산 지점 vs 총량   R1 판정 기준이 1차의 *체크포인트* 였다. 성립 안 함
    --eval-bytes        기본 1MB. 1차는 20MB 를 명시했다. 시간 추정이 51% 낮았다
    --max-bytes         기본 5MB. 1차 기준값은 2MB. **S1 게이트가 거짓 실패했다**
    --pool-docs         기본 30,000. 노이즈 run 은 5,000. 돌리기 직전에 잡았다

셋은 막았고 하나는 당했다. 당한 건은 위험했다 — 게이트가 "untie 가 깨졌다,
멈춘다" 를 출력했고, 코드 BPB 가 7.6% 움직인 게 이상해서 겨우 잡았다.
그대로 믿었으면 하루를 날렸다.

**사람이 매번 config 를 대조하는 것으로는 못 막는다.** 도구로 만든다.

## 무엇을 보나

config 의 필드를 셋으로 나눈다.

    치명   측정값 자체를 바꾼다. 다르면 비교 불가
    시간   벽시계만 바꾼다. 달라도 된다 (평가 간격 등)
    신원   당연히 다르다 (model, revision)

`seed` 는 기본적으로 **치명** 이다. 조건 간 비교는 같은 seed 여야 한다.
sigma 를 재려고 일부러 seed 를 바꾼 3 run 을 비교할 때만 `--allow seed` 를 준다.

## 기록이 없는 필드

옛 run 의 config 에는 `eval_budget` 이 없다 (2026-09-15 에 추가했다).
**없는 것을 "같다" 로 취급하지 않는다** — "기록 없음" 으로 경고한다.
비교해도 되는지 알 수 없다는 뜻이고, 그것이 정확한 상태다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "experiments" / "runs"

# 측정값 자체를 바꾸는 필드. 다르면 두 run 의 숫자를 나란히 놓을 수 없다.
CRITICAL = {
    # CPT
    "budget_bytes", "budget_tokens", "seq_len", "micro_bs", "accum", "lr",
    "lr_schedule", "pool_docs", "skip_docs", "optimizer", "dtype",
    "grad_checkpointing", "eval_budget", "seed",
    # eval (bpb.py)
    "max_bytes", "split",
}

# 벽시계만 바꾼다. 평가는 비파괴적이고 BPB 를 안 건드린다.
TIMING = {"eval_bytes"}

# 당연히 다르다.
IDENTITY = {"model", "revision", "name", "purpose"}


def load(run_id: str) -> dict:
    p = RUNS / run_id / "config.json"
    if not p.exists():
        raise SystemExit(f"{p} 가 없다. run_id 를 확인하라")
    return json.loads(p.read_text(encoding="utf-8"))


def load_argv(run_id: str) -> str:
    """원장의 argv 컬럼. **config 보다 이쪽이 진실이다** — 실제로 준 플래그다.

    2026-09-15 에 config 만 보고는 1차 노이즈 run 의 --eval-budget 을 알 수 없었다.
    argv 에는 그대로 남아 있었고, 거기서 1차 본 run 이 2MB, 노이즈 run 이 1MB 를
    썼다는 것이 드러났다.
    """
    import csv as _csv
    led = ROOT / "experiments" / "LEDGER.tsv"
    with led.open(encoding="utf-8") as fh:
        for r in _csv.DictReader(fh, delimiter="	"):
            if r["run_id"] == run_id and r["status"] == "ok":
                return r.get("argv", "") or ""
    return ""


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="두 run 을 비교해도 되는지 config 로 확인한다")
    ap.add_argument("runs", nargs="+", help="run_id 둘 이상")
    ap.add_argument("--allow", nargs="*", default=[],
                    help="달라도 되는 필드 (예: seed — sigma 측정용 3 run 비교)")
    args = ap.parse_args(argv)

    if len(args.runs) < 2:
        raise SystemExit("run_id 를 둘 이상 줘라")

    allow = set(args.allow)
    cfgs = {r: load(r) for r in args.runs}
    keys = sorted(set().union(*(set(c) for c in cfgs.values())))

    fatal: list = []
    unrecorded: list = []
    rows: list = []

    missing_all: list = []
    for k in keys:
        vals = [cfgs[r].get(k, "<기록 없음>") for r in args.runs]
        if len(set(map(repr, vals))) == 1:
            # 전부 같다. 단 **전부 기록이 없는** 경우는 "같다" 가 아니라 "모른다" 다.
            if vals[0] == "<기록 없음>" and k in CRITICAL:
                missing_all.append(k)
            continue
        if k in IDENTITY:
            kind = "신원"
        elif k in TIMING:
            kind = "시간"
        elif k in allow:
            kind = "허용"
        elif k in CRITICAL:
            kind = "**치명**"
            fatal.append(k)
        else:
            kind = "미분류"
            fatal.append(k)                # 모르는 필드는 치명으로 다룬다
        if "<기록 없음>" in map(str, vals):
            unrecorded.append(k)
        rows.append((k, kind, vals))

    w = max(len(r) for r in args.runs)
    print("비교 대상")
    for r in args.runs:
        print(f"  {r}")
    if allow:
        print(f"달라도 된다고 지정한 필드: {', '.join(sorted(allow))}")
    print()

    if not rows:
        print("모든 필드가 같다.")
    else:
        print(f"{'필드':22s} {'분류':8s} " + " ".join(f"{r[:w]:>{w}s}" for r in args.runs))
        print("-" * (32 + (w + 1) * len(args.runs)))
        for k, kind, vals in rows:
            print(f"{k:22s} {kind:8s} " + " ".join(f"{str(v)[:w]:>{w}s}" for v in vals))
    print()

    print("원장 argv — config 보다 이쪽이 진실이다")
    for r in args.runs:
        a = load_argv(r)
        print(f"  {r}")
        print(f"    {a if a else '<원장에 argv 없음>'}")
    print()

    if missing_all:
        print("모든 run 에서 기록이 없는 치명 필드: " + ", ".join(sorted(missing_all)))
        print("  **\"같다\" 가 아니라 \"둘 다 모른다\" 다.** 위 argv 로 직접 확인하라.")
        print()

    if unrecorded:
        print("기록 없는 필드: " + ", ".join(sorted(set(unrecorded))))
        print("  옛 run 의 config 에 없던 값이다. **\"같다\" 가 아니라 \"모른다\" 다.**")
        print("  그 필드가 측정값을 바꾸는 것이면 비교 근거를 따로 적어야 한다.")
        print()

    if fatal:
        print(f"  비교 불가 — 측정값을 바꾸는 필드가 다르다: {', '.join(sorted(set(fatal)))}")
        print()
        print("  두 run 의 숫자를 나란히 놓지 마라. 조건을 맞춰 다시 돌리거나,")
        print("  왜 비교 가능한지를 기록에 명시하라.")
        print("  (docs/DESIGN_DELTA.md 3-8 · PLAN.md \"R1 판정 기준 정정\")")
        return 1

    print("  비교 가능 — 측정값을 바꾸는 필드가 전부 같다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
