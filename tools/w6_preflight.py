"""6주차 WSD run 을 돌리기 전에 빈틈을 찾는다 (GPU 0시간).

    .conda/python.exe tools/w6_preflight.py --smoke    # GPU 시험 실행 전
    .conda/python.exe tools/w6_preflight.py            # 본 실행 전

동결 설정은 `tools/w6_spec.py` (PLAN.md "6주차 동결"). 본 실행 전에는:

- argv 가 기준(5주차 재학습)과 `--lr-schedule` · `--tag` 만 다른가 (원장 argv 와 플래그 단위)
- 계획 config 를 `cpt.build_config` 로 실제로 만들어 기준 config 와 대조 — 치명 필드 중
  `lr_schedule` 하나만 달라야 한다
- 기준 이후 `src` · `configs` 를 건드린 커밋이 없는가
- `wsd_lr_by_bytes` 기본값이 동결과 같은가 (config 에는 이름만 남는다)
- GPU 시험 실행이 ok 이고 LR 이 예측과 같은가
- 기준 체크포인트가 있는가 (문서별 비교용)

나머지는 5주차와 같은 점검이다 (프로세스 · 잠금 · 시각 · 환경 · 디스크).
"""

from __future__ import annotations

import argparse
import calendar
import inspect
import json
import shlex
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import w6_spec as S  # noqa: E402
from tools.upd_preflight import (  # noqa: E402
    FAIL, PASS, WARN, code_commits_since, git, ledger_rows, ok_row, running_training_pids,
)
from tools.w5_preflight import argv_diff  # noqa: E402
from tools.warm_preflight import config_diff  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
PLANNED_HOURS = 3.0 * 1.3


def schedule_defaults_match() -> list:
    """wsd_lr_by_bytes 의 기본값이 동결과 다른 이름들."""
    from src.training.callbacks import wsd_lr_by_bytes
    p = inspect.signature(wsd_lr_by_bytes).parameters
    want = {"warmup_frac": S.WARMUP_FRAC, "min_frac": S.MIN_FRAC, "decay_frac": S.DECAY_FRAC}
    return [k for k, v in want.items() if p[k].default != v]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="6주차 사전 점검")
    ap.add_argument("--smoke", action="store_true", help="GPU 시험 실행 전 점검만")
    args = ap.parse_args(argv)

    from src.utils import env as env_mod, gpulock
    from src.utils.clock import latest_valid_check

    rows = ledger_rows()
    res: list = []
    add = lambda n, s, d: res.append((n, s, d))  # noqa: E731

    dirty = git("status", "--porcelain", "--", "src", "configs")
    add("측정 코드가 깨끗한가", PASS if not dirty else FAIL, "깨끗함" if not dirty else dirty)
    running = running_training_pids()
    add("돌고 있는 학습 프로세스", WARN if running is None else (PASS if not running else FAIL),
        "없음" if not running else f"PID {running}")
    held = gpulock.read_lock(ROOT)
    add("GPU 잠금", PASS if held is None else (FAIL if gpulock.pid_alive(int(held.get('pid', -1))) else WARN),
        "없음" if held is None else f"{held}")
    bad = schedule_defaults_match()
    add("WSD 기본값 = 동결", PASS if not bad else FAIL, "같음" if not bad else f"다름: {bad}")

    if args.smoke:
        sts = [r["status"] for r in rows if r["run_id"] == S.SMOKE_RUN]
        add(f"run_id {S.SMOKE_RUN}", PASS if "ok" not in sts else WARN,
            "비어 있음" if not sts else f"이미 있음 {sts}")
    else:
        from tools.wsd_pairs import _tsv, CURVE, smoke_check
        sm_ok = ok_row(rows, S.SMOKE_RUN) is not None
        sm, _ = smoke_check(_tsv(CURVE)) if sm_ok else (None, [])
        add("GPU 시험 실행", PASS if sm else FAIL,
            "ok · LR 이 예측과 같다" if sm else ("원장 ok 행 없음" if not sm_ok else "LR 확인 실패 — wsd_pairs 표를 본다"))
        from src.training.cpt import build_config, build_parser
        for rid, (base, run_argv) in S.WSD_RUNS.items():
            sts = [r["status"] for r in rows if r["run_id"] == rid]
            add(f"run_id {rid}", PASS if not sts else FAIL, "비어 있음" if not sts else f"이미 있음 {sts}")
            b = ok_row(rows, base)
            if b is None:
                add(f"기준 {base}", FAIL, "원장 ok 행 없음")
                continue
            d = argv_diff(run_argv, b["argv"])
            want = ["--lr-schedule", "--tag"]
            add(f"argv vs {base}", PASS if d == want else FAIL,
                "--lr-schedule · --tag 만 다르다" if d == want else f"다른 플래그: {d}")
            planned = build_config(build_parser().parse_args(shlex.split(run_argv)))
            base_cfg = json.loads((RUNS / base / "config.json").read_text(encoding="utf-8"))
            cd = [k for k, _, _ in config_diff(planned, base_cfg)]
            add(f"config vs {base}", PASS if cd == ["lr_schedule"] else FAIL,
                "lr_schedule 만 다르다" if cd == ["lr_schedule"] else f"치명 필드 차이: {cd}")
            commits = code_commits_since(b["git_commit"])
            add(f"측정 코드 계보 since {base}", PASS if not commits else FAIL,
                "변경 없음" if not commits else ", ".join(commits))
            ck = ROOT / "artifacts" / "models" / base / "model.safetensors"
            add(f"기준 체크포인트 {base}", PASS if ck.exists() else WARN,
                "있음" if ck.exists() else "없음 — 문서 단위 비교를 못 한다")

    add("환경 등록", PASS if env_mod.is_registered(ROOT) else FAIL, "")
    chk = latest_valid_check(ROOT)
    hours = 0.2 if args.smoke else PLANNED_HOURS
    if chk is None:
        add("시각 검증", FAIL, "24시간 안의 기록 없음 — tools/check_clock.py --record")
    else:
        left = 24 - (time.time() - calendar.timegm(time.strptime(chk["ts_utc"], "%Y-%m-%dT%H:%M:%SZ"))) / 3600
        add("시각 검증", PASS if left >= hours else WARN, f"남은 {left:.1f}h / 계획 {hours:.1f}h")
    free = shutil.disk_usage(ROOT).free
    add("디스크 여유", PASS if free > 10 * 1024 ** 3 else FAIL, f"{free / 1024 ** 3:.0f}GB")

    for n, s, d in res:
        print(f"  [{s}] {n}  — {d}")
    return 1 if any(s == FAIL for _, s, _ in res) else 0


if __name__ == "__main__":
    raise SystemExit(main())
