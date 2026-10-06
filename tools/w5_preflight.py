"""5주차 Final Test 대표 체크포인트 재학습을 돌리기 전에 빈틈을 찾는다 (GPU 0시간).

    .conda/python.exe tools/w5_preflight.py

동결 설정은 `tools/w5_spec.py` 의 FT_RUNS (PLAN.md "5주차 동결").
R5 seed 42 run 과 argv 가 `--tag` · `--save` 만 다른지 원장 argv 와 토큰 단위로 대조하고,
계획 config 를 `cpt.build_config` 로 실제로 만들어 R5 config 와 치명 필드를 대조한다.
나머지는 3·4주차 사전 점검과 같은 점검이다 (프로세스 · 잠금 · 시각 · 환경 · 디스크).
"""

from __future__ import annotations

import calendar
import json
import shlex
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import w5_spec as S  # noqa: E402
from tools.upd_preflight import FAIL, PASS, WARN, git, ledger_rows, ok_row, running_training_pids  # noqa: E402
from tools.warm_preflight import config_diff  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
PLANNED_HOURS = 3.0 * 1.3


def argv_diff(planned: str, prior: str) -> list:
    """두 argv 의 차이 (플래그 단위). --tag 값과 --save 유무만 달라야 한다."""
    def parse(s):
        t = shlex.split(s)
        d, i = {}, 0
        while i < len(t):
            if i + 1 < len(t) and not t[i + 1].startswith("--"):
                d[t[i]] = t[i + 1]
                i += 2
            else:
                d[t[i]] = True
                i += 1
        return d
    a, b = parse(planned), parse(prior)
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def main() -> int:
    from src.training.cpt import build_config, build_parser
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

    for rid, (prior, argv) in S.FT_RUNS.items():
        sts = [r["status"] for r in rows if r["run_id"] == rid]
        add(f"run_id {rid}", PASS if not sts else FAIL, "비어 있음" if not sts else f"이미 있음 {sts}")
        p = ok_row(rows, prior)
        if p is None:
            add(f"기준 {prior}", FAIL, "원장 ok 행 없음")
            continue
        d = argv_diff(argv, p["argv"])
        add(f"argv vs {prior}", PASS if d == ["--save", "--tag"] else FAIL,
            "--tag · --save 만 다르다" if d == ["--save", "--tag"] else f"다른 플래그: {d}")
        planned = build_config(build_parser().parse_args(shlex.split(argv)))
        prior_cfg = json.loads((RUNS / prior / "config.json").read_text(encoding="utf-8"))
        cd = config_diff(planned, prior_cfg)
        allowed = {"warmup_bytes", "pool_extend_docs"}       # 옛 config 에 없던 필드 (4주차와 같은 근거)
        bad = [k for k, _, _ in cd if k not in allowed]
        add(f"config vs {prior}", PASS if not bad else FAIL,
            ("허용한 차이만: " + ", ".join(k for k, _, _ in cd)) if not bad else f"동결 밖: {bad}")

    add("환경 등록", PASS if env_mod.is_registered(ROOT) else FAIL, "")
    chk = latest_valid_check(ROOT)
    if chk is None:
        add("시각 검증", FAIL, "24시간 안의 기록 없음 — tools/check_clock.py --record")
    else:
        left = 24 - (time.time() - calendar.timegm(time.strptime(chk["ts_utc"], "%Y-%m-%dT%H:%M:%SZ"))) / 3600
        add("시각 검증", PASS if left >= PLANNED_HOURS else WARN, f"남은 {left:.1f}h / 계획 {PLANNED_HOURS:.1f}h")
    free = shutil.disk_usage(ROOT).free
    add("디스크 여유", PASS if free > 10 * 1024 ** 3 else FAIL, f"{free / 1024 ** 3:.0f}GB")

    for n, s, d in res:
        print(f"  [{s}] {n}  — {d}")
    return 1 if any(s == FAIL for _, s, _ in res) else 0


if __name__ == "__main__":
    raise SystemExit(main())
