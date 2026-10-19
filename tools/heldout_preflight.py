"""Final Test 를 열기 전에 빈틈을 찾는다 (GPU 0시간). **이 도구는 그 파일을 열지 않는다.**

    .conda/python.exe tools/heldout_preflight.py

- 개봉 태그가 아직 없는가 (있으면 이미 열린 것이다)
- spec 상수가 PLAN "5주차 동결" D 와 같은가 (대표 둘 · 예산 · seq · bootstrap)
- 대표 체크포인트가 등록된 그것인가 (디렉토리 해시 = artifacts.tsv)
- 평가 run_id 가 비어 있는가, 이름에 lockbox 단어가 없는가 (커밋 훅이 거부한다)
- 평가 파일이 있고 예산보다 큰가 — **존재와 크기만** 본다
- 측정 코드 · 프로세스 · 잠금 · 시각 · 디스크 (3~6주차와 같은 점검)
"""

from __future__ import annotations

import calendar
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import heldout_spec as S  # noqa: E402
from tools.heldout_pairs import checkpoint_hash_ok  # noqa: E402
from tools.upd_preflight import FAIL, PASS, WARN, git, ledger_rows, running_training_pids  # noqa: E402

DOCS_DIR = ROOT / "data" / "interim" / "docs"
PLANNED_HOURS = 1.0
# PLAN D 의 값 (여기와 spec 이 갈리면 둘 중 하나가 동결을 바꾼 것이다)
PLAN_D = {"ko": 20_000_000, "en": 2_000_000, "code": 2_000_000}


def spec_problems() -> list:
    bad = []
    if {lang: b for lang, _, b in S.FILES} != PLAN_D:
        bad.append("언어별 예산이 PLAN D 와 다르다")
    if S.SEQ_LEN != 2048 or S.B != 10_000 or S.RNG_SEED != 0:
        bad.append("seq_len · bootstrap 설정이 PLAN D 와 다르다")
    if sorted(r for _, r in S.CHECKPOINTS.values()) != [
            "cpt_c0_qwen_ft_seed42", "cpt_t2b_mean_ft_seed42"]:
        bad.append("대표 체크포인트가 PLAN D 의 둘이 아니다")
    for n in S.CHECKPOINTS:
        if "final_test" in S.doc_run_id(n):
            bad.append(f"{S.doc_run_id(n)}: 이름에 lockbox 단어가 있다")
    return bad


def main() -> int:
    from src.utils import env as env_mod, gpulock
    from src.utils.clock import latest_valid_check

    res: list = []
    add = lambda n, s, d: res.append((n, s, d))  # noqa: E731
    tag = git("tag", "--list", "final-test-opened")
    add("개봉 태그", PASS if not tag else FAIL, "아직 없음" if not tag else "이미 있다 — 이미 열렸다")
    bad = spec_problems()
    add("spec = PLAN D", PASS if not bad else FAIL, "같음" if not bad else " · ".join(bad))
    rows = ledger_rows()
    for n in S.CHECKPOINTS:
        h = checkpoint_hash_ok(n)
        add(f"체크포인트 해시 {n}", PASS if h else FAIL,
            "등록값과 같다" if h else ("다르다" if h is False else "등록값 없음"))
        rid = S.doc_run_id(n)
        sts = [r["status"] for r in rows if r["run_id"] == rid]
        add(f"run_id {rid}", PASS if not sts else FAIL, "비어 있음" if not sts else f"이미 있음 {sts}")
    for lang, fname, budget in S.FILES:
        p = DOCS_DIR / fname
        size = p.stat().st_size if p.exists() else 0     # 크기만 본다. 열지 않는다
        add(f"평가 파일 {lang}", PASS if size > budget * 2 else FAIL,
            f"{size / 1e6:.1f}MB (예산 {budget / 1e6:.0f}MB)" if size else "없음")

    dirty = git("status", "--porcelain", "--", "src", "configs", "tools")
    add("코드가 깨끗한가 (src · configs · tools)", PASS if not dirty else FAIL,
        "깨끗함" if not dirty else dirty)
    running = running_training_pids()
    add("돌고 있는 학습 프로세스", WARN if running is None else (PASS if not running else FAIL),
        "없음" if not running else f"PID {running}")
    held = gpulock.read_lock(ROOT)
    add("GPU 잠금", PASS if held is None else (FAIL if gpulock.pid_alive(int(held.get('pid', -1))) else WARN),
        "없음" if held is None else f"{held}")
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
