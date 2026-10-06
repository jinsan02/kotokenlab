"""4주차 신규 행 warm-start 를 돌리기 **전에** 빈틈을 찾는다 (GPU 0시간).

    .conda/python.exe tools/warm_preflight.py           # 전부 (토큰화 포함, 1분 안팎)
    .conda/python.exe tools/warm_preflight.py --quick   # 데이터 흐름 재현 생략

동결 설정은 `tools/warm_spec.py` (PLAN.md "4주차 run 설정 동결").
3주차 `tools/upd_preflight.py` 의 검증된 부품(데이터 흐름 재현 · 프로세스 검사)을
그대로 쓴다. 이번에 더한 것 셋:

- **계획 config 를 실제로 만들어 직접 CPT config 와 대조한다.** `cpt.build_config`
  로 계획 argv 의 config 를 만들고, `compare_runs` 의 치명 필드 분류로 차이를 낸다.
  `warm_spec.ALLOW_FIELDS` 밖의 차이가 하나라도 있으면 실패다. 3주차 seed 42 의
  `warmup_bytes` 처럼 **돌린 뒤에야** 드러나는 일을 막는다 (MISTAKES 10-02)
- **코드 계보를 근거 목록과 대조한다.** 직접 CPT(7683b83) 이후 src·configs 커밋이
  `warm_spec.REVIEWED_COMMITS` 와 정확히 같아야 한다. 새 커밋이 끼면 막는다
- **GPU 잠금 파일** 이 살아 있는 프로세스의 것이면 막는다 (`src/utils/gpulock.py`)

판정하지 않는다. "돌려도 되는가" 만 본다.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import warm_spec as W  # noqa: E402
from tools.upd_preflight import (  # noqa: E402
    FAIL, PASS, POOL, WARN, Stream, byte_table, doc_arrays, git, ledger_rows,
    load_tokenizer, ok_row, order_indices, parse_argv, running_training_pids,
)

RUNS = ROOT / "experiments" / "runs"
ARTIFACTS = ROOT / "experiments" / "artifacts.tsv"
OUT_MD = ROOT / "reports" / "tables" / "warm_preflight.md"
OUT_JSON = ROOT / "reports" / "tables" / "warm_preflight.json"
SEQ_LEN, TOKENS_PER_UPDATE = 2048, 2048 * 2 * 8
PLANNED_HOURS = 1.3 * 1.3          # seed42 한 run (gate 뒤 두 run 은 그때 다시 점검한다)
MIN_FREE_VRAM_MB = 12_000
MIN_FREE_DISK = 10 * 1024 ** 3


def planned_config(seed: int) -> dict:
    from src.training.cpt import build_config, build_parser
    return build_config(build_parser().parse_args(W.argv(seed)))


def config_diff(planned: dict, direct: dict) -> list:
    """compare_runs 의 치명 필드 가운데 값이 다르거나 한쪽에만 있는 것."""
    from tools.compare_runs import CRITICAL
    out = []
    for k in sorted(CRITICAL & (set(planned) | set(direct))):
        a, b = planned.get(k, "<기록 없음>"), direct.get(k, "<기록 없음>")
        if a != b:
            out.append((k, a, b))
    return out


def static_checks(rows: list) -> list:
    res: list = []
    add = lambda n, s, d: res.append((n, s, d))  # noqa: E731

    dirty = git("status", "--porcelain", "--", "src", "configs")
    add("측정 코드가 깨끗한가", PASS if not dirty else FAIL,
        "src·configs 에 커밋 안 된 변경 없음" if not dirty else dirty.replace("\n", " · "))

    running = running_training_pids()
    add("돌고 있는 학습 프로세스", WARN if running is None else (PASS if not running else FAIL),
        "프로세스 목록을 못 읽었다" if running is None else
        ("없음" if not running else "이미 돌고 있다: PID " + ", ".join(map(str, running))))

    from src.utils import gpulock
    held = gpulock.read_lock(ROOT)
    if held is None:
        add("GPU 잠금", PASS, "잠금 파일 없음")
    else:
        pid = int(held.get("pid", -1) or -1)
        alive = gpulock.pid_alive(pid)
        add("GPU 잠금", FAIL if alive else WARN,
            f"PID {pid} · {held.get('run_id')} — " +
            ("살아 있다. 끝난 뒤 시작한다" if alive else "죽은 잠금이다. 다음 run 이 회수한다"))

    for s in W.SEEDS:
        rid = W.run_id(s)
        sts = [r["status"] for r in rows if r["run_id"] == rid]
        # 결과 없이 끝난 시도(fail·abort)뿐이면 같은 id 로 다시 돌아도 된다 — 집계는 ok
        # 행만 읽는다. start 수가 종료 행 수보다 많으면 끝나지 않은 시도가 있다는 뜻이다.
        unterminated = sts.count("start") > sum(sts.count(x) for x in ("ok", "fail", "abort"))
        if not sts:
            st, msg = PASS, "비어 있음"
        elif "ok" in sts:
            st, msg = PASS, "이미 돌았다 (ok)"
        elif unterminated:
            st, msg = FAIL, f"끝나지 않은 시도가 있다 {sts} — 원인을 먼저 본다"
        else:
            st, msg = PASS, f"이전 시도가 결과 없이 끝났다 {sts} — 같은 id 로 다시 돈다"
        add(f"run_id seed{s}", st, f"{rid} — {msg}")

    direct_commit = None
    for s in W.SEEDS:
        d, c = ok_row(rows, f"{W.DIRECT}{s}"), ok_row(rows, f"{W.CTRL}{s}")
        add(f"비교 대상 seed{s}", PASS if d and c else FAIL,
            f"직접 CPT {'있음' if d else '없음'} · C0 {'있음' if c else '없음'}")
        if not d:
            continue
        direct_commit = direct_commit or d["git_commit"]
        cfg_p = RUNS / f"{W.DIRECT}{s}" / "config.json"
        direct = json.loads(cfg_p.read_text(encoding="utf-8"))
        diff = config_diff(planned_config(s), direct)
        outside = [k for k, _, _ in diff if k not in W.ALLOW_FIELDS]
        add(f"계획 config vs 직접 CPT seed{s}", PASS if not outside else FAIL,
            ("허용한 차이만: " + ", ".join(k for k, _, _ in diff)) if not outside else
            "동결 밖 차이: " + "; ".join(f"{k} 계획={a} 직접={b}" for k, a, b in diff
                                        if k in outside))
        a = parse_argv(d.get("argv", ""))
        want = {"--model": W.MODEL, "--budget-bytes": str(W.BUDGET_BYTES),
                "--pool-docs": str(W.POOL_DOCS), "--eval-bytes": str(W.EVAL_BYTES),
                "--eval-budget": str(W.EVAL_BUDGET), "--lr-schedule": "constant"}
        bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
        add(f"직접 CPT seed{s} 공유 플래그", PASS if not bad else FAIL,
            "모델·예산·풀·평가·스케줄 같음" if not bad else
            "; ".join(f"{k} 직접={x} 계획={y}" for k, (x, y) in bad.items()))

    if direct_commit:
        raw = git("log", "--format=%h", f"{direct_commit}..HEAD", "--", "src", "configs")
        seen = [h for h in raw.split() if h]
        unknown = [h for h in seen if not any(h.startswith(r) or r.startswith(h)
                                              for r in W.REVIEWED_COMMITS)]
        missing = [r for r in W.REVIEWED_COMMITS
                   if not any(h.startswith(r) or r.startswith(h) for h in seen)]
        add("직접 CPT 이후 측정 경로 커밋", PASS if not unknown and not missing else FAIL,
            f"{len(seen)}개 모두 근거 있음" if not unknown and not missing else
            ("근거 없는 커밋: " + ", ".join(unknown) if unknown else "") +
            (" · 목록에만 있는 커밋: " + ", ".join(missing) if missing else ""))

    from src.training.cpt import load_damaged_rows
    vocab = json.loads((ROOT / W.MODEL / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    try:
        warm_rows = load_damaged_rows(ROOT / W.WARM_ROWS)
        ok = len(warm_rows) == W.N_WARM_ROWS and max(warm_rows) < vocab
        add("새 행 목록", PASS if ok else FAIL,
            f"{len(warm_rows):,}행 · 최대 id {max(warm_rows):,} < vocab {vocab:,}")
    except (OSError, ValueError) as e:
        add("새 행 목록", FAIL, str(e))

    reg = None
    with ARTIFACTS.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["kind"] == "checkpoint" and r["name"] == W.MODEL_ARTIFACT_NAME:
                reg = r["artifact_sha256"]
    from src.utils.hashing import sha256_dir, short
    got = sha256_dir(ROOT / W.MODEL)
    add("모델 체크포인트 해시", PASS if reg and got == reg else FAIL,
        f"{short(got)} vs 등록 {short(reg) if reg else '없음'}")

    from src.utils import env as env_mod
    env_ok = env_mod.is_registered(ROOT)
    add("환경 등록", PASS if env_ok else FAIL, "등록됨" if env_ok else "미등록")

    from src.utils.clock import latest_valid_check
    chk = latest_valid_check(ROOT)
    if chk is None:
        add("시각 검증", FAIL, "24시간 안의 성공 기록 없음 — tools/check_clock.py --record")
    else:
        t = calendar.timegm(time.strptime(chk["ts_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        left = 24 - (time.time() - t) / 3600
        add("시각 검증", PASS if left >= PLANNED_HOURS else WARN,
            f"남은 유효시간 {left:.1f}h / 계획 {PLANNED_HOURS:.1f}h")

    free = shutil.disk_usage(ROOT).free
    add("디스크 여유", PASS if free >= MIN_FREE_DISK else FAIL, f"{free / 1024 ** 3:.0f}GB")

    smi = shutil.which("nvidia-smi")
    if smi:
        q = subprocess.run([smi, "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        try:
            used, total = (int(x) for x in q.splitlines()[0].split(","))
            add("GPU 여유", PASS if total - used >= MIN_FREE_VRAM_MB else WARN,
                f"{total - used:,}MB 비어 있음 (필요 약 {MIN_FREE_VRAM_MB:,}MB)")
        except (ValueError, IndexError):
            add("GPU 여유", WARN, f"nvidia-smi 출력 해석 실패: {q!r}")
    else:
        add("GPU 여유", WARN, "nvidia-smi 없음")
    return res


def replay_and_predict(rows: list) -> tuple:
    """직접 CPT 3 run 을 바이트 단위로 재현한 뒤, warm run 의 종료·단계 경계를 예측한다."""
    from src.training.cpt import load_pool

    res: list = []
    pred: dict = {}
    pool = load_pool(POOL, W.POOL_DOCS, 0)
    tk = load_tokenizer(W.MODEL, None)
    table, _ = byte_table(tk)
    print(f"  토큰화  {W.MODEL}  {W.POOL_DOCS:,}문서 ...", flush=True)
    arrs = doc_arrays(tk, pool, table)

    ok_all = True
    for s in W.SEEDS:
        r = ok_row(rows, f"{W.DIRECT}{s}")
        st = Stream(arrs, order_indices(W.POOL_DOCS, 0, s), SEQ_LEN)
        n, want = int(r["tokens_seen"]), int(r["raw_bytes_seen"])
        got = st.bytes_at(n)
        ok_all &= got == want
        res.append((f"재현 {W.DIRECT}{s}", PASS if got == want else FAIL,
                    f"토큰 {n:,} -> 바이트 {got:,} vs 원장 {want:,}"))
    if not ok_all:
        res.append(("warm run 예측", FAIL, "재현이 어긋나 예측을 믿을 수 없다"))
        return res, pred

    for s in W.SEEDS:
        st = Stream(arrs, order_indices(W.POOL_DOCS, 0, s), SEQ_LEN)
        max_u = st.full_tokens // TOKENS_PER_UPDATE
        k_end = st.first_update_at(W.BUDGET_BYTES, TOKENS_PER_UPDATE, max_u)
        k_sw = st.first_update_at(W.WARM_BYTES, TOKENS_PER_UPDATE, max_u)
        if not k_end or not k_sw:
            res.append((f"예측 seed{s}", FAIL, "풀이 예산을 못 채운다"))
            continue
        pred[str(s)] = {
            "run_id": W.run_id(s),
            "end_update": k_end, "end_raw_bytes": st.bytes_at(k_end * TOKENS_PER_UPDATE),
            "end_tokens": k_end * TOKENS_PER_UPDATE,
            "switch_update": k_sw, "switch_raw_bytes": st.bytes_at(k_sw * TOKENS_PER_UPDATE),
            "switch_tokens": k_sw * TOKENS_PER_UPDATE,
        }
        p = pred[str(s)]
        res.append((f"예측 seed{s}", PASS,
                    f"2단계 전환 update {k_sw:,} ({p['switch_raw_bytes']:,} B) · "
                    f"종료 update {k_end:,} ({p['end_raw_bytes']:,} B)"))
    return res, pred


def write(res: list, pred: dict, quick: bool) -> None:
    from src.utils.hashing import sha256_file, short
    L: list = []
    w = L.append
    w("# 4주차 사전 점검 — 신규 행 warm-start\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/warm_preflight.py`")
    w("> 동결 설정은 `tools/warm_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md)")
    w("> \"4주차 run 설정 동결\". GPU 0시간. 판정하지 않는다 — 돌려도 되는지만 본다.\n")
    w(f"HEAD `{git('rev-parse', '--short', 'HEAD')}` · 풀 sha256 `{short(sha256_file(POOL))}`\n")
    w("| 점검 | 상태 | 내용 |")
    w("|---|---|---|")
    for n, s, d in res:
        w(f"| {n} | **{s}** | {d} |")
    w("")
    if quick:
        w("`--quick` — 데이터 흐름 재현과 예측을 건너뛰었다.\n")
    if pred:
        w("## 예측 (`warm_preflight.json`)\n")
        w("| seed | 2단계 전환 update | 전환 원문 바이트 | 종료 update | 종료 원문 바이트 |")
        w("|---:|---:|---:|---:|---:|")
        for s, p in pred.items():
            w(f"| {s} | {p['switch_update']:,} | {p['switch_raw_bytes']:,} | "
              f"{p['end_update']:,} | {p['end_raw_bytes']:,} |")
        w("")
        w("## 실행 명령 (gate seed 부터, 한 번에 하나)\n")
        w("```")
        for s in W.SEEDS:
            w(W.command(s))
        w("```\n")
        w("seed 123 · 2026 은 seed 42 가 gate 를 통과할 때만 돌린다 (PLAN).\n")
    w("## 한계\n")
    w("- 재현은 원문 바이트 회계와 데이터 순서를 검증한다. GPU 연산·난수는 보지 않는다")
    w("- 1단계에서 옛 행이 그대로인지는 실제 run 이 단계 전환 때 비트 단위로 확인한다")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    if pred:
        OUT_JSON.write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="4주차 사전 점검")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    rows = ledger_rows()
    res = static_checks(rows)
    pred: dict = {}
    if not args.quick:
        more, pred = replay_and_predict(rows)
        res += more
    for n, s, d in res:
        print(f"  [{s}] {n}  — {d}")
    write(res, pred, args.quick)
    print(f"썼다 {OUT_MD.relative_to(ROOT)}")
    return 1 if any(s == FAIL for _, s, _ in res) else 0


if __name__ == "__main__":
    raise SystemExit(main())
