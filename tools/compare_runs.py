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

## config 가 같아도 코드가 다를 수 있다 (2026-09-20 에 추가)

2주차에 `cpt_c0_qwen_r5_seed42` 와 `seed123` 을 비교하면서 드러났다. config
필드는 `--allow seed` 로 전부 통과하는데, 그 사이 `src/training/cpt.py` 가
+232/-24 줄 바뀌어 있었다. **이 도구는 그것을 못 봤다.** 사람이 커밋 5개를
손으로 대조해서 "실제로 동작을 바꾼 것은 `2da591e` 하나" 를 확인했다.

손으로 하면 다음에 또 빠뜨린다. 이제 두 run 의 `git_commit` 사이에서
**측정 코드**(`src/`, `configs/`)를 건드린 커밋을 세고, 있으면 치명으로 다룬다.
`tools/`·`tests/`·`docs/` 만 바뀐 경우는 걸리지 않는다 — 측정에 안 닿는다.

정당한 이유가 있으면 `--allow code` 를 준다. `seed` 와 같은 취급이다:
**기본은 막고, 사람이 근거를 갖고 열어야 한다.**

`git_dirty=1` 인 run 은 그 커밋만으로 코드를 복원할 수 없다. 막지는 않지만
경고한다 — 이미 돌아간 run 을 되돌릴 수는 없고, 한계로 적어야 할 사실이다.
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
    # P3 에서 추가. 워밍업 길이와 데이터 순서를 바꾸므로 치명이다
    "warmup_bytes", "pool_extend_docs",
    # eval (bpb.py)
    "max_bytes", "split",
}

# 벽시계만 바꾼다. 평가는 비파괴적이고 BPB 를 안 건드린다.
# eval_at / save_at 은 지점을 더 잴 뿐이고 학습 경로를 건드리지 않는다.
TIMING = {"eval_bytes", "eval_at", "save_at", "damaged_rows"}

# 당연히 다르다.
IDENTITY = {"model", "revision", "name", "purpose"}

# 측정값을 만들어내는 코드. 여기가 바뀌면 config 가 같아도 숫자가 달라질 수 있다.
# tools/ 는 뺀다 — 원장을 읽어 표를 적을 뿐 학습·평가 경로에 없다.
MEASURE_PATHS = ("src", "configs")

# --allow 에 줄 수 있는 가짜 필드. 코드 계보 차이를 통과시킨다.
ALLOW_CODE = "code"


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


def load_lineage(run_id: str) -> tuple:
    """원장의 (git_commit, git_dirty). 없으면 (None, None)."""
    import csv as _csv
    led = ROOT / "experiments" / "LEDGER.tsv"
    with led.open(encoding="utf-8", newline="") as fh:
        for r in _csv.DictReader(fh, delimiter="\t", quoting=_csv.QUOTE_NONE):
            if r["run_id"] == run_id and r["status"] == "ok":
                sha = (r.get("git_commit") or "").strip()
                dirty = (r.get("git_dirty") or "").strip()
                return (sha or None, dirty or None)
    return (None, None)


def _git(*a) -> tuple:
    """(성공?, 출력). git 이 없거나 커밋을 모르면 조용히 실패를 돌려준다."""
    import subprocess
    try:
        p = subprocess.run(("git", "-C", str(ROOT)) + a, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    except OSError:
        return (False, "")
    return (p.returncode == 0, (p.stdout or "").strip())


def known_commit(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}")[0]


def measure_commits(sha_a: str, sha_b: str) -> list:
    """두 커밋 사이에서 **측정 코드** 를 건드린 커밋. 양쪽 방향을 다 본다.

    `A...B` 대칭차를 쓴다 — 어느 쪽이 앞인지 몰라도 되고, 한쪽에만 있는
    커밋도 잡힌다.
    """
    ok, out = _git("log", "--oneline", "--no-decorate", f"{sha_a}...{sha_b}",
                   "--", *MEASURE_PATHS)
    if not ok or not out:
        return []
    return out.splitlines()


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="두 run 을 비교해도 되는지 config 로 확인한다")
    ap.add_argument("runs", nargs="+", help="run_id 둘 이상")
    ap.add_argument("--allow", nargs="*", default=[],
                    help="달라도 되는 필드 (예: seed — sigma 측정용 3 run 비교). "
                         "'code' 를 주면 측정 코드가 바뀐 것도 통과시킨다 — "
                         "커밋마다 옛 동작이 재현되는지 직접 확인한 뒤에만 쓴다")
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

    # 코드 계보 — config 가 같아도 측정 코드가 다르면 숫자가 달라질 수 있다.
    lineage = {r: load_lineage(r) for r in args.runs}
    print("코드 계보 — config 가 같아도 여기가 다르면 숫자가 달라질 수 있다")
    dirty_runs: list = []
    for r in args.runs:
        sha, dirty = lineage[r]
        mark = ""
        if dirty == "1":
            dirty_runs.append(r)
            mark = "  dirty=1 (아래 참조)"
        print(f"  {r}  {sha[:12] if sha else '<원장에 없음>'}{mark}")
    print()

    base = args.runs[0]
    base_sha = lineage[base][0]
    code_diff: list = []
    unknown_sha: list = []
    for r in args.runs[1:]:
        sha = lineage[r][0]
        if not base_sha or not sha:
            unknown_sha.append(r)
            continue
        if sha == base_sha:
            continue
        if not (known_commit(base_sha) and known_commit(sha)):
            unknown_sha.append(r)
            continue
        hits = measure_commits(base_sha, sha)
        if hits:
            code_diff.append((r, hits))

    if unknown_sha:
        print("  커밋을 확인할 수 없는 run: " + ", ".join(unknown_sha))
        print("  **\"코드가 같다\" 가 아니라 \"모른다\" 다.**")
        print()
    if dirty_runs:
        from src.utils.gitinfo import dirty_is_code_scoped
        for r in dirty_runs:
            scoped = dirty_is_code_scoped(lineage[r][0] or "")
            if scoped is True:
                print(f"  {r}: **커밋되지 않은 코드 위에서 돌았다.** 그 커밋만으로")
                print("    결과를 재현할 수 없다. 한계로 같이 적어라")
            else:
                print(f"  {r}: a478426(2026-09-17) 이전이라 dirty 정의에 원장·리포트가")
                print("    포함됐다. 이 1 은 대개 그 run 이 append 한 원장 행이다 —")
                print("    코드 오염이라고 단정하지 않되, 아니라고도 단정하지 않는다")
        print()
    if code_diff:
        for r, hits in code_diff:
            print(f"  {base} <-> {r} 사이에 {', '.join(MEASURE_PATHS)} 를 "
                  f"건드린 커밋 {len(hits)}개")
            for line in hits:
                print(f"    {line}")
        print()
        if ALLOW_CODE in allow:
            print("  --allow code 로 통과시켰다. **왜 비교 가능한지 기록에 적어라** —")
            print("  커밋마다 기본값이 옛 동작을 재현하는지 직접 확인해야 한다.")
            print()
        else:
            fatal.append("<코드 계보>")

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
