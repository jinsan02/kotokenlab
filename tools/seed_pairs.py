"""상수 LR 168.5MB 의 **seed 쌍** 잔차를 원장에서 낸다 (학습 없음).

    .conda/python.exe tools/seed_pairs.py

## 왜

핵심 결과인 "T2b 종점이 C0 보다 +0.345 BPB 높다" 는 지금까지
**T2b 3 seed 를 C0 seed 42 하나에 대 놓고** 낸 값이다. 2026-09-19 감사가
"seed 42 의 우연인가" 를 최소 추가 실험 1번으로 올렸고, 2주차에
`cpt_c0_qwen_r5_seed{123,2026}` 을 채워 대응 쌍을 만든다.

이 도구는 그 쌍을 읽어 **잔차 Bf-Cf 를 seed 안에서 짝지어** 계산한다.

## 무엇을 적나

```
B0   T2b step0 ko     수술 직후, CPT 전 (seed 무관 — 같은 체크포인트)
U0   C0  step0 ko     수술 전 원본
Bf   T2b final ko     CPT 종점
Cf   C0  final ko     같은 seed 의 대조군 종점
d    Bf - Cf          잔차. **이것을 우선 본다** (RULES 14b amendment)
R    (B0-Bf)/(B0-Cf)  회복률. 보조 지표
```

## 이 도구가 하지 않는 것

**판정하지 않는다.** 동결 절(PLAN.md 2주차)이 정한 대로 이것은 가설 검정이
아니라 재현 확인이다. 새 경계를 만들지 않고, 세 seed 의 평균·표본 SD·paired
차이를 기술 통계로 적은 뒤 seed 42 가 그 분포 안에 있는지만 본다.

n=3 이다. 자유도 2 의 sigma 추정 95% 구간은 대략 [0.52 sigma, 6.3 sigma] 이고,
t 구간도 정규성을 가정한 **탐색적** 값이다. 일반적 결론으로 쓰지 않는다.

## 회계 차이를 같이 적는다

`2da591e` 이후 run 은 완전한 update 경계에서 멈춘다. seed 42 는 그 전에
돌았으므로 예산 종점이 한 update 미만 다르다. 표에 `tokens_applied` 와
`updates` 를 실어 그 차이가 보이게 한다 — 숨기면 나중에 못 찾는다.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LM = ROOT / "experiments" / "lm_metrics.tsv"
LEDGER = ROOT / "experiments" / "LEDGER.tsv"
OUT = ROOT / "reports" / "tables" / "seed_pairs.md"

SEEDS = (42, 123, 2026)
TREAT = "cpt_t2b_mean_r5_seed"
CTRL = "cpt_c0_qwen_r5_seed"
DOMAINS = (("ko", "한국어"), ("en", "영어"), ("code", "코드"))

# chi-square df=2 의 95% 구간 -> sigma 추정의 불확실성 배수 (noise_floor.py 와 같음)
SIGMA_CI_LO, SIGMA_CI_HI = 0.5207, 6.2847
# t(2) 양측 95%
T_CRIT_DF2 = 4.302653


def read_bpb() -> dict:
    """(run_id, checkpoint, domain) -> bpb."""
    with LM.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    return {(r["run_id"], r["checkpoint"], r["domain"]): float(r["bpb"])
            for r in rows if r["split"] == "dev"}


def read_ok_runs() -> dict:
    with LEDGER.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))
    return {r["run_id"]: r for r in rows if r["status"] == "ok"}


def num(row: dict, key: str) -> str:
    """원장 값이 NA 이면 NA 그대로 적는다. 없는 값을 채우지 않는다."""
    v = row.get(key, "NA")
    if v in ("", "NA", None):
        return "NA"
    try:
        return f"{int(v):,}"
    except ValueError:
        return str(v)


def recovery(b0: float, bf: float, cf: float):
    """R = (B0-Bf)/(B0-Cf). **손상 축이 아니면 None 을 돌려준다.**

    분모 `B0-Cf` 가 0 이하이면 통제군조차 수술 직후보다 그 축에서 나쁘다는
    뜻이라 "회복" 이 성립하지 않는다. 그대로 나누면 음수/음수가 그럴듯한
    백분율을 만든다 (RULES 14b 의 "영어·코드는 R 에 넣지 않는다" 를 집행).
    """
    denom = b0 - cf
    if denom <= 0:
        return None
    return (b0 - bf) / denom


def describe(vals: list) -> dict:
    """평균·표본 SD·탐색적 t 95% CI. n<2 면 SD 를 내지 않는다."""
    n = len(vals)
    if n == 0:
        # R 이 정의되지 않는 축에서는 빈 목록이 온다. 0 을 지어내지 않는다.
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(vals)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lo": None, "hi": None}
    sd = statistics.stdev(vals)
    half = T_CRIT_DF2 * sd / math.sqrt(n) if n == 3 else float("nan")
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="상수 LR seed 쌍 잔차")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    bpb = read_bpb()
    ok = read_ok_runs()

    o: list = []
    w = o.append
    w("# seed 쌍 잔차 — 상수 LR 168.5MB\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/seed_pairs.py`")
    w("> 원본은 `experiments/lm_metrics.tsv` 의 dev 행과 `experiments/LEDGER.tsv`")
    w("> 의 `ok` 행이다. 손으로 고치지 마라.\n")
    w("`Bf-Cf` 를 **같은 seed 안에서** 짝지어 낸다. 2026-09-19 감사가 올린")
    w("최소 추가 실험 1번(핵심 결과가 seed 42 의 우연인가)에 답하기 위한 표다.\n")
    w("**판정표가 아니다.** 동결 절(PLAN.md 2주차)대로 재현 확인이며, 새 경계를")
    w("만들지 않는다. 기술 통계를 적고 seed 42 가 분포 안에 있는지만 본다.\n")

    have = [s for s in SEEDS
            if f"{TREAT}{s}" in ok and f"{CTRL}{s}" in ok]
    missing = [s for s in SEEDS
               if f"{TREAT}{s}" not in ok or f"{CTRL}{s}" not in ok]
    if missing:
        w("## 아직 없는 쌍\n")
        for s in missing:
            lack = [r for r in (f"{TREAT}{s}", f"{CTRL}{s}") if r not in ok]
            w(f"- seed {s}: `" + "` · `".join(lack) + "` 의 `ok` 행이 없다")
        w("")
    if not have:
        w("완성된 쌍이 하나도 없다. run 이 끝난 뒤 다시 돌린다.\n")
        dest = Path(args.out)
        dest.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
        print("\n".join(o))
        print(f"\n썼다 {dest}")
        return 0

    w("## 학습 회계\n")
    w("`2da591e` 이후 run 은 완전한 update 경계에서 멈춘다. seed 42 는 그 전에")
    w("돌았으므로 관찰량과 반영량이 갈린다. 그 차이를 표에 남긴다.\n")
    w("| run | 관찰 토큰 | 반영 토큰 | update | wall_sec | 커밋 |")
    w("|---|---:|---:|---:|---:|---|")
    dirty: list = []
    for s in SEEDS:
        for pref in (CTRL, TREAT):
            rid = f"{pref}{s}"
            if rid not in ok:
                continue
            r = ok[rid]
            sha = (r.get("git_commit") or "")[:12] or "NA"
            if (r.get("git_dirty") or "").strip() == "1":
                dirty.append(rid)
                sha += " *dirty*"
            w(f"| `{rid}` | {num(r, 'tokens_seen')} | {num(r, 'tokens_applied')} | "
              f"{num(r, 'updates')} | {float(r['wall_sec']):,.0f} | `{sha}` |")
    w("")
    if dirty:
        from src.utils.gitinfo import dirty_is_code_scoped
        code_scoped = [d for d in dirty
                       if dirty_is_code_scoped(ok[d].get("git_commit", "")) is True]
        old_scoped = [d for d in dirty if d not in code_scoped]
        w("**`dirty` 인 run 이 있다.**\n")
        if old_scoped:
            w("- " + " · ".join(f"`{d}`" for d in old_scoped))
            w("  — `a478426`(2026-09-17) **이전** 에 돌았다. 그때의 `git_dirty` 는")
            w("  원장·리포트 파일도 dirty 로 셌으므로, 이 `1` 은 대개 그 run 이")
            w("  직접 append 한 원장 행이다. **코드가 더러웠다는 뜻이 아니다** —")
            w("  다만 둘을 구분할 수 없으므로 \"같은 코드였다\" 고 단정하지도 않는다")
        if code_scoped:
            w("- " + " · ".join(f"`{d}`" for d in code_scoped))
            w("  — 새 정의(코드 기준)에서 dirty 다. **커밋되지 않은 코드 위에서**")
            w("  **돌았다.** 그 커밋만으로 결과를 재현할 수 없다")
        w("")
    w("`NA` 는 회계 수정 이전 run 이라 그 열이 기록되지 않았다는 뜻이다.")
    w("**사후에 복원하지 않는다** — 복원할 수 없는 값이기 때문이다.\n")

    # 통제군의 update 수가 seed 마다 다르면 Cf 차이에 seed 와 학습량이 섞인다.
    # 그 교란을 표가 스스로 드러내게 한다 — 사람이 눈으로 찾게 두지 않는다.
    upd = {}
    for s in have:
        v = ok[f"{CTRL}{s}"].get("updates", "NA")
        upd[s] = int(v) if v not in ("", "NA", None) else None
    known = {s: u for s, u in upd.items() if u is not None}
    # 값이 갈리거나(확인된 불일치), 일부가 NA 라(같다고 확인할 수 없다) 경고한다.
    # "모른다" 를 "같다" 로 읽지 않기 위해 둘을 같이 취급한다.
    if len(set(known.values())) > 1 or len(known) < len(have):
        w("### 통제군 학습량이 seed 마다 같지 않다\n")
        w("`Cf` 차이에 **seed 효과와 학습량 차이가 섞인다.** 분리할 수 없으므로")
        w("나란히 적는다.\n")
        w("| seed | C0 update | C0 반영 토큰 | Cf (한국어) |")
        w("|---:|---:|---:|---:|")
        for s in have:
            cf = bpb[(f"{CTRL}{s}", "final", "ko")]
            r = ok[f"{CTRL}{s}"]
            w(f"| {s} | {num(r, 'updates')} | {num(r, 'tokens_applied')} | {cf:.6f} |")
        w("")
        w("update 를 덜 밟은 run 이 `Cf` 가 높으면(나쁘면) 방향이 맞는 것이지만,")
        w("**n=3 에 seed 와 교란되어 있어 크기를 가를 수 없다.** 이 차이를")
        w("\"회계 수정의 효과\" 라고 부르지 않는다. 3주차의 동일 update 대조가")
        w("이 축을 따로 잰다.\n")

    for dom, dom_ko in DOMAINS:
        w(f"## {dom_ko} (`{dom}`)\n")
        b0 = bpb.get((f"{TREAT}{have[0]}", "step0", dom))
        u0 = bpb.get((f"{CTRL}{have[0]}", "step0", dom))
        w(f"수술 직후 `B0` = {b0:.6f} · 수술 전 원본 `U0` = {u0:.6f}"
          if b0 is not None and u0 is not None else "`B0`/`U0` 를 찾지 못했다")
        w("")
        w("| seed | Bf (T2b) | Cf (C0) | **d = Bf-Cf** | B0-Cf (R 의 분모) | R (보조) |")
        w("|---:|---:|---:|---:|---:|---:|")
        ds: list = []
        rs: list = []
        for s in have:
            bf = bpb[(f"{TREAT}{s}", "final", dom)]
            cf = bpb[(f"{CTRL}{s}", "final", dom)]
            d = bf - cf
            ds.append(d)
            denom = b0 - cf
            r = recovery(b0, bf, cf)
            if r is None:
                cell = "정의 안 됨"
            else:
                rs.append(r)
                cell = f"{r:.2%}"
            w(f"| {s} | {bf:.6f} | {cf:.6f} | **{d:+.6f}** | {denom:+.6f} | {cell} |")
        w("")
        if not rs:
            w("이 축에서는 `B0-Cf` 가 0 이하라 **R 이 정의되지 않는다.** 대조군 C0")
            w("조차 수술 직후 모델보다 이 축에서 나쁘다 — 한국어 CPT 가 이 축을")
            w("같이 끌어내렸다는 뜻이고, \"회복률\" 이라는 말이 성립하는 구간이")
            w("아니다. 여기서는 `d` 만 읽는다.\n")

        st = describe(ds)
        if st["sd"] is None:
            w(f"쌍이 {st['n']}개뿐이라 표본 SD 를 내지 않는다.\n")
            continue
        w(f"평균 **{st['mean']:+.6f}** · 표본 SD {st['sd']:.6f} · "
          f"탐색적 t 95% CI [{st['lo']:+.6f}, {st['hi']:+.6f}]\n")
        w(f"sigma 추정 자체의 95% 구간은 대략 "
          f"[{st['sd'] * SIGMA_CI_LO:.6f}, {st['sd'] * SIGMA_CI_HI:.6f}] 다 (df=2).\n")

        if 42 in have:
            d42 = ds[have.index(42)]
            lo, hi = min(ds), max(ds)
            inside = lo <= d42 <= hi
            span = hi - lo
            w(f"**seed 42 의 d 는 {d42:+.6f}** 이고 세 값의 범위는 "
              f"[{lo:+.6f}, {hi:+.6f}] (폭 {span:.6f}) 다. "
              + ("분포 안에 있다." if inside else "**분포 밖이다 — 원인을 찾는다.**")
              + "\n")

        rst = describe(rs)
        if rst["sd"] is not None:
            w(f"R 은 보조 지표다: 평균 {rst['mean']:.2%} · 표본 SD "
              f"{rst['sd'] * 100:.2f}%p\n")

        # 과거 표와의 차이: 예전에는 T2b 세 seed 를 C0 seed 42 하나에 댔다.
        cf42 = bpb.get((f"{CTRL}42", "final", dom))
        if cf42 is not None and len(have) > 1:
            old = [bpb[(f"{TREAT}{s}", "final", dom)] - cf42 for s in have]
            ost = describe(old)
            w("### 예전 방식과 비교\n")
            w("지금까지는 T2b 세 seed 를 **C0 seed 42 하나** 에 댔다. 같은 데이터로")
            w("두 방식을 나란히 둔다.\n")
            w("| 방식 | 평균 d | 표본 SD |")
            w("|---|---:|---:|")
            w(f"| C0 seed 42 고정 (예전) | {ost['mean']:+.6f} | "
              + (f"{ost['sd']:.6f}" if ost["sd"] is not None else "NA") + " |")
            w(f"| seed 쌍 (지금) | {st['mean']:+.6f} | {st['sd']:.6f} |")
            w("")
            w(f"평균 차이는 {st['mean'] - ost['mean']:+.6f} 다. 예전 방식은 C0 의")
            w("seed 분산을 잔차에 그대로 싣고 있었다 — 그 크기가 이만큼이다.\n")

    w("## 읽는 법\n")
    w("- **d 를 우선 본다.** R 은 분모에 `B0-Cf` 가 들어가 C0 종점이 움직이면")
    w("  같이 움직인다 (RULES 14b amendment)")
    w("- n=3 이다. SD 도 t 구간도 **탐색적** 이며 일반적 결론으로 쓰지 않는다")
    w("- 영어·코드의 잔차는 `noise_floor.md` 의 sigma 해상도 근처이거나 그 아래다.")
    w("  한국어와 같은 무게로 읽지 않는다")
    w("- 이 표는 **고정 예산 종점** 의 비교다. 점근선 추정이 아니다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(o))
    print(f"\n썼다 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
