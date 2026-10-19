"""6주차 — WSD 와 상수 LR 의 판정표 (P3-B, amendment 6주차, GPU 0시간).

    .conda/python.exe tools/wsd_pairs.py

동결 설정은 `tools/w6_spec.py` (PLAN.md "6주차 동결 — 2026-10-19"). **결과를 보기 전에**
만든 집계기다. 손으로 고친 표가 아니다.

## 무엇을 확인하고 나서 판정하나

- WSD run 이 원장에 ok 로 있고, argv 가 동결과 글자까지 같고, 측정 코드가 깨끗했는가
- 기준(5주차 재학습) 이후 `src` · `configs` 를 건드린 커밋이 없는가
- config 가 기준과 **lr_schedule 하나만** 다른가
- **감쇠 전 구간의 run 안 재현**: 원문 DECAY_START 이하의 평가 지점에서 WSD 의 한국어 BPB 가
  기준과 REPRO_TOL 안인가 — 그 구간의 LR 은 상수와 같으므로 같아야 한다

하나라도 어긋나면 그 쪽 판정을 내지 않고 이유를 적는다.

## 판정

    경계  max(2σ, FLOOR)   σ 는 R5 3 seed 의 표본 SD (원장에서 계산)
    T2b 절대   ΔB = Bf(T2b WSD) - Bf(T2b 상수)
    잔차       Δd = d(WSD) - d(상수),  d = BPB(T2b) - BPB(C0)
    개선 Δ < -경계 · 구별 불가 |Δ| <= 경계 · 악화 Δ > 경계
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import w6_spec as S  # noqa: E402

LM = ROOT / "experiments" / "lm_metrics.tsv"
LEDGER = ROOT / "experiments" / "LEDGER.tsv"
CURVE = ROOT / "experiments" / "train_curve.tsv"
RUNS = ROOT / "experiments" / "runs"
OUT = ROOT / "reports" / "tables" / "wsd_pairs.md"
DOMAINS = (("ko", "한국어"), ("en", "영어"), ("code", "코드"))


def _tsv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def lm_rows() -> list:
    return [r for r in _tsv(LM) if r["split"] == "dev"]


def bpb_at(lm: list, run_id: str, checkpoint: str, domain: str):
    hits = [float(r["bpb"]) for r in lm if r["run_id"] == run_id
            and r["checkpoint"] == checkpoint and r["domain"] == domain]
    return hits[-1] if hits else None


def bound(sigma) -> float:
    """판정 경계. sigma 가 없으면 바닥만 쓴다 — 그 사실을 표에 적는다."""
    return max(2 * sigma, S.FLOOR) if sigma is not None else S.FLOOR


def verdict(delta, b: float):
    """세 갈래 + 모름. 작을수록 좋다 (BPB)."""
    if delta is None:
        return None
    if delta < -b:
        return "개선"
    if delta > b:
        return "악화"
    return "구별 불가"


def sigmas(lm: list) -> dict:
    """R5 3 seed 의 σ_B (T2b 최종 ko) 와 σ_d (seed 쌍 d). 셋이 다 있어야 낸다."""
    b = [bpb_at(lm, r, "final", "ko") for r in S.SIGMA_T2B]
    c = [bpb_at(lm, r, "final", "ko") for r in S.SIGMA_C0]
    out = {"sigma_b": None, "sigma_d": None}
    if all(x is not None for x in b):
        out["sigma_b"] = statistics.stdev(b)
    if all(x is not None for x in b + c):
        out["sigma_d"] = statistics.stdev([x - y for x, y in zip(b, c)])
    return out


def predecay_repro(lm: list, wsd_run: str, base_run: str, domain: str = "ko") -> tuple:
    """감쇠 전 평가 지점의 run 안 재현. (True/False/None, [(checkpoint, raw, wsd, base)])."""
    base = {r["checkpoint"]: float(r["bpb"]) for r in lm
            if r["run_id"] == base_run and r["domain"] == domain}
    rows = []
    for r in lm:
        if (r["run_id"] != wsd_run or r["domain"] != domain or r["checkpoint"] == "final"
                or int(r["raw_bytes_seen"]) > S.DECAY_START):
            continue
        rows.append((r["checkpoint"], int(r["raw_bytes_seen"]), float(r["bpb"]),
                     base.get(r["checkpoint"])))
    if not rows or any(x[3] is None for x in rows):
        return None, rows
    return all(abs(w - b) <= S.REPRO_TOL for _, _, w, b in rows), rows


def smoke_check(curve: list) -> tuple:
    """GPU 시험 실행의 LR 이 wsd_lr_by_bytes 예측과 같은가. (True/False/None, 상세)."""
    from src.training.callbacks import wsd_lr_by_bytes
    rows = [r for r in curve if r["run_id"] == S.SMOKE_RUN and r["lr"] not in ("", "NA")]
    if not rows:
        return None, ["시험 실행의 train_curve 행이 없다"]
    detail, ok, decayed = [], True, False
    for r in rows:
        raw = int(float(r["raw_bytes_seen"]))
        want = wsd_lr_by_bytes(raw, S.SMOKE_BUDGET, S.PEAK_LR, S.WARMUP_FRAC,
                               min_frac=S.MIN_FRAC, decay_frac=S.DECAY_FRAC)
        got = float(r["lr"])
        good = abs(got - want) <= S.SMOKE_LR_RTOL * want
        ok &= good
        decayed |= raw > S.SMOKE_BUDGET * (1 - S.DECAY_FRAC) and got < S.PEAK_LR
        detail.append(f"원문 {raw:,} B  lr {got:.4e}  예측 {want:.4e}  {'같음' if good else '다름'}")
    if not decayed:
        return False, detail + ["감쇠 구간을 지난 행이 없다 — 감쇠 경로를 확인하지 못했다"]
    return ok, detail


def _code_commits(a: str, b: str) -> list:
    out = subprocess.run(["git", "log", "--format=%h", f"{a}..{b}", "--", "src", "configs"],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    return [x for x in out.stdout.split() if x] if out.returncode == 0 else ["<git 실패>"]


def validate(wsd_run: str, ok: dict) -> list:
    """판정에 쓰기 전의 문제 목록. 비어 있으면 통과."""
    from tools.warm_preflight import config_diff
    base_run, argv = S.WSD_RUNS[wsd_run]
    r, b = ok.get(wsd_run), ok.get(base_run)
    if r is None:
        return ["아직 없음"]
    if b is None:
        return [f"기준 {base_run} 의 ok 행이 없다"]
    bad = []
    if r["argv"].strip() != argv:
        bad.append("argv 가 동결과 다르다")
    if r.get("git_dirty") != "0":
        bad.append(f"git_dirty={r.get('git_dirty')}")
    commits = _code_commits(b["git_commit"], r["git_commit"])
    if commits:
        bad.append("기준 이후 측정 코드 커밋: " + ", ".join(commits))
    try:
        cw = json.loads((RUNS / wsd_run / "config.json").read_text(encoding="utf-8"))
        cb = json.loads((RUNS / base_run / "config.json").read_text(encoding="utf-8"))
        extra = [k for k, _, _ in config_diff(cw, cb) if k != "lr_schedule"]
        if extra:
            bad.append(f"config 가 lr_schedule 밖에서도 다르다: {extra}")
    except FileNotFoundError as e:
        bad.append(f"config 없음: {e.filename}")
    return bad


def f6(x) -> str:
    return "—" if x is None else f"{x:.6f}"


def sf6(x) -> str:
    return "—" if x is None else f"{x:+.6f}"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="6주차 WSD 판정표")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    lm = lm_rows()
    ok = {r["run_id"]: r for r in _tsv(LEDGER) if r["status"] == "ok"}
    sg = sigmas(lm)
    L: list = []
    w = L.append
    w("# 6주차 — WSD 와 상수 LR (168.5MB, seed 42)\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/wsd_pairs.py`")
    w("> 동결 설정 `tools/w6_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md) \"6주차 동결\".")
    w("> 손으로 고치지 마라.\n")
    w(f"WSD: 워밍업 {S.WARMUP_FRAC:.0%} → 상수 {S.PEAK_LR:g} → 원문 {S.DECAY_START:,} B"
      f"(예산의 {1 - S.DECAY_FRAC:.0%})부터 코사인 감쇠 → 끝에서 {S.PEAK_LR * S.MIN_FRAC:g}. "
      "기준은 5주차 재학습(상수 LR)이다.\n")

    w("## GPU 시험 실행\n")
    sm, sd = smoke_check(_tsv(CURVE))
    w(f"`{S.SMOKE_RUN}` — "
      + {True: "**LR 이 예측과 같다**", False: "**LR 확인 실패**", None: "아직 없음"}[sm]
      + " (결과로 인용하지 않는다)\n")
    if sd:
        w("```")
        L.extend(sd)
        w("```\n")

    w("## run 검증\n")
    w("| 조건 | WSD run | 기준 | 상태 | 감쇠 전 재현 |")
    w("|---|---|---|---|---|")
    usable = {}
    for cond, run in S.WSD.items():
        problems = validate(run, ok)
        rep, _ = predecay_repro(lm, run, S.BASE[cond])
        rep_s = {True: "통과", False: "**실패**", None: "—"}[rep]
        usable[cond] = not problems and rep is True
        w(f"| {cond} | `{run}` | `{S.BASE[cond]}` | "
          f"{'통과' if not problems else ' · '.join(problems)} | {rep_s} |")
    w("")
    for cond, run in S.WSD.items():
        _, rows = predecay_repro(lm, run, S.BASE[cond])
        if rows:
            w(f"감쇠 전 지점 ({cond}, 한국어, 폭 ±{S.REPRO_TOL}):\n")
            w("| 지점 | 원문 바이트 | WSD | 상수 | 차이 |")
            w("|---|---:|---:|---:|---:|")
            for ck, raw, a, b in rows:
                w(f"| {ck} | {raw:,} | {a:.6f} | {f6(b)} | "
                  f"{sf6(None if b is None else a - b)} |")
            w("")

    w("## 판정 (한국어)\n")
    bb, bd = bound(sg["sigma_b"]), bound(sg["sigma_d"])
    w(f"σ_B = {f6(sg['sigma_b'])} · σ_d = {f6(sg['sigma_d'])} (R5 3 seed, 원장) → "
      f"경계 ΔB {bb:.6f} · Δd {bd:.6f} (= max(2σ, {S.FLOOR}))\n")
    v = {k: {d: {"const": bpb_at(lm, S.BASE[k], "final", d),
                 "wsd": bpb_at(lm, S.WSD[k], "final", d)} for d, _ in DOMAINS}
         for k in ("c0", "t2b")}
    both = usable["c0"] and usable["t2b"]
    dB = (v["t2b"]["ko"]["wsd"] - v["t2b"]["ko"]["const"]) if usable["t2b"] else None
    d_c = (v["t2b"]["ko"]["const"] - v["c0"]["ko"]["const"])
    d_w = (v["t2b"]["ko"]["wsd"] - v["c0"]["ko"]["wsd"]) if both else None
    dd = None if d_w is None else d_w - d_c
    vb, vd = verdict(dB, bb), verdict(dd, bd)
    w("| | 상수 | WSD | 차이 | 판정 |")
    w("|---|---:|---:|---:|---|")
    w(f"| T2b 절대 BPB | {f6(v['t2b']['ko']['const'])} | "
      f"{f6(v['t2b']['ko']['wsd'] if usable['t2b'] else None)} | {sf6(dB)} | **{vb or '—'}** |")
    w(f"| C0 BPB (기술) | {f6(v['c0']['ko']['const'])} | "
      f"{f6(v['c0']['ko']['wsd'] if usable['c0'] else None)} | "
      f"{sf6((v['c0']['ko']['wsd'] - v['c0']['ko']['const']) if usable['c0'] else None)} | — |")
    w(f"| 잔차 d | {sf6(d_c)} | {sf6(d_w)} | {sf6(dd)} | **{vd or '—'}** |")
    w("")
    b0 = bpb_at(lm, S.BASE["t2b"], "step0", "ko")
    if both and b0 is not None:
        rc = (b0 - v["t2b"]["ko"]["const"]) / (b0 - v["c0"]["ko"]["const"])
        rw = (b0 - v["t2b"]["ko"]["wsd"]) / (b0 - v["c0"]["ko"]["wsd"])
        w(f"R (보조, RULES 14b, B0 {b0:.6f}): 상수 {rc:.2%} · WSD {rw:.2%}\n")
    if vb is None and vd is None:
        w("**판정 없음** — 위 검증을 통과한 run 이 없다.\n")
    else:
        follow = "개선" in (vb, vd)
        w(("**후속: C0 · T2b 를 seed " + " · ".join(map(str, S.FOLLOWUP_SEEDS))
           + " 으로 더 돌린다** (PLAN 6주차 동결 '후속'). 그 전에 PLAN 에 run 목록을 더한다.\n")
          if follow else "**후속 없음** — 두 판정 모두 개선이 아니다. seed 42 로 닫는다.\n")

    w("## 영어 · 코드 (기술만)\n")
    w("| 축 | d 상수 | d WSD | Δd |")
    w("|---|---:|---:|---:|")
    for d, name in DOMAINS[1:]:
        dc = v["t2b"][d]["const"] - v["c0"][d]["const"] if None not in (
            v["t2b"][d]["const"], v["c0"][d]["const"]) else None
        dw = (v["t2b"][d]["wsd"] - v["c0"][d]["wsd"]) if both and None not in (
            v["t2b"][d]["wsd"], v["c0"][d]["wsd"]) else None
        w(f"| {name} | {sf6(dc)} | {sf6(dw)} | "
          f"{sf6(None if None in (dc, dw) else dw - dc)} |")
    w("")

    w("## 문서 단위 구간 (기술만, 판정에 쓰지 않는다)\n")
    try:
        from tools import doc_bootstrap as D, w5_spec as W5
        docs = {n: D.load_docs_run(S.doc_run_id(n)) for n in S.CHECKPOINTS}
        rep = {n: D.reproduction_run(S.CHECKPOINTS[n][1], docs[n]) if docs[n] else {}
               for n in S.CHECKPOINTS}
        okd = {n: bool(rep[n]) and all(rep[n].values()) for n in S.CHECKPOINTS}
        w("| 비교 | a | b | 차이 (한국어) | 95% 구간 |")
        w("|---|---|---|---:|---|")
        for label, a, b in S.DOC_PAIRS:
            if not (okd[a] and okd[b]):
                w(f"| {label} | {a} | {b} | — | 문서별 평가 없음 또는 원장 재현 실패 |")
                continue
            if D.pair_problem(docs[a]["ko"], docs[b]["ko"]):
                w(f"| {label} | {a} | {b} | — | 짝 불일치 |")
                continue
            r = D.paired_bootstrap(docs[a]["ko"], docs[b]["ko"], W5.B, W5.RNG_SEED)
            w(f"| {label} | {a} | {b} | {r['diff']:+.6f} | [{r['lo']:+.6f}, {r['hi']:+.6f}] |")
    except ImportError:
        w("numpy 가 없어 계산하지 않았다.")
    w("")

    w("## 한계\n")
    w("- seed 42 하나다. 경계의 σ 는 상수 LR R5 3 seed 에서 빌렸다 — WSD 자신의 seed 분산이 아니다")
    w("- 감쇠 길이(20%)와 바닥(10%)은 하나만 시험했다. 다른 설정의 WSD 를 대표하지 않는다")
    w("- **Final Test 대표 조건은 바뀌지 않는다** (PLAN 5주차 D). 이 표는 dev 의 처방 결과다")
    w("- dev 는 닳은 집합이다. 절대값보다 같은 dev 에서의 조건 간 차이를 읽는다")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {Path(args.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
