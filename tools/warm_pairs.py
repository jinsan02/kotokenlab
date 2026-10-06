"""4주차 신규 행 warm-start 를 원장에서 집계하고 PLAN 의 gate 로 판정한다 (학습 없음).

    .conda/python.exe tools/warm_pairs.py

동결 설정은 `tools/warm_spec.py` (PLAN.md "4주차 run 설정 동결 — 2026-10-06").
**결과를 보기 전에 만들었다** — 경계를 새로 만들지 않고 등록된 것만 계산한다.

## 무엇을 계산하나

seed 마다 같은 seed 의 직접 CPT 와 C0 를 짝짓는다. 지표는 한국어 dev BPB 종점.

```
Cf        C0 상수 LR 최종                       (cpt_c0_qwen_r5_seed<s>)
d_direct  직접 CPT 최종 - Cf                    (cpt_t2b_mean_r5_seed<s>)
d_warm    warm-start 최종 - Cf                  (cpt_t2b_mean_warm_seed<s>)
delta     d_direct - d_warm      양수면 warm-start 가 낫다
```

## 판정하기 전에 run 을 검증한다

하나라도 어긋나면 그 seed 는 판정에서 빠지고 이유를 표에 적는다.

- 원장 argv 가 동결 명령과 같다 · git_dirty=0 · 직접 CPT 이후 측정 경로 커밋이
  `warm_spec.REVIEWED_COMMITS` 안이다
- 사전 점검 예측(`warm_preflight.json`)과 같다 — 종료 update·원문 바이트, 2단계 전환
  update·원문 바이트(원장 note 의 warm_end_*)
- **2단계 전환 때 옛 행 최대 차이가 0** 이다 (note 의 warm_old_rows_max_diff)
- 학습 전 B0 가 직접 CPT 와 같다 · 단계 경계 평가 행이 하나 있다

## 판정

gate 는 seed 42 의 delta 하나로 정한다: `delta >= 0.010` 이면 seed 123·2026 을 더 돌리고,
아니면 멈춘다. 세 seed 가 모두 있으면 평균·표본 SD·탐색적 t(2) 95% CI 를 함께 적는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import warm_spec as W  # noqa: E402
from tools.seed_pairs import describe  # noqa: E402

LM = ROOT / "experiments" / "lm_metrics.tsv"
LEDGER = ROOT / "experiments" / "LEDGER.tsv"
RUNS = ROOT / "experiments" / "runs"
PRED = ROOT / "reports" / "tables" / "warm_preflight.json"
OUT = ROOT / "reports" / "tables" / "warm_pairs.md"
DOMAINS = (("ko", "한국어"), ("en", "영어"), ("code", "코드"))
B0_TOL = 1e-6


def _tsv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def bpb_at(lm: list, run_id: str, checkpoint: str, domain: str):
    hits = [float(r["bpb"]) for r in lm if r["run_id"] == run_id
            and r["checkpoint"] == checkpoint and r["domain"] == domain]
    return hits[-1] if hits else None


def switch_row(lm: list, run_id: str, domain: str = "ko"):
    """단계 경계(--eval-at WARM_BYTES)의 평가 행. 정확히 하나여야 한다."""
    hits = [r for r in lm if r["run_id"] == run_id and r["domain"] == domain
            and r["checkpoint"] not in ("final", "step0")
            and W.WARM_BYTES <= int(r["raw_bytes_seen"]) < W.WARM_BYTES + W.EVAL_AT_SLACK]
    return hits[0] if len(hits) == 1 else None


def note_fields(note: str) -> dict:
    """원장 note 의 warm_end_* 값."""
    out = {}
    for k in ("warm_end_step", "warm_end_bytes", "warm_end_tokens"):
        m = re.search(rf"{k}=(\d+)", note or "")
        out[k] = int(m.group(1)) if m else None
    m = re.search(r"warm_old_rows_max_diff=([0-9.eE+-]+)", note or "")
    out["warm_old_rows_max_diff"] = float(m.group(1)) if m else None
    return out


def delta(d_direct, d_warm):
    if d_direct is None or d_warm is None:
        return None
    return d_direct - d_warm


def _code_commits(a: str, b: str) -> list:
    out = subprocess.run(["git", "log", "--format=%h", f"{a}..{b}", "--", "src", "configs"],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    return [x for x in out.stdout.split() if x] if out.returncode == 0 else ["<git 실패>"]


def validate(seed: int, ok: dict, lm: list, pred: dict) -> list:
    rid = W.run_id(seed)
    r = ok.get(rid)
    if r is None:
        return ["아직 없음"]
    p: list = []
    if r.get("argv", "") != W.argv_str(seed):
        p.append("argv 가 동결 명령과 다르다")
    if r.get("git_dirty") != "0":
        p.append(f"git_dirty={r.get('git_dirty')}")
    d = ok.get(f"{W.DIRECT}{seed}")
    if d is not None:
        extra = [h for h in _code_commits(d["git_commit"], r["git_commit"])
                 if not any(h.startswith(c) or c.startswith(h) for c in W.REVIEWED_COMMITS)]
        if extra:
            p.append("근거 없는 측정 경로 커밋: " + ", ".join(extra))
    cfg_p = RUNS / rid / "config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        if cfg.get("warm_bytes") != W.WARM_BYTES or cfg.get("warm_rows") != W.WARM_ROWS:
            p.append(f"config warm 필드 {cfg.get('warm_rows')} · {cfg.get('warm_bytes')}")
    else:
        p.append("config.json 없음")
    nf = note_fields(r.get("note", ""))
    if nf["warm_old_rows_max_diff"] != 0.0:
        p.append(f"옛 행 최대 차이 {nf['warm_old_rows_max_diff']} (0 이어야 한다)")
    pr = pred.get(str(seed))
    if pr is None:
        p.append("사전 점검 예측 없음 (warm_preflight.json)")
    else:
        checks = (("updates", r.get("updates"), pr["end_update"]),
                  ("종료 원문", r.get("raw_bytes_seen"), pr["end_raw_bytes"]),
                  ("반영 토큰", r.get("tokens_applied"), pr["end_tokens"]),
                  ("전환 step", nf["warm_end_step"], pr["switch_update"]),
                  ("전환 원문", nf["warm_end_bytes"], pr["switch_raw_bytes"]))
        for name, got, want in checks:
            if str(got) != str(want):
                p.append(f"{name} {got} != 예측 {want}")
    if switch_row(lm, rid) is None:
        p.append("단계 경계 평가 행이 없거나 하나가 아니다")
    b0, b0_d = bpb_at(lm, rid, "step0", "ko"), bpb_at(lm, f"{W.DIRECT}{seed}", "step0", "ko")
    if b0 is None or b0_d is None or abs(b0 - b0_d) > B0_TOL:
        p.append(f"B0 {b0} vs 직접 CPT {b0_d}")
    return p


def per_seed(seed: int, lm: list) -> dict:
    out: dict = {}
    for dom, _ in DOMAINS:
        cf = bpb_at(lm, f"{W.CTRL}{seed}", "final", dom)
        bd = bpb_at(lm, f"{W.DIRECT}{seed}", "final", dom)
        bw = bpb_at(lm, W.run_id(seed), "final", dom)
        dd = None if bd is None or cf is None else bd - cf
        dw = None if bw is None or cf is None else bw - cf
        out[dom] = {"cf": cf, "direct": bd, "warm": bw, "d_direct": dd, "d_warm": dw,
                    "delta": delta(dd, dw)}
    sw = switch_row(lm, W.run_id(seed))
    out["switch_bpb"] = float(sw["bpb"]) if sw else None
    return out


def curve(lm: list, run_id: str) -> dict:
    """원문 MB(20MB 격자, 반올림) -> 한국어 BPB."""
    pts = {}
    for r in lm:
        if r["run_id"] == run_id and r["domain"] == "ko" and r["checkpoint"].startswith("step") \
                and r["checkpoint"] != "step0":
            mb = round(int(r["raw_bytes_seen"]) / 20e6) * 20
            if abs(int(r["raw_bytes_seen"]) / 1e6 - mb) < 1.0:
                pts[mb] = float(r["bpb"])
    return pts


def f6(x) -> str:
    return "NA" if x is None else f"{x:.6f}"


def sf6(x) -> str:
    return "NA" if x is None else f"{x:+.6f}"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="4주차 warm-start 집계")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    ok = {r["run_id"]: r for r in _tsv(LEDGER) if r["status"] == "ok"}
    lm = [r for r in _tsv(LM) if r["split"] == "dev"]
    pred = json.loads(PRED.read_text(encoding="utf-8")) if PRED.exists() else {}
    checks = {s: validate(s, ok, lm, pred) for s in W.SEEDS}
    done = [s for s in W.SEEDS if checks[s] != ["아직 없음"]]
    valid = [s for s in done if not checks[s]]
    data = {s: per_seed(s, lm) for s in done}

    L: list = []
    w = L.append
    w("# 4주차 — 신규 행 warm-start vs 직접 CPT (T2b 상수 LR 168.5MB)\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/warm_pairs.py`")
    w("> 동결 설정 `tools/warm_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md) \"4주차 run")
    w("> 설정 동결\". 사전 점검 [`warm_preflight.md`](warm_preflight.md). 손으로 고치지 마라.\n")
    w(f"첫 {W.WARM_BYTES / 1e6:.1f}MB 는 새 행 {W.N_WARM_ROWS:,} 만 학습하고(몸통·옛 행 얼림), "
      f"나머지는 전체를 학습한다. 총 원문은 직접 CPT 와 같은 {W.BUDGET_BYTES / 1e6:.1f}MB —")
    w("1단계 비용이 예산 안에 들어 있다.\n")

    w("## run 검증\n")
    w("| seed | run | 상태 |")
    w("|---:|---|---|")
    for s in W.SEEDS:
        st = ("통과" if s in valid else "아직 없음" if checks[s] == ["아직 없음"]
              else "**제외** — " + " · ".join(checks[s]))
        w(f"| {s} | `{W.run_id(s)}` | {st} |")
    w("")

    if not done:
        w("**아직 돈 run 이 없다.** 판정 없음.\n")
    else:
        w("## 한국어 잔차 (seed 짝)\n")
        w("| seed | Cf (C0) | 직접 CPT | warm-start | d_direct | d_warm | **delta** |")
        w("|---:|---:|---:|---:|---:|---:|---:|")
        for s in done:
            k = data[s]["ko"]
            w(f"| {s} | {f6(k['cf'])} | {f6(k['direct'])} | {f6(k['warm'])} | "
              f"{sf6(k['d_direct'])} | {sf6(k['d_warm'])} | **{sf6(k['delta'])}** |")
        w("")
        w("delta = d_direct − d_warm. **양수면 warm-start 가 낫다.**\n")

        w("## 판정\n")
        g = data[W.GATE_SEED]["ko"]["delta"] if W.GATE_SEED in valid else None
        verdict = W.gate(g)
        w(f"**gate (seed {W.GATE_SEED}): {verdict}** — delta {sf6(g)} vs 경계 +{W.FLOOR:.3f}.")
        if verdict == "통과":
            w("PLAN 대로 seed 123 · 2026 을 돌린다.")
        elif verdict == "미달":
            w("PLAN 대로 seed 42 하나로 멈춘다. 효과 방향을 보고 경계를 바꾸지 않는다.")
        deltas = [data[s]["ko"]["delta"] for s in valid if data[s]["ko"]["delta"] is not None]
        if len(deltas) == len(W.SEEDS):
            d = describe(deltas)
            w(f"\n세 seed: delta 평균 {sf6(d['mean'])} · 표본 SD {d['sd']:.6f} · "
              f"탐색적 t(2) 95% CI [{sf6(d['lo'])}, {sf6(d['hi'])}]. "
              f"평균이 경계 {'이상' if d['mean'] >= W.FLOOR else '미만'}이다.")
        w("")

        w("## 곡선 — 같은 원문 지점의 한국어 BPB (기술만)\n")
        w("1단계는 몸통이 얼어 있으므로 초반에는 직접 CPT 보다 느릴 수 있다. 판정은 종점만 본다.\n")
        for s in done:
            cw, cd = curve(lm, W.run_id(s)), curve(lm, f"{W.DIRECT}{s}")
            pts = sorted(set(cw) & set(cd))
            if not pts:
                continue
            w(f"seed {s} (단계 경계 {W.WARM_BYTES / 1e6:.1f}MB 의 warm BPB "
              f"{f6(data[s]['switch_bpb'])})\n")
            w("| 원문 MB | " + " | ".join(str(p) for p in pts) + " |")
            w("|---|" + "---:|" * len(pts))
            w("| 직접 CPT | " + " | ".join(f"{cd[p]:.4f}" for p in pts) + " |")
            w("| warm-start | " + " | ".join(f"{cw[p]:.4f}" for p in pts) + " |")
            w("")

        w("## 영어 · 코드 (기술만)\n")
        w("1단계는 옛 행과 몸통을 건드리지 않으므로 영어·코드 손상이 덜할 수 있다.\n")
        w("| seed | 축 | Cf | 직접 CPT | warm-start | delta |")
        w("|---:|---|---:|---:|---:|---:|")
        for s in done:
            for dom, name in DOMAINS[1:]:
                x = data[s][dom]
                w(f"| {s} | {name} | {f6(x['cf'])} | {f6(x['direct'])} | {f6(x['warm'])} | "
                  f"{sf6(x['delta'])} |")
        w("")

    w("## 한계\n")
    w("- 직접 CPT 는 `7683b83` 코드로 돌았다. 그 뒤 측정 경로 커밋은 전부 근거를 달았다")
    w("  (`warm_spec.REVIEWED_COMMITS`) — 직접 CPT config 해시가 재현됨은 테스트로 확인했다")
    w("- 1단계는 옛 행 이동량이 0 임을 확인하지만, 2단계의 옛 행·새 행 이동량은 새 행만")
    w("  `train_curve` 의 `upd_w_ratio_dmg` 로 남는다")
    w("- 학습률·스케줄은 두 단계가 같다 (상수 1e-5). 1단계 전용 학습률은 시험하지 않았다")
    w("- n=3 이다. t 구간은 정규성을 가정한 탐색적 값이다 (RULES 14c)")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {Path(args.out).relative_to(ROOT)}  (검증 통과 {len(valid)}/{len(W.SEEDS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
