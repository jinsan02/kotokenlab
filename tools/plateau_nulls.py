"""회복 정체가 자명한 수렴인지 검정한다 (학습 없음, 원장만 읽는다).

    .conda/python.exe tools/plateau_nulls.py

## 왜 필요한가

1차 헤드라인은 "회복률이 세 조건에서 65.4 / 65.9 / 68.2% 로 겹친다" 다.
여기에 대한 가장 강한 반론은 실험이 아니라 한 문장이다 —

    "고정 예산 최적화가 남은 격차의 일정 비율을 메우는 건 수렴이 원래 그렇지 않나?"

맞는 지적이다. 1차 완화 dL/dD = -k(L - L*) 를 풀면 R(D) = 1 - e^(-kD) 이고,
이 R 은 **초기 손상 L0 와 무관하다.** 즉 귀무모형이 이미 "조건이 달라도 R 은
같다" 를 예측한다. 이 논증을 반박하지 못하면 헤드라인이 통째로 없어진다.

이 스크립트는 귀무모형 둘을 **정량으로** 친다. 새 학습은 없다 —
1차 run 의 checkpoint 곡선이 이미 원장에 있다.

## 검정 셋

N1  1차 완화        -ln(1-R) 이 D 에 선형이어야 한다. 유효 k 가 상수여야 한다
N2  LR 정규화 완화   dL/dD = -lr(D)*k0*(L-L*) 라면 -ln(1-R) 이 integral(lr dD) 에 선형
N3  예산 외삽        지수모형으로 R=65.44% 에서 예산 x1.433 을 주면 무엇을 예측하나

N2 가 핵심이다. N1 이 깨지는 것은 코사인 LR 이 0 으로 가니 당연할 수 있다.
LR 로 정규화하고도 감속이 남으면 **그 감속은 스케줄이 아니다.**

## 하지 않는 것 — 점근선 주장

L(D) = L_inf + A*D^(-alpha) 적합도 같이 낸다. 다만 예산 범위가 0.92 decade 뿐이라
잔차가 L_inf 에 대해 단조이고 내부 최소가 없다. **식별되지 않는다.**
"무한 예산에서 몇 %에서 멈춘다" 는 이 데이터로 말할 수 없고, 그 사실을 표에 적는다.

## 이것은 사후 분석이다

1차 사전 등록에 없던 분석이다 (RULES 14). 논문에서 exploratory 로 표시한다.
대신 여기서 나온 수치는 R5 의 **사전 등록 예측** 으로 바꿔 docs/PLAN.md 에
날짜와 함께 올린다 — 사후 관측을 사전 예측으로 승격시키는 유일한 정직한 경로다.
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
TC = ROOT / "experiments" / "train_curve.tsv"
OUT = ROOT / "reports" / "tables" / "plateau_nulls.md"

# 회복률의 앵커는 이 run 의 final ko BPB 다. 손으로 적지 않고 원장에서 읽는다.
ANCHOR_RUN = "cpt_c0_qwen_main_seed42"

# (라벨, run_id, B0). B0 는 Pre-CPT BPB — reports/tables/precpt_bpb.md 의 측정값이다.
CONDS = [
    ("C0 대조", "cpt_c0_qwen_main_seed42", 1.156880),
    ("T2b30k", "cpt_t2b_mean_main_seed42", 2.380297),
    ("T2b10k", "cpt_t2b10k_mean_main_seed42", 2.191800),
    ("등토큰", "cpt_t2b_mean_eqtok_seed42", 2.380297),
]

REF = "cpt_t2b_mean_main_seed42"
REF_B0 = 2.380297
EQTOK = "cpt_t2b_mean_eqtok_seed42"


def read_curve(run_id: str, lang: str = "ko") -> list:
    """(raw_bytes, bpb) 를 예산 순으로. step0 은 뺀다 — 학습 전 지점은 B0 로 따로 쓴다."""
    with LM.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = [(int(r["raw_bytes_seen"]), float(r["bpb"])) for r in rows
           if r["run_id"] == run_id and r["domain"] == lang and r["checkpoint"] != "step0"]
    if not out:
        raise SystemExit(f"원장에 {run_id} / {lang} 의 checkpoint 행이 없다")
    return sorted(out)


def read_lr(run_id: str) -> list:
    """(raw_bytes, lr). train_curve 는 평가 지점마다 한 행이다."""
    with TC.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = [(int(r["raw_bytes_seen"]), float(r["lr"])) for r in rows
           if r["run_id"] == run_id and r["lr"] not in ("", "NA")]
    if len(out) < 2:
        raise SystemExit(f"{run_id} 의 train_curve 에 lr 이 없다")
    return sorted(out)


def interp(x: float, pts: list) -> float:
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1] if x > pts[-1][0] else pts[0][1]


def cumulative_lr(lr_pts: list) -> list:
    """integral(lr d(bytes)) 를 MB 단위로 사다리꼴 적분."""
    cum = [(lr_pts[0][0], 0.0)]
    for a, b in zip(lr_pts, lr_pts[1:]):
        cum.append((b[0], cum[-1][1] + 0.5 * (a[1] + b[1]) * (b[0] - a[0]) / 1e6))
    return cum


def powerlaw_profile(curve: list, floors: list) -> list:
    """L(D) = L_inf + A*D^-alpha 를 L_inf 격자마다 로그-로그 회귀. 식별성 확인용."""
    xs_d = [b / 1e6 for b, _ in curve]
    ls = [v for _, v in curve]
    prof = []
    for f in floors:
        if f >= min(ls):
            prof.append((f, None, None))
            continue
        xs = [math.log(d) for d in xs_d]
        ys = [math.log(v - f) for v in ls]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        b = my - a * mx
        ss = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
        prof.append((f, -a, ss))
    return prof


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="회복 정체의 귀무모형 검정")
    ap.add_argument("--out", default=str(OUT), help="보고표 경로")
    args = ap.parse_args(argv)

    anchor = read_curve(ANCHOR_RUN)[-1][1]
    out: list = []
    w = out.append

    w("# 회복 정체는 자명한 수렴인가 — 귀무모형 검정\n")
    w("> **사후 분석이다.** 1차 사전 등록에 없다 (RULES 14). 학습은 없고")
    w("> `experiments/lm_metrics.tsv` 와 `train_curve.tsv` 만 읽는다.")
    w("> 재현: `.conda/python.exe tools/plateau_nulls.py`\n")
    w(f"회복률 R = (B0 - B(D)) / (B0 - {anchor:.6f}).")
    w(f"앵커 {anchor:.6f} 는 `{ANCHOR_RUN}` 의 final 한국어 BPB 다 (원장에서 읽었다).\n")

    w("## N1 — 1차 완화 모형\n")
    w("`dL/dD = -k(L-L*)` 이면 `R = 1-e^(-kD)` 이고 **k 가 상수** 다.")
    w("이 모형은 초기 손상과 무관하게 R 이 같다고 **이미 예측한다** — 그래서")
    w('"세 조건의 R 이 겹친다" 만으로는 아무것도 반박하지 못한다.')
    w("검정할 것은 유효 k 가 정말 상수인가다.\n")
    w("| 조건 | 쓴 지점 | 초기 k (첫 구간) | 말기 k (끝 구간) | 비 |")
    w("|---|---:|---:|---:|---:|")
    dropped: list = []
    for label, rid, b0 in CONDS:
        c = read_curve(rid)
        d = b0 - anchor
        # R >= 1 이면 log 가 정의되지 않는다. C0 는 앵커 자신이라 끝에서 그렇게 된다.
        y = [(b, -math.log(1 - (b0 - v) / d)) for b, v in c if (b0 - v) / d < 1 - 1e-9]
        n_drop = len(c) - len(y)
        if n_drop:
            dropped.append((label, n_drop, len(c)))
        if len(y) < 3:
            w(f"| {label} | {len(y)}/{len(c)} | — | — | **정의 안 됨** |")
            continue
        k0 = (y[1][1] - y[0][1]) / ((y[1][0] - y[0][0]) / 1e6)
        k1 = (y[-1][1] - y[-2][1]) / ((y[-1][0] - y[-2][0]) / 1e6)
        w(f"| {label} | {len(y)}/{len(c)} | {k0:.5f} /MB | {k1:.6f} /MB | **{k0 / k1:.0f}배** |")
    w("")
    w("**k 는 상수가 아니다.** 수백 배 떨어진다. 1차 완화 모형은 기각된다.\n")
    if dropped:
        w("> 지점을 버린 조건이 있다: " +
          ", ".join(f"{lab} {n}/{tot}" for lab, n, tot in dropped) + ".")
        w("> `R >= 1` 이면 `-ln(1-R)` 이 정의되지 않는다. **C0 는 앵커 자신이라**")
        w("> 끝에서 구성상 `R = 1` 이 된다 — C0 는 이 검정의 대조군이 될 수 없다.")
        w("> 남은 지점의 k 비만 참고로 싣는다.\n")

    w("## N2 — LR 로 정규화한 완화 모형 (핵심)\n")
    w("N1 이 깨지는 것은 코사인 LR 이 0 으로 내려가니 당연할 수 있다.")
    w("`dL/dD = -lr(D)*k0*(L-L*)` 라면 `-ln(1-R)` 이 **integral(lr dD)** 에 선형이다.")
    w("LR 로 나누고도 감속이 남으면 그 감속은 스케줄이 아니다.\n")
    curve = read_curve(REF)
    lr_pts = read_lr(REF)
    cum = cumulative_lr(lr_pts)
    w(f"`{REF}` 기준.\n")
    w("| 예산 MB | R % | lr | d(-ln(1-R))/dD | **/ d(int lr)** |")
    w("|---:|---:|---:|---:|---:|")
    prev = None
    first = last = None
    for b, v in curve:
        r = (REF_B0 - v) / (REF_B0 - anchor)
        y = -math.log(1 - r)
        i = interp(b, cum)
        lr = interp(b, lr_pts)
        if prev is None:
            w(f"| {b / 1e6:.0f} | {r * 100:.2f} | {lr:.2e} | — | — |")
        else:
            k = (y - prev[1]) / ((b - prev[0]) / 1e6)
            if i > prev[2]:
                ki = (y - prev[1]) / (i - prev[2])
                first = ki if first is None else first
                last = ki
                w(f"| {b / 1e6:.0f} | {r * 100:.2f} | {lr:.2e} | {k:.5f} | **{ki:.0f}** |")
            else:
                w(f"| {b / 1e6:.0f} | {r * 100:.2f} | {lr:.2e} | {k:.5f} | — |")
        prev = (b, y, i)
    lr0, lr1 = lr_pts[0][1], lr_pts[-1][1]
    w("")
    w(f"**LR 로 정규화해도 {first / last:.0f}배 감속이 남는다.** 같은 구간에서")
    w(f"lr 자체는 {lr0 / lr1:.1f}배밖에 안 떨어진다 ({lr0:.2e} -> {lr1:.2e}).")
    w("즉 감속의 대부분은 LR 스케줄이 아니다.\n")

    w("## N3 — 예산 외삽\n")
    r30 = (REF_B0 - curve[-1][1]) / (REF_B0 - anchor)
    eq = read_curve(EQTOK)
    scale = eq[-1][0] / curve[-1][0]
    pred = 1 - (1 - r30) ** scale
    r_eq = (REF_B0 - eq[-1][1]) / (REF_B0 - anchor)
    w(f"지수모형으로 R={r30 * 100:.2f}% 에서 예산 x{scale:.3f} 을 주면:\n")
    w("```")
    w(f"지수모형 예측   {pred * 100:.2f}%")
    w(f"실측 등토큰     {r_eq * 100:.2f}%")
    w(f"과예측          {(pred - r_eq) * 100:+.2f} %p")
    w("```\n")
    w("> 등토큰 run 은 LR 축이 `cosine_by_tokens` 로 혼자 다르다.")
    w("> 같은 표에 놓을 때 반드시 병기한다.\n")

    w("## 하지 않는 주장 — 점근선\n")
    w("`L(D) = L_inf + A*D^-alpha` 를 L_inf 격자마다 적합했다.\n")
    w("| L_inf 가정 | alpha | 잔차 SS |")
    w("|---:|---:|---:|")
    for f, a, ss in powerlaw_profile(curve, [0.30, 0.60, 0.90, 1.1375, 1.30, 1.45, 1.53]):
        w(f"| {f:.4f} | {a:.3f} | {ss:.2e} |" if a is not None else f"| {f:.4f} | — | 불가 |")
    span = math.log10(curve[-1][0] / curve[0][0])
    w("")
    w("**잔차가 L_inf 에 대해 단조다 — 내부 최소가 없다.** 예산 범위가")
    w(f"{curve[0][0] / 1e6:.0f}~{curve[-1][0] / 1e6:.0f}MB = **{span:.2f} decade** 뿐이라")
    w('점근선이 식별되지 않는다. "무한 예산에서 몇 %에서 멈춘다" 는 이 데이터로')
    w("**말할 수 없다.** 논문에서 주장하지 않는다.\n")

    w("## 논문이 주장할 수 있는 정확한 형태\n")
    w("> 관측된 회복은 1차 완화보다 수백 배 빠르게 감속하며, LR 스케줄로")
    w(f"> 정규화해도 {first / last:.0f}배 감속이 남는다. 손상 크기 1.19배 ·")
    w("> 토큰당 노출 4.4배가 다른 조건에서도 회복 *비율* 은 0.5%p 안에서 같다.")
    w("> 점근 거동은 이 예산 범위에서 식별되지 않는다.\n")
    w('이것이 "수렴이 원래 그렇지 않나" 에 대한 답이다. 정체의 *존재* 가 아니라')
    w("**감속의 형태** 가 귀무모형과 어긋난다.\n")

    w("## 한계\n")
    w("- **사후 분석이다.** 논문에서 exploratory 로 표시한다")
    w("- **seed 42 단독.** 곡선 하나에서 나온 기울기다")
    w("- 앵커가 C0 의 *최종* 값이라 이른 D 에서는 대조군도 아직 못 간 지점과 잰다")
    w("- N2 는 AdamW 2차 모멘트 적응을 모형에 넣지 않았다. 감속의 일부는")
    w('  optimizer 상태일 수 있고, 그것은 "LR 이 아니다" 와 모순되지 않는다')
    w("- 등토큰은 LR 축이 달라 N3 의 외삽 대상으로 완전하지 않다")

    dest = Path(args.out)
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"썼다  {dest}")
    print(f"  N2 핵심:  LR 정규화 후에도 {first / last:.0f}배 감속 (lr 자체는 {lr0 / lr1:.1f}배)")
    print(f"  N3:       지수모형 {pred * 100:.2f}% vs 실측 {r_eq * 100:.2f}%  ({(pred - r_eq) * 100:+.2f}%p)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
