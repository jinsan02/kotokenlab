"""사전 등록 판정을 원장에서 다시 계산한다 (학습 없음).

    .conda/python.exe tools/verdicts.py

## 왜 도구로 만드는가

sigma 와 같은 결함이 판정에도 있었다 — `cpt_main.md` 와 `phase4.md` 의
"7.2 sigma", "348 sigma", "320 sigma" 는 전부 사람이 계산해 마크다운에 적은
값이다. sigma 를 전체 정밀도로 고치면(`tools/noise_floor.py`) 이 숫자들도 같이
움직인다. 손으로 다시 적으면 같은 실수를 반복한다.

## 두 잣대를 나란히 낸다

**1. 사전 등록 규칙 (2 sigma)** — 2026-08-29 에 고정했다. 이것이 판정이다.
   결과를 보고 바꾸지 않는다 (RULES 14).

**2. n=3 예측구간** — 사전 등록 규칙의 *한계* 를 드러내기 위한 진단이다.

두 번째가 필요한 이유: 판정은 seed 42 **한 번의 관측** 을 n=3 에서 추정한
sigma 와 비교한다. 그런데 `sigma` 자체가 자유도 2 의 추정량이라 불확실하다.
새 관측 하나에 대한 95% 예측구간은 `2 sigma` 가 아니라

    t(0.975, df=2) * sigma * sqrt(1 + 1/3)  =  4.30 * 1.155 * sigma  =  4.97 sigma

**2.5배 넓다.** 2 sigma 는 통과하지만 4.97 sigma 는 못 넘는 판정이 있으면
그것은 "차이 있음" 이 아니라 **"이 설계로는 분해되지 않음"** 이다.

사전 등록을 소급해서 바꾸지 않는다. 등록 규칙의 판정을 그대로 싣고,
예측구간을 못 넘는 칸에 표시를 단다. 논문에는 둘 다 적는다.

## 어느 sigma 를 쓰는가

사전 등록대로 **두 조건 중 큰 쪽** 이다 (`noise_floor.md` "보고 규칙").
sigma 는 17.5MB 에서 잰 하한이고 본 비교는 168.5MB 다 — 그 사실도 표에 적는다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.noise_floor import FAMILIES, LANGS, SEEDS, read_finals, sigmas  # noqa: E402

OUT = ROOT / "reports" / "tables" / "verdicts.md"

# 새 관측 하나에 대한 95% 예측구간 계수. t(0.975, df=2) * sqrt(1 + 1/n), n=3.
PI_K = 4.9684

BASELINE = ("C0", "cpt_c0_qwen_main_seed42")

# (라벨, run_id, sigma 를 빌려올 노이즈 조건)
CONDS = [
    ("T2a 제거만", "cpt_t2a_none_main_seed42", "T2a (제거만)"),
    ("T2b N=30,000", "cpt_t2b_mean_main_seed42", "T2b (수술+E1)"),
    ("T2b N=10,000", "cpt_t2b10k_mean_main_seed42", "T2b N=10,000"),
    ("T2b 등토큰", "cpt_t2b_mean_eqtok_seed42", "T2b (수술+E1)"),
]
SIGMA_OF_BASELINE = "C0 (원본)"


def condition_sigmas(finals: dict) -> dict:
    """(노이즈 조건 라벨, domain) -> 전체 정밀도 sigma."""
    out = {}
    for label, pref in FAMILIES:
        for dom, _ko in LANGS:
            vals = [finals[(f"{pref}{s}", dom)] for s in SEEDS
                    if (f"{pref}{s}", dom) in finals]
            if len(vals) == 3:
                out[(label, dom)] = sigmas(vals)[0]
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="사전 등록 판정을 원장에서 재계산")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    finals = read_finals()
    sig = condition_sigmas(finals)
    o: list = []
    w = o.append

    w("# 사전 등록 판정 — 원장에서 재계산\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/verdicts.py`")
    w("> sigma 는 [`noise_floor.md`](noise_floor.md), 값은 `experiments/lm_metrics.tsv`.")
    w("> 손으로 고치지 마라.\n")
    w("판정 규칙은 **2026-08-29 사전 등록** 이다 — `|Delta| <= 2 sigma` 면 구별 불가,")
    w("넘으면 개선/악화. sigma 는 두 조건 중 **큰 쪽** 을 쓴다 (2026-08-30 확정).\n")
    w(f"`PI` 열은 판정이 아니라 **진단** 이다. n=3 에서 추정한 sigma 로 새 관측")
    w(f"하나를 재는 95% 예측구간은 `{PI_K:.2f} sigma` 라 2 sigma 보다 2.5배 넓다.")
    w("`PI` 가 `아니오` 면 사전 등록 규칙은 통과했지만 **이 설계로는 분해되지**")
    w("**않는 차이** 라는 뜻이다.\n")

    base_id = BASELINE[1]
    w("## 168.5MB Equal-Raw-Data — 각 조건 vs C0\n")
    w("| 조건 | 언어 | C0 | 조건 | Delta | Delta % | sigma | sigma 배 | 등록 판정 | PI |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    flips: list = []
    for label, rid, sig_label in CONDS:
        for dom, dom_ko in LANGS:
            b = finals.get((base_id, dom))
            v = finals.get((rid, dom))
            if b is None or v is None:
                w(f"| {label} | {dom_ko} | — | — | — | — | — | — | 행 없음 | — |")
                continue
            s = max(sig.get((sig_label, dom), 0.0), sig.get((SIGMA_OF_BASELINE, dom), 0.0))
            d = v - b
            k = abs(d) / s if s else float("inf")
            verdict = "구별 불가" if k <= 2 else ("**악화**" if d > 0 else "**개선**")
            pi = "예" if k > PI_K else ("—" if k <= 2 else "**아니오**")
            if 2 < k <= PI_K:
                flips.append((label, dom_ko, k))
            w(f"| {label} | {dom_ko} | {b:.6f} | {v:.6f} | {d:+.6f} | {d/b*100:+.3f}% | "
              f"{s:.6f} | {k:.1f} | {verdict} | {pi} |")
    w("")
    w("> 등토큰은 예산 축이 다르고 (`cosine_by_tokens`, 원문 241.4MB) LR 스케줄도")
    w("> 혼자 다르다. 같은 표에 있지만 **나란히 읽으면 안 된다.**\n")

    if flips:
        w("## 사전 등록은 통과했으나 예측구간을 못 넘는 칸\n")
        for label, dom_ko, k in flips:
            w(f"- **{label} / {dom_ko}** — {k:.1f} sigma. 2 sigma 는 넘지만 "
              f"{PI_K:.2f} sigma 에 못 미친다")
        w("")
        w("이 칸들은 **\"차이 있음\" 으로 인용하지 않는다.** sigma 가 자유도 2 의")
        w("추정량이라 배수 자체의 불확실성이 그만큼 크다. 논문에는 등록 판정과")
        w("함께 이 사실을 적는다.\n")

    w("## 한계\n")
    w("- **sigma 는 17.5MB 에서 잰 하한이다.** 본 비교는 168.5MB 라 실제 sigma 가")
    w("  더 클 수 있다. 한국어 판정(300 sigma 이상)은 3배여도 유지된다")
    w("- **본 run 은 seed 42 단독이다.** 그래서 예측구간 진단이 필요하다")
    w("- 등토큰의 sigma 는 N=30,000 조건 것을 빌렸다. 등토큰 자체의 노이즈")
    w("  플로어는 잰 적이 없다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"썼다  {dest}")
    if flips:
        print("  예측구간 미달:", ", ".join(f"{a}/{b} {k:.1f}s" for a, b, k in flips))
    else:
        print("  모든 등록 판정이 예측구간도 넘는다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
