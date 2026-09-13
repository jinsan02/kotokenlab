"""조건별 노이즈 플로어 sigma 를 원장에서 계산한다 (학습 없음).

    .conda/python.exe tools/noise_floor.py

## 왜 도구로 만드는가

sigma 는 이 프로젝트에서 **하중이 가장 큰 숫자** 다. 모든 판정이
`Delta > 2 sigma` 로 내려가고, 논문의 348 sigma · 308 sigma · 320 sigma 가 전부
여기서 나온다. 그런데 2026-09-13 까지 sigma 를 계산하는 코드가 저장소에
없었다 — 사람이 계산해서 마크다운에 적었다.

그 결과 두 표가 **다른 정밀도** 를 썼다.

    noise_floor.md    BPB 를 소수 4자리로 반올림한 값에서 계산
    phase4.md         전체 정밀도

C0 한국어가 0.000058 (반올림) vs 0.000066 (전체) 로 갈리고, T2a 영어는
반올림하면 **0.000000** 이 되어 2 sigma 판정에서 0 으로 나눌 뻔했다.
en/code 에 반복되는 0.000058 은 측정값이 아니라 **반올림 양자화 바닥**
(0.0001/sqrt(3) = 0.0000577) 이다.

이 도구가 그것을 닫는다. 표는 코드가 적는다 (RULES).

## 무엇을 계산하는가

동일 config 를 seed 42/123/2026 으로 돌린 run 세 개의 final BPB 에서
**표본 표준편차(n-1)** 를 낸다. 언어(ko/en/code) x 조건마다 따로 낸다.

비교용으로 4자리 반올림 sigma 도 같이 낸다 — 과거 표가 어디서 왔는지
드러내기 위해서고, 판정에는 **전체 정밀도만** 쓴다.

## n=3 의 한계를 같이 적는다

자유도 2 다. sigma 추정 자체의 95% CI 가 대략 [0.52 sigma, 6.3 sigma] 이므로
**1e-4 수준의 sigma 끼리 비교하는 것은 의미가 없다.** en/code 판정은
"해상도 아래" 로 보고한다. 한국어 판정은 격차가 300 sigma 이상이라 어느 값을
써도 흔들리지 않는다.
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
OUT = ROOT / "reports" / "tables" / "noise_floor.md"

SEEDS = (42, 123, 2026)
LANGS = (("ko", "한국어"), ("en", "영어"), ("code", "코드"))

# (라벨, run_id 접두사). noise2 는 C0 의 재측정본이고 보고에 쓰는 쪽이다.
FAMILIES = [
    ("C0 (원본)", "cpt_c0_qwen_noise2_seed"),
    ("T2a (제거만)", "cpt_t2a_none_noise_seed"),
    ("T2b (수술+E1)", "cpt_t2b_mean_noise_seed"),
    ("T2b N=10,000", "cpt_t2b10k_mean_noise_seed"),
]

# chi-square df=2 의 95% 구간. sigma 추정의 불확실성을 적기 위한 상수다.
#   lower = sqrt(2 / 7.3778)   upper = sqrt(2 / 0.05064)
CI_LO, CI_HI = 0.5207, 6.2847


def read_finals() -> dict:
    """(run_id, domain) -> final BPB."""
    with LM.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {(r["run_id"], r["domain"]): float(r["bpb"])
            for r in rows if r["checkpoint"] == "final"}


def read_ledger_ok() -> dict:
    with LEDGER.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {r["run_id"]: r for r in rows if r["status"] == "ok"}


def sigmas(vals: list) -> tuple:
    """전체 정밀도 sigma 와 4자리 반올림 sigma. 둘 다 표본(n-1) 이다."""
    return statistics.stdev(vals), statistics.stdev([round(v, 4) for v in vals])


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="조건별 sigma 를 원장에서 계산한다")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    finals = read_finals()
    ledger = read_ledger_ok()
    o: list = []
    w = o.append

    w("# 노이즈 플로어 — 조건별 sigma_BPB\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/noise_floor.py`")
    w("> 원본은 `experiments/lm_metrics.tsv` 의 final 행이다. 손으로 고치지 마라.\n")
    w("동일 config 를 seed 42 / 123 / 2026 으로 3회. 각 17.5MB 원문, 약 11분.\n")
    w("**이걸 먼저 재지 않으면 이후 어떤 비교도 해석할 수 없다.** \"1.207 vs 1.198\" 이")
    w("의미 있는 차이인지 판단할 근거가 없다.\n")
    w("seed 는 **문서 순서만** 바꾼다. 풀(5,000문서 20.0MB)을 예산(17.5MB)에 맞춰")
    w("모든 seed 가 같은 문서를 거의 다 보게 했다. 실제 비교에서는 모든 조건이")
    w("Equal-Raw-Data 로 같은 데이터를 보므로 데이터 선택 분산은 조건 간 차이의")
    w("원인이 아니다.\n")

    w("## 결과\n")
    w("| 조건 | 언어 | seed 42 | seed 123 | seed 2026 | 평균 | **sigma** | **2 sigma** |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    table: dict = {}
    missing: list = []
    for label, pref in FAMILIES:
        for dom, dom_ko in LANGS:
            vals = []
            for s in SEEDS:
                v = finals.get((f"{pref}{s}", dom))
                if v is None:
                    missing.append(f"{pref}{s}/{dom}")
                else:
                    vals.append(v)
            if len(vals) != 3:
                w(f"| {label} | {dom_ko} | — | — | — | — | **행 없음** | — |")
                continue
            full, r4 = sigmas(vals)
            table[(label, dom)] = (vals, full, r4)
            bold = "**" if dom == "ko" else ""
            w(f"| {label} | {dom_ko} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | "
              f"{statistics.mean(vals):.6f} | {bold}{full:.6f}{bold} | {bold}{full*2:.6f}{bold} |")
    w("")
    if missing:
        w(f"> 원장에 없던 조합: {', '.join(missing)}\n")

    w("## 과거 표가 왜 달랐나 — 반올림에서 계산했다\n")
    w("2026-09-13 이전 표는 BPB 를 **소수 4자리로 반올림한 값** 에서 sigma 를 냈다")
    w("(`phase4.md` 의 T2b N=10,000 만 전체 정밀도였다). 차이는 이렇다.\n")
    w("| 조건 | 언어 | 전체 정밀도 | 4자리 반올림 | |")
    w("|---|---|---:|---:|---|")
    quant = 0.0001 / math.sqrt(3)
    for (label, dom), (_v, full, r4) in table.items():
        dom_ko = dict(LANGS)[dom]
        flag = ""
        if r4 == 0.0:
            flag = "**0 으로 나눌 뻔했다**"
        elif abs(r4 - quant) < 1e-6:
            flag = "반올림 양자화 바닥"
        elif abs(full - r4) / max(full, 1e-12) > 0.05:
            flag = f"{(r4-full)/full*100:+.0f}%"
        w(f"| {label} | {dom_ko} | {full:.6f} | {r4:.6f} | {flag} |")
    w("")
    w(f"**`0.000058` 은 측정값이 아니다.** 세 값 중 둘이 같고 하나가 0.0001 다르면")
    w(f"반올림 sigma 는 항상 `0.0001/sqrt(3)` = **{quant:.6f}** 가 된다. 과거 표에")
    w("이 값이 반복된 것은 en/code 의 분산이 4자리 아래에 있었다는 뜻이지,")
    w("세 조건의 sigma 가 실제로 같았다는 뜻이 아니다.\n")

    w("## sigma 는 조건마다 다르다 — 하나로 퉁칠 수 없다\n")
    ko = {lab: table[(lab, "ko")][1] for lab, _ in FAMILIES if (lab, "ko") in table}
    if ko:
        big = max(ko.values()); small = min(ko.values())
        w(f"한국어에서 가장 큰 sigma 가 가장 작은 것의 **{big/small:.1f}배** 다.")
        w("수술로 손상된 상태(BPB 2.38)에서 출발하면 seed 에 따라 회복 궤적이")
        w("갈리기 때문이다.\n")
    w("**T2a 가 C0 와 비슷하다는 것이 원인을 짚어 준다.** T2a 도 30,000개를 제거하는")
    w("수술을 받았지만 sigma 가 C0 와 같은 자릿수다. sigma 를 키우는 것은")
    w("\"수술을 받았는가\" 가 아니라 **\"수술로 손상됐는가\"** 다.\n")

    w("## 보고 규칙\n")
    w("조건별 sigma 를 각각 쓴다. 두 조건을 비교할 때는 **큰 쪽 sigma** 를 기준으로")
    w("하고, `Delta > 2 sigma` 일 때만 \"차이 있음\", 그 미만은 \"구별 불가\" 로 쓴다.\n")
    w("> **이 규칙은 2026-08-30 에 확정됐다** — sigma 를 처음 잰 직후이자 본 CPT")
    w("> 비교(08-31) 전이다. 원래 사전 등록(08-29)은 \"sigma 는 Step 5 에서 잰")
    w("> 노이즈 플로어\" 라고만 적었고 조건별이라는 말이 없었다. 방향은 보수적이지만")
    w("> (큰 쪽을 쓰면 차이를 선언하기 더 어렵다) **논문에 \"전부 사전 등록\" 이라고")
    w("> 뭉뚱그리면 안 된다.**\n")

    w("### n=3 의 해상도 한계\n")
    w("자유도 2 다. sigma 추정 자체의 95% 신뢰구간이 이만큼 넓다.\n")
    w("| 조건 | 언어 | sigma | 95% CI |")
    w("|---|---|---:|---|")
    for (label, dom), (_v, full, _r4) in table.items():
        if dom == "ko":
            w(f"| {label} | 한국어 | {full:.6f} | {full*CI_LO:.6f} ~ {full*CI_HI:.6f} |")
    w("")
    w("**따라서 1e-4 미만의 sigma 끼리 비교하는 것은 의미가 없다.**")
    w("en/code 에서 관측된 sigma 차이는 추정 잡음 안이다. 보고는 이렇게 한다:\n")
    w("```")
    w("한국어   조건별 sigma 를 그대로 쓴다. 격차가 300 sigma 이상이라 안 흔들린다")
    w("en/code  1e-4 미만 격차는 '해상도 아래' 로 보고한다. sigma 배수를 인용하지 않는다")
    w("```\n")

    w("## Equal-Raw-Data 가 작동한다는 증거\n")
    w("같은 17.5MB 예산에서 본 토큰 수 (원장 `ok` 행).\n")
    w("| 조건 | 원문 바이트 | 토큰 | 토큰/바이트 |")
    w("|---|---:|---:|---:|")
    for label, pref in FAMILIES:
        r = ledger.get(f"{pref}42")
        if not r:
            continue
        b, t = int(r["raw_bytes_seen"]), int(r["tokens_seen"])
        w(f"| {label} | {b:,} | {t:,} | {t/b:.6f} |")
    c0 = ledger.get("cpt_c0_qwen_noise2_seed42")
    t2b = ledger.get("cpt_t2b_mean_noise_seed42")
    if c0 and t2b:
        ratio = (int(t2b["tokens_seen"]) / int(t2b["raw_bytes_seen"])) / \
                (int(c0["tokens_seen"]) / int(c0["raw_bytes_seen"]))
        w("")
        w(f"T2b / C0 = **{ratio:.4f}** — 예산을 바이트로 세니 압축이 좋은 조건이")
        w("**같은 원문** 을 더 적은 토큰으로 지나간다. 토큰으로 셌다면 T2b 가")
        w(f"{1/ratio-1:.1%} 더 많은 원문을 보고도 같은 예산이라 불릴 뻔했다.\n")

    w("## 한계\n")
    w("- **예산이 17.5MB 다.** 본 실험(168.5MB)에서는 궤적이 더 갈릴 여지가 있어")
    w("  sigma 가 이보다 클 수 있다. **하한으로 다뤄라**")
    w("- **3 seed 라 sigma 추정 자체의 불확실성이 크다** (자유도 2, 위 CI 표)")
    w("- 문서 집합이 동일해 데이터 선택 분산이 빠져 있다. Equal-Raw-Data 설계에서는")
    w("  의도된 배제다")
    w("- 학습 전 BPB 는 이 표에 없다. `precpt_bpb.md` 에 있고, 노이즈 run 대부분이")
    w("  `step0` 행을 남기지 않았다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"썼다  {dest}")
    for (label, dom), (_v, full, r4) in table.items():
        if dom == "ko":
            print(f"  {label:16s} ko  sigma {full:.6f}  (반올림본 {r4:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
