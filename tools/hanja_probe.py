"""한자 프로브 D1·D2·D3 판정을 원장에서 계산해 표로 쓴다 (GPU 0시간).

    .conda/python.exe tools/hanja_probe.py

등록은 `docs/PLAN.md` "downstream 과 한자 프로브" (2026-09-13),
"D1·D2 실행 설정 보충" (2026-09-16), "D3 실행 조건" (2026-09-17).
**등록 비교는 T2a vs C0 뿐이다.** T2b 는 descriptive 로만 싣는다.

    D1  부분집합 **전체 합계** tok_per_byte. 도메인 행은 합산에만 쓰고 인용하지 않는다
        (라벨 정확도 ~55%). |T2a/C0 - 1| <= 2% 면 예측 적중
    D2  lm_metrics split=dev_hanja 의 BPB. |T2a/C0 - 1| <= 5% 면 예측 적중.
        이 부분집합의 sigma 는 없다 — 5% 미만 차이를 "차이" 로 주장하지 않는다
    D3  predictions.tsv 로 paired bootstrap. 규칙은 capability.verdict

아직 없는 입력은 "미실행" 으로 적고 멈추지 않는다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.capability import (  # noqa: E402
    MARGIN, bootstrap_ci, paired_diff_ci, verdict,
)

EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "tables" / "hanja_probe.md"

D1_RUN = "tok_bench_d1"
D1_TOK = {"C0": "qwen_original", "T2a": "t2a_v1_n30000", "T2b": "t2b_v2_n30000"}
MODELS = {"C0": "cpt_c0_qwen_main_seed42", "T2a": "cpt_t2a_none_main_seed42",
          "T2b": "cpt_t2b_mean_main_seed42", "T2b10k": "cpt_t2b10k_mean_main_seed42"}
D1_BOUND = 0.02
D2_BOUND = 0.05


def _rows(name: str) -> list:
    with (EXP / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def _ok_runs() -> set:
    return {r["run_id"] for r in _rows("LEDGER.tsv") if r["status"] == "ok"}


def d1(ok: set) -> dict:
    """{조건: (n_tokens, n_bytes)} — 도메인 행을 합친다."""
    if D1_RUN not in ok:
        return {}
    out = {}
    for cond, ver in D1_TOK.items():
        rs = [r for r in _rows("tokenizer_metrics.tsv")
              if r["run_id"] == D1_RUN and r["tokenizer_version"] == ver
              and r["split"] == "dev_hanja"]
        if rs:
            out[cond] = (sum(int(r["n_tokens"]) for r in rs),
                         sum(int(r["n_bytes"]) for r in rs))
    return out


def d2(ok: set) -> dict:
    """{조건: (bpb, n_bytes)}. 전체 정밀도는 total_nll / n_bytes 로 다시 구한다."""
    import math

    out = {}
    rows = _rows("lm_metrics.tsv")
    for cond, m in MODELS.items():
        rid = f"eval_bpb_{m}_d2"
        if rid not in ok:
            continue
        rs = [r for r in rows if r["run_id"] == rid and r["split"] == "dev_hanja"
              and r["domain"] == "ko"]
        if len(rs) != 1:
            raise SystemExit(f"{rid}: dev_hanja ko 행이 {len(rs)}개다")
        r = rs[0]
        nb = int(r["n_bytes"])
        out[cond] = (float(r["total_nll"]) / (math.log(2) * nb), nb)
    return out


def d3(ok: set) -> dict:
    """{조건: [(subject,row,correct,has_hanja,n_shot), ...]} — 문항 순서 그대로."""
    out = {}
    for cond in ("C0", "T2a", "T2b"):
        rid = f"eval_kmmlu_{MODELS[cond]}_d3"
        p = EXP / "runs" / rid / "predictions.tsv"
        if rid not in ok or not p.exists():
            continue
        with p.open(encoding="utf-8", newline="") as fh:
            out[cond] = [(r["subject"], int(r["row"]), int(r["correct"]), int(r["has_hanja"]),
                          int(r["n_shot"]))
                         for r in csv.DictReader(fh, delimiter="\t")]
    return out


def main() -> int:
    ok = _ok_runs()
    L = ["# 한자 프로브 D1·D2·D3",
         "",
         "`tools/hanja_probe.py` 가 원장에서 쓴다. 손으로 고치지 않는다.",
         "등록: `docs/PLAN.md` \"downstream 과 한자 프로브\" 와 그 보충 두 절.",
         "**등록 비교는 T2a vs C0 뿐이다.** T2b 는 descriptive.",
         "",
         "부분집합 `dev_hanja` 는 균질하지 않다 (한국어 한자 병기 · 일본어 지명 · 중국어 인용).",
         "\"한국어 한자\" 만의 효과로 읽지 않는다.",
         ""]

    # ── D1 ────────────────────────────────────────────────────────────────
    L += ["## D1 — 표현 비용 (tok_per_byte, 부분집합 전체 합계)", ""]
    t = d1(ok)
    if "C0" not in t or "T2a" not in t:
        L += ["미실행", ""]
    else:
        L += ["| 조건 | 토큰 | 바이트 | tok/byte | C0 대비 |", "|---|---:|---:|---:|---:|"]
        base = t["C0"][0] / t["C0"][1]
        for c, (nt, nb) in t.items():
            v = nt / nb
            L.append(f"| {c} | {nt:,} | {nb:,} | {v:.6f} | {v / base - 1:+.2%} |")
        rel = (t["T2a"][0] / t["T2a"][1]) / base - 1
        hit = abs(rel) <= D1_BOUND
        L += ["", f"**판정: T2a {rel:+.2%} — "
              f"{'예측 적중 (2% 안)' if hit else '예측 반증 (2% 밖)'}.**",
              "도메인 행은 합산에만 썼다 (라벨 정확도 ~55%).", ""]

    # ── D2 ────────────────────────────────────────────────────────────────
    L += ["## D2 — 한자 밀집 한국어 BPB (`--max-bytes 7000000`)", ""]
    b = d2(ok)
    if "C0" not in b or "T2a" not in b:
        L += ["미실행" + (f" ({len(b)}/4 완료)" if b else ""), ""]
    else:
        L += ["| 조건 | 바이트 | BPB | C0 대비 |", "|---|---:|---:|---:|"]
        for c, (v, nb) in b.items():
            L.append(f"| {c} | {nb:,} | {v:.6f} | {v / b['C0'][0] - 1:+.2%} |")
        rel = b["T2a"][0] / b["C0"][0] - 1
        hit = abs(rel) <= D2_BOUND
        L += ["", f"**판정: T2a {rel:+.2%} — "
              f"{'예측 적중 (5% 안)' if hit else '예측 반증 (5% 밖)'}.**",
              "이 부분집합의 sigma 는 없다. 5% 미만 차이를 \"차이\" 로 주장하지 않는다.",
              "바이트 수가 조건마다 다른 것은 조각 첫 토큰을 빼기 때문이다 (bpb.py 문서).", ""]

    # ── D3 ────────────────────────────────────────────────────────────────
    L += ["## D3 — KMMLU 6과목 (법률·의학·시사), 5-shot, log-likelihood", ""]
    p = d3(ok)
    if "C0" not in p or "T2a" not in p:
        L += ["미실행" + (f" ({', '.join(p)} 완료)" if p else ""), ""]
    else:
        keys = [(x[0], x[1], x[4]) for x in p["C0"]]
        for c in p:
            if [(x[0], x[1], x[4]) for x in p[c]] != keys:
                raise SystemExit(f"D3 {c}: 문항 순서나 shot 수가 C0 와 다르다")
        L += ["| 조건 | 문항 | 정확도 | 95% CI |", "|---|---:|---:|---|"]
        acc = {}
        for c, rs in p.items():
            x = [r[2] for r in rs]
            lo, hi = bootstrap_ci(x)
            acc[c] = (sum(x) / len(x), lo, hi)
            L.append(f"| {c} | {len(x):,} | {acc[c][0]:.4f} | [{lo:.4f}, {hi:.4f}] |")
        a = [r[2] for r in p["T2a"]]
        c0 = [r[2] for r in p["C0"]]
        d, dlo, dhi = paired_diff_ci(a, c0)
        v = verdict(acc["C0"][1], dlo, dhi)
        L += ["", f"T2a − C0 = **{d:+.2%}p**, paired 95% CI [{dlo:+.2%}p, {dhi:+.2%}p], "
              f"동등성 경계 ±{MARGIN:.0%}p",
              "", f"**판정: {v}.**", ""]
        if v == "측정 불가 — 바닥":
            L += [f"C0 의 CI 하한 {acc['C0'][1]:.4f} 가 찍기(0.25) 이하다. "
                  "이 규모에서는 KMMLU 로 추론 차이를 잴 수 없다.", ""]

        L += ["### descriptive — 판정에 쓰지 않는다", "",
              "과목별 (라벨이 아니라 KMMLU 의 과목 구분이다)", "",
              "| 과목 | 문항 | " + " | ".join(p) + " |",
              "|---|---:|" + "---:|" * len(p)]
        for s in dict.fromkeys(k[0] for k in keys):
            idx = [i for i, k in enumerate(keys) if k[0] == s]
            cells = [f"{sum(p[c][i][2] for i in idx) / len(idx):.4f}" for c in p]
            L.append(f"| {s} | {len(idx):,} | " + " | ".join(cells) + " |")
        hidx = [i for i, r in enumerate(p["C0"]) if r[3]]
        L += ["", f"한자를 1자 이상 품은 문항: {len(hidx):,} / {len(keys):,} "
              f"({len(hidx) / len(keys):.1%})"]
        if hidx:
            hd, hlo, hhi = paired_diff_ci([a[i] for i in hidx], [c0[i] for i in hidx])
            L.append(f"그 문항만의 T2a − C0 = {hd:+.2%}p [{hlo:+.2%}p, {hhi:+.2%}p] "
                     "— **사후 분할이다. 판정에 쓰지 않는다.**")
        # capability.tsv 의 n_shot 은 등록 명목값(5)이다. 실제 분포는 여기서만 보인다.
        shots = {}
        for r in p["C0"]:
            shots[r[4]] = shots.get(r[4], 0) + 1
        L += ["", "실제 shot 수 (C0 토크나이저 2048 토큰 기준, 모든 모델에 같은 프롬프트): "
              + " · ".join(f"{k}-shot {v:,}" for k, v in sorted(shots.items(), reverse=True))]
        L += ["", "CI 는 문항 표본 오차만 반영한다. 학습 seed 분산은 없다 (seed 42 단독).", ""]

    OUT.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("\n".join(L))
    print(f"\n썼다 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
