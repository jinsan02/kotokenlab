"""핵심 주장 표 — 3차(amendment) 종료 시점에 무엇을 주장하고 무엇을 철회했나 (GPU 0시간).

    .conda/python.exe tools/claims.py [--no-compression]

amendment §8 7주차 완료 기준 "core claim 표". 숫자는 전부 원장과 문서별 NLL 파일에서
**다른 도구의 함수로** 다시 낸다 — 손으로 옮긴 수치는 없다. 주장마다 통제축(무엇을 같게
했나) · 불확실성의 종류(문서 표본 / 학습 seed / 문항 표본 / 측정 반복) · 상태를 붙인다.

상태
    Final Test 확증   본 적 없는 문서에서 사전 등록 기준을 넘었다 (PLAN 5주차 D)
    dev 재현          dev 에서 seed 3쌍으로 재현됐다 (Final Test 는 안 봤다)
    dev 판정          dev 에서 사전 등록 경계로 판정했다 (seed 42)
    dev 측정          판정 경계 없이 잰 값이다
    철회              이전 문서의 주장을 amendment 로 거둬들였다 (본문은 보존)
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "tables" / "core_claims.md"
SEEDS = (42, 123, 2026)


def _fin(lm, run, dom="ko"):
    from tools.wsd_pairs import bpb_at
    return bpb_at(lm, run, "final", dom)


def claim_compression() -> dict:
    from tools.overview import KO_DOMAINS, tok_per_byte
    q = tok_per_byte("tok_bench_v2", "qwen_original", KO_DOMAINS)
    t = tok_per_byte("tok_bench_v2", "t2b_v2_n30000", KO_DOMAINS)
    return {"id": "C1", "claim": "T2b 는 한국어 토큰 수를 줄인다 (크기 보존 치환 N=30,000)",
            "value": f"dev tok/byte {q:.4f} -> {t:.4f} ({t / q - 1:+.1%})",
            "axis": "같은 원문 · 토크나이저만", "unc": "전수 측정 (표본 오차 없음)",
            "status": "dev 측정", "src": "tokenizer_metrics (tok_bench_v2) · heldout.md 압축률"}


def claim_system() -> dict:
    import csv
    p = ROOT / "experiments" / "system_bench.tsv"
    with p.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
                if r["mode"] == "raw_prompt" and r["run_id"] in ("sys_c0_qwen_v1", "sys_t2b_mean_v1")]
    by = {}
    for r in rows:
        by.setdefault(int(r["raw_chars"]), {})[r["run_id"]] = float(r["prefill_ms_mean"])
    d = [v["sys_t2b_mean_v1"] / v["sys_c0_qwen_v1"] - 1 for _, v in sorted(by.items())
         if len(v) == 2]
    return {"id": "C2", "claim": "같은 원문을 넣으면 T2b 의 prefill 이 짧다",
            "value": f"prefill {min(d):+.1%} ~ {max(d):+.1%} ({len(d)}개 길이)",
            "axis": "raw_prompt — 같은 원문, 토큰 수 다름, **품질 미매칭**",
            "unc": "측정 반복(n_runs) 안의 분산만, 배치 1 · GPU 한 대",
            "status": "dev 측정", "src": "system_bench.md"}


def claim_gap_seeds(lm) -> dict:
    d = [_fin(lm, f"cpt_t2b_mean_r5_seed{s}") - _fin(lm, f"cpt_c0_qwen_r5_seed{s}") for s in SEEDS]
    return {"id": "C3", "claim": "같은 원문 168.5MB CPT 후 T2b 의 한국어 dev BPB 가 C0 보다 나쁘다 (잔차 d)",
            "value": f"d 평균 {statistics.mean(d):+.6f} · seed SD {statistics.stdev(d):.6f} (3쌍)",
            "axis": "같은 원문 168.5MB · 상수 LR · 같은 seed 짝 (compute 는 다르다)",
            "unc": "학습 seed (n=3)", "status": "dev 재현", "src": "seed_pairs.md"}


def claim_heldout() -> dict:
    from tools import doc_bootstrap as D, heldout_spec as H
    from tools.heldout_pairs import _tsv, LEDGER, validate, verdict
    ok = {r["run_id"]: r for r in _tsv(LEDGER) if r["status"] == "ok"}
    docs = {n: D.load_docs_run(H.doc_run_id(n)) for n in H.CHECKPOINTS}
    # 2026-10-19 외부 검토 ③: 확증의 목적어를 좁힌다 — seed 42 대표 쌍 하나다.
    base = {"id": "C4", "claim": "사전에 고른 seed 42 대표 쌍에서, 처음 보는 한국어 문서에도 양의 잔차가 있다 (Final Test)",
            "axis": "C3 과 같다 · 대표 체크포인트 seed 42 둘", "unc": "문서 표본 (seed 는 C3 의 SD)",
            "src": "heldout.md"}
    if not all(docs.values()) or validate(docs, ok):
        return base | {"value": "—", "status": "미확인 (평가 없음 또는 검증 실패)"}
    r = D.paired_bootstrap(docs["t2b"]["ko"], docs["c0"]["ko"], H.B, H.RNG_SEED)
    v = verdict(r["lo"])
    return base | {"value": f"d {r['diff']:+.6f} [{r['lo']:+.6f}, {r['hi']:+.6f}] ({r['n']:,}문서)",
                   "status": "Final Test 확증" if v == "확증" else "Final Test 미확증"}


def claim_cosine_vs_const(lm) -> dict:
    cos = _fin(lm, "cpt_t2b_mean_main_seed42") - _fin(lm, "cpt_c0_qwen_main_seed42")
    con = _fin(lm, "cpt_t2b_mean_r5_seed42") - _fin(lm, "cpt_c0_qwen_r5_seed42")
    return {"id": "C5", "claim": "1차 격차의 일부는 코사인 스케줄의 몫이다 — '65% 벽' 은 식별되지 않았다",
            "value": f"d 코사인 {cos:+.6f} -> 상수 {con:+.6f} ({con - cos:+.6f})",
            "axis": "같은 원문 · 스케줄만 (seed 42)", "unc": "학습 seed (seed 42 하나)",
            "status": "dev 측정", "src": "cpt_main.md · seed_pairs.md · amendment §3"}


def claim_update() -> dict:
    from tools import upd_pairs as U
    lm = U.lm_rows()
    ko = [U.per_seed(s, lm)["ko"] for s in SEEDS]
    rho = [k["rho"] for k in ko]
    du = [k["d_upd"] for k in ko]
    # 2026-10-19 외부 검토 ①: "update 부족이 원인이 아니다" 는 인과 비약이라 관측으로 좁혔다.
    return {"id": "C6", "claim": "시험한 update 보정(같은 1,523 update)만으로는 격차가 해소되지 않았다",
            "value": f"같은 update 에서 d {statistics.mean(du):+.6f} · 줄어든 몫 rho {statistics.mean(rho):.1%}"
                     " (seed 별 평균, 원인의 기여율이 아니다)",
            "axis": "같은 optimizer update 수 (compute proxy, FLOPs 동일 아님)",
            "unc": "학습 seed (n=3)", "status": "dev 판정 (예측 적중)", "src": "upd_pairs.md"}


def claim_warm(lm) -> dict:
    cf = _fin(lm, "cpt_c0_qwen_r5_seed42")
    dd = _fin(lm, "cpt_t2b_mean_r5_seed42") - cf
    dw = _fin(lm, "cpt_t2b_mean_warm_seed42") - cf
    return {"id": "C7", "claim": "같은 총예산 안에서 새 행을 먼저 학습시키는 배분(첫 20%)은 직접 CPT 보다 낫지 않았다",
            "value": f"delta {dd - dw:+.6f} (gate +0.010 미달)", "axis": "총 원문 168.5MB (준비 비용 포함)",
            "unc": "학습 seed (seed 42 하나)", "status": "dev 판정 (예측 적중)", "src": "warm_pairs.md"}


def claim_wsd() -> dict:
    from tools import wsd_pairs as P, w6_spec as W6
    lm = P.lm_rows()
    f = {k: {s: P.bpb_at(lm, r, "final", "ko") for s, r in (("c", W6.BASE[k]), ("w", W6.WSD[k]))}
         for k in ("c0", "t2b")}
    db = f["t2b"]["w"] - f["t2b"]["c"]
    dd = db - (f["c0"]["w"] - f["c0"]["c"])
    return {"id": "C8", "claim": "시험한 WSD(마지막 20% 감쇠)는 상수 LR 보다 낫지 않다",
            "value": f"T2b {db:+.6f} · 잔차 {dd:+.6f} (경계 0.005, 둘 다 악화)",
            "axis": "같은 원문 168.5MB · 스케줄만", "unc": "학습 seed (seed 42 하나, σ 는 상수 LR 에서 빌림)",
            "status": "dev 판정 (예측 빗나감)", "src": "wsd_pairs.md"}


def claim_t2a(lm) -> dict:
    from src.evaluation.capability import paired_diff_ci
    from tools.contamination_check import scope_line
    from tools.hanja_probe import _ok_runs, d3
    p = d3(_ok_runs())
    rel = _fin(lm, "cpt_t2a_none_main_seed42") / _fin(lm, "cpt_c0_qwen_main_seed42") - 1
    kv = "KMMLU 미확인"
    if "T2a" in p and "C0" in p:
        d, lo, hi = paired_diff_ci([r[2] for r in p["T2a"]], [r[2] for r in p["C0"]])
        kv = f"KMMLU 6과목 {d:+.2%}p [{lo:+.2%}p, {hi:+.2%}p]"
    return {"id": "C9", "claim": "제거만 한 T2a 는 시험한 aggregate 지표에서 저하가 검출되지 않았다",
            "value": f"한국어 dev BPB {rel:+.3%} · {kv}",
            "axis": "같은 원문 · 코사인 seed 42 · vocab 축소만",
            "unc": "문항 표본 (KMMLU) · seed 하나. " + (scope_line() or "오염 검사 미확인"),
            "status": "dev 측정 (\"무손상\" 은 철회)", "src": "cpt_main.md · hanja_probe.md"}


WITHDRAWN = (
    ("W1", "회복률 65% 는 보편적 벽이다", "코사인 스케줄의 유한 종점이었다 (C5). 점근선은 미식별"),
    ("W2", "R1 이 치환 특유 현상을 입증했다", "행 수 · 노출 · 토큰화가 함께 달랐다 — 두 구성의 궤적이 달랐을 뿐"),
    ("W3", "tie 는 원인이 아니다", "17.5MB · 그 설정에서 5%p 급 효과가 없었을 뿐 (paired +0.444%p)"),
    ("W4", "CPT 행 이식은 완벽한 초기화 상한이다", "몸통-행 상호적응을 깨는 compatibility stress test 다"),
    ("W5", "Common Crawl 중복은 C0 에 유리하다", "겹침의 크기와 방향을 재지 않았다"),
    ("W6", "E1 은 FVT 와 동등하다", "standalone 표면이 있는 25,821개에서만 일치"),
    ("W7", "168.5MB 는 같은 compute 다", "같은 원문량이다. 토큰 · update 가 다르다 (C6 이 따로 잰다)"),
)


def rows(compression: bool = True) -> list:
    from tools.wsd_pairs import lm_rows
    lm = lm_rows()
    out = [claim_compression()] if compression else []
    out += [claim_system(), claim_gap_seeds(lm), claim_heldout(), claim_cosine_vs_const(lm),
            claim_update(), claim_warm(lm), claim_wsd(), claim_t2a(lm)]
    return out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="핵심 주장 표")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-compression", action="store_true")
    args = ap.parse_args(argv)
    L: list = []
    w = L.append
    w("# 핵심 주장 표 — 3차(amendment) 종료 시점\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/claims.py` — 수치는 원장과 문서별")
    w("> NLL 파일에서 각 도구의 함수로 다시 냈다. 손으로 고치지 마라.\n")
    w("한국어 품질 지표는 고정 token-context dev BPB 다 (amendment §2). 잔차 d = BPB(T2b) − BPB(C0).")
    w("불확실성은 종류가 다르면 섞지 않는다.\n")
    w("## 주장\n")
    w("| | 주장 | 수치 | 통제축 | 불확실성 | 상태 | 근거 |")
    w("|---|---|---|---|---|---|---|")
    for r in rows(not args.no_compression):
        w(f"| {r['id']} | {r['claim']} | {r['value']} | {r['axis']} | {r['unc']} | **{r['status']}** | {r['src']} |")
    w("")
    w("## 철회한 주장 (본문은 각 문서에 보존, amendment §3)\n")
    w("| | 이전 주장 | 지금 말할 수 있는 것 |")
    w("|---|---|---|")
    for i, a, b in WITHDRAWN:
        w(f"| {i} | {a} | {b} |")
    w("")
    w("## 읽는 법\n")
    w("- **Final Test 확증** 은 C4 하나뿐이다. 나머지는 dev 결과다 — dev 는 여러 번 본 집합이다")
    w("- 시스템 수치(C2)는 품질이 나쁜 모델의 비용이다. 품질을 맞춘 서비스 개선이 아니다")
    w("- compute 를 맞춘 비교는 C6 의 update 수 proxy 뿐이다 (FLOPs 는 재지 않았다)")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
