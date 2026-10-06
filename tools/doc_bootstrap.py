"""문서 단위 paired bootstrap — BPB 차이의 구간 (amendment §4 · 5주차, 학습 없음).

    .conda/python.exe tools/doc_bootstrap.py

동결 설정은 `tools/w5_spec.py` (PLAN.md "5주차 동결"). 입력은
`src/evaluation/doc_nll.py` 가 남긴 `experiments/runs/<eval run>/doc_nll.tsv` 다.

## 무엇을 계산하나

두 체크포인트 a, b 를 **같은 dev 문서** 로 짝짓는다. 문서를 복원 추출하고 매 표본에서

    BPB = sum(NLL) / (ln2 * sum(채점 바이트))

를 각각 다시 계산해 차이 BPB(a) - BPB(b) 의 분포를 얻는다 (amendment §4 가 정한 방식 —
문서 BPB 의 평균이 아니다). b 가 C0 면 그 차이가 잔차 d 다.

## 무엇을 확인하고 나서 계산하나

- 두 평가가 **같은 문서 집합을 같은 순서로** 봤는가 (문서 번호 · 원문 바이트가 같다)
- 각 평가가 그 체크포인트를 만든 학습 run 의 **원장 dev BPB 를 재현** 했는가
  (doc_nll 이 원장 note 에 `<lang>_vs_ledger=` 로 남긴다. 반올림 6자리에서 같아야 한다)

하나라도 어긋나면 그 쌍은 계산하지 않고 이유를 적는다.

## 이 구간이 말하는 것과 말하지 않는 것

dev **문서 표본** 의 불확실성이다. 학습 seed 의 불확실성은 담지 않는다 — 그것은
seed 쌍 표(`seed_pairs.md`, `upd_pairs.md`)가 따로 낸다 (amendment §9 의 5번).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import w5_spec as S  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
LEDGER = ROOT / "experiments" / "LEDGER.tsv"
OUT = ROOT / "reports" / "tables" / "doc_bootstrap.md"
LANGS = ("ko", "en", "code")
LN2 = math.log(2)


def ok_note(run_id: str):
    """eval run 의 ok 행 note. 없으면 None."""
    with LEDGER.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
                if r["run_id"] == run_id and r["status"] == "ok"]
    return rows[-1].get("note", "") if rows else None


def reproduced(note: str, lang: str):
    """원장 재현 여부: True / False / None(기록 없음)."""
    m = re.search(rf"\b{lang}_vs_ledger=([+-]?[0-9.]+)", note or "")
    if not m:
        return None
    return abs(float(m.group(1))) < 5e-7


def load_docs(name: str) -> dict:
    """lang -> [(doc, nll, bytes, raw_bytes)] (문서 번호 순). 파일이 없으면 빈 dict."""
    p = RUNS / S.doc_run_id(name) / "doc_nll.tsv"
    if not p.exists():
        return {}
    out: dict = {}
    with p.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            out.setdefault(r["lang"], []).append(
                (int(r["doc"]), float(r["nll"]), int(r["bytes"]), int(r["raw_bytes"])))
    for v in out.values():
        v.sort()
    return out


def pair_problem(a: list, b: list):
    """짝지을 수 없으면 이유, 있으면 None."""
    if len(a) != len(b):
        return f"문서 수 {len(a)} != {len(b)}"
    for x, y in zip(a, b):
        if x[0] != y[0] or x[3] != y[3]:
            return f"문서 {x[0]} 의 번호·원문 바이트가 다르다"
    return None


def bpb(nll, nbytes) -> float:
    return float(sum(nll)) / (LN2 * float(sum(nbytes)))


def paired_bootstrap(a: list, b: list, n_boot: int, seed: int) -> dict:
    """차이 BPB(a)-BPB(b) 의 점추정과 95% 백분위 구간."""
    import numpy as np
    na = np.array([x[1] for x in a]); ba = np.array([x[2] for x in a], dtype=np.float64)
    nb = np.array([x[1] for x in b]); bb = np.array([x[2] for x in b], dtype=np.float64)
    point = bpb(na, ba) - bpb(nb, bb)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    step = 1000
    for k in range(0, n_boot, step):
        idx = rng.integers(0, n, size=(min(step, n_boot - k), n))
        da = na[idx].sum(1) / (LN2 * ba[idx].sum(1))
        db = nb[idx].sum(1) / (LN2 * bb[idx].sum(1))
        diffs[k:k + len(idx)] = da - db
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n": n, "bpb_a": bpb(na, ba), "bpb_b": bpb(nb, bb), "diff": point,
            "lo": float(lo), "hi": float(hi), "se": float(diffs.std(ddof=1))}


def status(name: str, docs: dict) -> str:
    if not docs:
        return "아직 없음"
    note = ok_note(S.doc_run_id(name))
    marks = {lang: reproduced(note, lang) for lang in LANGS}
    bad = [lang for lang, v in marks.items() if v is False]
    if bad:
        return "원장 재현 실패: " + ", ".join(bad)
    if any(v is None for v in marks.values()):
        return "원장 대조 기록 없음"
    return "통과"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="문서 단위 paired bootstrap")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    names = sorted({n for _, a, b, _ in S.PAIRS for n in (a, b)})
    docs = {n: load_docs(n) for n in names}
    st = {n: status(n, docs[n]) for n in names}

    L: list = []
    w = L.append
    w("# 문서 단위 paired bootstrap — dev BPB 차이의 구간\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/doc_bootstrap.py`")
    w("> 동결 설정 `tools/w5_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md) \"5주차 동결\".")
    w(f"> dev 원문 언어별 {S.EVAL_BYTES / 1e6:.0f}MB, 복원 추출 {S.B:,}회 (seed {S.RNG_SEED}),"
      " 95% 백분위 구간.\n")
    w("차이 = BPB(a) − BPB(b). b 가 C0 면 잔차 d 다. **문서 표본의 불확실성만 담는다** —")
    w("학습 seed 의 불확실성은 seed 쌍 표가 따로 낸다.\n")

    w("## 체크포인트 평가\n")
    w("| 이름 | 체크포인트 | 평가 run | 상태 |")
    w("|---|---|---|---|")
    for n in names:
        w(f"| {n} | `{S.CHECKPOINTS[n][0]}` | `{S.doc_run_id(n)}` | {st[n]} |")
    w("")

    for lang in LANGS:
        w(f"## {dict(ko='한국어', en='영어', code='코드')[lang]}\n")
        w("| 비교 | a | b | 문서 | BPB(a) | BPB(b) | 차이 | 95% 구간 |")
        w("|---|---|---|---:|---:|---:|---:|---|")
        for label, a, b, _ in S.PAIRS:
            if st[a] != "통과" or st[b] != "통과":
                w(f"| {label} | {a} | {b} | — | — | — | — | 계산 안 함 ({a}: {st[a]} · {b}: {st[b]}) |")
                continue
            prob = pair_problem(docs[a].get(lang, []), docs[b].get(lang, []))
            if prob:
                w(f"| {label} | {a} | {b} | — | — | — | — | 짝 불일치: {prob} |")
                continue
            r = paired_bootstrap(docs[a][lang], docs[b][lang], S.B, S.RNG_SEED)
            w(f"| {label} | {a} | {b} | {r['n']:,} | {r['bpb_a']:.6f} | {r['bpb_b']:.6f} | "
              f"{r['diff']:+.6f} | [{r['lo']:+.6f}, {r['hi']:+.6f}] |")
        w("")

    w("## 비교의 뜻\n")
    for label, a, b, why in S.PAIRS:
        w(f"- **{label}** — {why}")
    w("")
    w("## 한계\n")
    w("- dev 문서는 1차부터 여러 번 본 집합이다 — 낙관 편향이 있을 수 있다 (Final Test 를 따로 둔 이유)")
    w("- 고정 token-context 평가다. 같은 원문 byte-context 비교가 아니다 (amendment §2)")
    w("- 1차 코사인 run 들은 seed 42 하나이고, 상수 LR 대표 조건도 seed 42 재학습 하나다")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {Path(args.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
