"""Final Test 판정표 — PLAN "5주차 동결" D 를 집행한다 (GPU 0시간, 개봉 뒤에만 의미가 있다).

    .conda/python.exe tools/heldout_pairs.py

설정은 `tools/heldout_spec.py`. **결과를 보기 전에** 만든 집계기다. 손으로 고친 표가 아니다.
입력은 `doc_nll --split heldout` 이 남긴 문서별 NLL 두 개다.

## 판정하기 전에 확인하는 것

- 두 평가 run 이 원장에 ok 이고 note 에 heldout 표시가 있는가
- 평가한 체크포인트가 등록된 그것인가 — 디렉토리 해시가 `artifacts.tsv` 의 등록값과 같은가
- 두 평가가 같은 문서를 같은 순서로 봤는가 (문서 번호 · 원문 바이트)
- 언어별 원문 예산에 닿았는가

하나라도 어긋나면 판정을 내지 않고 이유를 적는다.

## 판정 (D 의 3)

한국어 d = BPB(T2b) - BPB(C0) 의 문서 paired bootstrap 95% 구간 하한 > 0 이면 **확증**,
아니면 **미확증**. 크기는 dev 값과 나란히 기술만 한다. 학습 seed 불확실성(dev seed 쌍 SD)은
따로 한 줄로 적고 섞지 않는다 (D 의 5).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import heldout_spec as S  # noqa: E402
from tools import doc_bootstrap as D  # noqa: E402

LEDGER = ROOT / "experiments" / "LEDGER.tsv"
ARTIFACTS = ROOT / "experiments" / "artifacts.tsv"
DOCS_DIR = ROOT / "data" / "interim" / "docs"
OUT = ROOT / "reports" / "tables" / "heldout.md"
LANG_NAME = {"ko": "한국어", "en": "영어", "code": "코드"}


def _tsv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def verdict(lo):
    """확증 / 미확증 / 모름(None)."""
    if lo is None:
        return None
    return "확증" if lo > 0 else "미확증"


def budget_reached(rows: list, budget: int) -> bool:
    return sum(r[3] for r in rows) >= budget


def checkpoint_hash_ok(name: str):
    """True/False/None(등록값 없음). 디렉토리 해시를 다시 계산해 등록값과 대조한다."""
    from src.utils.hashing import sha256_dir
    path, train_run = S.CHECKPOINTS[name]
    regs = [r for r in _tsv(ARTIFACTS) if r["run_id"] == train_run and r["path"] == path]
    if not regs:
        return None
    return sha256_dir(ROOT / path) == regs[-1]["artifact_sha256"]


def validate(docs: dict, ok: dict) -> list:
    """문제 목록. 비어 있으면 통과."""
    bad = []
    for name in S.CHECKPOINTS:
        rid = S.doc_run_id(name)
        r = ok.get(rid)
        if r is None:
            bad.append(f"{rid}: 원장 ok 행 없음")
            continue
        if "heldout" not in (r.get("note") or ""):
            bad.append(f"{rid}: note 에 heldout 표시 없음")
        h = checkpoint_hash_ok(name)
        if h is not True:
            bad.append(f"{name}: 체크포인트 해시 {'불일치' if h is False else '등록값 없음'}")
        for lang, _, budget in S.FILES:
            rows = docs[name].get(lang, [])
            if not rows or not budget_reached(rows, budget):
                bad.append(f"{name}/{lang}: 예산 {budget:,} B 에 닿지 않았다")
    if not bad:
        for lang, _, _ in S.FILES:
            p = D.pair_problem(docs["t2b"][lang], docs["c0"][lang])
            if p:
                bad.append(f"{lang}: 짝 불일치 — {p}")
    return bad


def compression(lang: str, fname: str, rows: list) -> dict:
    """평가한 바로 그 문서들의 tok/byte 를 두 토크나이저로 다시 센다 (CPU).

    evaluate 와 같은 규칙으로 읽는다 — 빈 줄을 건너뛰고 문서 번호 순. 원문 바이트가
    문서별 파일과 다르면 같은 문서가 아니므로 멈춘다.
    """
    from transformers import AutoTokenizer
    toks = {n: AutoTokenizer.from_pretrained(str(ROOT / S.CHECKPOINTS[n][0]))
            for n in S.CHECKPOINTS}
    n_tok = {n: 0 for n in toks}
    n_bytes = 0
    i = 0
    with (DOCS_DIR / fname).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if i >= len(rows):
                break
            text = json.loads(line)["text"]
            raw = len(text.encode("utf-8"))
            if raw != rows[i][3]:
                raise SystemExit(f"{lang} 문서 {i}: 원문 바이트가 문서별 파일과 다르다")
            n_bytes += raw
            for n, t in toks.items():
                n_tok[n] += len(t(text, add_special_tokens=False)["input_ids"])
            i += 1
    return {n: n_tok[n] / n_bytes for n in toks} | {"bytes": n_bytes, "docs": i}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Final Test 판정표")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-compression", action="store_true", help="압축률 재계산을 건너뛴다")
    args = ap.parse_args(argv)

    ok = {r["run_id"]: r for r in _tsv(LEDGER) if r["status"] == "ok"}
    docs = {n: D.load_docs_run(S.doc_run_id(n)) for n in S.CHECKPOINTS}
    problems = validate(docs, ok)
    L: list = []
    w = L.append
    w("# Final Test — 대표 조건 둘의 1회 평가\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/heldout_pairs.py`")
    w("> 규칙 [`docs/PLAN.md`](../../docs/PLAN.md) \"5주차 동결\" D · 설정 `tools/heldout_spec.py`.")
    w("> 손으로 고치지 마라. 개봉은 `final-test-opened` 태그의 커밋이다.\n")
    w("| | 체크포인트 | 평가 run |")
    w("|---|---|---|")
    for n, (p, _) in S.CHECKPOINTS.items():
        w(f"| {n} | `{p}` | `{S.doc_run_id(n)}` |")
    w("")
    w("## 검증\n")
    if problems:
        for p in problems:
            w(f"- **{p}**")
        w("\n**판정 없음** — 위 문제가 풀리기 전에는 수치를 읽지 않는다.\n")
        Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
        print(f"썼다 {args.out} (판정 없음)")
        return 1
    w("통과 — 원장 ok · 체크포인트 해시 = 등록값 · 문서 짝 일치 · 언어별 예산 도달.\n")

    r = {lang: D.paired_bootstrap(docs["t2b"][lang], docs["c0"][lang], S.B, S.RNG_SEED)
         for lang, _, _ in S.FILES}
    k = r["ko"]
    v = verdict(k["lo"])
    w("## 1차 — 한국어 잔차 d (D 의 3)\n")
    w("| | 문서 | BPB(C0) | BPB(T2b) | d | 95% 문서 구간 | 판정 |")
    w("|---|---:|---:|---:|---:|---|---|")
    w(f"| Final Test | {k['n']:,} | {k['bpb_b']:.6f} | {k['bpb_a']:.6f} | **{k['diff']:+.6f}** | "
      f"[{k['lo']:+.6f}, {k['hi']:+.6f}] | **{v}** |")
    dev = {n: D.load_docs(S.DEV_NAMES[n]) for n in S.CHECKPOINTS}
    if all(D.status(S.DEV_NAMES[n], dev[n]) == "통과" for n in S.CHECKPOINTS):
        dv = D.paired_bootstrap(dev["t2b"]["ko"], dev["c0"]["ko"], S.B, S.RNG_SEED)
        w(f"| dev (참고) | {dv['n']:,} | {dv['bpb_b']:.6f} | {dv['bpb_a']:.6f} | {dv['diff']:+.6f} | "
          f"[{dv['lo']:+.6f}, {dv['hi']:+.6f}] | — |")
        w("")
        w(f"Final Test − dev 의 d 차이 **{k['diff'] - dv['diff']:+.6f}** — dev 를 여러 번 본 데서 온"
          " 낙관 편향의 기술값이다 (판정에 쓰지 않는다).")
    w("")
    w(f"확증 기준: 구간 하한 > 0. 하한 {k['lo']:+.6f}.\n")

    from tools import wsd_pairs as P
    sg = P.sigmas(P.lm_rows())
    w("**학습 seed 불확실성 (따로 보고, D 의 5)**: dev 상수 LR seed 쌍 3개의 d 표본 SD "
      + (f"{sg['sigma_d']:.6f}" if sg["sigma_d"] is not None else "미확인")
      + " — Final Test 는 seed 42 체크포인트 하나만 평가했다. 위 구간에는 이 몫이 없다.\n")

    w("## 부차 — 영어 · 코드 (D 의 4, 기술만)\n")
    w("| 축 | 문서 | BPB(C0) | BPB(T2b) | d | 95% 문서 구간 |")
    w("|---|---:|---:|---:|---:|---|")
    for lang in ("en", "code"):
        x = r[lang]
        w(f"| {LANG_NAME[lang]} | {x['n']:,} | {x['bpb_b']:.6f} | {x['bpb_a']:.6f} | "
          f"{x['diff']:+.6f} | [{x['lo']:+.6f}, {x['hi']:+.6f}] |")
    w("")
    w("## 부차 — 압축률 (평가한 그 문서들, tok/byte)\n")
    if args.no_compression:
        w("건너뛰었다 (--no-compression).\n")
    else:
        w("| 축 | 문서 | 원문 바이트 | C0 (Qwen) | T2b | T2b/C0 - 1 |")
        w("|---|---:|---:|---:|---:|---:|")
        for lang, fname, _ in S.FILES:
            c = compression(lang, fname, docs["c0"][lang])
            w(f"| {LANG_NAME[lang]} | {c['docs']:,} | {c['bytes']:,} | {c['c0']:.6f} | "
              f"{c['t2b']:.6f} | {c['t2b'] / c['c0'] - 1:+.2%} |")
        w("")
    w("## 한계\n")
    w("- 체크포인트는 seed 42 하나씩이다. seed 불확실성은 dev 에서 빌렸다")
    w("- 고정 token-context 평가다 (amendment §2). 같은 원문 byte-context 비교가 아니다")
    w("- Final Test 는 dev · train 과 문서 단위로 나뉘었고 split 전에 전역 중복 제거를 했다"
      " (split_audit). Qwen 사전학습과의 겹침은 모른다")
    w("- 이 평가는 한 번만 한다. 이 표 이후 조건 · 경계 · 체크포인트 · 평가량을 바꾸지 않는다")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
