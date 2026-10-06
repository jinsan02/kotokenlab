"""제거 토큰 노출 층화 — T2a 의 제거가 문서에 따라 다르게 작용했는가 (amendment §6 · 5주차).

    .conda/python.exe tools/exposure_strata.py

동결 설정은 `tools/w5_spec.py` (PLAN.md "5주차 동결"). 학습 없음.

## 왜

D2 의 "한자 밀집" 부분집합은 **문자** 의 한자 밀도로 골랐다. 그것은 T2a 가 지운 토큰에
얼마나 노출됐는지와 같지 않다 (amendment §1 "한자 밀집도가 제거 토큰 영향량이다" 수용).
여기서는 **원본 Qwen 토크나이저로 토큰화했을 때 T2a 가 지운 토큰이 몇 번 나오는가** 로
dev 문서를 나눈다.

## 구간 (결과를 보기 전에 정한 규칙)

    0      제거 토큰이 한 번도 안 나온 문서
    low    1 .. m
    high   > m           m = 제거 토큰이 1번 이상 나온 문서들의 출현 수 중앙값

구간마다 BPB(T2a) − BPB(C0) 를 그 구간 문서만으로 내고, 같은 방식의 문서 단위 paired
bootstrap 으로 구간을 붙인다. 문서가 `MIN_DOCS_PER_BIN` 보다 적은 구간은 수치를 비교하지
않는다. **탐색적 분석이다** — 판정 경계를 두지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from tools import w5_spec as S  # noqa: E402
from tools.doc_bootstrap import load_docs, pair_problem, paired_bootstrap, status  # noqa: E402

OUT = ROOT / "reports" / "tables" / "exposure_strata.md"
DEV = ROOT / "data" / "interim" / "docs" / "dev.jsonl"


def removed_ids() -> set:
    """prune 목록의 token_id. TSV 의 token 칸에 따옴표가 있어 탭으로 직접 쪼갠다."""
    out = set()
    with (ROOT / S.PRUNE_LIST).open(encoding="utf-8", newline="") as fh:
        head = fh.readline().rstrip("\r\n").split("\t")
        k = head.index("token_id")
        for line in fh:
            out.add(int(line.rstrip("\r\n").split("\t")[k]))
    return out


def dev_texts(n: int) -> list:
    """dev.jsonl 의 앞 n 문서 (빈 줄 제외 — bpb.evaluate 와 같은 번호)."""
    out = []
    with DEV.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(json.loads(line)["text"])
            if len(out) >= n:
                break
    return out


def bins_for(counts: list) -> tuple:
    """(구간 이름 목록, m). m 은 0 이 아닌 출현 수의 중앙값."""
    nz = [c for c in counts if c > 0]
    m = statistics.median(nz) if nz else 0
    names = ["0" if c == 0 else ("low" if c <= m else "high") for c in counts]
    return names, m


def main(argv: list | None = None) -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="제거 토큰 노출 층화")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    a, b = S.STRATA_PAIR
    da, db = load_docs(a), load_docs(b)
    sa, sb = status(a, da), status(b, db)

    L: list = []
    w = L.append
    w("# 제거 토큰 노출 층화 — T2a vs C0 (dev 한국어)\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/exposure_strata.py`")
    w("> 동결 설정 `tools/w5_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md) \"5주차 동결\".")
    w("> **탐색적 분석** — 판정 경계가 없다.\n")
    w(f"비교: `{a}` − `{b}` (1차 코사인 seed 42). 노출 = 원본 Qwen 토크나이저로 토큰화했을 때")
    w(f"T2a 가 지운 토큰(`{S.PRUNE_LIST}`)이 나온 횟수.\n")

    if sa != "통과" or sb != "통과":
        w(f"**아직 계산하지 않는다** — {a}: {sa} · {b}: {sb}\n")
    else:
        prob = pair_problem(da.get("ko", []), db.get("ko", []))
        if prob:
            w(f"**짝 불일치** — {prob}\n")
        else:
            rows_a, rows_b = da["ko"], db["ko"]
            texts = dev_texts(len(rows_a))
            tok = AutoTokenizer.from_pretrained(S.BASE_TOKENIZER[0], revision=S.BASE_TOKENIZER[1])
            rm = removed_ids()
            counts = [sum(1 for i in tok(t, add_special_tokens=False)["input_ids"] if i in rm)
                      for t in texts]
            for x, t in zip(rows_a, texts):
                if x[3] != len(t.encode("utf-8")):
                    w("**문서 대응 불일치** — dev.jsonl 의 순번과 doc_nll 이 다르다\n")
                    break
            else:
                names, m = bins_for(counts)
                nz = sum(1 for c in counts if c > 0)
                w(f"문서 {len(counts):,} 중 제거 토큰이 1번 이상 나온 문서 {nz:,} "
                  f"({nz / len(counts):.1%}), 총 출현 {sum(counts):,}회. 구간 경계 m = {m}.\n")
                w("| 구간 | 문서 | 출현 수 범위 | BPB(T2a) | BPB(C0) | 차이 | 95% 구간 |")
                w("|---|---:|---|---:|---:|---:|---|")
                for name in ("0", "low", "high"):
                    idx = [i for i, n in enumerate(names) if n == name]
                    rng = ("0" if name == "0" else
                           f"1–{m}" if name == "low" else f"> {m}")
                    if len(idx) < S.MIN_DOCS_PER_BIN:
                        w(f"| {name} | {len(idx):,} | {rng} | — | — | — | 문서 {S.MIN_DOCS_PER_BIN} 미만 — 비교 안 함 |")
                        continue
                    r = paired_bootstrap([rows_a[i] for i in idx], [rows_b[i] for i in idx],
                                         S.B, S.RNG_SEED)
                    w(f"| {name} | {r['n']:,} | {rng} | {r['bpb_a']:.6f} | {r['bpb_b']:.6f} | "
                      f"{r['diff']:+.6f} | [{r['lo']:+.6f}, {r['hi']:+.6f}] |")
                w("")
    w("## 한계\n")
    w("- 1차 코사인 seed 42 하나다. 구간은 문서 표본의 불확실성만 담는다")
    w("- 노출은 **원본** 토크나이저 기준이다. T2a 토크나이저는 그 자리를 다른 토큰으로 쪼갠다")
    w("- 구간이 문서 길이와 섞인다 — 긴 문서일수록 출현 수가 크다. 밀도가 아니라 횟수다")
    w("- 한자 문자 밀도(D2)와는 다른 축이다")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {Path(args.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
