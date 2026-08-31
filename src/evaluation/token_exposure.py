"""새 토큰이 학습 중 몇 번 발화하는가 (스펙 §26 의 해석 도구).

왜 필요한가
    본 CPT 에서 T2b 는 C0 를 따라잡지 못했다 (한국어 BPB 1.5671 vs 1.1375).
    "초기화가 나빴나, 노출이 부족했나" 를 가르려면 새 토큰이 실제로 몇 번
    gradient 를 받았는지 세야 한다. 총량이 아니라 **토큰당** 분포가 답을 준다 —
    실측에서 새 토큰이 전체 토큰의 43% 를 차지하는데도 중앙값 발화는 143회였다.
    소수의 머리가 다 먹기 때문이다.

    이 숫자가 reports/tables/cpt_main.md 의 결론 근거이므로, 손으로 돌린
    일회성 스크립트가 아니라 저장소 안에 둔다.

사용
    .conda/python.exe -m src.evaluation.token_exposure \
        --model artifacts/models/kot2b_v2_n30000_mean \
        --id-map artifacts/tokenizers/kot2b_v2_n30000/id_map.json \
        --pool-docs 50000 --sample-bytes 20000000 --full-bytes 168500000

CPT 와 **같은 문서를 같은 순서로** 본다 (load_pool + seed shuffle). 전량을
토큰화하면 오래 걸리므로 앞의 sample-bytes 만 세고 full-bytes 로 환산한다.
환산은 문서를 섞어 뽑았으므로 타당하지만, **'0회' 버킷만은 상한이다** —
표본에서 0 이면 환산해도 0 이라, 뒤쪽 문서에서 발화하는 것이 섞여 있다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.training.cpt import load_pool  # noqa: E402

BUCKETS = ((0, 0, "0회 (한 번도 안 나옴)"), (1, 99, "1~99"),
           (100, 999, "100~999"), (1000, 9999, "1,000~9,999"),
           (10000, 10 ** 15, "10,000 이상"))


def count_fires(tokenizer, docs, new_ids: set, sample_bytes: int,
                batch: int = 500) -> tuple:
    """(발화 Counter, 본 바이트, 총 토큰). CPT 와 같은 순서로 훑는다."""
    cnt: Counter = Counter()
    total_tok = seen = 0
    buf: list = []

    def flush():
        nonlocal total_tok
        for ids in tokenizer(buf, add_special_tokens=False)["input_ids"]:
            cnt.update(ids)
            total_tok += len(ids)

    for text in docs:
        seen += len(text.encode("utf-8"))
        buf.append(text)
        if len(buf) >= batch:
            flush()
            buf = []
        if seen >= sample_bytes:
            break
    if buf:
        flush()
    return cnt, seen, total_tok


def main(argv: list | None = None) -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="새 토큰의 학습 중 발화 분포")
    ap.add_argument("--model", required=True)
    ap.add_argument("--id-map", required=True,
                    help="artifacts/tokenizers/<tok>/id_map.json")
    ap.add_argument("--pool-docs", type=int, default=50_000)
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-bytes", type=int, default=20_000_000)
    ap.add_argument("--full-bytes", type=int, default=168_500_000,
                    help="환산 대상 — 실제 CPT 예산")
    ap.add_argument("--out", default=None, help="토큰별 발화수 TSV 경로")
    args = ap.parse_args(argv)

    idmap = json.loads(Path(args.id_map).read_text(encoding="utf-8"))
    new_ids = {int(k) for k in idmap["map"]}
    print(f"새 토큰 {len(new_ids):,}개  (mode {idmap.get('mode')})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    docs = load_pool(ROOT / "data" / "interim" / "docs" / "train.jsonl",
                     args.pool_docs, args.skip_docs)
    random.Random(args.seed).shuffle(docs)      # CPT 와 같은 순서

    cnt, seen, total_tok = count_fires(tokenizer, docs, new_ids,
                                       args.sample_bytes)
    if not seen:
        raise SystemExit("문서를 하나도 못 읽었다. 코퍼스 경로를 확인하라")

    scale = args.full_bytes / seen
    fires = {i: cnt.get(i, 0) for i in new_ids}
    tot_new = sum(fires.values())
    vals = sorted(fires.values())

    print(f"표본 {seen / 1e6:.1f}MB, 토큰 {total_tok:,}  (환산 배율 {scale:.2f})")
    print(f"새 토큰 총 발화 {tot_new:,} = 전체 토큰의 "
          f"{100 * tot_new / total_tok:.2f}%")
    print(f"환산: 전체에서 약 {int(tot_new * scale):,}회")
    print()
    print(f"토큰당 발화(환산)  중앙값 {vals[len(vals) // 2] * scale:,.0f}   "
          f"평균 {statistics.mean(vals) * scale:,.0f}   "
          f"최대 {vals[-1] * scale:,.0f}")
    print()
    print("발화 횟수 분포 (전체 환산)")
    for lo, hi, label in BUCKETS:
        n = sum(1 for v in vals if lo <= v * scale <= hi)
        note = "   <- 표본 기준이라 상한이다" if lo == hi == 0 else ""
        print(f"  {label:22} {n:6,}개  ({100 * n / len(vals):5.1f}%){note}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("token_id\ttoken\tfires_sample\tfires_scaled\n")
            for i in sorted(fires, key=lambda k: -fires[k]):
                tok = tokenizer.convert_ids_to_tokens(i)
                fh.write(f"{i}\t{tok}\t{fires[i]}\t{int(fires[i] * scale)}\n")
        print(f"\n토큰별 표: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
