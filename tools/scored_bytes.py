"""BPB 가 **원문의 어디를 채점하고 어디를 빼는지** 조건별로 잰다 (GPU 0시간).

    .conda/python.exe tools/scored_bytes.py

## 왜

`bpb.py` 는 문서를 토큰화해 2,048 토큰 비중첩 조각으로 자르고, 각 조각의 **첫
토큰을 뺀** 나머지를 채점한다. 첫 토큰은 예측 대상이 아니기 때문이다.

그런데 조각 경계는 **토큰 기준** 이라 토크나이저마다 다르다. 그래서

- 어느 원문 위치가 빠지는지가 조건마다 다르고
- 최종 분모(`n_bytes`)도 조건마다 조금씩 다르다

실제로 D2 에서 같은 787문서를 재는데 C0 6,701,611B · T2a 6,701,664B ·
T2b 6,701,313B 로 갈렸다 (0.005% 범위). 2026-09-19 감사가 이것을 "동일
byte-context 비교가 아니다" 로 지적했고, 이 도구가 그 크기를 숫자로 만든다.

**모델을 올리지 않는다.** 토큰화만 하면 되는 계산이라 GPU 가 필요 없고, 따라서
어떤 체크포인트에서도 같은 값이다.

## 무엇을 세나

```
총 바이트       문서 원문 UTF-8 길이 (토크나이저와 무관한 기준값)
채점 바이트     각 조각의 2번째 토큰부터가 나타내는 바이트
빠진 바이트     조각 첫 토큰 + 마지막 불완전 조각(2토큰 미만)
경계 수         조각 수 = 첫 토큰이 빠진 횟수
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT = ROOT / "reports" / "tables" / "scored_bytes.md"
DOCS = ROOT / "data" / "interim" / "docs"

# D2 와 같은 조건 셋. 모델이 아니라 **토크나이저** 만 쓴다.
CONDITIONS = (
    ("C0", "artifacts/models/cpt_c0_qwen_main_seed42"),
    ("T2a", "artifacts/models/cpt_t2a_none_main_seed42"),
    ("T2b", "artifacts/models/cpt_t2b_mean_main_seed42"),
)


def measure(tok, path: Path, max_bytes: int, seq_len: int, byte_len_fn) -> dict:
    """bpb.evaluate 와 **같은 순서로** 세되 NLL 은 계산하지 않는다."""
    total_bytes = scored_bytes = 0
    first_token_bytes = 0.0
    dropped_tail = 0.0
    n_chunks = n_docs = 0
    seen = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            raw = len(text.encode("utf-8"))
            seen += raw
            total_bytes += raw
            n_docs += 1
            ids = tok(text, add_special_tokens=False)["input_ids"]
            for i in range(0, len(ids), seq_len):
                chunk = ids[i:i + seq_len]
                if len(chunk) < 2:
                    dropped_tail += byte_len_fn(chunk)
                    continue
                n_chunks += 1
                first_token_bytes += byte_len_fn(chunk[:1])
                scored_bytes += byte_len_fn(chunk[1:])
            if seen >= max_bytes:
                break

    return {"n_docs": n_docs, "n_chunks": n_chunks, "total_bytes": total_bytes,
            "scored_bytes": scored_bytes, "first_token_bytes": first_token_bytes,
            "dropped_tail": dropped_tail}


def main(argv: list | None = None) -> int:
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import AutoTokenizer

    from src.evaluation.bpb import token_byte_length

    ap = argparse.ArgumentParser(description="BPB 채점 바이트 진단")
    ap.add_argument("--split", default="dev_hanja",
                    help="data/interim/docs/<split>.jsonl (기본은 D2 가 쓴 것)")
    ap.add_argument("--max-bytes", type=int, default=7_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    path = DOCS / f"{args.split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} 가 없다")
    alphabet = set(ByteLevel.alphabet())

    results = {}
    for name, model_path in CONDITIONS:
        tok = AutoTokenizer.from_pretrained(str(ROOT / model_path))
        cache: dict = {}

        def byte_len_fn(ids: list, _tok=tok, _cache=cache) -> int:
            total = 0
            for i in ids:
                v = _cache.get(i)
                if v is None:
                    v = token_byte_length(_tok.convert_ids_to_tokens(int(i)), alphabet)
                    _cache[i] = v
                total += v
            return total

        results[name] = measure(tok, path, args.max_bytes, args.seq_len, byte_len_fn)
        print(f"  {name} 완료")

    o: list = []
    w = o.append
    w("# BPB 는 원문의 어디를 채점하는가\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/scored_bytes.py`")
    w("> 모델을 올리지 않는다 — 토큰화만으로 나오는 값이라 체크포인트와 무관하다.\n")
    w("`bpb.py` 는 2,048 토큰 비중첩 조각의 **첫 토큰을 빼고** 채점한다. 조각")
    w("경계가 토큰 기준이라 **어디가 빠지는지가 조건마다 다르다.** 그 크기를 잰다.\n")
    w(f"대상: `{args.split}` · 상한 {args.max_bytes / 1e6:.2f}MB · "
      f"seq_len {args.seq_len}\n")
    w("| 조건 | 문서 | 조각 | 원문 바이트 | 채점 바이트 | 첫 토큰으로 빠짐 | 꼬리 조각 | 채점 비율 |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        w(f"| {name} | {r['n_docs']:,} | {r['n_chunks']:,} | {r['total_bytes']:,} | "
          f"{r['scored_bytes']:,} | {r['first_token_bytes']:,.0f} | "
          f"{r['dropped_tail']:,.0f} | {r['scored_bytes'] / r['total_bytes']:.4%} |")
    w("")

    base = results["C0"]["scored_bytes"]
    w("## 조건 간 분모 차이\n")
    w("| 조건 | 채점 바이트 | C0 대비 |")
    w("|---|---:|---:|")
    for name, r in results.items():
        w(f"| {name} | {r['scored_bytes']:,} | {r['scored_bytes'] / base - 1:+.4%} |")
    w("")
    spread = (max(r["scored_bytes"] for r in results.values())
              - min(r["scored_bytes"] for r in results.values()))
    w(f"최대 차이는 **{spread:,} 바이트 ({spread / base:.4%})** 다.\n")
    w("## 읽는 법\n")
    w("- 이 차이는 **분모** 의 차이다. 조건 간 BPB 격차(수십 %)보다 서너 자릿수")
    w("  작으므로 1차·2차의 큰 결론을 뒤집지 않는다")
    w("- 다만 **같은 원문 위치를 채점한 비교가 아니다.** 고정 token-context")
    w("  배포 조건의 비교이고, 동일 byte-context likelihood 는 아직 측정하지 않았다")
    w("- 빠지는 양의 대부분은 조각 첫 토큰이다. 조각이 길수록(seq_len 이 클수록)")
    w("  비율이 줄어든다 — 같은 seq_len 을 쓰는 한 조건 간 비교에는 공평하다")
    w("- 소수점 차이를 근거로 조건을 고르지 않는다. 이 표는 **크기를 알아 두기**")
    w("  위한 것이다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(o))
    print(f"\n썼다 {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
