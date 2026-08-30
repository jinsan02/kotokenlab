"""T2b 기증 토큰 채굴 — 한국어 고효율 토큰 후보 (스펙 §11~13, §99).

무엇을 하는가
    한국어 코퍼스를 **기존 Qwen 토크나이저로** 토큰화한 뒤, 인접한 토큰 쌍의
    빈도를 센다. 가장 자주 붙어 나오는 쌍이 곧 "합치면 가장 이득인 토큰" 이다.

왜 별도 BPE 를 학습하지 않는가
    별도 한국어 BPE 를 학습하면 그 토큰들이 Qwen vocab 으로 표현될 보장이 없고,
    몇 조각으로 쪼개질지도 제각각이라 슬롯 회계가 흐트러진다. T2b 의 정의는
    **vocab 크기 유지** 이므로 회계가 어긋나면 주장 자체가 깨진다.

    인접쌍 채굴은 그 문제가 없다.

        새 토큰 1개 = merge 규칙 1개 = 슬롯 1개

    부품 두 개가 이미 vocab 에 있으므로 merge 규칙을 그대로 쓸 수 있고,
    이득도 정확히 "출현 횟수 x 1토큰" 으로 계산된다. BPE 가 원래 쓰는 탐욕
    기준을 Qwen vocab 위에서 한 겹 더 적용하는 것과 같다.

한계 (Phase A)
    한 번만 채굴한다. 새로 추가한 토큰이 다시 이웃과 합쳐질 기회는 보지 않는다.
    여러 라운드를 돌리면 이득이 더 커지지만 라운드마다 코퍼스를 다시 토큰화해야
    한다. 단일 라운드 이득이 부족하면 그때 Phase B 로 간다.

    표본은 기본 300MB 다. 상위 2만 쌍의 순위를 정하는 데는 충분하고
    (그 구간 빈도가 수천 회 단위), 전체 4.2GB 를 돌리는 비용을 아낀다.

    .conda/python.exe -m src.tokenizer.train --tag v1 --top 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import numpy as np  # noqa: E402

from src.utils.tracking import RunContext, make_run_id  # noqa: E402

TAB = chr(9)
NL = chr(10)


def mine_pairs(tokenizer, path: Path, max_bytes: int, vocab_size: int,
               batch: int = 512) -> tuple:
    """**pretokenizer 세그먼트 안쪽** 인접 토큰 쌍의 빈도를 센다.

    경계를 넘는 쌍은 세면 안 된다. Qwen 의 Split 정규식은 숫자를 한 자씩
    고립시키고(`\\p{N}`) 글자와 구두점을 다른 세그먼트로 가른다. 그래서
    " 1", "20", "다." 같은 쌍은 merge rank 를 어떻게 줘도 **절대 발화하지
    않는다** — BPE 는 세그먼트 안에서만 병합하기 때문이다.

    처음에는 이 확인 없이 세었고, 채굴 빈도 10만회 이상인 최상위 기증자 6개가
    발화율 0% 였다. 슬롯 30,000개 중 12,311개가 그렇게 낭비됐다.

    토큰의 문자 오프셋과 세그먼트 시작 위치를 대조해서 거른다.
    """
    pre = tokenizer.backend_tokenizer.pre_tokenizer
    chunks: list = []
    seen_bytes = 0
    n_docs = n_tokens = n_cross = 0
    buf: list = []

    def flush() -> None:
        nonlocal n_tokens, n_cross
        if not buf:
            return
        enc = tokenizer(buf, add_special_tokens=False, return_offsets_mapping=True)
        for text, ids, offs in zip(buf, enc["input_ids"], enc["offset_mapping"]):
            if len(ids) < 2:
                continue
            arr = np.asarray(ids, dtype=np.int64)
            n_tokens += arr.size
            starts = {span[0] for _, span in pre.pre_tokenize_str(text)}
            # 쌍 (i, i+1) 은 i+1 이 새 세그먼트를 열지 않을 때만 유효하다
            keep = np.fromiter(
                (offs[j + 1][0] not in starts for j in range(len(ids) - 1)),
                dtype=bool, count=len(ids) - 1)
            n_cross += int((~keep).sum())
            pairs = (arr[:-1] * vocab_size + arr[1:])[keep]
            if pairs.size:
                chunks.append(pairs)
        buf.clear()

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            seen_bytes += len(text.encode("utf-8"))
            n_docs += 1
            buf.append(text)
            if len(buf) >= batch:
                flush()
                if n_docs % 51_200 == 0:
                    print(f"      {n_docs:,}문서  {seen_bytes / 1e6:.0f}MB  "
                          f"{n_tokens:,}토큰")
            if seen_bytes >= max_bytes:
                break
    flush()

    if not chunks:
        raise RuntimeError(f"{path} 에서 인접쌍을 하나도 못 모았다")
    keys = np.concatenate(chunks)
    del chunks
    uniq, counts = np.unique(keys, return_counts=True)
    print(f"      세그먼트 내 인접쌍 {keys.size:,}개, 서로 다른 쌍 {uniq.size:,}개")
    print(f"      경계를 넘어 버린 쌍 {n_cross:,}개 "
          f"({n_cross / max(n_cross + keys.size, 1):.1%}) — 발화 불가능한 것들이다")
    return uniq, counts, n_docs, seen_bytes, n_tokens


def main(argv: list | None = None) -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="T2b 기증 토큰 채굴")
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--max-bytes", type=int, default=300_000_000)
    ap.add_argument("--top", type=int, default=50_000, help="기증 풀 크기")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    vocab = tok.get_vocab()
    id2tok = {i: t for t, i in vocab.items()}
    vocab_size = max(vocab.values()) + 1

    path = ROOT / "data" / "interim" / "docs" / f"{args.split}.jsonl"
    if not path.exists():
        print(f"{path} 가 없다", file=sys.stderr)
        return 1

    config = {
        "repo": args.repo, "revision": args.revision, "split": args.split,
        "max_bytes": args.max_bytes, "top": args.top, "rounds": 1,
        "purpose": "t2b_donor_pool",
    }
    run_id = make_run_id("tok", "donors", args.tag)

    with RunContext(run_id, phase="tok", config=config,
                    skip_env_check=args.skip_env_check) as run:
        print(f"[1/2] 한국어 인접쌍 채굴  {path.name}  최대 "
              f"{args.max_bytes / 1e6:.0f}MB")
        uniq, counts, n_docs, seen_bytes, n_tokens = mine_pairs(
            tok, path, args.max_bytes, vocab_size, args.batch)

        print(f"[2/2] 상위 {args.top:,}개 선별")
        order = np.argsort(counts)[::-1][:args.top]
        out_dir = ROOT / "artifacts" / "vocab_stats" / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"donors_{args.top}.tsv"
        cols = ("rank", "left_id", "right_id", "new_token", "count", "gain")
        n_written = 0
        total_gain = 0
        n_dup = 0
        # 서로 다른 쌍이 같은 문자열로 합쳐질 수 있다 — ("ab","c") 와 ("a","bc")
        # 는 둘 다 "abc" 다. 둘 다 넣으면 뒤엣것이 앞엣것의 슬롯을 덮어써서
        # vocab 이 조용히 줄고, 크기 유지라는 T2b 의 정의가 깨진다.
        emitted: set = set()
        with out.open("w", encoding="utf-8", newline=NL) as fh:
            fh.write(TAB.join(cols) + NL)
            for rank, i in enumerate(order, 1):
                key = int(uniq[i])
                left, right = divmod(key, vocab_size)
                lt, rt = id2tok.get(left), id2tok.get(right)
                if lt is None or rt is None:
                    continue
                new_token = lt + rt
                if new_token in vocab:
                    continue          # 이미 있는 토큰이면 이득이 없다
                if new_token in emitted:
                    n_dup += 1
                    continue
                emitted.add(new_token)
                gain = int(counts[i])  # 쌍 하나를 합치면 출현마다 1토큰씩 준다
                total_gain += gain
                n_written += 1
                fh.write(TAB.join((
                    str(rank), str(left), str(right),
                    json.dumps(new_token, ensure_ascii=False),
                    str(int(counts[i])), str(gain))) + NL)

        print(f"{NL}{out}  {n_written:,}행  (중복 문자열 {n_dup:,}개 제외)")
        print(f"  표본 {n_docs:,}문서 {seen_bytes / 1e6:.0f}MB {n_tokens:,}토큰")
        # 이 값은 **상한** 이다. 인접쌍은 서로 겹친다 — (A,B) 와 (B,C) 가 같은
        # B 를 두고 경쟁하므로 둘 다 발화할 수 없다. 단순 합산은 크게 부풀려진다.
        # 실제 이득은 만든 토크나이저로 Level 1 을 재야 알 수 있다.
        print(f"  이득 상한 {total_gain:,}토큰 "
              f"({total_gain / max(n_tokens, 1):.2%} — 쌍 겹침을 무시한 값이라 "
              f"실제보다 크다. 진짜 이득은 Level 1 로 잰다)")

        run.raw_bytes_seen = seen_bytes
        run.tokens_seen = n_tokens
        run.note = (f"donor pool {n_written} from {uniq.size} distinct pairs, "
                    f"gain_upper_bound {total_gain} "
                    f"({total_gain / max(n_tokens, 1):.4f}, overlap ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
