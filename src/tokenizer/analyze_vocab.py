"""vocab 구성·빈도 분포 분석 (스펙 §18, §19).

pruning 후보를 고르려면 세 가지를 알아야 한다.

1. **코퍼스 사용 빈도** — 스펙 §19 의 "저빈도" 조건.
2. **언어별 빈도** — §19 의 "한국어/영어/code 핵심 token 아님" 조건.
   한국어에서 안 쓰인다고 지웠다가 영어·코드에서 쓰이는 토큰이면
   Candidate Gate 의 regression 조건을 스스로 무너뜨린다. 그래서 세 코퍼스를
   따로 센다.
3. **merge DAG 에서 잎인가** — 스펙에 없는 조건이라 여기서 못 박는다.

   BPE 에서 토큰 ``AB`` 는 merge 규칙 ``(A,B) -> AB`` 로만 만들어진다.
   ``A`` 를 지우면서 규칙을 남기면 토크나이저가 깨지고, 규칙을 지우면 ``AB`` 가
   도달 불가능한 死 슬롯이 된다. 연쇄적으로 번진다. 그래서 다른 merge 의
   부품으로 쓰이지 않는 **잎(leaf) 토큰에서만** 고른다.
   Qwen2.5 는 151,643개 중 111,464개(73.5%)가 잎이라 여유가 충분하다.

산출물은 artifacts/vocab_stats/<tag>/token_stats.tsv 한 장이고,
prune.py 와 substitute.py 가 이것만 읽는다.

    .conda/python.exe -m src.tokenizer.analyze_vocab --split train --tag v1
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

from src.tokenizer.protected import protected_token_ids  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

TAB = chr(9)
NL = chr(10)

# (라벨, 파일 이름) — 없는 파일은 건너뛴다.
CORPORA = (
    ("ko", "{split}.jsonl"),
    ("en", "{split}_control_english.jsonl"),
    ("code", "{split}_control_code.jsonl"),
)


def merge_component_tokens(backend) -> set:
    """다른 merge 규칙의 부품으로 쓰이는 토큰 문자열 집합."""
    model = json.loads(backend.to_str())["model"]
    used: set = set()
    for pair in model["merges"]:
        left, right = pair.split(" ") if isinstance(pair, str) else pair
        used.add(left)
        used.add(right)
    return used


def count_corpus(tokenizer, path: Path, vocab_size: int, batch: int = 512) -> tuple:
    """파일 하나를 토큰화해 ID 빈도를 센다. 문서를 통째로 메모리에 올리지 않는다."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    n_docs = n_chars = n_tokens = 0
    buf: list = []

    def flush() -> None:
        nonlocal n_tokens
        if not buf:
            return
        for ids in tokenizer(buf, add_special_tokens=False)["input_ids"]:
            arr = np.asarray(ids, dtype=np.int64)
            counts[:] += np.bincount(arr, minlength=vocab_size)[:vocab_size]
            n_tokens += arr.size
        buf.clear()

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            n_docs += 1
            n_chars += len(text)
            buf.append(text)
            if len(buf) >= batch:
                flush()
                if n_docs % 51_200 == 0:
                    print(f"      {n_docs:,}문서  {n_tokens:,}토큰")
    flush()
    return counts, n_docs, n_chars, n_tokens


def main(argv: list | None = None) -> int:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description="vocab 빈도 + merge DAG 분석")
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    vocab = tok.get_vocab()
    vocab_size = max(vocab.values()) + 1
    id2tok = {i: t for t, i in vocab.items()}

    used = merge_component_tokens(tok.backend_tokenizer)
    protected = protected_token_ids(tok)
    print(f"{args.repo}  vocab {len(vocab):,} (id 최대 {vocab_size - 1:,})")
    print(f"  merge 부품 {len(used):,}개, 보호 토큰 {len(protected):,}개")

    base = ROOT / "data" / "interim" / "docs"
    config = {
        "repo": args.repo, "revision": args.revision, "split": args.split,
        "batch": args.batch, "purpose": "pruning_candidate_stats",
    }
    run_id = make_run_id("tok", "vocabstats", args.tag)

    with RunContext(run_id, phase="tok", config=config,
                    skip_env_check=args.skip_env_check) as run:
        per_lang: dict = {}
        totals: dict = {}
        for lang, pattern in CORPORA:
            path = base / pattern.format(split=args.split)
            if not path.exists():
                print(f"  [건너뜀] {path.name} 없음")
                continue
            print(f"  [{lang}] {path.name}  {path.stat().st_size / 1e6:.0f} MB")
            counts, n_docs, n_chars, n_tokens = count_corpus(
                tok, path, vocab_size, args.batch)
            per_lang[lang] = counts
            totals[lang] = (n_docs, n_chars, n_tokens)
            nz = int((counts > 0).sum())
            print(f"      {n_docs:,}문서  {n_tokens:,}토큰  사용된 vocab {nz:,}")

        if not per_lang:
            raise RuntimeError(f"{base} 에 {args.split} 코퍼스가 없다")

        total = np.zeros(vocab_size, dtype=np.int64)
        for counts in per_lang.values():
            total += counts

        out_dir = ROOT / "artifacts" / "vocab_stats" / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "token_stats.tsv"
        cols = ("token_id", "token", "count_total", "count_ko", "count_en",
                "count_code", "is_leaf", "is_protected")
        zeros = np.zeros(vocab_size, dtype=np.int64)
        c_ko = per_lang.get("ko", zeros)
        c_en = per_lang.get("en", zeros)
        c_cd = per_lang.get("code", zeros)
        n_leaf = 0
        with out.open("w", encoding="utf-8", newline=NL) as fh:
            fh.write(TAB.join(cols) + NL)
            for tid in range(vocab_size):
                token = id2tok.get(tid)
                if token is None:
                    continue          # config.vocab_size 와의 빈칸
                leaf = int(token not in used)
                n_leaf += leaf
                fh.write(TAB.join((
                    str(tid),
                    json.dumps(token, ensure_ascii=False),   # 탭·개행 안전
                    str(int(total[tid])), str(int(c_ko[tid])),
                    str(int(c_en[tid])), str(int(c_cd[tid])),
                    str(leaf), str(int(tid in protected)),
                )) + NL)

        unused = int((total == 0).sum())
        print(f"{NL}token_stats.tsv  {len(id2tok):,}행 -> {out}")
        print(f"  잎 {n_leaf:,}  보호 {len(protected):,}  전 코퍼스 미사용 {unused:,}")

        run.raw_bytes_seen = sum(t[1] for t in totals.values())
        run.tokens_seen = sum(t[2] for t in totals.values())
        run.note = ("vocab stats: " + " ".join(
            f"{k}={v[0]}docs/{v[2]}tok" for k, v in totals.items())
            + f" leaf={n_leaf} unused={unused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
