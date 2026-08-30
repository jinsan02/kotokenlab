"""T2a(축소) / T2b(치환) 토크나이저 생성 (스펙 §12, §103).

두 조건은 **같은 pruning 목록** 을 쓴다. 목록이 다르면 "축소 vs 치환" 이 아니라
서로 다른 두 토크나이저를 비교하는 것이 되어 신규성 주장이 성립하지 않는다.

    T2a  제거만          vocab 151,665 - N   embedding resize 필요
    T2b  제거 + 치환      vocab 151,665 유지   embedding 형상 불변

구현 메모
    merge 규칙은 ID 가 아니라 **토큰 문자열 쌍** 을 참조한다. 덕분에 T2a 에서
    ID 를 다시 매길 때 merges 는 손대지 않아도 된다. 고쳐야 하는 것은 vocab 의
    번호와 added_tokens 의 id 뿐이다.

    제거할 때는 그 토큰을 **결과로 만드는** merge 규칙도 함께 지운다. 남겨두면
    vocab 에 없는 토큰을 만들려 드는 규칙이 남는다. 부품으로 쓰이는 규칙은
    지울 필요가 없다 — prune.py 가 잎 토큰만 고르므로 그런 규칙이 없다.

    새 merge 는 목록 끝(가장 늦은 rank)에 붙인다. 더 이른 규칙이 부품을 먼저
    가져가면 새 토큰은 영영 안 나온다. 그래서 만들고 나서 실제 인코딩으로
    활성 여부를 확인하고 활성률을 보고한다.

    .conda/python.exe -m src.tokenizer.substitute --tag v1 --mode t2b -n 10000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.tokenizer.protected import (  # noqa: E402
    assert_byte_roundtrip, assert_protected_survive, protected_token_ids)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

TAB = chr(9)
NL = chr(10)


def _read_tsv(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip(NL).split(TAB)
        for line in fh:
            rows.append(dict(zip(header, line.rstrip(NL).split(TAB))))
    return rows


def _merge_parts(entry) -> tuple:
    return tuple(entry.split(" ")) if isinstance(entry, str) else tuple(entry)


def build(base_json: dict, pruned: list, donors: list, mode: str) -> tuple:
    """수정된 tokenizer.json 과 ID 매핑을 만든다.

    pruned  : [(token_id, token_str), ...]  두 조건이 공유하는 제거 목록
    donors  : [(left_token, right_token, new_token), ...]  T2b 만 사용
    """
    tj = json.loads(json.dumps(base_json))       # 깊은 복사
    model = tj["model"]
    vocab: dict = model["vocab"]
    n_before = len(vocab)
    prune_tokens = {t for _, t in pruned}

    missing = prune_tokens - set(vocab)
    if missing:
        raise ValueError(f"vocab 에 없는 토큰을 지우려 한다: {sorted(missing)[:5]}")

    # 지울 토큰을 결과로 만드는 merge 규칙을 함께 제거한다
    kept_merges = []
    dropped = 0
    for entry in model["merges"]:
        left, right = _merge_parts(entry)
        if left + right in prune_tokens:
            dropped += 1
            continue
        kept_merges.append(entry)

    for token in prune_tokens:
        vocab.pop(token, None)

    id_map: dict = {}
    if mode == "t2b":
        slots = sorted(tid for tid, _ in pruned)
        if len(donors) < len(slots):
            raise ValueError(
                f"기증자가 {len(donors):,}개뿐인데 빈 슬롯은 {len(slots):,}개다")
        as_str = isinstance(model["merges"][0], str) if model["merges"] else True
        for tid, (left, right, new_token) in zip(slots, donors):
            vocab[new_token] = tid
            kept_merges.append(left + " " + right if as_str else [left, right])
            id_map[str(tid)] = {"token": new_token, "left": left, "right": right}
        n_vocab = max(vocab.values()) + 1
        # 서로 다른 쌍이 같은 문자열로 합쳐지면 (("ab","c") 와 ("a","bc") 가
        # 둘 다 "abc") 뒤엣것이 앞엣것의 슬롯을 덮어써서 vocab 이 조용히 준다.
        # 크기 유지가 T2b 의 정의이므로 여기서 멈춘다.
        if len(vocab) != n_before or n_vocab != n_before:
            raise AssertionError(
                f"T2b 는 크기를 유지해야 한다: {n_before:,} -> vocab {len(vocab):,}, "
                f"최대 id+1 {n_vocab:,}. 기증 토큰에 중복이 있는지 확인하라")
    else:
        # T2a — 번호를 다시 매긴다. merges 는 문자열을 참조하므로 손대지 않는다.
        survivors = sorted(vocab.items(), key=lambda kv: kv[1])
        vocab.clear()
        for new_id, (token, old_id) in enumerate(survivors):
            vocab[token] = new_id
            if new_id != old_id:
                id_map[str(old_id)] = new_id
        for entry in tj.get("added_tokens", []):
            if entry["content"] in vocab:
                entry["id"] = vocab[entry["content"]]
        n_vocab = len(vocab)

    model["merges"] = kept_merges
    return tj, id_map, {"merges_dropped": dropped, "vocab_size": n_vocab,
                        "merges": len(kept_merges)}


def activation_rate(tok, donors: list, sample_texts: list) -> float:
    """새 토큰이 실제 인코딩에서 나오는 비율. rank 경쟁에 져서 안 나올 수 있다."""
    wanted = {t for _, _, t in donors}
    if not wanted:
        return 1.0
    vocab = tok.get_vocab()
    ids = {vocab[t] for t in wanted if t in vocab}
    seen: set = set()
    for text in sample_texts:
        seen.update(i for i in tok.encode(text, add_special_tokens=False) if i in ids)
    return len(seen) / max(len(ids), 1)


def main(argv: list | None = None) -> int:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    ap = argparse.ArgumentParser(description="T2a / T2b 토크나이저 생성")
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--mode", choices=("t2a", "t2b"), required=True)
    ap.add_argument("-n", "--num", type=int, required=True)
    ap.add_argument("--donor-pool", type=int, default=50_000)
    ap.add_argument("--sample-docs", type=int, default=300)
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    stats_dir = ROOT / "artifacts" / "vocab_stats" / args.tag
    prune_path = stats_dir / ("prune_" + str(args.num) + ".tsv")
    if not prune_path.exists():
        print(f"{prune_path} 가 없다. -m src.tokenizer.prune -n {args.num} 먼저.",
              file=sys.stderr)
        return 1
    pruned = [(int(r["token_id"]), json.loads(r["token"]))
              for r in _read_tsv(prune_path)]

    base_tok = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    donors: list = []
    if args.mode == "t2b":
        donor_path = stats_dir / ("donors_" + str(args.donor_pool) + ".tsv")
        if not donor_path.exists():
            print(f"{donor_path} 가 없다. -m src.tokenizer.train 먼저.",
                  file=sys.stderr)
            return 1
        id2tok = {i: t for t, i in base_tok.get_vocab().items()}
        for r in _read_tsv(donor_path):
            new_token = json.loads(r["new_token"])
            left, right = id2tok[int(r["left_id"])], id2tok[int(r["right_id"])]
            donors.append((left, right, new_token))
            if len(donors) >= len(pruned):
                break

    before_protected = protected_token_ids(base_tok)
    base_json = json.loads(base_tok.backend_tokenizer.to_str())

    version = "ko" + args.mode + "_" + args.tag + "_n" + str(args.num)
    config = {"repo": args.repo, "revision": args.revision, "mode": args.mode,
              "num_pruned": args.num, "tag": args.tag,
              "donor_pool": args.donor_pool if args.mode == "t2b" else None}
    run_id = make_run_id("tok", args.mode, args.tag, "n" + str(args.num))

    with RunContext(run_id, phase="tok", config=config,
                    skip_env_check=args.skip_env_check) as run:
        print(f"[1/3] {args.mode} 생성  제거 {len(pruned):,}  치환 {len(donors):,}")
        tj, id_map, info = build(base_json, pruned, donors, args.mode)
        print(f"      vocab {len(base_json['model']['vocab']):,} -> "
              f"{info['vocab_size']:,}")
        print(f"      merges {len(base_json['model']['merges']):,} -> "
              f"{info['merges']:,}  (규칙 {info['merges_dropped']:,}개 제거)")

        print("[2/3] 무결성 검사")
        backend = Tokenizer.from_str(json.dumps(tj, ensure_ascii=False))
        new_tok = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token=base_tok.bos_token, eos_token=base_tok.eos_token,
            pad_token=base_tok.pad_token, unk_token=base_tok.unk_token)

        # merge 일관성 — 모든 규칙의 부품과 결과가 vocab 에 있어야 한다
        v = set(tj["model"]["vocab"])
        bad = []
        for entry in tj["model"]["merges"]:
            parts = _merge_parts(entry)
            if not (set(parts) <= v) or "".join(parts) not in v:
                bad.append(entry)
        if bad:
            raise AssertionError(f"끊어진 merge 규칙 {len(bad):,}개: {bad[:3]}")
        print(f"      merge 일관성 OK  ({len(tj['model']['merges']):,}개)")

        assert_byte_roundtrip(new_tok)
        print("      바이트 왕복 OK")
        if args.mode == "t2b":
            assert_protected_survive(new_tok, before_protected)
            print("      보호 토큰 생존 OK")

        rate = 1.0
        if donors:
            docs_path = ROOT / "data" / "interim" / "docs" / "dev.jsonl"
            texts = []
            with docs_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    texts.append(json.loads(line)["text"])
                    if len(texts) >= args.sample_docs:
                        break
            rate = activation_rate(new_tok, donors, texts)
            print(f"      기증 토큰 활성률 {rate:.1%}  (dev {len(texts)}문서 기준)")

        print("[3/3] 저장")
        out_dir = ROOT / "artifacts" / "tokenizers" / version
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        new_tok.save_pretrained(str(out_dir))
        (out_dir / "id_map.json").write_text(
            json.dumps({"mode": args.mode, "base": args.repo,
                        "revision": args.revision, "num_pruned": args.num,
                        "map": id_map}, ensure_ascii=False, indent=2),
            encoding="utf-8", newline=NL)
        tok_sha = sha256_file(out_dir / "tokenizer.json")
        print(f"      {out_dir}")
        print(f"      tokenizer_sha256 = {tok_sha}")

        run.extra["tokenizer_version"] = version
        run.extra["vocab_size"] = info["vocab_size"]
        run.extra["tokenizer_sha256"] = tok_sha
        run.note = (f"{args.mode} n={args.num} vocab={info['vocab_size']} "
                    f"merges={info['merges']} activation={rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
