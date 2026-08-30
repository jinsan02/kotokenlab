"""BPB — bits per byte (스펙 §37~39).

    BPB = TotalNLL / (ln2 x RawBytes)

**토크나이저가 다른 모델을 비교할 수 있는 유일한 지표다.** token PPL 로 비교하면
토큰을 크게 자르는 쪽이 자동으로 유리해져서, 압축을 개선했다는 사실이 성능
개선으로 둔갑한다. 분모를 원문 바이트로 두면 그 우회로가 막힌다
(docs/RULES.md 3번).

Pre-CPT 로 쓰는 법
    수술 직후 학습 없이 재면 **초기화 방법의 효과만** 보인다. 스펙 §20 이
    요구하는 효과 분리가 여기서 이뤄진다 — 이 시점의 BPB 차이는 토크나이저
    효과도 정렬 효과도 CPT 효과도 아니고 오직 초기화 차이다.

세는 방법
    문서를 토큰화해 seq_len 조각으로 자르고, 각 조각에서 **두 번째 토큰부터**
    NLL 을 더한다 (첫 토큰은 예측 대상이 아니다). 바이트도 같은 범위만 센다.
    조각 경계에서 문맥이 끊기는 손해는 모든 조건에 똑같이 적용되므로 비교를
    왜곡하지 않는다.

    바이트는 토크나이저와 무관하게 원문 UTF-8 길이다. 그래서 조건마다 토큰 수는
    달라도 분모는 같다.

    .conda/python.exe -m src.evaluation.bpb --model artifacts/models/... --max-bytes 5000000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.utils.tracking import RunContext, make_run_id  # noqa: E402

CORPORA = (("ko", "{split}.jsonl"),
           ("en", "{split}_control_english.jsonl"),
           ("code", "{split}_control_code.jsonl"))


def token_byte_length(token: str, alphabet: set) -> int:
    """토큰 하나가 나타내는 **원문 바이트 수**.

    ByteLevel BPE 의 토큰 문자열은 바이트를 유니코드 문자로 옮겨 적은 것이다.
    "다" 는 3바이트인데 vocab 에는 'ëĭ¤' 로 들어 있고, 이 3글자가 곧 3바이트다.
    len(token.encode("utf-8")) 를 쓰면 6 이 나와 한국어 분모가 두 배로 부풀고
    BPB 가 계통적으로 틀린다 — 언어마다 부풀림 비율이 달라서 비교도 깨진다.

    ByteLevel 알파벳 안의 문자만으로 이뤄졌으면 글자 수가 곧 바이트 수다.
    그렇지 않으면 added token (<|endoftext|> 등) 이므로 실제 UTF-8 길이를 쓴다.
    """
    if token and all(ch in alphabet for ch in token):
        return len(token)
    return len(token.encode("utf-8"))


def evaluate(model, tokenizer, path: Path, max_bytes: int, seq_len: int,
             device: str, byte_len_fn) -> dict:
    import torch

    total_nll = 0.0
    total_bytes = 0
    total_chars = 0
    total_tokens = 0
    seen_bytes = 0
    n_docs = 0

    def score(ids: list) -> None:
        nonlocal total_nll, total_bytes, total_chars, total_tokens
        for i in range(0, len(ids), seq_len):
            chunk = ids[i:i + seq_len]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk], device=device)
            with torch.no_grad():
                logits = model(x).logits.float()
            lp = torch.log_softmax(logits[0, :-1], dim=-1)
            tgt = x[0, 1:]
            nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            total_nll += float(nll.sum())
            total_tokens += len(chunk) - 1
            total_bytes += byte_len_fn(chunk[1:])
            # bpc 는 부차 지표다. 토크나이저가 다르면 BPB 로만 비교한다.
            total_chars += len(tokenizer.decode(chunk[1:]))

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            seen_bytes += len(text.encode("utf-8"))
            n_docs += 1
            score(tokenizer(text, add_special_tokens=False)["input_ids"])
            if seen_bytes >= max_bytes:
                break

    ln2 = math.log(2)
    bpb = total_nll / (ln2 * total_bytes) if total_bytes else float("nan")
    bpc = total_nll / (ln2 * total_chars) if total_chars else float("nan")
    return {"n_docs": n_docs, "n_bytes": total_bytes, "n_chars": total_chars,
            "n_tokens": total_tokens, "total_nll": total_nll, "bpb": bpb,
            "bpc": bpc,
            "token_ppl": math.exp(total_nll / total_tokens) if total_tokens else float("nan")}


def main(argv: list | None = None) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="Pre-CPT / CPT BPB")
    ap.add_argument("--model", required=True, help="경로 또는 repo id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None, help="원장에 남길 이름")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-bytes", type=int, default=5_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tag", default="precpt")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    name = args.name or Path(args.model).name
    kw = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", **kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # 강제하지 않으면 8,192 토큰에서 메모리 7.1배, 시간 10.3배가 된다
    # (docs/RULES.md 4번, reports/tables/resource_probe.md)
    backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

    blen = {}

    from tokenizers.pre_tokenizers import ByteLevel
    alphabet = set(ByteLevel.alphabet())

    def byte_len_fn(ids: list) -> int:
        total = 0
        for i in ids:
            v = blen.get(i)
            if v is None:
                v = token_byte_length(tokenizer.convert_ids_to_tokens(int(i)),
                                      alphabet)
                blen[i] = v
            total += v
        return total

    config = {"model": args.model, "revision": args.revision, "split": args.split,
              "max_bytes": args.max_bytes, "seq_len": args.seq_len,
              "purpose": "pre_cpt_bpb"}
    run_id = make_run_id("eval", "bpb", name, args.tag)

    base = ROOT / "data" / "interim" / "docs"
    with RunContext(run_id, phase="eval", config=config,
                    skip_env_check=args.skip_env_check) as run:
        print(f"{name}  vocab {len(tokenizer):,}  "
              f"params {sum(p.numel() for p in model.parameters()):,}")
        agg_nll = agg_bytes = agg_tokens = 0
        with sdpa_kernel(backends):
            for lang, pattern in CORPORA:
                path = base / pattern.format(split=args.split)
                if not path.exists():
                    continue
                m = evaluate(model, tokenizer, path, args.max_bytes,
                             args.seq_len, device, byte_len_fn)
                agg_nll += m["total_nll"]
                agg_bytes += m["n_bytes"]
                agg_tokens += m["n_tokens"]
                print(f"  {lang:<5} {m['n_docs']:>6,}문서 {m['n_bytes'] / 1e6:>6.2f}MB "
                      f"{m['n_tokens']:>9,}토큰   BPB {m['bpb']:.4f}  "
                      f"tokPPL {m['token_ppl']:.2f}")
                # 어느 모델인지는 run_id (eval_bpb_<name>_<tag>) 에 들어 있다.
                run.log("lm_metrics", checkpoint="pre_cpt", tokens_seen=0,
                        raw_bytes_seen=0, split=args.split, domain=lang,
                        n_bytes=m["n_bytes"], total_nll=round(m["total_nll"], 4),
                        bpb=round(m["bpb"], 6), bpc=round(m["bpc"], 6),
                        token_ppl=round(m["token_ppl"], 4))

        total_bpb = agg_nll / (math.log(2) * agg_bytes)
        print(f"  {'전체':<5} {'':>6} {agg_bytes / 1e6:>13.2f}MB "
              f"{agg_tokens:>9,}토큰   BPB {total_bpb:.4f}")
        run.raw_bytes_seen = agg_bytes
        run.tokens_seen = agg_tokens
        run.extra["tokenizer_version"] = name
        run.note = f"pre-CPT BPB {total_bpb:.4f} on {args.split} {agg_bytes}B"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
