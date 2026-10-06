"""문서별 NLL — 문서 단위 paired bootstrap 의 입력 (스펙 §36 BPB · amendment §4 · 5주차).

    .conda/python.exe -m src.evaluation.doc_nll --model artifacts/models/<run_id> \
        --source-run <run_id> --tag dev2mb

저장된 CPT 체크포인트를 CPT 의 dev 평가와 **같은 함수 · 같은 인자** 로 다시 평가해
(`bpb.evaluate`, dev 원문 언어별 2MB, seq_len 2048), 문서마다 NLL · 채점 바이트를
`experiments/runs/<run_id>/doc_nll.tsv` 에 남긴다.

**원장 재현을 스스로 확인한다.** `--source-run` 을 주면 그 학습 run 의 원장 최종 dev
BPB 와 문서별 합으로 다시 낸 BPB 를 대조해 차이를 원장 note 에 적는다. 같은 문서 ·
같은 계산이므로 같아야 한다 — 다르면 문서별 값으로 낸 bootstrap 을 믿을 수 없다.

판정하지 않는다. 비교는 `tools/doc_bootstrap.py` 가 한다.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.utils.tracking import RunContext, make_run_id  # noqa: E402

DEV = (("ko", "dev.jsonl"),
       ("en", "dev_control_english.jsonl"),
       ("code", "dev_control_code.jsonl"))
COLUMNS = ("lang", "doc", "nll", "bytes", "tokens", "raw_bytes")


def ledger_final_bpb(run_id: str) -> dict:
    """학습 run 의 원장 최종 dev BPB (언어별). 없으면 빈 dict."""
    path = ROOT / "experiments" / "lm_metrics.tsv"
    out: dict = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["run_id"] == run_id and r["checkpoint"] == "final" and r["split"] == "dev":
                out[r["domain"]] = float(r["bpb"])
    return out


def bpb_from_docs(rows: list) -> float:
    nll = sum(r["nll"] for r in rows)
    b = sum(r["bytes"] for r in rows)
    return nll / (math.log(2) * b) if b else float("nan")


def main(argv: list | None = None) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluation.bpb import evaluate, token_byte_length

    ap = argparse.ArgumentParser(description="문서별 NLL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--source-run", default=None, help="이 체크포인트를 만든 학습 run_id")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-bytes", type=int, default=2_000_000,
                    help="언어별 dev 원문 예산 (CPT 의 --eval-budget 과 같게)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tag", default="dev2mb")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    name = args.name or Path(args.model).name
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    alphabet = set(ByteLevel.alphabet())
    blen: dict = {}

    def byte_len_fn(ids: list) -> int:
        total = 0
        for i in ids:
            v = blen.get(i)
            if v is None:
                v = token_byte_length(tokenizer.convert_ids_to_tokens(int(i)), alphabet)
                blen[i] = v
            total += v
        return total

    config = {"model": args.model, "split": "dev", "max_bytes": args.max_bytes,
              "seq_len": args.seq_len, "source_run": args.source_run,
              "purpose": "doc_nll"}
    run_id = make_run_id("eval", "docnll", name, args.tag)
    want = ledger_final_bpb(args.source_run) if args.source_run else {}

    with RunContext(run_id, phase="eval", config=config,
                    skip_env_check=args.skip_env_check) as run:
        dev_dir = ROOT / "data" / "interim" / "docs"
        all_rows: list = []
        notes = []
        with sdpa_kernel(backends):
            for lang, fname in DEV:
                docs: list = []
                m = evaluate(model, tokenizer, dev_dir / fname, args.max_bytes,
                             args.seq_len, device, byte_len_fn, docs_out=docs)
                re_bpb = bpb_from_docs(docs)
                for d in docs:
                    all_rows.append({"lang": lang, **d})
                run.log("lm_metrics", checkpoint="docnll", tokens_seen=0, raw_bytes_seen=0,
                        split="dev", domain=lang, n_bytes=m["n_bytes"],
                        total_nll=round(m["total_nll"], 4), bpb=round(m["bpb"], 6),
                        bpc=round(m["bpc"], 6), token_ppl=round(m["token_ppl"], 4))
                line = f"{lang} {len(docs)}문서 BPB {m['bpb']:.6f} (문서합 {re_bpb:.6f})"
                if lang in want:
                    diff = round(m["bpb"], 6) - want[lang]
                    line += f" · 원장 {want[lang]:.6f} 차이 {diff:+.6f}"
                    notes.append(f"{lang}_vs_ledger={diff:+.6f}")
                print("  " + line)

        out = run.dir / "doc_nll.tsv"
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            for r in all_rows:
                fh.write("\t".join([r["lang"], str(r["doc"]), repr(r["nll"]), str(r["bytes"]),
                                    str(r["tokens"]), str(r["raw_bytes"])]) + "\n")
        run.note = f"doc_nll {len(all_rows)}행 source={args.source_run} " + " ".join(notes)
        print(f"  썼다 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
