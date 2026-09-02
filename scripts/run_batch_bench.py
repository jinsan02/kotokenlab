"""Q6-E — 배치 처리량. 사전 등록은 docs/PLAN.md "Phase 6" E 절.

    .conda/python.exe scripts/run_batch_bench.py \
        --model artifacts/models/cpt_c0_qwen_main_seed42 --name c0_qwen --tag v1

Q6 본 측정은 배치 1 단일 요청이었다. 그래서 "메모리를 30% 아끼므로 요청을
1.4배 받는다" 를 말할 수 없었다 — 동시 처리 수와 처리량은 KV 절감과 다른
속도로 움직인다. 배치가 커지면 메모리가 아니라 **연산** 이 먼저 병목이 된다.

두 가지를 잰다.

    최대 배치   같은 한국어 원문을 몇 개까지 동시에 넣을 수 있는가.
                OOM 이 날 때까지 올린다. 이것이 "요청을 몇 개 받는가" 다.
    처리량      배치마다 초당 몇 개를 prefill 하는가.

**raw_prompt 모드만 쓴다.** 같은 원문을 넣어야 사용자가 겪는 차이가 된다.
equal_tokens 로 배치를 재면 조건 간 차이가 정의상 0 이다.

prefill 만 잰다. 생성 경로는 Q6 에서 잡음이 5.7~17.5% 로 못 읽는 것이
확인됐고, 배치를 얹으면 더 나빠진다.

OOM 을 잡은 뒤에는 캐시를 비우고 다음으로 넘어간다. 비우지 않으면 파편화가
남아 이후 측정이 실제보다 작은 배치에서 죽는다.
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

from src.evaluation import latency, memory  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

# 배치를 재는 원문 길이. Q6 에서 이득이 가장 컸던 40,000자 근처를 쓴다.
RAW_CHARS = 20_000
BATCHES = (1, 2, 4, 8, 16, 32, 64)


def load_prompt(max_chars: int) -> str:
    path = ROOT / "data" / "interim" / "docs" / "dev.jsonl"
    buf: list = []
    total = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            buf.append(text)
            total += len(text)
            if total >= max_chars:
                break
    return "".join(buf)[:max_chars]


def main(argv: list | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="배치 처리량")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--raw-chars", type=int, default=RAW_CHARS)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    name = args.name or Path(args.model).name
    kw = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", **kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    model.config.use_cache = True
    cfg = model.config

    text = load_prompt(args.raw_chars)
    one = tokenizer(text, add_special_tokens=False,
                    return_tensors="pt")["input_ids"]
    n_tok = one.shape[1]

    config = {
        "model": args.model, "revision": args.revision,
        "raw_chars": args.raw_chars, "input_tokens": n_tok,
        "batches": list(BATCHES), "warmup": args.warmup, "runs": args.runs,
        "dtype": "bfloat16", "attn_prefill": "efficient+cudnn",
        "purpose": "q6e_batch_throughput",
    }
    run_id = make_run_id("sys", name, "batch" + args.tag)

    with RunContext(run_id, phase="sys", config=config,
                    skip_env_check=args.skip_env_check) as run:
        print(f"{name}  vocab {cfg.vocab_size:,}  원문 {args.raw_chars:,}자 "
              f"-> {n_tok:,}토큰  KV {memory.kv_cache_mb(cfg, n_tok):.1f}MB/시퀀스")

        max_ok = 0
        for b in BATCHES:
            ids = one.repeat(b, 1).to(device)
            memory.reset()
            try:
                pf = latency.prefill(model, ids, args.warmup, args.runs)
            except torch.cuda.OutOfMemoryError:
                print(f"  배치 {b:3d}  OOM — 여기서 멈춘다")
                del ids
                torch.cuda.empty_cache()
                break
            peak = memory.peaks()
            thr = b / (pf["mean"] / 1000)
            kv = memory.kv_cache_mb(cfg, n_tok) * b
            max_ok = b
            print(f"  배치 {b:3d}  prefill {pf['mean']:8.1f}ms (p95 {pf['p95']:8.1f})  "
                  f"처리량 {thr:6.2f} seq/s  KV {kv:7.1f}MB  peak {peak['peak_alloc_mb']:7.0f}MB")

            run.log("system_bench", model=args.model, tokenizer_version=name,
                    mode="raw_prompt", raw_chars=args.raw_chars,
                    raw_bytes=len(text.encode("utf-8")),
                    input_tokens=n_tok, gen_tokens=0,
                    n_warmup=args.warmup, n_runs=args.runs,
                    tokenize_ms_mean=None,
                    prefill_ms_mean=round(pf["mean"], 4),
                    prefill_ms_p95=round(pf["p95"], 4),
                    ttft_ms_mean=None, ttft_ms_p95=None,
                    decode_tok_s_mean=None,
                    total_ms_mean=None, total_ms_std=round(pf["std"], 4),
                    kv_cache_mb_est=round(kv, 4),
                    peak_alloc_mb=round(peak["peak_alloc_mb"], 1),
                    peak_reserved_mb=round(peak["peak_reserved_mb"], 1),
                    batch_size=b,
                    throughput_docs_s=round(thr, 4))
            del ids
            torch.cuda.empty_cache()

        print(f"  최대 배치 {max_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
