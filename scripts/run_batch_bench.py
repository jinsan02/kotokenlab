"""Q6-E — 배치 처리량. 사전 등록은 docs/PLAN.md "Phase 6" E 절.

    .conda/python.exe scripts/run_batch_bench.py \
        --model artifacts/models/cpt_c0_qwen_main_seed42 --name c0_qwen --tag v1

Q6 본 측정은 배치 1 단일 요청이었다. 그래서 "메모리를 30% 아끼므로 요청을
1.4배 받는다" 를 말할 수 없었다 — 동시 처리 수와 처리량은 KV 절감과 다른
속도로 움직인다. 배치가 커지면 메모리가 아니라 **연산** 이 먼저 병목이 된다.

두 가지를 잰다.

    최대 배치   같은 한국어 원문을 몇 개까지 동시에 넣을 수 있는가.
                **OOM 을 기다리면 안 된다.** Windows WDDM 드라이버는 장치
                메모리가 모자라면 시스템 RAM 으로 흘린다 — 죽지 않고
                느려지기만 한다. 실측에서 배치 26 이 peak 17,495MB 로 장치
                총량 16,303MB 를 넘겼는데도 OOM 이 안 났고, 처리량만
                3.85 -> 2.04 seq/s 로 무너졌다. 그대로 두면 "최대 배치 48"
                이라는 무의미한 답이 나온다.

                그래서 **peak_alloc 이 장치 총량을 넘는 순간** 을 경계로 쓴다.
                그것이 원래 OOM 으로 표시하려던 "안 들어간다" 지점이다.
    처리량      배치마다 초당 몇 개를 prefill 하는가.

**raw_prompt 모드만 쓴다.** 같은 원문을 넣어야 사용자가 겪는 차이가 된다.
equal_tokens 로 배치를 재면 조건 간 차이가 정의상 0 이다.

prefill 만 잰다. 생성 경로는 Q6 에서 잡음이 5.7~17.5% 로 못 읽는 것이
확인됐고, 배치를 얹으면 더 나빠진다.

OOM 을 잡은 뒤에는 캐시를 비우고 다음으로 넘어간다. 비우지 않으면 파편화가
남아 이후 측정이 실제보다 작은 배치에서 죽는다.

**최대 배치의 절대값은 데스크톱이 쓰는 VRAM 에 딸려 있다.** 브라우저가 1GB 를
잡고 있으면 그만큼 일찍 죽는다. 그래서 두 조건을 연달아 재고 **절대값보다
비(比)를 주로 보고한다** — 같은 조건에서 잰 비는 데스크톱 사용량이 상수로
빠지므로 훨씬 튼튼하다. 시작 시점의 여유 VRAM 도 config 에 남긴다.
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
# 2의 거듭제곱으로 잡으면 안 된다. Q6 실측으로 배치당 증가분을 뽑으면 C0 는
# 약 659MB/시퀀스라 배치 20~22 에서, T2b 는 약 462MB 라 28~30 에서 한계다 —
# 둘 다 16 을 통과하고 32 에서 죽어 **1.4배 차이가 같은 칸에 묻힌다.**
# 그 차이가 이 실험의 핵심 숫자이므로 한계 근처를 촘촘하게 훑는다.
BATCHES = (1, 2, 4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 48)


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

    free_mb = (torch.cuda.mem_get_info()[0] / 1024 / 1024
               if torch.cuda.is_available() else 0)
    config = {
        "model": args.model, "revision": args.revision,
        "free_vram_mb_at_start": round(free_mb, 1),
        "raw_chars": args.raw_chars, "input_tokens": n_tok,
        "batches": list(BATCHES), "warmup": args.warmup, "runs": args.runs,
        "dtype": "bfloat16", "attn_prefill": "efficient+cudnn",
        "purpose": "q6e_batch_throughput",
    }
    run_id = make_run_id("sys", name, "batch" + args.tag)

    with RunContext(run_id, phase="sys", config=config,
                    skip_env_check=args.skip_env_check) as run:
        print(f"{name}  vocab {cfg.vocab_size:,}  원문 {args.raw_chars:,}자 "
              f"-> {n_tok:,}토큰  KV {memory.kv_cache_mb(cfg, n_tok):.1f}MB/시퀀스"
              f"  시작 여유 {free_mb:,.0f}MB")

        total_mb = (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    if torch.cuda.is_available() else 0)
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
            spilled = peak["peak_alloc_mb"] > total_mb
            if not spilled:
                max_ok = b
            print(f"  배치 {b:3d}  prefill {pf['mean']:8.1f}ms (p95 {pf['p95']:8.1f})  "
                  f"처리량 {thr:6.2f} seq/s  KV {kv:7.1f}MB  "
                  f"peak {peak['peak_alloc_mb']:7.0f}MB"
                  + ("  <- 장치 초과. 여기서 멈춘다" if spilled else ""))

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
            if spilled:
                break

        print(f"  장치 안에 들어가는 최대 배치 {max_ok}  (총 {total_mb:,.0f}MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
