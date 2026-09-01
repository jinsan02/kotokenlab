"""Q6 — 시스템 벤치마크 (스펙 §49, §50). 사전 등록은 docs/PLAN.md.

    .conda/python.exe scripts/run_system_bench.py --model artifacts/models/cpt_c0_qwen_main_seed42 \
        --name c0_qwen --tag v1

두 모드로 잰다.

    raw_prompt     같은 **한국어 원문** 을 넣는다. T2b 는 그 글을 30% 적은
                   토큰으로 처리하므로 시퀀스가 짧아진다. 실제 사용자가 겪는 차이다.
    equal_tokens   같은 **토큰 수** 를 넣는다. 개선이 시퀀스 길이에서 오는지
                   vocab 크기에서 오는지를 가른다.

`equal_tokens` 의 T2b vs C0 는 **널 대조군** 이다. vocab 도 아키텍처도 같으니
차이가 나오면 안 된다. 나오면 계측기가 고장난 것이므로 결과를 믿기 전에 이걸 본다.

프롬프트는 dev 에서 만든다. final_test 는 열지 않는다 (docs/RULES.md 2번).
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

# 한국어 원문 길이(자). 40,000자는 약 27,300 토큰으로 실측 안전 상한 근처다.
RAW_CHARS = (5_000, 10_000, 20_000, 40_000)

# equal_tokens 모드에서 모든 조건에 똑같이 넣는 토큰 수.
EQUAL_TOKENS = (2_048, 4_096, 8_192, 16_384)


def load_prompt(max_chars: int) -> str:
    """dev 문서를 이어 붙여 목표 길이의 한국어 원문을 만든다."""
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

    ap = argparse.ArgumentParser(description="Q6 시스템 벤치마크")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--prefill-warmup", type=int, default=20)
    ap.add_argument("--prefill-runs", type=int, default=100)
    # 생성 경로는 한 회가 수 초다. 100회면 몇 시간이 되므로 줄이고, 줄인 사실은
    # n_warmup/n_runs 컬럼에 남는다 (docs/PLAN.md "측정 규약").
    ap.add_argument("--gen-warmup", type=int, default=5)
    ap.add_argument("--gen-runs", type=int, default=20)
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
    # 학습 때 껐던 값이 체크포인트에 실려 올 수 있다. 생성 경로를 재려면 켜야 한다.
    model.config.use_cache = True

    cfg = model.config
    config = {
        "model": args.model, "revision": args.revision,
        "gen_tokens": args.gen_tokens,
        "prefill_warmup": args.prefill_warmup, "prefill_runs": args.prefill_runs,
        "gen_warmup": args.gen_warmup, "gen_runs": args.gen_runs,
        "raw_chars": list(RAW_CHARS), "equal_tokens": list(EQUAL_TOKENS),
        "dtype": "bfloat16",
        "attn_prefill": "efficient+cudnn",
        "attn_decode": "efficient+cudnn+math",
        "purpose": "q6_system_bench",
    }
    run_id = make_run_id("sys", name, args.tag)

    with RunContext(run_id, phase="sys", config=config,
                    skip_env_check=args.skip_env_check) as run:
        w_mb = memory.weight_mb(model)
        print(f"{name}  vocab {cfg.vocab_size:,}  가중치 {w_mb:,.0f}MB  "
              f"KV {memory.kv_bytes_per_token(cfg):,}B/토큰")

        def measure(mode: str, ids, raw_chars, raw_bytes, text):
            n_tok = ids.shape[1]
            memory.reset()
            # 백엔드 정책은 latency 가 경로마다 직접 정한다. 여기서 감싸면
            # decode 가 융합 커널 안에 갇혀 "No available kernel" 로 죽는다.
            pf = latency.prefill(model, ids,
                                 args.prefill_warmup, args.prefill_runs)
            gen = latency.generate(model, ids, args.gen_tokens,
                                   args.gen_warmup, args.gen_runs)
            peak = memory.peaks()
            tok_ms = (latency.tokenize_ms(tokenizer, text)["mean"]
                      if text is not None else None)
            kv = memory.kv_cache_mb(cfg, n_tok)

            print(f"  {mode:12} {n_tok:6,}토큰  prefill {pf['mean']:7.1f}ms "
                  f"(p95 {pf['p95']:7.1f})  TTFT {gen['ttft']['mean']:7.1f}ms  "
                  f"decode {gen['decode_tok_s_mean']:5.1f}tok/s  "
                  f"KV {kv:6.1f}MB  peak {peak['peak_alloc_mb']:6.0f}MB")

            run.log("system_bench", model=args.model,
                    tokenizer_version=name, mode=mode,
                    raw_chars=raw_chars, raw_bytes=raw_bytes,
                    input_tokens=n_tok, gen_tokens=args.gen_tokens,
                    n_warmup=args.prefill_warmup, n_runs=args.prefill_runs,
                    tokenize_ms_mean=None if tok_ms is None else round(tok_ms, 4),
                    prefill_ms_mean=round(pf["mean"], 4),
                    prefill_ms_p95=round(pf["p95"], 4),
                    ttft_ms_mean=round(gen["ttft"]["mean"], 4),
                    ttft_ms_p95=round(gen["ttft"]["p95"], 4),
                    decode_tok_s_mean=round(gen["decode_tok_s_mean"], 4),
                    total_ms_mean=round(gen["total"]["mean"], 4),
                    total_ms_std=round(gen["total"]["std"], 4),
                    kv_cache_mb_est=round(kv, 4),
                    peak_alloc_mb=round(peak["peak_alloc_mb"], 1),
                    peak_reserved_mb=round(peak["peak_reserved_mb"], 1))

        # ── raw_prompt: 같은 원문. 토큰 수는 토크나이저가 정한다 ────────────
        for n_chars in RAW_CHARS:
            text = load_prompt(n_chars)
            ids = tokenizer(text, add_special_tokens=False,
                            return_tensors="pt")["input_ids"].to(device)
            measure("raw_prompt", ids, len(text), len(text.encode("utf-8")), text)

        # ── equal_tokens: 같은 토큰 수. 원문은 조건마다 다르다 ─────────────
        # 어휘 분포가 지연에 섞이지 않도록 실제 한국어를 잘라 쓴다. 무작위 id 는
        # 존재하지 않는 토큰열이라 캐시 거동이 실제와 달라질 수 있다.
        long_text = load_prompt(RAW_CHARS[-1] * 2)
        full = tokenizer(long_text, add_special_tokens=False,
                         return_tensors="pt")["input_ids"]
        for n_tok in EQUAL_TOKENS:
            if full.shape[1] < n_tok:
                print(f"  (equal_tokens {n_tok:,} 건너뜀 — dev 원문이 짧다)")
                continue
            ids = full[:, :n_tok].to(device)
            measure("equal_tokens", ids, None, None, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
