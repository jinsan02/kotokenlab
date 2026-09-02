"""Continued Pretraining 루프 (스펙 §26, §79).

    .conda/python.exe -m src.training.cpt --model Qwen/Qwen2.5-0.5B --seed 42 \
        --budget-bytes 18000000 --tag noise

Equal-Raw-Data (기본)
    예산을 토큰이 아니라 **원문 바이트** 로 센다. 토크나이저가 다른 조건을
    tokens_seen 으로 맞추면 압축이 좋은 쪽이 같은 예산에서 더 적은 원문을 보게
    되어, 압축 개선이 곧 데이터 손해로 바뀐다. 바이트로 맞추면 모든 조건이
    **같은 글** 을 본다 ([RULES.md](../../docs/RULES.md) 12).

Equal Token Budget (--budget-tokens, 스펙 §32~33)
    보완 축이다. 같은 **연산·문맥** 을 주면 압축이 좋은 쪽이 더 많은 원문을 본다 —
    실무에서 토크나이저를 바꾸는 이유가 그것이다. 예산도 LR 스케줄도 토큰 기준이
    된다. 조건마다 tok/byte 가 일정하므로 '진행률 대비 LR' 곡선은 바이트 기준과
    같은 모양이고, 따라서 이미 돌린 바이트 기준 run 과 비교가 성립한다.

노이즈 플로어
    같은 config 를 seed 만 바꿔 여러 번 돌리면 sigma_BPB 가 나온다. 이걸 먼저
    재지 않으면 이후 어떤 비교도 해석할 수 없다 — "1.207 vs 1.198" 이 의미
    있는 차이인지 판단할 근거가 없기 때문이다. seed 는 **문서 순서** 와 torch
    난수를 바꾼다. 문서 풀은 같게 두어, 데이터 선택이 아니라 순서와 비결정성만
    변수가 되게 한다.

attention 은 반드시 EFFICIENT + CUDNN 안에서 돈다. 강제하지 않으면 8,192 토큰에서
메모리 7.1배, 시간 10.3배가 된다 (reports/tables/resource_probe.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.training.callbacks import CurveLogger, cosine_lr_by_bytes  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402


def load_pool(path: Path, n_docs: int, skip: int = 0) -> list:
    """앞의 skip 개를 건너뛰고 n_docs 개를 읽어 문서 풀을 만든다.

    skip 은 정렬 단계가 이미 본 문서를 CPT 가 다시 보지 않게 한다. 겹치면
    CPT 예산의 일부가 재학습이 되어 조건 간 비교는 유지되더라도 "168.5MB 를
    학습했다" 는 서술이 부정확해진다.

    풀을 고정하고 **순서만** seed 로 섞는다. seed 마다 다른 문서를 뽑으면
    데이터 선택 분산까지 섞여 들어가는데, 실제 비교에서는 모든 조건이 같은
    데이터를 보므로(Equal-Raw-Data) 그 분산은 조건 간 차이의 원인이 아니다.
    """
    docs = []
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            seen += 1
            if seen <= skip:
                continue
            docs.append(json.loads(line)["text"])
            if len(docs) >= n_docs:
                break
    return docs


def pack(tokenizer, docs: list, seq_len: int, eos_id: int, byte_len_fn):
    """문서를 이어 붙여 seq_len 조각으로 자른다. 조각의 **정확한** 원문 바이트를 함께 준다.

    처음에는 문서의 바이트/토큰 비율을 조각에 곱해 배분했다. 조각에는 이전 문서의
    토큰이 섞여 있는데 현재 문서의 비율을 쓰므로 어긋난다 — 한국어만이면 -0.38%,
    한영 혼합이면 +0.73% 로 **부호까지 구성에 따라 뒤집혔다.**

    크기보다 부호가 문제다. C0 와 T2b 는 같은 글에 대해 바이트/토큰 비율이 다르니
    같은 예산에서 실제로는 다른 분량을 보게 되고, Equal-Raw-Data 라는 통제축이
    조용히 깨진다 (RULES 12). 노이즈 플로어가 sigma=0.0001 로 좁아서 0.7% 의
    데이터량 차이면 조건 간 차이를 그것만으로 만들어낼 수 있다.

    그래서 토큰마다 실제 바이트를 더한다. ByteLevel BPE 는 무손실 바이트 분절이라
    토큰 바이트의 합이 원문 바이트와 정확히 일치한다. EOS 는 코퍼스 원문이 아니므로
    0 바이트로 센다 (byte_len_fn 을 만드는 쪽 책임).
    """
    buf: list = []
    for text in docs:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buf.extend(ids + [eos_id])
        while len(buf) >= seq_len:
            chunk, buf = buf[:seq_len], buf[seq_len:]
            yield chunk, byte_len_fn(chunk)


def main(argv: list | None = None) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluation.bpb import evaluate, token_byte_length

    ap = argparse.ArgumentParser(description="Continued Pretraining")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-bytes", type=int, default=18_000_000)
    ap.add_argument("--budget-tokens", type=int, default=0,
                    help="0 이 아니면 토큰 예산으로 돈다 (스펙 §32~33 등토큰). "
                         "예산도 LR 스케줄도 토큰 기준이 된다")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pool-docs", type=int, default=30_000)
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="정렬 단계가 이미 본 문서 수. 겹쳐 학습하지 않기 위해")
    ap.add_argument("--eval-bytes", type=int, default=1_000_000,
                    help="dev BPB 를 몇 바이트마다 잴지")
    ap.add_argument("--eval-budget", type=int, default=1_000_000,
                    help="평가에 쓸 dev 원문 바이트 (언어별)")
    ap.add_argument("--tag", default="noise")
    ap.add_argument("--save", action="store_true",
                    help="학습된 모델을 artifacts/models/<run_id>/ 에 저장한다")
    ap.add_argument("--allow-short", action="store_true",
                    help="예산을 못 채워도 통과시킨다 (조건 간 비교에는 쓰지 마라)")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    name = args.name or Path(args.model).name
    kw = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", **kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    eos_id = tokenizer.eos_token_id or 0
    config = {
        "model": args.model, "revision": args.revision, "seed": args.seed,
        "budget_bytes": args.budget_bytes,
        "budget_tokens": args.budget_tokens, "seq_len": args.seq_len,
        "micro_bs": args.micro_bs, "accum": args.accum, "lr": args.lr,
        "pool_docs": args.pool_docs, "skip_docs": args.skip_docs,
        "optimizer": "adamw8bit",
        "dtype": "bfloat16", "grad_checkpointing": True,
        "lr_schedule": ("cosine_by_tokens" if args.budget_tokens
                        else "cosine_by_raw_bytes"),
    }
    run_id = make_run_id("cpt", name, args.tag, seed=args.seed)

    with RunContext(run_id, phase="cpt", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check) as run:
        import bitsandbytes as bnb

        from tokenizers.pre_tokenizers import ByteLevel
        alphabet = set(ByteLevel.alphabet())
        blen: dict = {}

        # EOS 는 문서 경계 표시일 뿐 코퍼스 원문이 아니다. 0 바이트로 세지 않으면
        # 문서가 짧을수록 예산이 빨리 닳아 조건 간 데이터량이 어긋난다.
        blen[eos_id] = 0

        def byte_len_fn(ids: list) -> int:
            total = 0
            for i in ids:
                v = blen.get(i)
                if v is None:
                    v = token_byte_length(
                        tokenizer.convert_ids_to_tokens(int(i)), alphabet)
                    blen[i] = v
                total += v
            return total

        pool_path = ROOT / "data" / "interim" / "docs" / "train.jsonl"
        docs = load_pool(pool_path, args.pool_docs, args.skip_docs)
        rng = random.Random(args.seed)
        rng.shuffle(docs)                     # seed 는 **순서만** 바꾼다
        budget_txt = (f"{args.budget_tokens / 1e6:.1f}M 토큰" if args.budget_tokens
                      else f"{args.budget_bytes / 1e6:.1f}MB 원문")
        print(f"{name}  seed {args.seed}  문서 풀 {len(docs):,}  예산 {budget_txt}")

        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

        # 모듈별 기울기 노름. 스키마에 컬럼이 있는데 오래 비워 뒀다.
        # 회복이 멎을 때 임베딩 기울기가 죽는지 살아 있는지가 기전을 가른다 —
        # 죽으면 최적화 문제이고, 살아 있는데도 BPB 가 안 내려가면 표현 공간
        # 문제다. tie_word_embeddings 라 lm_head 는 embed_tokens 와 같은
        # 텐서이고 named_parameters 가 중복을 지우므로 한 번만 잡힌다.
        GROUPS = {"emb": [], "attn": [], "ffn": []}
        for pname, param in model.named_parameters():
            if "embed_tokens" in pname:
                GROUPS["emb"].append(param)
            elif "self_attn" in pname:
                GROUPS["attn"].append(param)
            elif "mlp" in pname:
                GROUPS["ffn"].append(param)

        def group_norms() -> dict:
            """클리핑 **전에** 부른다. clip_grad_norm_ 은 grad 를 제자리에서 줄인다."""
            out = {}
            for key, params in GROUPS.items():
                g = [p.grad for p in params if p.grad is not None]
                out[key] = float(torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(x) for x in g]))) if g else None
            return out
        backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
        curve = CurveLogger(run, args.eval_bytes)

        dev_dir = ROOT / "data" / "interim" / "docs"

        # 한국어만 보면 파국적 망각이 진행돼도 학습 중에는 안 보이고 끝나서야
        # 드러난다. 스펙 §17 의 regression 조건이 영어·코드에 걸려 있으므로
        # 평가 지점마다 셋을 다 본다.
        DEV = (("ko", "dev.jsonl"),
               ("en", "dev_control_english.jsonl"),
               ("code", "dev_control_code.jsonl"))

        def dev_bpb() -> dict:
            model.eval()
            out: dict = {}
            with sdpa_kernel(backends):
                for lang, fname in DEV:
                    path = dev_dir / fname
                    if not path.exists():
                        continue
                    m = evaluate(model, tokenizer, path, args.eval_budget,
                                 args.seq_len, device, byte_len_fn)
                    out[lang] = m["bpb"]
            model.train()
            return out

        def fmt(d: dict) -> str:
            return "  ".join(f"{k} {v:.4f}" for k, v in d.items())

        base = dev_bpb()
        base_bpb = base["ko"]
        print(f"  학습 전 dev BPB  {fmt(base)}")

        # 학습 전 지점을 반드시 원장에 남긴다. 예전에는 출력만 하고 버려서,
        # '얼마나 회복했나' 를 말하려면 나중에 같은 예산으로 다시 재야 했다.
        # 평가 예산이 run 마다 다르면 사후 측정과 대조도 못 한다.
        for lang, v in base.items():
            run.log("lm_metrics", checkpoint="step0", tokens_seen=0,
                    raw_bytes_seen=0, split="dev", domain=lang,
                    n_bytes=None, total_nll=None, bpb=round(v, 6),
                    bpc=None, token_ppl=None)

        # 등토큰 예산이면 진행량·LR·중단 판정이 전부 토큰 기준이 된다.
        # 조건마다 tok/byte 가 일정하므로 '진행률 대비 LR' 곡선은 바이트 기준과
        # 같은 모양이다 — 즉 이미 돌린 바이트 기준 run 과 비교가 성립한다.
        by_tokens = args.budget_tokens > 0
        budget = args.budget_tokens if by_tokens else args.budget_bytes

        model.train()
        torch.cuda.reset_peak_memory_stats()
        step = micro = 0
        tokens_seen = 0
        raw_bytes = 0.0
        loss_acc = 0.0
        batch: list = []
        opt.zero_grad(set_to_none=True)
        stream = pack(tokenizer, docs, args.seq_len, eos_id, byte_len_fn)
        final_bpb = base_bpb

        with sdpa_kernel(backends):
            for chunk, take in stream:
                batch.append(chunk)
                raw_bytes += take
                tokens_seen += len(chunk)
                if len(batch) < args.micro_bs:
                    continue
                x = torch.tensor(batch, device=device)
                batch = []
                out = model(x, labels=x)
                (out.loss / args.accum).backward()
                loss_acc += float(out.loss)
                micro += 1

                if micro % args.accum == 0:
                    progress = tokens_seen if by_tokens else int(raw_bytes)
                    lr = cosine_lr_by_bytes(progress, budget, args.lr)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    gn = group_norms()          # 클리핑 전에 잰다
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    train_loss = loss_acc / args.accum
                    loss_acc = 0.0

                    if curve.due(int(raw_bytes)):
                        curve.mark(int(raw_bytes))
                        cur = dev_bpb()
                        final_bpb = cur["ko"]
                        peak = int(torch.cuda.max_memory_allocated() / 1e6)
                        print(f"  step {step:>4}  {raw_bytes / 1e6:>6.2f}MB  "
                              f"loss {train_loss:.4f}  {fmt(cur)}  "
                              f"lr {lr:.2e}  peak {peak}MB")
                        curve.log(step=step, tokens_seen=tokens_seen,
                                  raw_bytes_seen=int(raw_bytes),
                                  train_loss=train_loss, dev_bpb=final_bpb,
                                  lr=lr, grad_norm=gnorm, peak_vram_mb=peak,
                                  grad_norm_emb=gn["emb"],
                                  grad_norm_attn=gn["attn"],
                                  grad_norm_ffn=gn["ffn"])
                        for lang, v in cur.items():
                            run.log("lm_metrics", checkpoint=f"step{step}",
                                    tokens_seen=tokens_seen,
                                    raw_bytes_seen=int(raw_bytes), split="dev",
                                    domain=lang, n_bytes=None, total_nll=None,
                                    bpb=round(v, 6), bpc=None, token_ppl=None)

                if (tokens_seen if by_tokens else raw_bytes) >= budget:
                    break

        # 문서 풀이 예산보다 작으면 스트림이 먼저 끝나고 그냥 종료된다. 그러면
        # 조건마다 다른 분량을 학습하고도 모른 채 비교하게 된다 — Equal-Raw-Data 가
        # 깨지는 두 번째 경로다. 조용히 넘어가지 않는다.
        shortfall = 1.0 - (tokens_seen if by_tokens else raw_bytes) / budget
        if shortfall > 0.01:
            msg = (f"예산 미달: {raw_bytes / 1e6:.2f}MB / "
                   f"{args.budget_bytes / 1e6:.2f}MB ({shortfall:.1%} 부족). "
                   f"문서 풀({args.pool_docs:,}개)이 모자란다 — --pool-docs 를 늘려라.")
            if not args.allow_short:
                raise RuntimeError(msg)
            print(f"  [주의] {msg}")

        final = dev_bpb()
        final_bpb = final["ko"]
        print(f"  학습 후 dev BPB  {fmt(final)}")
        for lang, v in final.items():
            delta = (v / base[lang] - 1) * 100
            print(f"      {lang:<5} {base[lang]:.4f} -> {v:.4f}  ({delta:+.2f}%)")
            run.log("lm_metrics", checkpoint="final", tokens_seen=tokens_seen,
                    raw_bytes_seen=int(raw_bytes), split="dev", domain=lang,
                    n_bytes=None, total_nll=None, bpb=round(v, 6),
                    bpc=None, token_ppl=None)
        # 저장하지 않으면 103분 학습한 가중치를 버리게 되고, Step 7 시스템
        # 벤치마크와 Level 3 capability 가 쓸 체크포인트가 없어 다시 학습해야 한다.
        if args.save:
            out_dir = ROOT / "artifacts" / "models" / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            model.config.use_cache = True      # 학습 중 껐던 것을 되돌린다
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            from src.utils.hashing import sha256_file
            sha = sha256_file(out_dir / "model.safetensors")
            print(f"  저장 {out_dir}")
            print(f"  model_sha256 = {sha}")
            run.extra["tokenizer_sha256"] = sha

        run.tokens_seen = tokens_seen
        run.raw_bytes_seen = int(raw_bytes)
        run.extra["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / 1e6)
        run.note = (f"steps={step} ko {base_bpb:.4f}->{final_bpb:.4f} "
                    + " ".join(f"{k} {base[k]:.4f}->{final[k]:.4f}"
                               for k in final if k != "ko")
                    + f" lr={args.lr} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
