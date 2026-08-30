"""Embedding Alignment — backbone 을 얼리고 embedding 만 학습한다 (스펙 §25).

    .conda/python.exe -m src.training.alignment \
        --model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --tag v1

왜 CPT 앞에 따로 두는가
    수술 직후 T2b 의 dev BPB 는 2.38 로 기준선(1.16)의 2배다. 이 상태로 바로 full
    CPT 를 돌리면 손상된 embedding 이 **초기부터 backbone 에 나쁜 gradient 를
    밀어넣는다.** 그러면 우리가 재려는 "토크나이저 효과" 에 "embedding 복구가
    backbone 을 망친 효과" 가 섞여, 스펙 §20 이 금지하는 혼입이 그대로 일어난다.

    backbone 을 얼면 embedding 만 제자리를 찾는다. 그 다음에야 CPT 가 토크나이저
    효과를 측정하는 실험이 된다.

세 조건에 모두 적용한다
    T2b 에만 주면 T2b 가 원문을 더 본 셈이 되어 스펙 §30 의 통제
    ("Original Tokenizer + Same CPT vs Korean Tokenizer + Same CPT")가 깨진다.
    C0 와 T2a 는 새로 생긴 행이 없어 사실상 무효과지만, 데이터 노출량을 맞추기
    위해 같은 예산으로 돈다.

예산 사다리
    스펙은 1M / 5M / 10M / 20M 토큰을 제시한다. 우리는 **바이트로** 센다 —
    토큰으로 세면 압축이 좋은 조건이 같은 예산에서 원문을 덜 보게 되어
    Equal-Raw-Data 가 깨진다 (RULES 12). 사다리의 각 칸에서 dev 를 재고
    체크포인트를 남겨, 어느 지점에서 포화하는지 보고 고른다.

tie_word_embeddings 라 embedding 하나만 풀면 lm_head 도 함께 학습된다.
따로 풀면 묶임이 끊겨 파라미터 수가 달라지고 T2a 와의 비교가 깨진다.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.training.callbacks import CurveLogger, cosine_lr_by_bytes  # noqa: E402
from src.training.cpt import load_pool, pack  # noqa: E402
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

DEV = (("ko", "dev.jsonl"),
       ("en", "dev_control_english.jsonl"),
       ("code", "dev_control_code.jsonl"))

TRAIN_FILES = {"ko": "train.jsonl",
               "en": "train_control_english.jsonl",
               "code": "train_control_code.jsonl"}


def build_mixed_pool(root: Path, mix: dict, pool_docs: int, seed: int) -> tuple:
    """세 언어를 목표 바이트 비율로 섞은 문서 목록을 만든다.

    한국어만으로 정렬했더니 영어가 +8.8%, 코드가 +13.1% 나빠졌다. backbone 이
    얼어 있으면 모든 적응이 embedding 에서만 일어나는데, tie_word_embeddings 라
    그 행렬 하나가 모든 언어의 표현이자 출력 로짓 방향이다. 한국어 쪽으로 밀면
    영어·코드가 함께 밀려난다.

    영어·코드를 섞으면 그 토큰들이 **직접 gradient 신호를 받아** 제자리를 지킨다.
    T2b 의 새 토큰은 한국어 텍스트에만 나오므로 한국어가 여전히 주가 되어야 하고,
    영어·코드는 닻(anchor) 역할이다.

    문서를 하나씩 고를 때마다 **목표 비율에서 가장 뒤처진 언어**를 집는다.
    누적 바이트가 목표를 따라가므로 순서가 한쪽으로 몰리지 않는다.
    """
    from src.training.cpt import load_pool

    rng = random.Random(seed)
    total_frac = sum(v for v in mix.values() if v > 0)
    target = {k: v / total_frac for k, v in mix.items() if v > 0}

    # 목표 총 바이트를 한국어 풀 크기로 정한다. 그 다음 언어별 목표 바이트를
    # 실제 문서 길이로 나눠 필요한 문서 수를 잡는다 — 영어·코드는 문서가 훨씬
    # 짧아서 같은 문서 수로는 비율이 안 맞고, 풀이 먼저 소진되면 뒤쪽이 한국어로만
    # 채워져 목표에서 5% 씩 어긋났다.
    pools = {}
    budget_bytes = None
    for lang in sorted(target, key=lambda k: -target[k]):
        path = root / "data" / "interim" / "docs" / TRAIN_FILES[lang]
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없다")
        if budget_bytes is None:
            docs = load_pool(path, pool_docs)
            budget_bytes = sum(len(d.encode("utf-8")) for d in docs) / target[lang]
        else:
            want = budget_bytes * target[lang]
            docs, acc = [], 0
            with path.open("r", encoding="utf-8") as fh:
                import json as _json
                for line in fh:
                    if not line.strip():
                        continue
                    d = _json.loads(line)["text"]
                    docs.append(d)
                    acc += len(d.encode("utf-8"))
                    if acc >= want * 1.15:      # 15% 여유
                        break
        rng.shuffle(docs)
        pools[lang] = docs

    got = {k: 0.0 for k in pools}
    idx = {k: 0 for k in pools}
    out: list = []
    while True:
        avail = [k for k in pools if idx[k] < len(pools[k])]
        if not avail:
            break
        seen = sum(got.values()) or 1.0
        pick = min(avail, key=lambda k: got[k] / seen - target[k])
        d = pools[pick][idx[pick]]
        idx[pick] += 1
        got[pick] += len(d.encode("utf-8"))
        out.append(d)
    return out, got


def freeze_backbone(model) -> tuple:
    """embedding 을 뺀 전부를 얼린다. (학습 파라미터 수, 전체) 를 돌려준다."""
    emb = model.get_input_embeddings().weight
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p is emb:
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
    if trainable == 0:
        raise RuntimeError("embedding 을 못 찾았다 — 얼릴 대상이 잘못됐다")
    return trainable, total


def main(argv: list | None = None) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluation.bpb import evaluate, token_byte_length

    ap = argparse.ArgumentParser(description="Embedding Alignment (backbone frozen)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rungs", default="3400000,17000000,34000000,67000000",
                    help="바이트 사다리. 스펙 §25 의 1M/5M/10M/20M 토큰에 대응")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="embedding 만 학습하므로 CPT 보다 크게 잡는다")
    ap.add_argument("--pool-docs", type=int, default=22_000)
    ap.add_argument("--mix", default="ko=0.6,en=0.2,code=0.2",
                    help="정렬 코퍼스의 언어별 목표 바이트 비율")
    ap.add_argument("--eval-budget", type=int, default=1_000_000)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--save-final", action="store_true")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    budget = max(rungs)

    name = args.name or Path(args.model).name
    kw = {"revision": args.revision} if args.revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", **kw)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.config.use_cache = False

    trainable, total = freeze_backbone(model)
    eos_id = tokenizer.eos_token_id or 0

    config = {
        "model": args.model, "revision": args.revision, "seed": args.seed,
        "rungs": rungs, "seq_len": args.seq_len, "micro_bs": args.micro_bs,
        "accum": args.accum, "lr": args.lr, "pool_docs": args.pool_docs,
        "frozen_backbone": True, "optimizer": "adamw8bit", "mix": args.mix,
        "lr_schedule": "cosine_by_raw_bytes",
    }
    run_id = make_run_id("align", name, args.tag, seed=args.seed)

    with RunContext(run_id, phase="align", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check) as run:
        import bitsandbytes as bnb
        from tokenizers.pre_tokenizers import ByteLevel

        alphabet = set(ByteLevel.alphabet())
        blen: dict = {eos_id: 0}

        def byte_len_fn(ids: list) -> int:
            t = 0
            for i in ids:
                v = blen.get(i)
                if v is None:
                    v = token_byte_length(
                        tokenizer.convert_ids_to_tokens(int(i)), alphabet)
                    blen[i] = v
                t += v
            return t

        mix = {}
        for part in args.mix.split(","):
            k, v = part.split("=")
            mix[k.strip()] = float(v)
        docs, got = build_mixed_pool(ROOT, mix, args.pool_docs, args.seed)
        tot = sum(got.values()) or 1.0
        print(f"{name}  seed {args.seed}  학습 파라미터 {trainable:,} / {total:,} "
              f"({trainable / total:.1%})")
        print(f"  혼합 코퍼스 {len(docs):,}문서 {tot / 1e6:.0f}MB  "
              + "  ".join(f"{k} {got[k] / tot:.0%}" for k in got))
        print(f"  사다리 {[f'{r / 1e6:.1f}MB' for r in rungs]}")

        opt = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
        backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
        dev_dir = ROOT / "data" / "interim" / "docs"
        curve = CurveLogger(run, budget * 2)      # 사다리에서만 잰다

        def dev_bpb() -> dict:
            model.eval()
            out: dict = {}
            with sdpa_kernel(backends):
                for lang, fname in DEV:
                    p = dev_dir / fname
                    if p.exists():
                        out[lang] = evaluate(model, tokenizer, p, args.eval_budget,
                                             args.seq_len, device, byte_len_fn)["bpb"]
            model.train()
            return out

        def fmt(d: dict) -> str:
            return "  ".join(f"{k} {v:.4f}" for k, v in d.items())

        base = dev_bpb()
        print(f"  정렬 전  {fmt(base)}")
        run.log("lm_metrics", checkpoint="align_0", tokens_seen=0, raw_bytes_seen=0,
                split="dev", domain="ko", n_bytes=None, total_nll=None,
                bpb=round(base["ko"], 6), bpc=None, token_ppl=None)

        model.train()
        torch.cuda.reset_peak_memory_stats()
        step = micro = tokens_seen = 0
        raw_bytes = 0.0
        loss_acc = 0.0
        batch: list = []
        opt.zero_grad(set_to_none=True)
        pending = list(rungs)
        last = base

        with sdpa_kernel(backends):
            for chunk, take in pack(tokenizer, docs, args.seq_len, eos_id, byte_len_fn):
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
                    lr = cosine_lr_by_bytes(int(raw_bytes), budget, args.lr)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    train_loss = loss_acc / args.accum
                    loss_acc = 0.0

                    while pending and raw_bytes >= pending[0]:
                        rung = pending.pop(0)
                        last = dev_bpb()
                        peak = int(torch.cuda.max_memory_allocated() / 1e6)
                        print(f"  {rung / 1e6:>5.1f}MB  step {step:>4}  "
                              f"loss {train_loss:.4f}  {fmt(last)}  peak {peak}MB")
                        curve.log(step=step, tokens_seen=tokens_seen,
                                  raw_bytes_seen=int(raw_bytes),
                                  train_loss=train_loss, dev_bpb=last["ko"],
                                  lr=lr, grad_norm=gnorm, peak_vram_mb=peak)
                        for lang, v in last.items():
                            run.log("lm_metrics",
                                    checkpoint=f"align_{rung // 1_000_000}MB",
                                    tokens_seen=tokens_seen,
                                    raw_bytes_seen=int(raw_bytes), split="dev",
                                    domain=lang, n_bytes=None, total_nll=None,
                                    bpb=round(v, 6), bpc=None, token_ppl=None)

                if raw_bytes >= budget:
                    break

        short = 1.0 - raw_bytes / budget
        if short > 0.01:
            raise RuntimeError(
                f"예산 미달: {raw_bytes / 1e6:.2f}MB / {budget / 1e6:.2f}MB "
                f"({short:.1%} 부족). --pool-docs 를 늘려라.")

        # 예산 도달 break 는 매 chunk 마다 걸리므로, 마지막 사다리 칸이 다음 accum
        # 경계 전에 빠져나가 평가되지 않는다. 실제로 C0 는 67MB 까지 학습됐는데
        # 보고된 최종 수치가 34MB 값이었다 — 저장된 모델과 기록이 어긋난다.
        if pending:
            print(f"  (사다리 {[f'{r / 1e6:.1f}MB' for r in pending]} 가 "
                  f"평가 전에 예산에 닿았다. 최종 지점에서 다시 잰다)")
        last = dev_bpb()
        print(f"  정렬 후  {fmt(last)}  @ {raw_bytes / 1e6:.1f}MB")
        for lang, v in last.items():
            run.log("lm_metrics", checkpoint="align_final",
                    tokens_seen=tokens_seen, raw_bytes_seen=int(raw_bytes),
                    split="dev", domain=lang, n_bytes=None, total_nll=None,
                    bpb=round(v, 6), bpc=None, token_ppl=None)
        for lang, v in last.items():
            print(f"      {lang:<5} {base[lang]:.4f} -> {v:.4f} "
                  f"({(v / base[lang] - 1) * 100:+.2f}%)")

        if args.save_final:
            out_dir = ROOT / "artifacts" / "models" / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            model.config.use_cache = True
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            sha = sha256_file(out_dir / "model.safetensors")
            print(f"  저장 {out_dir}")
            print(f"  model_sha256 = {sha}")
            run.extra["tokenizer_sha256"] = sha

        run.tokens_seen = tokens_seen
        run.raw_bytes_seen = int(raw_bytes)
        run.extra["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / 1e6)
        run.note = (f"align steps={step} ko {base['ko']:.4f}->{last['ko']:.4f} "
                    + " ".join(f"{k} {base[k]:.4f}->{last[k]:.4f}"
                               for k in last if k != "ko")
                    + f" lr={args.lr} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
