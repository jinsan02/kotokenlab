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

from src.training.callbacks import SCHEDULES, CurveLogger  # noqa: E402
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


def load_damaged_rows(path: Path | str) -> list:
    """손상된(또는 새로 만들어진) 임베딩 행 id 목록.

    세 가지 출처를 다 받는다 — 어느 쪽을 줘야 하는지 기억할 필요가 없어야 한다.

        damage_rows.py 의 damaged_rows.json   {"rows": [...]}
        토크나이저의 id_map.json              {"map": {새 id: ...}}   T2b 의 새 행
        그냥 id 목록                          [1, 2, 3]
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict):
        d = d.get("rows", d.get("map", d))
        if isinstance(d, dict):
            d = list(d.keys())
    rows = sorted({int(x) for x in d})
    if not rows:
        raise ValueError(f"{path}: 행 목록이 비었다")
    return rows


def save_checkpoint(model, tokenizer, run_id: str, suffix: str,
                    raw_bytes: int) -> str:
    """artifacts/models/<run_id><suffix>/ 에 저장하고 sha256 을 돌려준다.

    중간 저장은 **학습을 멈추지 않는다** — `use_cache` 만 잠깐 되돌렸다가 끈다.
    켠 채로 두면 다음 forward 가 KV 를 들고 있어 VRAM 이 는다.
    """
    out_dir = ROOT / "artifacts" / "models" / f"{run_id}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    was = model.config.use_cache
    model.config.use_cache = True
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    model.config.use_cache = was
    from src.utils.hashing import sha256_file
    sha = sha256_file(out_dir / "model.safetensors")
    print(f"  저장 {out_dir}  ({raw_bytes / 1e6:.2f}MB 지점)")
    print(f"  model_sha256 = {sha}")
    return sha


def warmup_fraction(warmup_bytes: int, budget: int,
                    default_frac: float = 0.02) -> float:
    """워밍업 끝 지점을 **예산 비율** 이 아니라 **바이트** 로 고정할 수 있게 한다.

    비율로 두면 예산을 늘리는 순간 워밍업도 같이 늘어난다 — 168.5MB 의 2% 는
    3.37MB 인데 500MB 의 2% 는 10MB 다. 그러면 긴 run 의 앞부분이 짧은 run 과
    **첫 스텝부터** 다른 LR 을 밟아, "R5 의 연장" 이라고 부를 수 없게 된다.

    0 을 주면(기본) 지금까지와 똑같이 예산의 2% 다.
    """
    if warmup_bytes <= 0 or budget <= 0:
        return default_frac
    return warmup_bytes / budget


def order_pool(docs: list, seed: int, extra: list) -> list:
    """문서 순서를 정한다. **기존 풀의 순서는 확장해도 보존된다.**

    P2 까지는 풀 전체를 한 번에 섞었다. 그 상태로 풀을 50,000 -> 150,000 으로
    늘리면 첫 문서부터 순서가 바뀌어, 긴 run 의 앞 168.5MB 가 R5 와 다른 데이터가
    된다. 기존 풀을 같은 seed 로 섞은 결과를 그대로 두고 추가분을 뒤에 붙인다.
    추가분은 seed+1 로 섞는다 — 파일 순서(코퍼스 정렬)가 그대로 학습 순서가
    되지 않게 한다.

    `extra` 가 비면 P2 와 **완전히 같은 순서** 다.
    """
    base = list(docs)
    random.Random(seed).shuffle(base)
    if not extra:
        return base
    tail = list(extra)
    random.Random(seed + 1).shuffle(tail)
    return base + tail


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
    # R5 는 "65% 가 벽인가 lr 이 꺼진 것인가" 를 묻는다. 감쇠만 빼고 나머지는
    # 전부 같아야 그 물음에 답이 된다.
    ap.add_argument("--lr-schedule", choices=tuple(SCHEDULES), default="cosine")
    ap.add_argument("--warmup-bytes", type=int, default=0,
                    help="워밍업이 끝나는 원문 바이트. 0 이면 예산의 2%% (기존 동작). "
                         "예산이 다른 run 의 앞부분을 재현하려면 바이트로 고정한다")
    ap.add_argument("--pool-docs", type=int, default=30_000)
    ap.add_argument("--pool-extend-docs", type=int, default=0,
                    help="기존 풀 뒤에 붙일 문서 수. 기존 풀의 순서는 보존된다 "
                         "(P3-A 가 R5 의 앞부분을 재현하려면 필요하다)")
    ap.add_argument("--skip-docs", type=int, default=0,
                    help="정렬 단계가 이미 본 문서 수. 겹쳐 학습하지 않기 위해")
    ap.add_argument("--eval-bytes", type=int, default=1_000_000,
                    help="dev BPB 를 몇 바이트마다 잴지")
    ap.add_argument("--eval-budget", type=int, default=1_000_000,
                    help="평가에 쓸 dev 원문 바이트 (언어별)")
    ap.add_argument("--eval-at", type=int, action="append", default=None,
                    metavar="BYTES",
                    help="평가 간격에 안 걸리는 지점을 추가로 잰다 (여러 번 줄 수 있다)")
    ap.add_argument("--damaged-rows", default=None, metavar="JSON",
                    help="손상·신규 임베딩 행 목록. 그 행만의 기울기와 "
                         "update/weight 비를 train_curve 에 남긴다 (측정만 한다)")
    ap.add_argument("--save-at", type=int, default=0, metavar="BYTES",
                    help="그 지점을 지날 때 체크포인트를 저장한다 "
                         "(artifacts/models/<run_id>_at<N>mb)")
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
        # 평가 설정도 config 에 남긴다. eval_budget 은 보고되는 BPB 자체를
        # 정하므로, 기록이 없으면 두 run 의 최종값을 비교해도 되는지 알 수 없다.
        # 2026-09-15 에 1차 노이즈 run 의 config 에서 이 값을 찾다가 없어서
        # 겪었다 — 같은 날 --max-bytes 기본값 불일치로 게이트가 거짓 실패했다.
        "eval_budget": args.eval_budget, "eval_bytes": args.eval_bytes,
        # P3 에서 추가. 앞의 둘은 측정값을 바꾸고(워밍업 길이 · 데이터 순서),
        # 뒤의 둘은 벽시계만 바꾼다 (tools/compare_runs.py 가 그렇게 분류한다).
        "warmup_bytes": args.warmup_bytes,
        "pool_extend_docs": args.pool_extend_docs,
        "eval_at": sorted(args.eval_at or []), "save_at": args.save_at,
        "damaged_rows": args.damaged_rows,
        "optimizer": "adamw8bit",
        "dtype": "bfloat16", "grad_checkpointing": True,
        "lr_schedule": (f"{args.lr_schedule}_by_tokens" if args.budget_tokens
                        else f"{args.lr_schedule}_by_raw_bytes"),
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
        total_docs = args.pool_docs + max(args.pool_extend_docs, 0)
        pool = load_pool(pool_path, total_docs, args.skip_docs)
        # seed 는 **순서만** 바꾼다. 확장분은 기존 풀 뒤에 붙는다 — 그래야
        # 예산이 긴 run 의 앞부분이 짧은 run 과 같은 데이터를 같은 순서로 본다.
        docs = order_pool(pool[:args.pool_docs], args.seed, pool[args.pool_docs:])
        if args.pool_extend_docs:
            print(f"      문서 풀 확장  {args.pool_docs:,} + {len(pool) - args.pool_docs:,}"
                  f"  (앞 {args.pool_docs:,}개의 순서는 보존)")
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
        #
        # tie 를 끊으면(P2/Q7) lm_head 가 별도 파라미터가 되는데, 그러면 아래
        # 셋 중 어느 조건에도 안 걸려 **에러 없이 기록에서 빠진다.** 하필 Q7 의
        # 관심사가 출력 방향의 기울기라 그것만 사라지면 안 된다. 따로 잡는다.
        # tied run 에서는 head 그룹이 비어 있고 값이 NA 로 남는다 — 없는 것이
        # 맞다. 한 텐서에 두 경로가 합쳐져 있어 나눌 수 없기 때문이다.
        GROUPS = {"emb": [], "attn": [], "ffn": [], "head": []}
        for pname, param in model.named_parameters():
            if "embed_tokens" in pname:
                GROUPS["emb"].append(param)
            elif "lm_head" in pname:
                GROUPS["head"].append(param)
            elif "self_attn" in pname:
                GROUPS["attn"].append(param)
            elif "mlp" in pname:
                GROUPS["ffn"].append(param)
        ungrouped = [n for n, _ in model.named_parameters()
                     if not any(k in n for k in
                                ("embed_tokens", "lm_head", "self_attn", "mlp"))]
        print(f"      기울기 그룹  emb {len(GROUPS['emb'])}  head {len(GROUPS['head'])}"
              f"  attn {len(GROUPS['attn'])}  ffn {len(GROUPS['ffn'])}"
              f"  (norm/bias 등 {len(ungrouped)}개는 grad_norm 전체에만 든다)")

        # 손상 행 계측 (P3 W0-5). 목록을 안 주면 아무것도 하지 않는다.
        dmg_ids = None
        emb_w = model.get_input_embeddings().weight
        if args.damaged_rows:
            rows = load_damaged_rows(args.damaged_rows)
            bad = [i for i in rows if i >= emb_w.shape[0]]
            if bad:
                raise SystemExit(f"손상 행 id 가 임베딩 밖이다: {bad[:5]} "
                                 f"(vocab {emb_w.shape[0]})")
            dmg_ids = torch.tensor(rows, device=device)
            print(f"      손상 행 계측  {len(rows):,}행  {args.damaged_rows}")

        def group_norms() -> dict:
            """클리핑 **전에** 부른다. clip_grad_norm_ 은 grad 를 제자리에서 줄인다."""
            out = {}
            for key, params in GROUPS.items():
                g = [p.grad for p in params if p.grad is not None]
                out[key] = float(torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(x) for x in g]))) if g else None
            return out
        backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
        curve = CurveLogger(run, args.eval_bytes, extra_points=args.eval_at)

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
        if args.warmup_bytes and by_tokens:
            raise SystemExit("--warmup-bytes 는 바이트 예산에서만 쓴다 "
                             "(--budget-tokens 와 같이 주지 마라)")
        lr_fn = SCHEDULES[args.lr_schedule]
        warm = warmup_fraction(args.warmup_bytes, budget)
        print(f"      워밍업  {warm * budget / 1e6:.2f}MB  ({warm:.2%} 지점)")

        model.train()
        torch.cuda.reset_peak_memory_stats()
        saved_at = None
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
                    lr = lr_fn(progress, budget, args.lr, warm)
                    for g in opt.param_groups:
                        g["lr"] = lr
                    gn = group_norms()          # 클리핑 전에 잰다
                    dmg_g = dmg_before = None
                    if dmg_ids is not None and emb_w.grad is not None:
                        dmg_g = float(torch.linalg.vector_norm(emb_w.grad[dmg_ids]))
                        # AdamW 의 실제 이동량은 기울기 크기와 비례하지 않는다.
                        # 스텝 전후를 직접 빼서 잰다 (손상 행만이라 싸다).
                        dmg_before = emb_w.data[dmg_ids].clone()
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0))
                    opt.step()
                    dmg_ratio = None
                    if dmg_before is not None:
                        moved = float(torch.linalg.vector_norm(
                            emb_w.data[dmg_ids] - dmg_before))
                        scale = float(torch.linalg.vector_norm(dmg_before))
                        dmg_ratio = moved / scale if scale > 0 else None
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
                                  grad_norm_ffn=gn["ffn"],
                                  grad_norm_head=gn["head"],
                                  grad_norm_dmg=dmg_g,
                                  upd_w_ratio_dmg=dmg_ratio)
                        for lang, v in cur.items():
                            run.log("lm_metrics", checkpoint=f"step{step}",
                                    tokens_seen=tokens_seen,
                                    raw_bytes_seen=int(raw_bytes), split="dev",
                                    domain=lang, n_bytes=None, total_nll=None,
                                    bpb=round(v, 6), bpc=None, token_ppl=None)

                    # 중간 저장. 긴 run 의 중간 지점이 다른 실험의 입력이 된다
                    # (P3-F 가 A 의 168.5MB 체크포인트를 쓴다).
                    if args.save_at and raw_bytes >= args.save_at and not saved_at:
                        saved_at = save_checkpoint(
                            model, tokenizer, run_id,
                            f"_at{int(args.save_at / 1e6)}mb", int(raw_bytes))

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
            sha = save_checkpoint(model, tokenizer, run_id, "", int(raw_bytes))
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
