"""R1 의 K 를 손상 배율로 맞춘다 (P2, 학습 없음).

    .conda/python.exe scripts/calibrate_k.py --ks 0 100 200 300 500 1000

## 왜 보정이 필요한가

Day 0 스모크에서 K=1,000 이 ko BPB 를 1.1366 -> 3.5478 (3.12배) 로 만들었다.
1차 T2b 의 손상 배율은 **2.058배** (2.3803 / 1.1569) 였다. 손상 크기가 다르면
R1 의 회복률을 1차의 65.4% 와 나란히 놓을 수 없다 — 회복률의 분모가 초기
손상이기 때문이다.

그래서 K 를 **결과가 아니라 손상 배율로** 고른다. 여기에 학습은 없다. 고른 K 를
docs/PLAN.md 에 적고 나서 R1 을 시작한다. 학습 결과를 보고 K 를 다시 고르면
사전 등록이 아니다 (RULES 14).

## 무엇을 재는가

언어 셋을 다 잰다. T2b 의 손상은 한국어에 몰려 있어 영어가 거의 그대로였다
(0.8115 -> 0.8126). count_ko 상위 K개를 망가뜨리면 영어·코드도 함께 무너지는데
(K=1,000 에서 +27.6% / +29.7%), **그 부수 피해까지 T2b 와 비슷해야** R1 이
같은 실험이 된다. ko 배율만 맞추고 영어가 무너져 있으면 다른 손상이다.

모델을 한 번만 올리고 K 마다 원본 embedding 을 되돌려 다시 망가뜨린다.
K 하나당 산출물 1.2GB 를 저장하지 않으려는 것이다.

**K=0 을 반드시 넣는다.** 기준선을 인용하지 않고 같은 코드·같은 예산으로 직접
재야 배율이 내부적으로 일관된다 (SPEC_P2 5절 "널 대조군 먼저").
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402

from src.evaluation.bpb import CORPORA, evaluate, token_byte_length  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

# 1차 T2b 의 손상 배율. 2.3803 / 1.1569 (reports/tables/precpt_bpb.md)
TARGET_RATIO = 2.058


def main(argv: list | None = None) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(ROOT / "scripts"))
    import damage_rows as dr

    ap = argparse.ArgumentParser(description="R1 의 K 를 손상 배율로 보정")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--base-revision",
                    default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--how", default="mean", choices=dr.HOWS)
    ap.add_argument("--count-min", type=int, default=0,
                    help="행당 노출 하한 (count_ko). P3-D 는 T2b 새 토큰의 "
                         "노출 중앙값 143 의 10배 안 구간을 쓴다")
    ap.add_argument("--count-max", type=int, default=0,
                    help="행당 노출 상한 (count_ko)")
    ap.add_argument("--stats-tag", default="v1")
    ap.add_argument("--max-bytes", type=int, default=2_000_000,
                    help="언어별 원문 바이트. 1차 Pre-CPT 와 같은 2MB 가 기본")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="kcal")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, revision=args.base_revision, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    vocab = tok.get_vocab()
    id2tok = {i: t for t, i in vocab.items()}
    parents = dr.merge_parents(tok)
    stats = dr.load_stats(args.stats_tag)

    # 원본을 fp32 로 붙들어 둔다. bf16 으로 되돌리면 K 마다 반올림이 쌓여
    # 나중 K 가 앞 K 의 손상을 물려받는다.
    origin = model.get_input_embeddings().weight.detach().float().cpu().numpy().copy()

    blen: dict = {}
    from tokenizers.pre_tokenizers import ByteLevel
    alphabet = set(ByteLevel.alphabet())

    def byte_len_fn(ids: list) -> int:
        total = 0
        for i in ids:
            v = blen.get(i)
            if v is None:
                v = token_byte_length(tok.convert_ids_to_tokens(int(i)), alphabet)
                blen[i] = v
            total += v
        return total

    config = {"base": args.base, "base_revision": args.base_revision,
              "ks": list(args.ks), "how": args.how, "max_bytes": args.max_bytes,
              "seed": args.seed, "target_ratio": TARGET_RATIO,
              "select_rule": "count_ko_desc_unprotected_has_parents",
              "count_min": args.count_min, "count_max": args.count_max,
              "purpose": "p2_r1_k_calibration"}
    run_id = make_run_id("eval", "kcal", args.how, args.tag)
    base_dir = ROOT / "data" / "interim" / "docs"
    backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

    with RunContext(run_id, phase="eval", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check) as run:
        print(f"손상 {args.how}  목표 배율 {TARGET_RATIO}  "
              f"언어별 {args.max_bytes / 1e6:.1f}MB")
        print(f"{'K':>7} {'ko':>9} {'배율':>7} {'en':>9} {'코드':>9}"
              f" {'평균이동':>9}")
        print("-" * 56)

        results = []
        ko0 = None
        for k in args.ks:
            emb = origin.copy()
            info = {"mean_move": 0.0}
            if k > 0:
                rows = dr.pick_rows(stats, parents, id2tok, k,
                                    args.count_min, args.count_max)
                info = dr.damage(emb, rows, args.how, parents, vocab, args.seed)
            with torch.no_grad():
                model.get_input_embeddings().weight.copy_(
                    torch.from_numpy(emb).to(torch.bfloat16))
            if model.config.tie_word_embeddings:
                model.tie_weights()

            bpb = {}
            with sdpa_kernel(backends):
                for lang, pattern in CORPORA:
                    path = base_dir / pattern.format(split="dev")
                    if not path.exists():
                        continue
                    m = evaluate(model, tok, path, args.max_bytes,
                                 args.seq_len, device, byte_len_fn)
                    bpb[lang] = m["bpb"]
                    run.log("lm_metrics", checkpoint=f"k{k}",
                            tokens_seen=0, raw_bytes_seen=0, split="dev",
                            domain=lang, n_bytes=m["n_bytes"],
                            total_nll=round(m["total_nll"], 4),
                            bpb=round(m["bpb"], 6), bpc=round(m["bpc"], 6),
                            token_ppl=round(m["token_ppl"], 6))
            if ko0 is None:
                ko0 = bpb["ko"]
            ratio = bpb["ko"] / ko0
            results.append((k, bpb, ratio, info["mean_move"]))
            print(f"{k:>7,} {bpb['ko']:>9.4f} {ratio:>7.3f} {bpb['en']:>9.4f}"
                  f" {bpb.get('code', float('nan')):>9.4f} {info['mean_move']:>9.5f}")

        pick = min((r for r in results if r[0] > 0),
                   key=lambda r: abs(r[2] - TARGET_RATIO), default=None)
        if pick:
            k, bpb, ratio, _ = pick
            en_pct = 100 * (bpb["en"] / results[0][1]["en"] - 1)
            print(f"\n목표 {TARGET_RATIO} 에 가장 가까운 것: K={k:,} (배율 {ratio:.3f})")
            print(f"  이때 영어는 {en_pct:+.1f}% — 1차 T2b 는 +0.1% 였다")
            if abs(en_pct) > 5:
                print("  **부수 피해가 T2b 와 다르다.** ko 배율만 맞은 것이고,")
                print("  같은 손상이라고 말할 수 없다. PLAN 에 한계로 적어라")
            run.note = (f"k_calibration how={args.how} pick={k} ratio={ratio:.3f} "
                        f"en_pct={en_pct:+.1f} target={TARGET_RATIO}")
        run.extra["model"] = args.base
        run.extra["model_revision"] = args.base_revision
        run.extra["init_method"] = f"damage_{args.how}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
