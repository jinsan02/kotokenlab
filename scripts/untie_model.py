"""tie_word_embeddings 를 끊는다 (P2 의 R4 = Q7). 스펙 §20 의 통제축 밖 조작이다.

    .conda/python.exe scripts/untie_model.py \
        --model Qwen/Qwen2.5-0.5B --revision 060db64... --name untied_c0
    .conda/python.exe scripts/untie_model.py \
        --model artifacts/models/kot2b_v2_n30000_mean --name untied_t2b_mean

**수술 스크립트가 아니라 후처리다.** run_surgery.py 안에 --untie 분기를 넣지
않은 이유가 있다. 넣으면 untied 판을 얻을 때마다 수술을 다시 돌려야 하고, 그
순간 embedding 행렬이 1차 산출물과 비트 동일하다는 보장을 잃는다. 여기서는
**이미 있는 산출물을 입력으로 받으므로** 행렬이 물리적으로 같은 바이트다.
그래서 untied-C0 과 untied-T2b 가 정확히 같은 연산의 두 입력이 되고,
대조군 대칭이 공짜로 성립한다.

config 로 끊으면 안 된다
    AutoModelForCausalLM.from_pretrained(..., tie_word_embeddings=False) 는
    transformers 5.16 에서 lm_head.weight 를 **체크포인트에 없는 키** 로 보고
    무작위로 채운다 (std 0.0200 vs embedding 0.0156). 그 경로로 만든 untied-C0
    은 C0 이 아니라 출력층이 파괴된 C0 이고, Q7 은 tie 가 아니라 다른 것을
    재게 된다. tied 로 적재한 뒤 사본을 떼는 순서만 forward 를 보존한다.

무엇이 보존되고 무엇이 바뀌는가
    보존   forward. 같은 입력의 logits 가 tied 원본과 비트 단위로 같다.
           따라서 학습 전 BPB 도 같다 — 그것이 Q7 의 S1 정합성 게이트다.
    바뀜   파라미터 수. 0.5B 에서 494,032,768 -> 630,167,424 (+27.6%).
           이 증가분 때문에 **untied-C0 대조군 없이는 아무 말도 할 수 없다**
           (docs/PLAN.md "Q7").

산출물은 artifacts/models/<name>/ 에 두고 sha256 만 원장에 남긴다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402


def untie(model, torch) -> None:
    """lm_head 를 embedding 의 사본으로 떼어 낸다.

    detach().clone() 이라야 저장소가 갈라진다. 안 그러면 safetensors 가 공유
    텐서로 보고 하나만 쓰거나 거부한다.

    tie_weights(recompute_mapping=True) 를 부르는 이유는 config 를 바꾼 뒤
    내부 매핑을 다시 계산시키기 위해서다. config 가 False 이므로 재계산 결과가
    비고, 따라서 아무것도 다시 묶이지 않는다.
    """
    if not getattr(model.config, "tie_word_embeddings", False):
        raise SystemExit("이미 untied 다. 끊을 것이 없다")
    model.config.tie_word_embeddings = False
    model.lm_head.weight = torch.nn.Parameter(
        model.get_input_embeddings().weight.detach().clone())
    model.tie_weights(recompute_mapping=True)


def main(argv: list | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="tie_word_embeddings 를 끊는다")
    ap.add_argument("--model", required=True, help="hub repo 또는 로컬 산출물 경로")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--name", required=True, help="artifacts/models/<name>/")
    ap.add_argument("--tag", default="untie")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    out_dir = ROOT / "artifacts" / "models" / args.name
    if out_dir.exists():
        raise SystemExit(f"이미 있다: {out_dir}\n지우거나 --name 을 바꿔라")

    kw = {"revision": args.revision} if args.revision else {}
    config = {"model": args.model, "revision": args.revision,
              "name": args.name, "purpose": "q7_untie"}
    run_id = make_run_id("surgery", args.name, args.tag, seed=args.seed)

    with RunContext(run_id, phase="surgery", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check) as run:
        print(f"[1/3] 적재  {args.model}")
        tok = AutoTokenizer.from_pretrained(args.model, **kw)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, **kw)
        n_tied = sum(p.numel() for p in model.parameters())
        emb = model.get_input_embeddings().weight
        print(f"      tie={model.config.tie_word_embeddings}  "
              f"embedding {tuple(emb.shape)}  파라미터 {n_tied:,}")

        # 끊기 전의 logits 를 잡아 둔다. 이걸 안 하면 "forward 가 보존됐다" 를
        # 주장만 하고 확인은 안 하는 셈이 된다. 이 검사가 Q7 의 S1 게이트가
        # 딛고 서는 사실이다.
        probe = torch.tensor([[100, 200, 300, 400, 500]])
        with torch.no_grad():
            before = model(probe).logits.clone()

        print("[2/3] lm_head 분리")
        untie(model, torch)
        head = model.lm_head.weight
        assert emb.data_ptr() != head.data_ptr(), "저장소가 아직 공유 상태다"
        assert torch.equal(emb, head), "사본이 원본과 다르다"
        n_untied = sum(p.numel() for p in model.parameters())
        added = n_untied - n_tied
        print(f"      파라미터 {n_tied:,} -> {n_untied:,} "
              f"(+{added:,}, +{100 * added / n_tied:.1f}%)")

        with torch.no_grad():
            after = model(probe).logits
        if not torch.equal(before, after):
            raise AssertionError(
                "logits 가 바뀌었다. untie 가 forward 를 보존하지 못했다 — "
                "이 산출물로 Q7 을 돌리면 tie 가 아니라 다른 것을 재게 된다")
        print("      logits 비트 단위 동일 — forward 보존 확인")

        print("[3/3] 저장")
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))

        # 저장이 실제로 lm_head 를 실었는지 본다. 여기서 빠지면 재적재할 때
        # 조용히 무작위 초기화로 되돌아가고, 그 사실이 학습 몇 시간 뒤에야
        # BPB 로 드러난다.
        from safetensors import safe_open
        with safe_open(str(out_dir / "model.safetensors"), "pt") as fh:
            keys = list(fh.keys())
        if "lm_head.weight" not in keys:
            raise AssertionError(
                f"safetensors 에 lm_head.weight 가 없다 (텐서 {len(keys)}개). "
                "config 만 바뀌고 가중치가 안 실린 상태다")
        sha = sha256_file(out_dir / "model.safetensors")
        print(f"      {out_dir}")
        print(f"      텐서 {len(keys)}개  model_sha256 = {sha}")

        run.extra["model"] = args.model
        run.extra["vocab_size"] = int(emb.shape[0])
        run.extra["init_method"] = "untie_clone"
        run.note = (f"untie {args.model} -> {args.name} "
                    f"params={n_tied}->{n_untied} tensors={len(keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
