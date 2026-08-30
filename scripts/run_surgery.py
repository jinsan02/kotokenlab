"""토크나이저 교체 후 embedding 수술 (스펙 §20~22).

    .conda/python.exe scripts/run_surgery.py --tokenizer kot2b_v2_n30000 --init mean
    .conda/python.exe scripts/run_surgery.py --tokenizer kot2a_v1_n30000 --init none

무엇을 분리하는가
    스펙 §20 이 요구하는 것은 효과의 분리다. 토크나이저를 갈아끼우고 바로 CPT 하면
    토크나이저 효과 / 초기화 효과 / 정렬 효과 / CPT 효과가 섞여서 무엇이 기여했는지
    말할 수 없다. 이 스크립트는 **초기화까지만** 하고 멈춘다. 나온 체크포인트를
    학습 없이 평가하면 (Pre-CPT) 초기화 방법의 효과만 따로 보인다.

두 조건이 다르게 처리된다
    T2a  vocab 이 줄어 행렬을 다시 만든다. 새로 생기는 자리가 없으므로 --init none.
    T2b  크기가 같고 치환된 자리만 채운다. --init random|mean|weighted.

tie_word_embeddings 를 풀지 않는다. embedding 하나가 입력이자 출력이므로
lm_head 를 따로 만들면 파라미터 수가 달라져 T2a 와의 비교가 깨진다.
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

import numpy as np  # noqa: E402

from src.surgery.init_mean import component_ids, init_mean  # noqa: E402
from src.surgery.init_random import init_random, reference_stats  # noqa: E402
from src.surgery.init_weighted import SCHEMES, init_weighted  # noqa: E402
from src.surgery.resize import (  # noqa: E402
    assert_filled, id_mapping, padded_vocab_size, rearrange)
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

BASE_REPO = "Qwen/Qwen2.5-0.5B"
BASE_REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def load_token_stats(tag: str, vocab_size: int) -> tuple:
    """analyze_vocab 산출물에서 (바이트 길이, 코퍼스 빈도) 배열을 만든다."""
    path = ROOT / "artifacts" / "vocab_stats" / tag / "token_stats.tsv"
    nbytes = np.ones(vocab_size, dtype=np.int64)
    freq = np.zeros(vocab_size, dtype=np.int64)
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip(chr(10)).split(chr(9))
        ci = {c: i for i, c in enumerate(header)}
        for line in fh:
            c = line.rstrip(chr(10)).split(chr(9))
            tid = int(c[ci["token_id"]])
            if tid >= vocab_size:
                continue
            nbytes[tid] = max(len(json.loads(c[ci["token"]]).encode("utf-8")), 1)
            freq[tid] = int(c[ci["count_total"]])
    return nbytes, freq


def main(argv: list | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="embedding 수술")
    ap.add_argument("--tokenizer", required=True, help="artifacts/tokenizers/ 아래 이름")
    ap.add_argument("--init", required=True,
                    choices=("none", "random", "mean", "weighted"))
    ap.add_argument("--weight-scheme", default="freq", choices=SCHEMES)
    ap.add_argument("--stats-tag", default="v1")
    ap.add_argument("--rescale", action="store_true",
                    help="채운 행의 표준편차를 살아남은 행에 맞춘다")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    tok_dir = ROOT / "artifacts" / "tokenizers" / args.tokenizer
    if not tok_dir.exists():
        print(f"{tok_dir} 가 없다", file=sys.stderr)
        return 1
    id_map_blob = json.loads((tok_dir / "id_map.json").read_text(encoding="utf-8"))
    mode = id_map_blob["mode"]

    if mode == "t2a" and args.init != "none":
        print("T2a 는 새로 생기는 자리가 없다. --init none 이어야 한다.",
              file=sys.stderr)
        return 2
    if mode == "t2b" and args.init == "none":
        print("T2b 는 치환된 자리를 채워야 한다. --init none 은 NaN 을 남긴다.",
              file=sys.stderr)
        return 2

    version = f"{args.tokenizer}_{args.init}"
    if args.init == "weighted":
        version += f"-{args.weight_scheme}"
    if args.rescale:
        version += "-rescaled"
    config = {
        "base_repo": BASE_REPO, "base_revision": BASE_REV,
        "tokenizer": args.tokenizer, "mode": mode, "init": args.init,
        "weight_scheme": args.weight_scheme if args.init == "weighted" else None,
        "stats_tag": args.stats_tag, "seed": args.seed,
        "rescale": args.rescale,
    }
    run_id = make_run_id("surgery", args.tokenizer,
                         args.init + ("rescaled" if args.rescale else ""),
                         seed=args.seed)

    with RunContext(run_id, phase="surgery", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check) as run:
        print(f"[1/4] 모델·토크나이저 적재  ({mode}, init={args.init})")
        base_tok = AutoTokenizer.from_pretrained(BASE_REPO, revision=BASE_REV)
        new_tok = AutoTokenizer.from_pretrained(str(tok_dir))
        model = AutoModelForCausalLM.from_pretrained(
            BASE_REPO, revision=BASE_REV, torch_dtype=torch.float32)
        old_emb = model.get_input_embeddings().weight.detach().cpu().numpy()
        print(f"      원본 embedding {old_emb.shape}  tie={model.config.tie_word_embeddings}")

        print("[2/4] ID 매핑과 행 재배치")
        mapping = id_mapping(base_tok, new_tok)
        n_tokens = len(new_tok.get_vocab())
        if mode == "t2b":
            # 크기 유지가 T2b 의 정의다. 다시 계산하면 Qwen 이 여분으로 둔 256행을
            # 잘라내 형상이 바뀌고, "vocab 을 늘리지 않았다" 는 주장이 깨진다.
            new_size = old_emb.shape[0]
        else:
            new_size = padded_vocab_size(max(mapping.size, n_tokens))
        new_emb, todo = rearrange(old_emb, mapping, new_size)
        moved = int((mapping >= 0).sum())
        renumbered = int(((mapping >= 0) & (mapping != np.arange(mapping.size))).sum())
        pad_rows = set(range(mapping.size, new_size))
        print(f"      토큰 {n_tokens:,}  패딩 포함 {new_size:,} (128 배수)")
        print(f"      옮긴 행 {moved:,}  그중 번호가 바뀐 것 {renumbered:,}")
        print(f"      채워야 할 행 {len(todo) - len(pad_rows):,} (+패딩 {len(pad_rows):,})")

        print(f"[3/4] 초기화  {args.init}")
        fill = [i for i in todo if i not in pad_rows]
        if args.init == "none":
            if fill:
                raise AssertionError(f"init none 인데 빈 행이 {len(fill):,}개 있다")
        elif args.init == "random":
            init_random(new_emb, fill, seed=args.seed)
        else:
            comps = component_ids(id_map_blob["map"], base_tok.get_vocab())
            missing = set(fill) - set(comps)
            if missing:
                raise AssertionError(
                    f"부품 정보가 없는 행 {len(missing):,}개: {sorted(missing)[:5]}")
            if args.init == "mean":
                init_mean(new_emb, comps, old_emb)
            else:
                nbytes, freq = load_token_stats(args.stats_tag, old_emb.shape[0])
                init_weighted(new_emb, comps, old_emb, nbytes, freq,
                              scheme=args.weight_scheme)
        # 부품 평균은 노름이 줄어든다 — 두 벡터가 같은 방향이 아니면 평균의
        # 길이가 원본보다 짧다. 그대로 두면 E1 이 나쁠 때 "평균이 틀려서" 인지
        # "노름이 작아서" 인지 구별할 수 없다. E0 에서 분포를 맞춘 것과 같은
        # 이유로, 방향은 두고 크기만 살아남은 행에 맞추는 선택지를 둔다.
        if args.rescale and fill:
            kept = np.array(sorted(set(range(new_size)) - pad_rows - set(fill)))
            _, ref_std = reference_stats(new_emb, kept)
            _, cur_std = reference_stats(new_emb, np.asarray(fill))
            if cur_std > 0:
                new_emb[np.asarray(fill)] *= (ref_std / cur_std)
                print(f"      노름 보정  표준편차 {cur_std:.5f} -> {ref_std:.5f}")

        # 패딩 행은 학습에 쓰이지 않지만 NaN 을 남기면 저장·적재가 깨진다
        new_emb[sorted(pad_rows)] = 0.0
        assert_filled(new_emb)
        m, s = reference_stats(new_emb, np.array(sorted(set(range(new_size)) - pad_rows)))
        print(f"      채운 뒤 전체 평균 {m:+.5f}  표준편차 {s:.5f}")
        if fill:
            fm, fs = reference_stats(new_emb, np.asarray(fill))
            print(f"      채운 행만    평균 {fm:+.5f}  표준편차 {fs:.5f}")

        print("[4/4] 모델에 반영하고 저장")
        model.resize_token_embeddings(new_size)
        with torch.no_grad():
            model.get_input_embeddings().weight.copy_(torch.from_numpy(new_emb))
        model.config.vocab_size = new_size

        # T2a 는 번호를 다시 매기므로 config 의 특수 토큰 ID 가 옛 자리를 가리킨다.
        # 실제로 eos 가 151,643 을 가리키는데 vocab 은 121,728 까지였다 — 생성이
        # 깨지고 CPT 의 문서 경계도 어긋난다. BPB 는 특수 토큰을 안 써서 영향이
        # 없었지만, 산출물은 맞아야 한다. 토크나이저가 아는 번호로 다시 적는다.
        # 새 토크나이저의 속성만 보면 안 된다. Qwen 은 tokenizer 에 BOS 가 없는데
        # config.bos_token_id 는 151,643 을 들고 있어서, 그 경로로는 범위 밖 값이
        # 그대로 남는다. 옛 ID -> 토큰 문자열 -> 새 ID 로 되짚는다.
        base_id2tok = {i: s for s, i in base_tok.get_vocab().items()}
        new_vocab = new_tok.get_vocab()
        moved_ids = {}
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            old_id = getattr(model.config, attr, None)
            new_id = getattr(new_tok, attr, None)
            if new_id is None and old_id is not None:
                new_id = new_vocab.get(base_id2tok.get(old_id))
            if new_id is not None and old_id != new_id:
                setattr(model.config, attr, new_id)
                if getattr(model, "generation_config", None) is not None:
                    setattr(model.generation_config, attr, new_id)
                moved_ids[attr] = (old_id, new_id)
            elif old_id is not None and old_id >= new_size:
                raise AssertionError(
                    f"config.{attr}={old_id} 가 vocab {new_size:,} 밖인데 "
                    "옮길 자리를 못 찾았다")
        if moved_ids:
            for attr, (a, b) in moved_ids.items():
                print(f"      특수 토큰 재지정  {attr}  {a} -> {b}")

        if model.config.tie_word_embeddings:
            model.tie_weights()
        model = model.to(torch.bfloat16)

        out_dir = ROOT / "artifacts" / "models" / version
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        new_tok.save_pretrained(str(out_dir))
        n_params = sum(p.numel() for p in model.parameters())
        sha = sha256_file(out_dir / "model.safetensors")
        print(f"      {out_dir}")
        print(f"      파라미터 {n_params:,}  model_sha256 = {sha}")

        run.extra["tokenizer_version"] = args.tokenizer
        run.extra["vocab_size"] = new_size
        run.extra["init_method"] = args.init
        run.note = (f"{mode} init={args.init}"
                    f"{'-rescaled' if args.rescale else ''} vocab={new_size} "
                    f"filled={len(fill)} params={n_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
