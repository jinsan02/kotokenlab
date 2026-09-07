"""토크나이저를 그대로 두고 embedding 행 K개만 망가뜨린다 (P2 의 R1·R2·R3).

    .conda/python.exe scripts/damage_rows.py --k 10000 --how mean --name dmg_mean_k10000
    .conda/python.exe scripts/damage_rows.py --k 10000 --how permute --name dmg_perm_k10000

1차는 "토큰을 치환했더니 회복이 65% 에서 멎었다" 였다. 그런데 실제로 일어난
일은 **임베딩 행 K개가 망가진 상태에서 CPT 를 했다** 이고, 토크나이저 치환은
그 손상을 만드는 한 가지 방법일 뿐이다. 토크나이저를 안 건드리고 같은 벽이
나오면 65% 는 한국어 토크나이저의 성질이 아니라 tied-embedding 언어모델의
일반 성질이다 (docs/SPEC_P2.md §0).

## 어느 행을 망가뜨리는가 — 사전 등록된 규칙

**코퍼스 한국어 빈도(`count_ko`) 내림차순 상위 K개**, 아래 둘을 뺀 것.

    protected      byte fallback 256개와 special (RULES 12c)
    부모 없는 토큰  merges 에 `left+right` 로 나타나지 않는 원자 토큰.
                   부품 평균을 정의할 수 없다

빈도로 고르는 이유가 핵심이다. T2b 가 치환한 30,000개는 **코퍼스가 거의 안 쓰던**
토큰이었고(T2a 가 공짜였던 이유), 치환 뒤 그 자리에 들어간 것은 자주 발화하는
한국어 토큰이었다. 즉 1차의 손상은 "자주 쓰이는 행이 틀린 벡터를 들고 있다"
이다. 여기서 안 쓰이는 행을 골라 망가뜨리면 손상이 잡히지도 않고 회복률도
무의미해진다 — 지루한 이유로 R1 이 실패한다.

**세 손상 종류가 같은 행 집합을 쓴다.** 안 그러면 R2 가 손상 종류가 아니라
행 선택을 비교하게 된다. 그래서 부모 없는 토큰은 `mean` 만이 아니라 셋 다에서
뺀다.

## 손상 종류 (R2)

    random   살아남은 행의 (평균, 표준편차) 난수. 1차 E0 과 같다.
             의미를 **전혀** 안 남긴다
    mean     merges 가 말하는 두 부모 행의 평균. 1차 E1 과 같다.
             방향은 대충 맞고 노름은 짧아진다
    permute  K개 행을 자기들끼리 섞는다. 분포도 노름도 **정확히** 보존되고
             배정만 틀린다. 1차에 없던 조건이고, "노름이 틀려서" 와
             "배정이 틀려서" 를 가르는 유일한 대조군이다

산출물은 artifacts/models/<name>/ 에 두고 tools/register_artifact.py 로 등록한다.
토크나이저는 원본 그대로 복사되므로 C0 와 토큰 수가 같다 — 등원문이 곧 등토큰이다.
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

from src.surgery.init_random import init_random, reference_stats  # noqa: E402
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

HOWS = ("random", "mean", "permute")


def merge_parents(tokenizer) -> dict:
    """토큰 문자열 -> (왼쪽 부모, 오른쪽 부모).

    BPE 의 merges 는 `left right` 목록이고, 그 병합이 만든 토큰은 left+right 다.
    같은 문자열을 만드는 규칙이 여럿이면 **먼저 오는 것** 을 쓴다 — merges 는
    우선순위 순이라 실제 인코딩에서 이기는 규칙이 앞에 있다.
    """
    model = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
    out: dict = {}
    for pair in model["merges"]:
        left, right = pair.split(" ") if isinstance(pair, str) else pair
        out.setdefault(left + right, (left, right))
    return out


def load_stats(tag: str) -> list:
    """token_stats.tsv 를 읽는다.

    `token` 컬럼은 json.dumps 로 적혀 있다 (탭·개행 안전). 디코드하지 않고
    merges 와 대조하면 따옴표 때문에 거의 다 안 맞는다 — 후보가 14개로 줄어
    조용히 잘못된 행 집합이 나온다. 실제로 한 번 그랬다.
    """
    path = ROOT / "artifacts" / "vocab_stats" / tag / "token_stats.tsv"
    lines = path.read_text(encoding="utf-8").split(chr(10))
    header = lines[0].split(chr(9))
    out = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        row = dict(zip(header, ln.split(chr(9))))
        row["token"] = json.loads(row["token"])
        out.append(row)
    return out


def pick_rows(stats: list, parents: dict, id2tok: dict, k: int) -> list:
    """사전 등록된 규칙대로 K개 행을 고른다.

    토큰 문자열은 TSV 가 아니라 **토크나이저** 를 진실로 삼아 token_id 로
    되짚는다. 두 곳이 어긋나면 어긋난 채로 조용히 진행되기 때문이다.
    """
    cand = []
    for r in stats:
        if r["is_protected"] != "0":
            continue
        tid = int(r["token_id"])
        token = id2tok.get(tid)
        if token is None or token != r["token"]:
            continue
        if token not in parents:
            continue
        cand.append(r)
    cand.sort(key=lambda r: (-int(r["count_ko"]), int(r["token_id"])))
    if len(cand) < k:
        raise SystemExit(
            f"후보가 {len(cand):,}개뿐이라 K={k:,} 를 채울 수 없다. "
            "K 를 줄이거나 선택 규칙을 다시 등록해라")
    return cand[:k]


def damage(emb: np.ndarray, rows: list, how: str, parents: dict,
           vocab: dict, seed: int) -> dict:
    """emb 을 제자리에서 망가뜨리고 무엇을 했는지 돌려준다."""
    ids = np.array([int(r["token_id"]) for r in rows])
    before = emb[ids].copy()
    info = {"n": len(ids)}

    if how == "random":
        keep = np.array(sorted(set(range(emb.shape[0])) - set(ids.tolist())))
        _, ref_std = reference_stats(emb, keep)
        init_random(emb, ids.tolist(), seed=seed)
        info["ref_std"] = ref_std

    elif how == "permute":
        # 같은 벡터 집합을 자기들끼리 다시 배정한다. 고정점(자기 자신에게
        # 남는 행)이 있으면 그만큼 손상이 덜 들어가므로 없앤다.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        for _ in range(64):
            fixed = np.flatnonzero(order == np.arange(len(ids)))
            if fixed.size == 0:
                break
            # 고정점을 이웃과 맞바꾼다
            for i in fixed:
                j = (i + 1) % len(ids)
                order[i], order[j] = order[j], order[i]
        info["fixed_points"] = int((order == np.arange(len(ids))).sum())
        emb[ids] = before[order]

    else:  # mean — 1차 E1 과 같은 연산
        missing = 0
        for row_i, r in enumerate(rows):
            left, right = parents[r["token"]]
            li, ri = vocab.get(left), vocab.get(right)
            if li is None or ri is None:
                missing += 1
                continue
            emb[ids[row_i]] = (emb[li] + emb[ri]) / 2.0
        if missing:
            raise AssertionError(f"부모 ID 를 못 찾은 행 {missing}개")
        info["missing"] = missing

    moved = float(np.linalg.norm(emb[ids] - before, axis=1).mean())
    info["mean_move"] = moved
    info["std_before"] = float(before.std())
    info["std_after"] = float(emb[ids].std())
    return info


def main(argv: list | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="임베딩 행 K개만 망가뜨린다")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--base-revision",
                    default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--how", choices=HOWS, required=True)
    ap.add_argument("--name", required=True, help="artifacts/models/<name>/")
    ap.add_argument("--stats-tag", default="v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="damage")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    out_dir = ROOT / "artifacts" / "models" / args.name
    if out_dir.exists():
        raise SystemExit(f"이미 있다: {out_dir}\n지우거나 --name 을 바꿔라")

    config = {"base": args.base, "base_revision": args.base_revision,
              "k": args.k, "how": args.how, "stats_tag": args.stats_tag,
              "seed": args.seed, "select_rule": "count_ko_desc_unprotected_has_parents",
              "purpose": "p2_r1r2r3_row_damage"}
    run_id = make_run_id("surgery", args.name, args.tag, seed=args.seed)

    with RunContext(run_id, phase="surgery", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check) as run:
        print(f"[1/4] 적재  {args.base}")
        tok = AutoTokenizer.from_pretrained(args.base, revision=args.base_revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.base, revision=args.base_revision, dtype=torch.float32)
        emb = model.get_input_embeddings().weight.detach().cpu().numpy().copy()
        vocab = tok.get_vocab()
        print(f"      embedding {emb.shape}  tie={model.config.tie_word_embeddings}")

        print("[2/4] 행 선택")
        parents = merge_parents(tok)
        stats = load_stats(args.stats_tag)
        id2tok = {i: t for t, i in vocab.items()}
        rows = pick_rows(stats, parents, id2tok, args.k)
        pool = sum(1 for r in stats
                   if r["is_protected"] == "0"
                   and id2tok.get(int(r["token_id"])) == r["token"]
                   and r["token"] in parents)
        print(f"      후보 {pool:,}개 중 상위 {args.k:,}개")
        print(f"      가장 많이 쓰이는 것 count_ko={rows[0]['count_ko']}  "
              f"가장 적은 것 count_ko={rows[-1]['count_ko']}")

        print(f"[3/4] 손상  {args.how}")
        info = damage(emb, rows, args.how, parents, vocab, args.seed)
        print(f"      평균 이동 거리 {info['mean_move']:.5f}  "
              f"표준편차 {info['std_before']:.5f} -> {info['std_after']:.5f}")
        for key in ("fixed_points", "ref_std"):
            if key in info:
                print(f"      {key} = {info[key]}")

        print("[4/4] 저장")
        with torch.no_grad():
            model.get_input_embeddings().weight.copy_(torch.from_numpy(emb))
        # 토크나이저를 안 바꿨으므로 tie 도 그대로다. 여기서 풀면 R1 이
        # 손상과 tie 를 동시에 바꾸는 실험이 된다 — 그건 Q7 의 몫이다.
        if model.config.tie_word_embeddings:
            model.tie_weights()
        model = model.to(torch.bfloat16)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        sha = sha256_file(out_dir / "model.safetensors")
        print(f"      {out_dir}")
        print(f"      model_sha256 = {sha}")

        run.extra["model"] = args.base
        run.extra["model_revision"] = args.base_revision
        run.extra["vocab_size"] = int(emb.shape[0])
        run.extra["init_method"] = f"damage_{args.how}"
        run.note = (f"damage {args.how} k={args.k} pool={pool} "
                    f"move={info['mean_move']:.5f} "
                    f"std={info['std_before']:.5f}->{info['std_after']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
