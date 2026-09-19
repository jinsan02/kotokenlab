"""CPT 를 마친 임베딩 행을 원본 몸통에 옮겨 심는다 — P3-E "오라클 초기화 상한".

    .conda/python.exe scripts/transplant_rows.py \
        --donor artifacts/models/cpt_t2b_mean_main_seed42 \
        --id-map artifacts/tokenizers/kot2b_v2_n30000/id_map.json \
        --name oracle_t2b_main --tag p3e

## 무엇을 만드나

```
몸통 · 나머지 행   수술 직후의 T2b 모델 (artifacts/models/kot2b_v2_n30000_mean)
새 30,000행        **CPT 를 마친** T2b 의 같은 행 (donor)
토크나이저         T2b 그대로
```

즉 "초기화를 이보다 잘할 수는 없는 값" 을 넣은 조건이다. 여기서 CPT 를 돌려
회복이 얼마나 빨라지는지 보면, **초기화가 격차의 얼마를 설명하는가** 의 상한이
나온다. ZeTT 를 못 돌리는 대신 그 상한을 재는 것이 P3-E 다
([`docs/SPEC_P3.md`](../docs/SPEC_P3.md) §2 E).

## 옮기지 않는 것

**새 행 말고는 한 바이트도 건드리지 않는다.** 몸통이 함께 옮겨 오면 "초기화
효과" 가 아니라 "CPT 된 모델을 복사한 것" 이 되어 실험이 성립하지 않는다.
그래서 이식 뒤 **바뀌지 않아야 할 텐서가 원본과 비트 단위로 같은지 검사** 하고,
어긋나면 저장하지 않는다.

tie_word_embeddings 라 출력 행도 같이 따라간다 — 끊지 않는다. 끊으면 이식과
tie 를 동시에 바꾸는 실험이 된다 (그건 Q7 의 몫이었다).
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

from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

DEFAULT_BASE = ROOT / "artifacts" / "models" / "kot2b_v2_n30000_mean"
DEFAULT_MAP = ROOT / "artifacts" / "tokenizers" / "kot2b_v2_n30000" / "id_map.json"


def main(argv: list | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="CPT 된 행만 원본 몸통에 이식")
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="수술 직후 모델 (몸통 제공)")
    ap.add_argument("--donor", required=True, help="CPT 를 마친 모델 (행 제공)")
    ap.add_argument("--id-map", default=str(DEFAULT_MAP))
    ap.add_argument("--name", required=True, help="artifacts/models/<name>/")
    ap.add_argument("--tag", default="p3e")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    out_dir = ROOT / "artifacts" / "models" / args.name
    if out_dir.exists():
        raise SystemExit(f"이미 있다: {out_dir}\n지우거나 --name 을 바꿔라")

    spec = json.loads(Path(args.id_map).read_text(encoding="utf-8"))
    rows = sorted(int(k) for k in spec["map"])

    config = {"base": args.base, "donor": args.donor, "id_map": args.id_map,
              "n_rows": len(rows), "purpose": "p3e_oracle_init_ceiling"}
    run_id = make_run_id("surgery", args.name, args.tag)

    with RunContext(run_id, phase="surgery", config=config,
                    skip_env_check=args.skip_env_check) as run:
        print(f"[1/4] 적재  몸통 {args.base}")
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
        tok = AutoTokenizer.from_pretrained(args.base)
        donor = AutoModelForCausalLM.from_pretrained(args.donor, dtype=torch.float32)
        print(f"      행   {args.donor}")

        emb = model.get_input_embeddings().weight
        demb = donor.get_input_embeddings().weight
        if emb.shape != demb.shape:
            raise SystemExit(f"임베딩 모양이 다르다: {tuple(emb.shape)} vs {tuple(demb.shape)}")
        if max(rows) >= emb.shape[0]:
            raise SystemExit(f"행 id 가 임베딩 밖이다: {max(rows)} >= {emb.shape[0]}")

        print(f"[2/4] 이식  {len(rows):,}행")
        idx = torch.tensor(rows)
        before = emb.detach().clone()
        moved = float(torch.linalg.vector_norm(demb[idx] - before[idx]))
        with torch.no_grad():
            emb[idx] = demb[idx]
        if model.config.tie_word_embeddings:
            model.tie_weights()

        print("[3/4] 검사  건드리지 않은 것이 정말 그대로인가")
        keep = torch.ones(emb.shape[0], dtype=torch.bool)
        keep[idx] = False
        same_rows = torch.equal(emb.detach()[keep], before[keep])
        # 몸통은 애초에 손대지 않지만, 확인 없이 믿지 않는다.
        body = {n: p for n, p in model.named_parameters()
                if "embed_tokens" not in n and "lm_head" not in n}
        dbody = dict(donor.named_parameters())
        # 몸통이 donor 와 같아지면 "CPT 된 모델을 복사한 것" 이 된다. donor 는
        # CPT 를 거쳤으므로 정상이라면 **하나도** 같지 않아야 한다.
        n_body_equal_donor = sum(
            1 for n, p in body.items()
            if n in dbody and torch.equal(p.detach(), dbody[n].detach()))
        if not same_rows:
            raise SystemExit("이식 대상이 아닌 행이 바뀌었다 — 저장하지 않는다")
        if n_body_equal_donor:
            raise SystemExit(
                f"몸통 텐서 {n_body_equal_donor}개가 donor 와 같다. 이러면 "
                "초기화 효과가 아니라 CPT 된 모델을 복사한 것이 된다 — 저장하지 않는다")
        print(f"      이식 외 행 {int(keep.sum()):,}개 비트 일치")
        print(f"      몸통 텐서 {len(body)}개 중 donor 와 같은 것 {n_body_equal_donor}개"
              f"  (0 이어야 정상 — donor 는 CPT 를 거쳤다)")
        print(f"      이식된 행의 이동 거리(L2) {moved:.4f}")

        print("[4/4] 저장")
        model = model.to(torch.bfloat16)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        sha = sha256_file(out_dir / "model.safetensors")
        (out_dir / "transplanted_rows.json").write_text(
            json.dumps({"donor": args.donor, "base": args.base, "rows": rows},
                       ensure_ascii=False), encoding="utf-8", newline=chr(10))
        print(f"      {out_dir}")
        print(f"      model_sha256 = {sha}")

        run.extra["model"] = args.base
        run.extra["init_method"] = "transplant_cpt_rows"
        run.extra["vocab_size"] = int(emb.shape[0])
        run.note = (f"oracle transplant {len(rows)} rows from {Path(args.donor).name} "
                    f"moved_l2={moved:.4f} body_equal_donor={n_body_equal_donor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
