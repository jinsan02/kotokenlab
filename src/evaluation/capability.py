"""Level 3 — MC few-shot benchmark + bootstrap CI (스펙 §40, §52).

지금 구현한 것은 **D3 KMMLU 프로브** 하나다. 조건은 전부
`docs/PLAN.md` "D3 실행 조건" (2026-09-17, 데이터를 받기 전 커밋) 에 있다.
여기 상수는 그 등록을 옮긴 것이다. 바꾸면 사전 등록이 아니다.

    .conda/python.exe -m src.evaluation.capability \
        --model artifacts/models/cpt_c0_qwen_main_seed42 --prompt-tokenizer artifacts/models/cpt_c0_qwen_main_seed42

## 프롬프트 문자열을 모델마다 다시 만들지 않는다

shot 수는 **C0 토크나이저** 로 잰 길이가 2048 토큰 안에 들 때까지 줄인다.
평가하는 모델의 토크나이저로 재면 T2a 와 C0 가 같은 문항에서 다른 shot 수를
받을 수 있고, 그러면 토크나이저 차이가 shot 차이로 새어 들어온다.
그래서 `--prompt-tokenizer` 는 **항상 C0** 를 준다.

## 무엇을 남기나

- `capability.tsv` — 과목 합계(micro)와 과목별 정확도, 문항 bootstrap CI
- `experiments/runs/<run_id>/predictions.tsv` — 문항마다 예측·정답. **문항 본문은
  남기지 않는다** (KMMLU 는 CC-BY-ND 다. 과목과 행 번호면 다시 찾을 수 있다)

T2a 와 C0 의 paired 판정은 `tools/hanja_probe.py` 가 predictions 를 읽어 낸다.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

# ── 2026-09-17 PLAN.md 등록값 ──────────────────────────────────────────────
KMMLU_REPO = "HAERAE-HUB/KMMLU"
KMMLU_REVISION = "d61b3f19e552c576bf5960dd24289763edc36a88"
SUBJECTS: tuple[str, ...] = (      # D3 등록 (2026-09-17). 기본값을 바꾸지 않는다
    "Law", "Criminal-Law", "Patent", "Taxation",
    "Health", "Political-Science-and-Sociology",
)
# KMMLU 45과목 전체. P3-F 는 **D3 에서 이미 본 6과목을 빼고** 나머지만 쓴다
# (본 데이터로 등록하지 않는다 — docs/SPEC_P3.md §2 F).
ALL_SUBJECTS: tuple[str, ...] = (
    "Accounting", "Agricultural-Sciences", "Aviation-Engineering-and-Maintenance",
    "Biology", "Chemical-Engineering", "Chemistry", "Civil-Engineering",
    "Computer-Science", "Construction", "Criminal-Law", "Ecology", "Economics",
    "Education", "Electrical-Engineering", "Electronics-Engineering",
    "Energy-Management", "Environmental-Science", "Fashion", "Food-Processing",
    "Gas-Technology-and-Engineering", "Geomatics", "Health", "Industrial-Engineer",
    "Information-Technology", "Interior-Architecture-and-Design", "Korean-History",
    "Law", "Machine-Design-and-Manufacturing", "Management", "Maritime-Engineering",
    "Marketing", "Materials-Engineering", "Math", "Mechanical-Engineering",
    "Nondestructive-Testing", "Patent", "Political-Science-and-Sociology",
    "Psychology", "Public-Safety", "Railway-and-Automotive-Engineering",
    "Real-Estate", "Refrigerating-Machinery", "Social-Welfare", "Taxation",
    "Telecommunications-and-Wireless-Technology",
)
REST_SUBJECTS: tuple[str, ...] = tuple(s for s in ALL_SUBJECTS if s not in SUBJECTS)
SUBJECT_SETS = {"d3": SUBJECTS, "rest": REST_SUBJECTS, "all": ALL_SUBJECTS}
MAX_SHOTS = 5
CTX_LIMIT = 2048            # CPT seq_len
LETTERS = "ABCD"
CHANCE = 0.25
MARGIN = 0.02               # 동등성 경계 ±2%p
N_BOOT = 10_000
BOOT_SEED = 0
BENCHMARK = "kmmlu_d3"

DATA = ROOT / "data" / "raw" / "kmmlu" / "data"
REQUIRED = ("question", "answer", "A", "B", "C", "D")
PRED_COLUMNS = ("subject", "row", "n_shot", "n_prompt_tokens",
                "gold", "pred", "correct", "has_hanja")


def is_han(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F)


def read_items(path: Path) -> list:
    """KMMLU CSV 한 파일. 정답 1~4 를 0~3 으로 옮긴다."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: 열이 없다 {missing}")
        items = []
        for r in reader:
            ans = int(float(r["answer"]))
            if ans not in (1, 2, 3, 4):
                raise ValueError(f"{path.name}: 정답이 1~4 가 아니다 ({r['answer']!r})")
            items.append({"question": r["question"].strip(),
                          "choices": [r[k].strip() for k in LETTERS],
                          "gold": ans - 1})
    return items


def format_item(item: dict, with_answer: bool) -> str:
    lines = [item["question"]]
    lines += [f"{L}. {c}" for L, c in zip(LETTERS, item["choices"])]
    tail = f"정답: {LETTERS[item['gold']]}" if with_answer else "정답:"
    return "\n".join(lines + [tail])


def build_prompt(shots: list, item: dict, n_tokens, limit: int = CTX_LIMIT,
                 max_shots: int = MAX_SHOTS) -> tuple:
    """(prompt, k). 길이가 limit 을 넘으면 **앞쪽** 예시부터 뺀다.

    뒤쪽 예시가 문항에 가깝게 남는다. k=0 에서도 넘으면 그대로 둔다 (절단 안 함).
    길이는 이어붙일 답 토큰 하나를 위해 1 을 더해 잰다.
    """
    use = list(shots[:max_shots])
    target = format_item(item, with_answer=False)
    while True:
        parts = [format_item(s, with_answer=True) for s in use] + [target]
        prompt = "\n\n".join(parts)
        if not use or n_tokens(prompt) + 1 <= limit:
            return prompt, len(use)
        use = use[1:]


def bootstrap_ci(correct, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple:
    import numpy as np

    x = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple:
    """d = mean(a) - mean(b). 같은 문항 인덱스를 함께 뽑는다."""
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired 비교는 문항 수가 같아야 한다")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def verdict(c0_ci_lo: float, d_lo: float, d_hi: float,
            margin: float = MARGIN, chance: float = CHANCE) -> str:
    """등록 판정. 바닥 검사가 먼저다."""
    if c0_ci_lo <= chance:
        return "측정 불가 — 바닥"
    if -margin <= d_lo and d_hi <= margin:
        return "같다"
    if d_hi < 0:
        return "T2a 저하"
    if d_lo > 0:
        return "T2a 향상"
    return "분해되지 않음"


def score(model, prompt_ids: list, cont_ids: list, device: str) -> list:
    """보기마다 이어붙인 부분의 로그확률 합."""
    import torch

    if all(len(c) == 1 for c in cont_ids):
        x = torch.tensor([prompt_ids], device=device)
        with torch.no_grad():
            lp = torch.log_softmax(model(x).logits[0, -1].float(), dim=-1)
        return [float(lp[c[0]]) for c in cont_ids]
    out = []
    for c in cont_ids:
        x = torch.tensor([prompt_ids + c], device=device)
        with torch.no_grad():
            lp = torch.log_softmax(model(x).logits[0].float(), dim=-1)
        n = len(prompt_ids)
        out.append(float(sum(lp[n - 1 + i, t] for i, t in enumerate(c))))
    return out


def main(argv: list | None = None) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.tracking import RunContext, make_run_id

    ap = argparse.ArgumentParser(description="D3 KMMLU 프로브")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokenizer", required=True,
                    help="shot 수를 정하는 토크나이저. 항상 C0 를 준다")
    ap.add_argument("--name", default=None)
    ap.add_argument("--subjects", default="d3", choices=tuple(SUBJECT_SETS),
                    help="d3 = 등록된 6과목(기본) · rest = 나머지 39과목(P3-F) · all")
    ap.add_argument("--tag", default="d3")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    name = args.name or Path(args.model).name
    tok = AutoTokenizer.from_pretrained(args.model)
    ptok = AutoTokenizer.from_pretrained(args.prompt_tokenizer)

    def n_tokens(s: str) -> int:
        return len(ptok(s, add_special_tokens=False)["input_ids"])

    # 프롬프트를 모델을 올리기 전에 전부 만든다 — 모델과 무관해야 하므로.
    subjects = SUBJECT_SETS[args.subjects]
    work = []
    for subject in subjects:
        shots = read_items(DATA / f"{subject}-dev.csv")
        for i, item in enumerate(read_items(DATA / f"{subject}-test.csv")):
            prompt, k = build_prompt(shots, item, n_tokens)
            work.append((subject, i, item, prompt, k))

    cont_ids = [tok(" " + L, add_special_tokens=False)["input_ids"] for L in LETTERS]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]

    config = {"model": args.model, "prompt_tokenizer": args.prompt_tokenizer,
              "benchmark": BENCHMARK, "repo": KMMLU_REPO,
              "revision": KMMLU_REVISION, "subject_set": args.subjects,
              "subjects": list(subjects),
              "max_shots": MAX_SHOTS, "ctx_limit": CTX_LIMIT,
              "scoring": "loglik_letter", "n_boot": N_BOOT,
              "boot_seed": BOOT_SEED, "purpose": "d3_hanja_probe"}
    run_id = make_run_id("eval", "kmmlu", name, args.tag)

    with RunContext(run_id, phase="eval", config=config,
                    skip_env_check=args.skip_env_check) as run:
        print(f"{name}  vocab {len(tok):,}  문항 {len(work):,}  "
              f"보기 토큰 {[len(c) for c in cont_ids]}")
        rows = []
        n_tok_total = 0
        with sdpa_kernel(backends):
            for j, (subject, i, item, prompt, k) in enumerate(work):
                ids = tok(prompt, add_special_tokens=False)["input_ids"]
                n_tok_total += len(ids)
                lps = score(model, ids, cont_ids, device)
                pred = max(range(4), key=lambda a: lps[a])
                rows.append({"subject": subject, "row": i, "n_shot": k,
                             "n_prompt_tokens": len(ids), "gold": item["gold"],
                             "pred": pred, "correct": int(pred == item["gold"]),
                             "has_hanja": int(any(is_han(ch) for ch in format_item(item, False)))})
                if (j + 1) % 500 == 0:
                    acc = sum(r["correct"] for r in rows) / len(rows)
                    print(f"  {j + 1:>6,}/{len(work):,}  acc {acc:.4f}")

        with (run.dir / "predictions.tsv").open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\t".join(PRED_COLUMNS) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in PRED_COLUMNS) + "\n")

        groups = [(BENCHMARK if args.subjects == "d3" else f"{BENCHMARK}_{args.subjects}",
                   rows)] + [
            (f"kmmlu_{s}", [r for r in rows if r["subject"] == s]) for s in subjects]
        for bench, rs in groups:
            c = [r["correct"] for r in rs]
            lo, hi = bootstrap_ci(c)
            acc = sum(c) / len(c)
            print(f"  {bench:<44} n={len(c):>5}  acc {acc:.4f}  [{lo:.4f}, {hi:.4f}]")
            run.log("capability", checkpoint="final", benchmark=bench, lang="ko",
                    n_items=len(c), n_shot=MAX_SHOTS, metric="accuracy",
                    value=round(acc, 6), ci_lo=round(lo, 6), ci_hi=round(hi, 6))

        short = sum(1 for r in rows if r["n_shot"] < MAX_SHOTS)
        over = sum(1 for r in rows if r["n_prompt_tokens"] + 1 > CTX_LIMIT)
        print(f"  shot<{MAX_SHOTS} 문항 {short:,}  이 모델에서 {CTX_LIMIT} 초과 {over:,}")
        run.tokens_seen = n_tok_total
        run.extra["tokenizer_version"] = name
        acc = sum(r["correct"] for r in rows) / len(rows)
        run.note = (f"D3 KMMLU acc {acc:.4f} n={len(rows)} "
                    f"short_shot={short} over_ctx={over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
