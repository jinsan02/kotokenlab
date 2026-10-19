"""문서별 NLL — 문서 단위 paired bootstrap 의 입력 (스펙 §36 BPB · amendment §4 · 5주차).

    .conda/python.exe -m src.evaluation.doc_nll --model artifacts/models/<run_id> \
        --source-run <run_id> --tag dev2mb

저장된 CPT 체크포인트를 CPT 의 dev 평가와 **같은 함수 · 같은 인자** 로 다시 평가해
(`bpb.evaluate`, dev 원문 언어별 2MB, seq_len 2048), 문서마다 NLL · 채점 바이트를
`experiments/runs/<run_id>/doc_nll.tsv` 에 남긴다.

**원장 재현을 스스로 확인한다.** `--source-run` 을 주면 그 학습 run 의 원장 최종 dev
BPB 와 문서별 합으로 다시 낸 BPB 를 대조해 차이를 원장 note 에 적는다. 같은 문서 ·
같은 계산이므로 같아야 한다 — 다르면 문서별 값으로 낸 bootstrap 을 믿을 수 없다.

판정하지 않는다. 비교는 `tools/doc_bootstrap.py` 가 한다.

**`--split heldout`** (Final Test, 개봉 뒤에만): 평가 파일 · 언어별 예산 · 태그를
`tools/heldout_spec.py` 의 상수로 고정한다 (PLAN "5주차 동결" D). 비교할 원장 값이 없으므로
원장 대조는 하지 않고, lm_metrics 의 split 은 `final_test` 로 적는다. 판정은
`tools/heldout_pairs.py` 가 한다.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.utils.tracking import RunContext, make_run_id  # noqa: E402

DEV = (("ko", "dev.jsonl"),
       ("en", "dev_control_english.jsonl"),
       ("code", "dev_control_code.jsonl"))
COLUMNS = ("lang", "doc", "nll", "bytes", "tokens", "raw_bytes")


def _heldout():
    from tools import heldout_spec
    return heldout_spec


def plan_files(split: str, max_bytes: int) -> tuple:
    """(lang, 파일 이름, 원문 예산) 목록과 lm_metrics 의 split 값.

    dev 는 지금까지와 같다 (세 언어 모두 --max-bytes). heldout 은 spec 의 고정값만 쓴다 —
    --max-bytes 로 바꿀 수 없다.
    """
    if split == "dev":
        return [(lang, f, max_bytes) for lang, f in DEV], "dev"
    h = _heldout()
    return list(h.FILES), h.SPLIT_LABEL


def heldout_guard(args) -> None:
    """개봉 태그가 없으면 열지 않는다. 설정이 spec 과 다르면 멈춘다 (PLAN D 의 6)."""
    import subprocess
    h = _heldout()
    tag = subprocess.run(["git", "tag", "--list", "final-test-opened"], cwd=str(ROOT),
                         capture_output=True, text=True).stdout.strip()
    if not tag:
        raise SystemExit("final-test-opened 태그가 없다 — 개봉 결정 · 태그 전에는 열지 않는다")
    if args.tag != h.TAG or args.seq_len != h.SEQ_LEN:
        raise SystemExit(f"heldout 은 태그 {h.TAG} · seq_len {h.SEQ_LEN} 로 고정이다")
    if args.model not in {p for p, _ in h.CHECKPOINTS.values()}:
        raise SystemExit(f"{args.model} 은 대표 체크포인트가 아니다 (PLAN D 의 1)")


def ledger_final_bpb(run_id: str) -> dict:
    """학습 run 의 원장 최종 dev BPB (언어별). 없으면 빈 dict."""
    path = ROOT / "experiments" / "lm_metrics.tsv"
    out: dict = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["run_id"] == run_id and r["checkpoint"] == "final" and r["split"] == "dev":
                out[r["domain"]] = float(r["bpb"])
    return out


def bpb_from_docs(rows: list) -> float:
    nll = sum(r["nll"] for r in rows)
    b = sum(r["bytes"] for r in rows)
    return nll / (math.log(2) * b) if b else float("nan")


def main(argv: list | None = None) -> int:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluation.bpb import evaluate, token_byte_length

    ap = argparse.ArgumentParser(description="문서별 NLL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--source-run", default=None, help="이 체크포인트를 만든 학습 run_id")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-bytes", type=int, default=2_000_000,
                    help="언어별 dev 원문 예산 (CPT 의 --eval-budget 과 같게)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tag", default=None, help="기본 dev2mb (heldout 은 spec 의 태그)")
    ap.add_argument("--split", choices=("dev", "heldout"), default="dev")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)
    files, split_label = plan_files(args.split, args.max_bytes)
    if args.tag is None:
        args.tag = "dev2mb" if args.split == "dev" else _heldout().TAG
    if args.split == "heldout":
        heldout_guard(args)
    # 2026-10-07: CRLF 작업 목록에서 읽은 run_id 끝에 '\r' 이 붙어 원장을 못 찾았다.
    if args.source_run:
        args.source_run = args.source_run.strip()

    name = args.name or Path(args.model).name
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    alphabet = set(ByteLevel.alphabet())
    blen: dict = {}

    def byte_len_fn(ids: list) -> int:
        total = 0
        for i in ids:
            v = blen.get(i)
            if v is None:
                v = token_byte_length(tokenizer.convert_ids_to_tokens(int(i)), alphabet)
                blen[i] = v
            total += v
        return total

    config = {"model": args.model, "split": split_label,
              "max_bytes": args.max_bytes if args.split == "dev" else None,
              "seq_len": args.seq_len, "source_run": args.source_run,
              "purpose": "doc_nll"}
    if args.split == "heldout":
        config["budgets"] = {lang: b for lang, _, b in files}
    run_id = make_run_id("eval", "docnll", name, args.tag)
    # heldout 에는 대조할 원장 값이 없다 — 대조하지 않는다는 사실을 note 에 남긴다.
    want = ledger_final_bpb(args.source_run) if args.source_run and args.split == "dev" else {}
    if args.source_run and args.split == "dev" and not want:
        # 조용히 대조를 건너뛰지 않는다 — 그러면 "재현 확인" 이 빠진 채 결과가 남는다.
        raise SystemExit(f"--source-run {args.source_run!r} 의 원장 최종 dev BPB 가 없다 "
                         "(run_id 오타 · 공백 · 아직 안 끝난 run 인지 확인)")

    with RunContext(run_id, phase="eval", config=config,
                    skip_env_check=args.skip_env_check) as run:
        dev_dir = ROOT / "data" / "interim" / "docs"
        all_rows: list = []
        notes = []
        with sdpa_kernel(backends):
            for lang, fname, budget in files:
                docs: list = []
                m = evaluate(model, tokenizer, dev_dir / fname, budget,
                             args.seq_len, device, byte_len_fn, docs_out=docs)
                re_bpb = bpb_from_docs(docs)
                for d in docs:
                    all_rows.append({"lang": lang, **d})
                run.log("lm_metrics", checkpoint="docnll", tokens_seen=0, raw_bytes_seen=0,
                        split=split_label, domain=lang, n_bytes=m["n_bytes"],
                        total_nll=round(m["total_nll"], 4), bpb=round(m["bpb"], 6),
                        bpc=round(m["bpc"], 6), token_ppl=round(m["token_ppl"], 4))
                line = f"{lang} {len(docs)}문서 BPB {m['bpb']:.6f} (문서합 {re_bpb:.6f})"
                if lang in want:
                    diff = round(m["bpb"], 6) - want[lang]
                    line += f" · 원장 {want[lang]:.6f} 차이 {diff:+.6f}"
                    notes.append(f"{lang}_vs_ledger={diff:+.6f}")
                print("  " + line)

        out = run.dir / "doc_nll.tsv"
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            for r in all_rows:
                fh.write("\t".join([r["lang"], str(r["doc"]), repr(r["nll"]), str(r["bytes"]),
                                    str(r["tokens"]), str(r["raw_bytes"])]) + "\n")
        if args.split == "heldout":
            notes.append("ledger_check=none(heldout)")
        run.note = f"doc_nll {len(all_rows)}행 source={args.source_run} " + " ".join(notes)
        print(f"  썼다 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
