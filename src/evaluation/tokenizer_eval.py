"""Level 1 — 토크나이저 내재 평가 (스펙 §14~17, §35).

GPU 를 쓰기 전 CPU 단계에서 후보를 거른다. 여기서 떨어진 토크나이저는
학습으로 보내지 않는다 (스펙 §17 Candidate Gate).

**전체 평균 하나만 내지 않는다.** 도메인별로 한 행씩 남긴다 (스펙 §16).
"한국어 -30%, 영어 +2%, 코드 +17%" 같은 trade-off 는 평균에 가려진다.

측정:
    tok_per_char  tok_per_byte  bytes_per_tok  tok_per_eojeol
    fertility     = 토큰 수 / 형태소 수 (kiwipiepy 기준, 검토 A7)
    P50/P90/P95/P99/MAX 시퀀스 길이 — 꼬리가 context overflow 를 만든다 (스펙 §15)

    .conda/python.exe -m src.evaluation.tokenizer_eval \\
        --config configs/evaluation/tokenizer.yaml --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402

from src.data.morph import analyzer_version, count_morphemes  # noqa: E402
from src.data.normalize import byte_len, count_eojeol  # noqa: E402
from src.utils.hashing import sha256_obj  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

BATCH = 256


def load_docs(split: str, root: Path | None = None) -> list:
    """data/interim/docs/<split>.jsonl 를 읽는다 (파이프라인 산출물)."""
    path = Path(root or ROOT) / "data" / "interim" / "docs" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없다. 먼저 scripts/run_data_pipeline.py 를 돌려라."
        )
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load_tokenizer(spec: dict):
    from transformers import AutoTokenizer

    if spec.get("path"):
        return AutoTokenizer.from_pretrained(str(ROOT / spec["path"]))
    kwargs = {"revision": spec["revision"]} if spec.get("revision") else {}
    return AutoTokenizer.from_pretrained(spec["repo_id"], **kwargs)


def token_lengths(tokenizer, texts: list) -> np.ndarray:
    """문서별 토큰 수. 배치 인코딩으로 Rust 병렬성을 쓴다."""
    out = np.empty(len(texts), dtype=np.int64)
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        enc = tokenizer(chunk, add_special_tokens=False, verbose=False)["input_ids"]
        out[i:i + len(chunk)] = [len(x) for x in enc]
    return out


def evaluate(tokenizer, docs: list, fertility_sample: int, seed: int = 42) -> dict:
    """도메인 -> 지표 dict."""
    by_domain: dict = defaultdict(list)
    for d in docs:
        by_domain[d["domain"]].append(d)

    rng = np.random.default_rng(seed)
    results: dict = {}
    for domain, group in sorted(by_domain.items()):
        texts = [d["text"] for d in group]
        lens = token_lengths(tokenizer, texts)

        n_chars = sum(len(t) for t in texts)
        n_bytes = sum(byte_len(t) for t in texts)
        n_eojeol = sum(count_eojeol(t) for t in texts)
        n_tokens = int(lens.sum())

        # fertility 는 형태소 분석이 무거우므로 표본으로 잰다
        fert = float("nan")
        if fertility_sample > 0 and len(texts) > 0:
            k = min(fertility_sample, len(texts))
            idx = rng.choice(len(texts), size=k, replace=False)
            morphs = sum(count_morphemes(texts[i]) for i in idx)
            toks = int(lens[idx].sum())
            fert = toks / morphs if morphs else float("nan")

        results[domain] = {
            "n_docs": len(group), "n_chars": n_chars, "n_bytes": n_bytes,
            "n_eojeol": n_eojeol, "n_tokens": n_tokens,
            "tok_per_char": n_tokens / n_chars if n_chars else float("nan"),
            "tok_per_byte": n_tokens / n_bytes if n_bytes else float("nan"),
            "bytes_per_tok": n_bytes / n_tokens if n_tokens else float("nan"),
            "tok_per_eojeol": n_tokens / n_eojeol if n_eojeol else float("nan"),
            "fertility_mean": fert,
            "p50_len": int(np.percentile(lens, 50)),
            "p90_len": int(np.percentile(lens, 90)),
            "p95_len": int(np.percentile(lens, 95)),
            "p99_len": int(np.percentile(lens, 99)),
            "max_len": int(lens.max()),
        }
    return results


def gate(baseline: dict, candidate: dict, cfg: dict) -> list:
    """스펙 §17 Candidate Gate. 통과 못한 사유 목록을 돌려준다 (빈 리스트면 통과).

    한국어 도메인은 개선, 영어·코드는 악화 상한을 본다. 코퍼스에 영어·코드
    도메인이 없으면 그 조건은 **판정하지 않는다** — 통과로 치지 않는다.
    """
    fails: list = []
    ko_domains = [d for d in baseline if d not in ("ko_en_mixed", "code")]
    if ko_domains:
        b = sum(baseline[d]["n_tokens"] for d in ko_domains) / \
            sum(baseline[d]["n_chars"] for d in ko_domains)
        c = sum(candidate[d]["n_tokens"] for d in ko_domains if d in candidate) / \
            sum(candidate[d]["n_chars"] for d in ko_domains if d in candidate)
        gain = (b - c) / b
        need = float(cfg.get("korean_tok_per_char_improvement_min", 0.15))
        if gain < need:
            fails.append(f"한국어 압축 개선 {gain * 100:.1f}% < 요구 {need * 100:.0f}%")
    for domain, key, label in (("code", "code_degradation_max", "코드"),):
        if domain in baseline and domain in candidate:
            b = baseline[domain]["tok_per_char"]
            c = candidate[domain]["tok_per_char"]
            deg = (c - b) / b
            lim = float(cfg.get(key, 0.10))
            if deg > lim:
                fails.append(f"{label} 악화 {deg * 100:.1f}% > 상한 {lim * 100:.0f}%")
        else:
            fails.append(f"{domain} 도메인이 코퍼스에 없어 판정 불가")
    return fails


def main(argv: list | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description="Level 1 토크나이저 벤치마크")
    ap.add_argument("--config", default="configs/evaluation/tokenizer.yaml")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--fertility-sample", type=int, default=1500)
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    cfg_path = ROOT / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    specs = cfg["tokenizers"]

    docs = load_docs(args.split)
    print(f"{args.split} 문서 {len(docs):,}건, "
          f"{sum(byte_len(d['text']) for d in docs) / 1e6:.1f} MB\n")

    config = {
        "config_file": args.config, "split": args.split,
        "tokenizers": specs, "fertility_sample": args.fertility_sample,
        "morph_analyzer": analyzer_version(),
        "gate": cfg.get("gate", {}),
    }
    run_id = make_run_id("tok", "bench", args.tag)

    all_results: dict = {}
    with RunContext(run_id, phase="tok", config=config,
                    skip_env_check=args.skip_env_check) as run:
        for spec in specs:
            name = spec["name"]
            tok = load_tokenizer(spec)
            res = evaluate(tok, docs, args.fertility_sample)
            all_results[name] = res
            for domain, m in res.items():
                run.log("tokenizer_metrics", tokenizer_version=name,
                        tokenizer_sha256=sha256_obj(
                            {"repo": spec.get("repo_id") or spec.get("path"),
                             "rev": spec.get("revision", "NA")}),
                        split=args.split, domain=domain, **m)
            total_tok = sum(m["n_tokens"] for m in res.values())
            total_chars = sum(m["n_chars"] for m in res.values())
            print(f"  {name:<16} vocab {len(tok):>7,}  "
                  f"전체 tok/char {total_tok / total_chars:.4f}")
        run.note = f"{len(specs)} tokenizers x {args.split}"

    # ── 비교 표 ───────────────────────────────────────────────────────────
    names = list(all_results)
    base = names[0]
    domains = sorted({d for r in all_results.values() for d in r})

    print(f"\n{'=' * 78}\ntok/char (낮을수록 압축이 좋다) — 기준 {base}\n")
    head = f"{'domain':<16}{'docs':>7}" + "".join(f"{n:>15}" for n in names)
    print(head)
    print("-" * len(head))
    for domain in domains:
        row = f"{domain:<16}{all_results[base].get(domain, {}).get('n_docs', 0):>7,}"
        b = all_results[base].get(domain, {}).get("tok_per_char")
        for n in names:
            v = all_results[n].get(domain, {}).get("tok_per_char")
            if v is None:
                row += f"{'—':>15}"
            elif n == base or b is None:
                row += f"{v:>15.4f}"
            else:
                row += f"{v:>9.4f}{(v - b) / b * 100:>+6.1f}%"
        print(row)

    print(f"\n{'P95 시퀀스 길이':<16}{'':<7}" +
          "".join(f"{n:>15}" for n in names))
    for domain in domains:
        row = f"{domain:<16}{'':<7}"
        for n in names:
            v = all_results[n].get(domain, {}).get("p95_len")
            row += f"{v:>15,}" if v is not None else f"{'—':>15}"
        print(row)

    print(f"\n{'fertility (tok/형태소)':<24}" + "".join(f"{n:>15}" for n in names))
    for domain in domains:
        row = f"{domain:<24}"
        for n in names:
            v = all_results[n].get(domain, {}).get("fertility_mean")
            row += f"{v:>15.3f}" if v == v else f"{'—':>15}"
        print(row)

    print(f"\n{'=' * 78}\nCandidate Gate (기준 {base})\n")
    for n in names[1:]:
        fails = gate(all_results[base], all_results[n], cfg.get("gate", {}))
        mark = "통과" if not fails else "탈락"
        print(f"  {n:<16} {mark}")
        for f in fails:
            print(f"      - {f}")
    print(f"\n형태소 분석기: {analyzer_version()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
