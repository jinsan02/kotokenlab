"""영어·코드 대조군 코퍼스 (스펙 §16, §17 / docs/HANDOFF.md 3번).

지금 Candidate Gate 의 판정은 반쪽이다. 스펙 §17 은 한국어 압축 개선과 함께
**영어 악화 ≤5%, 코드 악화 ≤10%** 를 요구하는데, 그걸 잴 코퍼스가 없어서
"code 도메인이 코퍼스에 없어 판정 불가" 로 탈락시키고 있었다.

한국어 파이프라인과 같은 manifest 스키마, 같은 doc_id 해시 분할을 쓴다.
다른 것은 품질 필터 프로파일뿐이다 (src/data/quality.py 의 for_language).
코드는 반복 검사를 끈다 — import 줄과 보일러플레이트가 정당하게 반복되므로
한국어 SEO 스팸 기준을 그대로 적용하면 멀쩡한 파일이 전부 탈락한다.

    .conda/python.exe scripts/run_control_pipeline.py --lang en --max-docs 8000
    .conda/python.exe scripts/run_control_pipeline.py --lang code --max-docs 8000

산출물은 data/interim/docs/<split>_control.jsonl 에 **덧붙인다**. 한국어
파이프라인이 만든 <split>.jsonl 을 건드리지 않으므로 실행 순서에 의존하지 않는다.
Level 1 평가는 두 파일을 모두 읽는다.

학습용이 아니라 regression 측정용이다. 도메인당 dev 5MB 정도면 충분하다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402

from src.data import dedup as dd  # noqa: E402
from src.data.domain import latin_share  # noqa: E402
from src.data.normalize import byte_len, hangul_ratio, normalize_text  # noqa: E402
from src.data.quality import FilterStats, QualityConfig, check  # noqa: E402
from src.data.source import list_parquet, stream_parquet  # noqa: E402
from src.data.split import SplitConfig, assign  # noqa: E402
from src.utils import ledger  # noqa: E402
from src.utils.hashing import sha256_file, sha256_obj  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

# (repo, 파케이 경로 prefix, 읽을 컬럼, 본문 컬럼, 도메인 라벨)
SOURCES = {
    "en": ("HuggingFaceFW/fineweb-edu", "sample/10BT/",
           ["text", "id", "url", "dump"], "text", "english"),
    "code": ("codeparrot/github-code-clean", "data/train-",
             ["code", "repo_name", "path", "language", "license"], "code", "code"),
}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="영어·코드 대조군 코퍼스")
    ap.add_argument("--lang", choices=("en", "code"), required=True)
    ap.add_argument("--max-docs", type=int, default=8_000)
    # 기본 120MB 가 --max-docs 보다 먼저 걸려서 조용히 절반 규모가 나온 적이 있다
    # (코드 dev 4.64MB, 기준 5MB 미달). 이제 어느 쪽이 걸렸는지 아래에서 알린다.
    ap.add_argument("--max-bytes", type=int, default=120_000_000)
    ap.add_argument("--files", type=int, default=4, help="몇 개 파케이에 걸쳐 뽑을지")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    repo, prefix, columns, text_col, domain = SOURCES[args.lang]
    qcfg = QualityConfig.for_language(args.lang)
    mcfg = dd.MinHashConfig(seed=args.seed)
    scfg = SplitConfig(seed=args.seed)          # 한국어와 **같은** 분할 규칙

    print(f"{repo}  ({domain})  파케이 목록 조회 중...")
    all_paths = list_parquet(repo, prefix)
    if not all_paths:
        print(f"{repo} 에서 {prefix}*.parquet 을 못 찾았다", file=sys.stderr)
        return 1
    step = max(1, len(all_paths) // args.files)
    paths = all_paths[::step][:args.files]
    print(f"  전체 {len(all_paths)}개 중 {len(paths)}개를 고르게 골랐다\n")

    config = {
        "repo": repo, "prefix": prefix, "domain": domain, "lang": args.lang,
        "paths": paths, "max_docs": args.max_docs, "max_bytes": args.max_bytes,
        "quality": qcfg.__dict__, "minhash": mcfg.__dict__, "split": scfg.__dict__,
        "seed": args.seed, "purpose": "regression_control_only",
    }
    run_id = make_run_id("data", f"control{args.lang}", args.tag, seed=args.seed)

    interim = ROOT / "data" / "interim" / "docs"
    interim.mkdir(parents=True, exist_ok=True)

    with RunContext(run_id, phase="data", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check) as run:
        t0 = time.time()

        print(f"[1/4] 수집·정규화·필터  (프로파일 {args.lang})")
        stats = FilterStats()
        docs: list = []
        raw_bytes = 0
        for row in stream_parquet(repo, paths, columns, args.max_docs,
                                  args.max_bytes, text_column=text_col):
            raw = row.get(text_col) or ""
            raw_bytes += len(raw.encode("utf-8"))
            text = normalize_text(raw)
            reason = check(text, qcfg)
            stats.record(reason)
            if reason:
                continue
            doc_id = str(row.get("id") or row.get("path") or row["_fallback_id"])
            docs.append({
                "doc_id": f"{domain}:{doc_id}", "text": text, "domain": domain,
                "sha256": dd.content_sha256(text),
                "date": str(row.get("dump") or row.get("language") or "NA")[:10],
                "latin_share": round(latin_share(text), 4),
                "hangul_ratio": round(hangul_ratio(text), 4),
            })
        for line in stats.as_lines():
            print(line)
        if raw_bytes >= args.max_bytes and stats.total < args.max_docs:
            print(f"  [주의] --max-bytes ({args.max_bytes / 1e6:.0f}MB) 가 먼저 걸렸다. "
                  f"--max-docs {args.max_docs:,} 중 {stats.total:,} 건만 봤다. "
                  f"의도한 규모가 아니면 --max-bytes 를 올려라.")
        if not docs:
            raise RuntimeError(f"{args.lang} 프로파일로 통과한 문서가 없다 — 필터를 확인하라")

        print("[2/4] Exact dedup")
        keep, n_exact = dd.exact_dedup((d["doc_id"], d["sha256"]) for d in docs)
        docs = [d for d in docs if d["doc_id"] in keep]
        print(f"      제거 {n_exact:,}  남음 {len(docs):,}")

        print("[3/4] Near dedup")
        hasher = dd.MinHasher(mcfg)
        sigs = np.empty((len(docs), mcfg.num_perm), dtype=np.uint32)
        for i, d in enumerate(docs):
            sigs[i] = hasher.signature(d["text"])
        keep_near, n_near, n_clusters = dd.near_dedup(
            [d["doc_id"] for d in docs], sigs, mcfg)
        docs = [d for d in docs if d["doc_id"] in keep_near]
        print(f"      제거 {n_near:,} ({n_clusters:,}개 군집)  남음 {len(docs):,}")

        print("[4/4] 분할·저장 (한국어와 같은 doc_id 해시 규칙)")
        by_split: dict = defaultdict(list)
        for d in docs:
            by_split[assign(d["doc_id"], scfg)].append(d)

        shas = {}
        for sp, group in sorted(by_split.items()):
            group.sort(key=lambda d: d["doc_id"])
            nb = sum(byte_len(d["text"]) for d in group)
            name = f"{sp}_control_{domain}"
            path = ledger.manifest_path(name)
            if path.exists():
                path.unlink()
            ledger.append_manifest_rows(name, [
                {"doc_id": d["doc_id"], "source": repo, "domain": domain,
                 "date": d["date"], "language": args.lang, "sha256": d["sha256"],
                 "split": sp, "char_count": len(d["text"]),
                 "byte_count": byte_len(d["text"]),
                 "latin_share": d["latin_share"],
                 "hangul_ratio": d["hangul_ratio"]}
                for d in group])
            shas[sp] = sha256_file(path)

            out = interim / f"{sp}_control_{domain}.jsonl"
            with out.open("w", encoding="utf-8", newline="\n") as fh:
                for d in group:
                    fh.write(json.dumps(
                        {"doc_id": d["doc_id"], "domain": domain, "text": d["text"]},
                        ensure_ascii=False) + "\n")
            print(f"      {sp:<11} {len(group):>6,}문서  {nb / 1e6:7.2f} MB  "
                  f"-> {out.name}")

        run.raw_bytes_seen = sum(byte_len(d["text"]) for d in docs)
        run.extra["manifest_sha256"] = sha256_obj(shas)
        run.note = (f"{domain} control kept={stats.kept}/{stats.total} "
                    f"exact={n_exact} near={n_near}")
        print(f"\nmanifest_sha256 = {run.extra['manifest_sha256']}")
        print(f"소요 {time.time() - t0:.0f}초, 원본 {raw_bytes / 1e6:.0f}MB "
              f"-> 정제 {run.raw_bytes_seen / 1e6:.0f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
