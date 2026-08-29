"""Step 1 데이터 파이프라인 — 수집부터 문서 단위 분할까지 (스펙 §97).

    수집 → 정규화 → 품질 필터 → 도메인 라벨 → Exact dedup → Near dedup
         → manifest → 문서 단위 Train/Dev/Test 분할

**Split first, tokenize later** (docs/RULES.md 1번). 이 스크립트가 끝나고
manifest_sha256 이 확정되기 전에는 토크나이저를 학습하지 않는다.

    # 소규모 관통 (1주차)
    .conda/python.exe scripts/run_data_pipeline.py --max-docs 60000 --tag pilot

    # 전체 규모 (3~4주차)
    .conda/python.exe scripts/run_data_pipeline.py --max-bytes 5000000000 --tag v1

정규화된 본문은 data/interim/docs/<split>.jsonl 에 저장한다 (git 제외).
커밋되는 것은 data/manifests/<split>.tsv 뿐이다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402

from src.data import dedup as dd  # noqa: E402
from src.data.domain import DomainRules, classify  # noqa: E402
from src.data.normalize import byte_len, count_eojeol, normalize_text  # noqa: E402
from src.data.quality import FilterStats, QualityConfig, check  # noqa: E402
from src.data.split import SplitConfig, assign  # noqa: E402
from src.utils import ledger  # noqa: E402
from src.utils.hashing import sha256_file, sha256_obj  # noqa: E402
from src.utils.tracking import RunContext, make_run_id  # noqa: E402

DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "kor_Hang"


COLUMNS = ["text", "id", "url", "date", "dump", "language_score"]


def stream_docs(max_docs: int, max_bytes: int, shards: int = 1, seed: int = 42):
    """샤드 전체에 **고르게 퍼진** 표본을 읽는다.

    앞에서부터 순차로 읽으면 안 된다. 실제로 해보니 앞 2,868건 중 24% 가
    tripadvisor 한 호스트였다 — 파케이가 출처별로 뭉쳐 있기 때문이다.
    그 표본으로 도메인 분포나 압축률을 논하면 전부 틀린다.

    샤드 하나가 2,783개 row group (각 1,000행, 압축 2.6MB) 이므로,
    row group 을 일정 간격으로 골라 읽으면 파일 전체를 가로지르는 표본이 된다.
    HfFileSystem 이 range 요청을 하므로 4.8GB 를 다 받지 않는다.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rows_per_group = 1000
    want_groups = max(1, -(-max_docs // rows_per_group))  # 올림
    per_shard = max(1, -(-want_groups // shards))
    seen_bytes = 0
    emitted = 0

    for shard in range(shards):
        path = (f"datasets/{DATASET}/data/{CONFIG}/train/"
                f"000_{shard:05d}.parquet")
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            total = pf.metadata.num_row_groups
            step = max(1, total // per_shard)
            picked = list(range(0, total, step))[:per_shard]
            for rg in picked:
                table = pf.read_row_group(rg, columns=COLUMNS)
                for row in table.to_pylist():
                    if emitted >= max_docs or seen_bytes >= max_bytes:
                        return
                    text = row.get("text") or ""
                    seen_bytes += len(text.encode("utf-8"))
                    emitted += 1
                    yield {
                        "doc_id": row.get("id") or f"fw2_s{shard}_g{rg}_{emitted:09d}",
                        "text": text,
                        "url": row.get("url") or "",
                        "date": (row.get("date") or "")[:10],
                        "dump": row.get("dump") or "",
                        "language_score": row.get("language_score"),
                    }


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 1 데이터 파이프라인")
    ap.add_argument("--max-docs", type=int, default=60_000)
    ap.add_argument("--max-bytes", type=int, default=400_000_000)
    ap.add_argument("--tag", default="pilot", help="run_id 에 들어가는 이름")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rules", default="configs/data/domain_rules.yaml")
    ap.add_argument("--shards", type=int, default=1,
                    help="몇 개 샤드에 걸쳐 표본을 뽑을지")
    ap.add_argument("--skip-env-check", action="store_true")
    args = ap.parse_args(argv)

    rules_path = ROOT / args.rules
    rules = DomainRules.load(rules_path)
    qcfg = QualityConfig()
    mcfg = dd.MinHashConfig(seed=args.seed)
    scfg = SplitConfig(seed=args.seed)

    config = {
        "dataset": DATASET, "config": CONFIG,
        "max_docs": args.max_docs, "max_bytes": args.max_bytes,
        "shards": args.shards, "sampling": "row_group_strided",
        "domain_rules_version": rules.version,
        "domain_rules_sha256": sha256_file(rules_path),
        "quality": vars(qcfg) if hasattr(qcfg, "__dict__") else qcfg.__dict__,
        "minhash": mcfg.__dict__, "split": scfg.__dict__,
        "seed": args.seed,
    }
    run_id = make_run_id("data", args.tag, seed=args.seed)

    interim = ROOT / "data" / "interim" / "docs"
    interim.mkdir(parents=True, exist_ok=True)

    with RunContext(run_id, phase="data", config=config, seed=args.seed,
                    skip_env_check=args.skip_env_check,
                    note=f"FineWeb-2 {CONFIG} pilot") as run:
        t0 = time.time()

        # ── 1) 수집 + 정규화 + 품질 필터 + 도메인 ────────────────────────
        print(f"[1/5] 수집·정규화·필터  (최대 {args.max_docs:,}문서 / "
              f"{args.max_bytes / 1e6:.0f}MB)")
        stats = FilterStats()
        docs: list = []
        raw_bytes = 0
        for row in stream_docs(args.max_docs, args.max_bytes,
                               shards=args.shards, seed=args.seed):
            raw_bytes += len(row["text"].encode("utf-8"))
            text = normalize_text(row["text"])
            reason = check(text, qcfg, row.get("language_score"))
            stats.record(reason)
            if reason:
                continue
            domain, host = classify(row["url"], text, rules)
            docs.append({
                "doc_id": row["doc_id"], "text": text, "domain": domain,
                "host": host, "date": row["date"] or "NA",
                "sha256": dd.content_sha256(text),
            })
            if stats.total % 20_000 == 0:
                print(f"      {stats.total:,}건 처리, {len(docs):,}건 통과")
        for line in stats.as_lines():
            print(line)

        if not docs:
            raise RuntimeError("통과한 문서가 없다 — 필터가 너무 빡빡하다")

        # ── 2) Exact dedup ───────────────────────────────────────────────
        print("[2/5] Exact dedup (정규화 후 SHA256)")
        keep_exact, n_exact = dd.exact_dedup((d["doc_id"], d["sha256"]) for d in docs)
        docs = [d for d in docs if d["doc_id"] in keep_exact]
        print(f"      제거 {n_exact:,}  남음 {len(docs):,}")

        # ── 3) Near dedup ────────────────────────────────────────────────
        print(f"[3/5] Near dedup (MinHash {mcfg.num_perm}perm / "
              f"{mcfg.bands}bands / thr {mcfg.threshold})")
        hasher = dd.MinHasher(mcfg)
        sigs = np.empty((len(docs), mcfg.num_perm), dtype=np.uint32)
        for i, d in enumerate(docs):
            sigs[i] = hasher.signature(d["text"])
            if (i + 1) % 20_000 == 0:
                print(f"      시그니처 {i + 1:,}/{len(docs):,}")
        keep_near, n_near, n_clusters = dd.near_dedup(
            [d["doc_id"] for d in docs], sigs, mcfg)
        docs = [d for d in docs if d["doc_id"] in keep_near]
        print(f"      제거 {n_near:,} ({n_clusters:,}개 군집)  남음 {len(docs):,}")

        # ── 4) 문서 단위 분할 ────────────────────────────────────────────
        print("[4/5] 문서 단위 분할 (doc_id 해시 기반)")
        by_split: dict = defaultdict(list)
        for d in docs:
            d["split"] = assign(d["doc_id"], scfg)
            by_split[d["split"]].append(d)
        for sp in ("train", "dev", "final_test"):
            group = by_split.get(sp, [])
            nb = sum(byte_len(d["text"]) for d in group)
            print(f"      {sp:<11} {len(group):>7,}문서  {nb / 1e6:8.1f} MB")

        # 도메인 분포
        print("      도메인 분포 (train / dev):")
        dom_tr = Counter(d["domain"] for d in by_split.get("train", []))
        dom_dv = Counter(d["domain"] for d in by_split.get("dev", []))
        for dom in sorted(set(dom_tr) | set(dom_dv), key=lambda x: -dom_tr[x]):
            dev_bytes = sum(byte_len(d["text"]) for d in by_split.get("dev", [])
                            if d["domain"] == dom)
            print(f"        {dom:<16} {dom_tr[dom]:>7,} / {dom_dv[dom]:>6,}"
                  f"   dev {dev_bytes / 1e6:6.2f} MB")

        # ── 5) manifest + 본문 저장 ──────────────────────────────────────
        print("[5/5] manifest 와 본문 저장")
        manifest_shas = {}
        for sp, group in by_split.items():
            group.sort(key=lambda d: d["doc_id"])
            path = ledger.manifest_path(sp)
            if path.exists():
                path.unlink()          # manifest 는 파이프라인 산출물이라 재생성한다
            ledger.append_manifest_rows(sp, [
                {"doc_id": d["doc_id"], "source": "fineweb-2", "domain": d["domain"],
                 "date": d["date"], "language": "ko", "sha256": d["sha256"],
                 "split": sp, "char_count": len(d["text"]),
                 "byte_count": byte_len(d["text"])}
                for d in group
            ])
            manifest_shas[sp] = sha256_file(path)

            out = interim / f"{sp}.jsonl"
            with out.open("w", encoding="utf-8", newline="\n") as fh:
                for d in group:
                    fh.write(json.dumps(
                        {"doc_id": d["doc_id"], "domain": d["domain"],
                         "text": d["text"]}, ensure_ascii=False) + "\n")
            print(f"      {path.name:<16} {len(group):>7,}행  sha={manifest_shas[sp][:12]}")

        # 매니페스트 본체는 커밋하지 않는다 (전체 규모에서 수백 MB).
        # 사람이 검토할 수 있는 요약만 남긴다. 계보는 manifest_sha256 이 잡는다.
        summary = ROOT / "data" / "manifests" / "SUMMARY.tsv"
        agg: dict = defaultdict(lambda: [0, 0, 0])
        for sp, group in by_split.items():
            for d in group:
                cell = agg[(sp, d["domain"])]
                cell[0] += 1
                cell[1] += len(d["text"])
                cell[2] += byte_len(d["text"])
        with summary.open("w", encoding="utf-8", newline="
") as fh:
            fh.write("split	domain	n_docs	n_chars	n_bytes	manifest_sha256
")
            for (sp, dom), (nd, nc, nb) in sorted(agg.items()):
                fh.write(f"{sp}	{dom}	{nd}	{nc}	{nb}	{manifest_shas[sp]}
")
        print(f"      SUMMARY.tsv       {len(agg)}행 (커밋 대상)")

        combined = sha256_obj(manifest_shas)
        run.raw_bytes_seen = sum(byte_len(d["text"]) for d in docs)
        run.extra["manifest_sha256"] = combined
        run.note = (f"docs={len(docs)} exact_dup={n_exact} near_dup={n_near} "
                    f"kept={stats.kept}/{stats.total}")

        print(f"\nmanifest_sha256 = {combined}")
        print(f"소요 {time.time() - t0:.0f}초, 원본 {raw_bytes / 1e6:.0f}MB "
              f"-> 정제 {run.raw_bytes_seen / 1e6:.0f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
