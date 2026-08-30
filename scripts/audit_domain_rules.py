"""도메인 규칙을 조사하고 정확도를 잰다 (docs/HANDOFF.md 1~2번).

**규칙 기반 분류의 오류율을 모른 채 도메인별 결과를 주장하면 안 된다.**
스펙 §16 의 도메인별 평가는 이 라벨이 맞다는 전제 위에 서 있는데,
지금 그 전제가 검증된 적이 없다.

세 가지 모드:

    hosts   여러 샤드에 걸쳐 상위 호스트 빈도와 현재 규칙이 매긴 도메인을 본다.
            규칙을 고치기 전에 무엇이 web_general 로 새는지 먼저 봐야 한다.

    sample  seed 고정 무작위 표본을 TSV 로 뽑는다. gold_domain 은 사람이 채운다.
            이미 사람이 채운 파일은 덮어쓰지 않는다.

    score   채워진 gold_domain 과 predicted_domain 을 대조해 정확도와
            혼동 행렬을 낸다. 이 숫자가 리포트에 들어간다.

    .conda/python.exe scripts/audit_domain_rules.py --mode hosts --shards 4 --max-docs 20000
    .conda/python.exe scripts/audit_domain_rules.py --mode sample --shards 4 --max-docs 20000
    .conda/python.exe scripts/audit_domain_rules.py --mode score

한 샤드 앞부분만 읽으면 tripadvisor 한 호스트가 24% 를 차지한 전례가 있다.
그래서 --shards 는 2 이상을 권한다.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402

from scripts.run_data_pipeline import stream_docs  # noqa: E402
from src.data.domain import DomainRules, classify, classify_host, host_of  # noqa: E402
from src.data.normalize import normalize_text  # noqa: E402
from src.data.quality import QualityConfig, check  # noqa: E402

AUDIT_COLUMNS = (
    "sample_id", "url", "host", "text_preview",
    "predicted_domain", "gold_domain", "reviewer_note",
)
DEFAULT_AUDIT = ROOT / "reports" / "tables" / "domain_audit.tsv"


def collect(max_docs: int, max_bytes: int, shards: int, rules: DomainRules) -> list:
    """품질 필터와 **호스트 상한**을 통과한 문서만 모은다.

    파이프라인과 같은 조건이어야 한다. 감사 도구가 상한을 빼먹으면 조사 결과와
    실제 코퍼스가 어긋나서, 규칙을 잘못된 분포 위에서 고치게 된다.
    """
    qcfg = QualityConfig()
    host_cap = int(rules.spam.get("max_docs_per_host", 0) or 0)
    host_seen: Counter = Counter()
    kept: list = []
    seen = capped = 0
    for row in stream_docs(max_docs, max_bytes, shards=shards):
        seen += 1
        text = normalize_text(row["text"])
        if check(text, qcfg, row.get("language_score")):
            continue
        domain, host = classify(row["url"], text, rules)
        if host_cap and host:
            host_seen[host] += 1
            if host_seen[host] > host_cap:
                capped += 1
                continue
        kept.append({
            "doc_id": row["doc_id"], "url": row["url"], "host": host,
            "text": text, "domain": domain,
            "by_rule": classify_host(host, rules) is not None,
        })
    print(f"수집 {seen:,}건 -> 필터 통과 {len(kept):,}건 "
          f"({len(kept) / max(seen, 1) * 100:.1f}%)")
    if host_cap:
        print(f"  호스트 상한 {host_cap:,}건/호스트 로 {capped:,}건 제외\n")
    else:
        print()
    return kept


# ── hosts ─────────────────────────────────────────────────────────────────
def mode_hosts(docs: list, top: int) -> None:
    hosts = Counter(d["host"] for d in docs if d["host"])
    total = sum(hosts.values())
    domain_of = {d["host"]: d["domain"] for d in docs}
    by_rule = {d["host"]: d["by_rule"] for d in docs}

    print(f"상위 호스트 {top}개 (전체 {len(hosts):,}개 호스트, {total:,}문서)\n")
    print(f"{'#':>3}  {'문서':>6} {'비율':>7}  {'규칙':<5} {'도메인':<14} 호스트")
    print("-" * 82)
    for i, (host, n) in enumerate(hosts.most_common(top), 1):
        mark = "O" if by_rule.get(host) else "-"
        print(f"{i:>3}  {n:>6,} {n / total * 100:6.2f}%  {mark:<5} "
              f"{domain_of.get(host, '?'):<14} {host}")

    print("\n도메인 분포 (문서 수 기준)")
    dom = Counter(d["domain"] for d in docs)
    for name, n in dom.most_common():
        print(f"  {name:<16} {n:>7,}  {n / len(docs) * 100:6.2f}%")

    unmatched = Counter(d["host"] for d in docs
                        if d["domain"] == "web_general" and d["host"])
    if unmatched:
        share = sum(unmatched.values()) / len(docs) * 100
        print(f"\nweb_general 로 샌 상위 호스트 30개 (전체의 {share:.1f}%)")
        print("규칙을 보강할 후보다. 다만 표본에 과적합하지 마라 — 한 호스트가")
        print("표본의 몇 %를 차지하는지 먼저 보고, 쏠렸으면 샤드를 늘려 다시 재라.\n")
        for host, n in unmatched.most_common(30):
            print(f"  {n:>6,}  {n / len(docs) * 100:5.2f}%  {host}")


# ── sample ────────────────────────────────────────────────────────────────
def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ").replace("\r", " ")


def has_gold(path: Path) -> int:
    """사람이 채운 gold_domain 이 몇 개인지."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if "gold_domain" not in header:
            return 0
        idx = header.index("gold_domain")
        return sum(1 for ln in fh
                   if (f := ln.rstrip("\n").split("\t")) and len(f) > idx
                   and f[idx] not in ("", "NA"))


def mode_sample(docs: list, n: int, seed: int, path: Path, force: bool) -> int:
    filled = has_gold(path)
    if filled and not force:
        print(f"{path} 에 이미 사람이 채운 gold_domain {filled}건이 있다.\n"
              "덮어쓰지 않는다. 새로 뽑으려면 --force 를 주거나 파일명을 바꿔라.",
              file=sys.stderr)
        return 1

    rng = np.random.default_rng(seed)
    k = min(n, len(docs))
    idx = rng.choice(len(docs), size=k, replace=False)
    idx.sort()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(AUDIT_COLUMNS) + "\n")
        for rank, i in enumerate(idx, 1):
            d = docs[int(i)]
            fh.write("\t".join([
                f"s{rank:04d}", _escape(d["url"])[:200], d["host"] or "NA",
                _escape(d["text"][:180]), d["domain"], "", "",
            ]) + "\n")

    print(f"{path}\n  표본 {k}건 (seed={seed}). gold_domain 과 reviewer_note 를 사람이 채운다.")
    print(f"  가능한 라벨: news encyclopedia blog community technical code "
          f"ko_en_mixed web_general")
    print(f"\n채운 뒤:  .conda/python.exe scripts/audit_domain_rules.py --mode score")
    return 0


# ── score ─────────────────────────────────────────────────────────────────
def mode_score(path: Path) -> int:
    if not path.exists():
        print(f"{path} 가 없다. --mode sample 로 먼저 뽑아라.", file=sys.stderr)
        return 1
    rows: list = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) == len(header):
                rows.append(dict(zip(header, f)))

    labeled = [r for r in rows if r.get("gold_domain") not in ("", "NA", None)]
    if not labeled:
        print(f"{path} 에 채워진 gold_domain 이 없다. 사람이 먼저 라벨링해야 한다.",
              file=sys.stderr)
        return 1

    correct = sum(1 for r in labeled if r["gold_domain"] == r["predicted_domain"])
    acc = correct / len(labeled)
    print(f"라벨링 {len(labeled)}/{len(rows)}건   정확도 {acc * 100:.1f}% "
          f"({correct}/{len(labeled)})\n")

    conf: dict = defaultdict(Counter)
    for r in labeled:
        conf[r["gold_domain"]][r["predicted_domain"]] += 1
    print("혼동 행렬 (행=정답, 열=예측)")
    for gold in sorted(conf):
        wrong = {p: c for p, c in conf[gold].items() if p != gold}
        n_g = sum(conf[gold].values())
        right = conf[gold][gold]
        line = f"  {gold:<16} {right}/{n_g}"
        if wrong:
            line += "   오분류 -> " + ", ".join(
                f"{p}({c})" for p, c in sorted(wrong.items(), key=lambda x: -x[1]))
        print(line)

    print(f"\n이 정확도를 리포트에 적어라. 도메인별 결과를 주장할 때 "
          f"오류율 {(1 - acc) * 100:.1f}% 를 함께 밝힌다.")
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="도메인 규칙 조사·감사")
    ap.add_argument("--mode", choices=("hosts", "sample", "score"), required=True)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--max-docs", type=int, default=20_000)
    ap.add_argument("--max-bytes", type=int, default=150_000_000)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rules", default="configs/data/domain_rules.yaml")
    ap.add_argument("--out", default=str(DEFAULT_AUDIT))
    ap.add_argument("--force", action="store_true", help="사람이 채운 감사 파일도 덮어쓴다")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.mode == "score":
        return mode_score(out)

    rules = DomainRules.load(ROOT / args.rules)
    print(f"도메인 규칙 {rules.version}  |  샤드 {args.shards}개에 분산 표본\n")
    docs = collect(args.max_docs, args.max_bytes, args.shards, rules)
    if not docs:
        print("통과한 문서가 없다", file=sys.stderr)
        return 1

    if args.mode == "hosts":
        mode_hosts(docs, args.top)
        return 0
    return mode_sample(docs, args.sample_size, args.seed, out, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
