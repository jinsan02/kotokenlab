"""라벨링된 감사 표본으로 규칙을 채점한다 — 추가 라벨링 없이 반복 개발.

블라인드 감사 150건이 있으므로, 규칙을 고칠 때마다 사람을 다시 부르지 않아도
된다. 다만 같은 표본을 보며 고치면 그 표본에 과적합한다. 그래서 나눈다.

    dev      sample_id 해시 < 0.67   규칙을 고치면서 몇 번이든 본다
    holdout  나머지                   **마지막에 한 번만** 본다

holdout 을 여러 번 보면 dev 와 같아진다. --holdout 은 최종 측정에만 쓴다.

    .conda/python.exe scripts/eval_domain_rules.py                 # dev
    .conda/python.exe scripts/eval_domain_rules.py --holdout       # 최종 1회
    .conda/python.exe scripts/eval_domain_rules.py --errors        # 오분류 보기
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.domain import DomainRules, classify  # noqa: E402

DEV_FRACTION = 0.67


def bucket(sample_id: str) -> float:
    h = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, "big") / 2 ** 32


def load(path: Path) -> list:
    lines = [ln for ln in io.open(path, encoding="utf-8").read().split("\n") if ln.strip()]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def report(rows: list, rules: DomainRules, show_errors: bool) -> float:
    for r in rows:
        r["pred"] = classify(r["url"], r["text_preview"], rules)[0]

    gold = Counter(r["gold_domain"] for r in rows)
    pred = Counter(r["pred"] for r in rows)
    tp = Counter(r["gold_domain"] for r in rows if r["gold_domain"] == r["pred"])
    correct = sum(tp.values())
    acc = correct / len(rows)

    print(f"정확도 {acc * 100:.1f}%  ({correct}/{len(rows)})\n")
    print(f"{'도메인':<14}{'정답':>5}{'예측':>5}{'맞음':>5}{'정밀도':>9}{'재현율':>9}")
    print("-" * 50)
    for d in sorted(set(gold) | set(pred)):
        pr = f"{tp[d] / pred[d] * 100:>7.0f}%" if pred[d] else f"{'—':>8}"
        rc = f"{tp[d] / gold[d] * 100:>7.0f}%" if gold[d] else f"{'—':>8}"
        print(f"{d:<14}{gold[d]:>5}{pred[d]:>5}{tp[d]:>5}{pr}{rc}")

    if show_errors:
        print("\n오분류")
        by = defaultdict(list)
        for r in rows:
            if r["gold_domain"] != r["pred"]:
                by[(r["gold_domain"], r["pred"])].append(r)
        for (g, p), items in sorted(by.items(), key=lambda x: -len(x[1])):
            print(f"\n  정답 {g} -> 예측 {p}  ({len(items)}건)")
            for r in items[:4]:
                print(f"    {r['host'][:30]:<30} {r['text_preview'][:58]}")
    return acc


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="감사 표본으로 도메인 규칙 채점")
    ap.add_argument("--audit", default="reports/tables/domain_audit_v5.tsv")
    ap.add_argument("--rules", default="configs/data/domain_rules.yaml")
    ap.add_argument("--holdout", action="store_true",
                    help="holdout 으로 채점한다. 최종 1회만 써라")
    ap.add_argument("--all", action="store_true", help="dev+holdout 전체")
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args(argv)

    rows = [r for r in load(ROOT / args.audit)
            if r.get("gold_domain") not in ("", "NA", None)]
    if args.all:
        split, subset = "전체", rows
    elif args.holdout:
        split, subset = "holdout", [r for r in rows if bucket(r["sample_id"]) >= DEV_FRACTION]
        print("!! holdout 은 최종 측정용이다. 이 숫자를 보고 규칙을 고치면 "
              "holdout 이 dev 가 된다.\n")
    else:
        split, subset = "dev", [r for r in rows if bucket(r["sample_id"]) < DEV_FRACTION]

    rules = DomainRules.load(ROOT / args.rules)
    print(f"규칙 {rules.version}  |  {split} {len(subset)}건\n")
    report(subset, rules, args.errors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
