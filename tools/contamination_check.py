"""CPT 문서 풀이 KMMLU 문항과 겹치는지 센다 (학습 없음, GPU 0시간).

    .conda/python.exe tools/contamination_check.py
    .conda/python.exe tools/contamination_check.py --subjects rest

## 왜

`docs/DATA_SOURCES.md` 의 파이프라인 7단계("평가 문항과 n-gram 중복 문서 제거")는
**설계만 있고 구현되지 않았다** (2026-09-17 확인). 그래서 우리 CPT 코퍼스에
KMMLU 문항이 들어 있을 수 있고, 그 경우 과제 정확도는 기억의 몫을 포함한다.

D3 는 이미 돌았으므로 여기서는 **보고만** 한다. P3-F 는 판정 **전에** 이것을
돌리고, 겹친 문항이 있으면 그 문항을 뺀 판정도 함께 낸다
([`docs/PLAN.md`](../docs/PLAN.md) "P3-F").

## 무엇을 세나

```
기준    13-gram (문자 단위). 정규화는 공백 제거 + NFKC 소문자화
대상    문항의 **질문 본문** 만. 보기는 짧아 우연 일치가 많다
코퍼스  CPT 가 실제로 쓴 문서 풀 — train.jsonl 의 앞 --pool-docs 개
        (순서를 섞기 전이므로 집합은 같다)
```

**n-gram 하나가 걸리는 것으로는 아무 말도 못 한다.** 시험 문항에는 "다음 중 옳은
것은" 같은 상용구가 많아 그것만으로 문항의 절반이 걸린다 (실측: 6과목 44%,
그런데 걸린 n-gram 은 174종뿐이었다).

그래서 **덮임 비율** 을 같이 센다 — 그 문항의 전체 13-gram 중 몇 %가 풀에
나타났는가. 문항이 통째로 코퍼스에 있으면 이 값이 1 에 가깝다.

```
덮임 >= 50%   외운 것일 수 있다. P3-F 에서 빼고도 판정한다
덮임 20~50%   눈으로 본다
그 미만       상용구로 본다
```

Qwen 사전학습 데이터의 오염은 이 도구로 알 수 없다. **우리 CPT 풀만** 본다.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.capability import DATA, SUBJECT_SETS, read_items  # noqa: E402

OUT = ROOT / "reports" / "tables" / "contamination.md"
POOL = ROOT / "data" / "interim" / "docs" / "train.jsonl"
N = 13
SUSPECT = 0.5          # 이 이상 덮이면 "외웠을 수 있다" 로 따로 센다
WATCH = 0.2
MIN_GRAMS = 10         # 문항이 짧으면 덮임이 쉽게 올라간다 (4 gram 중 2개 = 50%)


def norm(s: str) -> str:
    """공백을 지우고 NFKC 소문자화. 줄바꿈·띄어쓰기 차이로 놓치지 않게."""
    return "".join(unicodedata.normalize("NFKC", s).lower().split())


def grams(s: str, n: int = N) -> set:
    t = norm(s)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="KMMLU 문항과 CPT 풀의 13-gram 겹침")
    ap.add_argument("--subjects", default="d3", choices=tuple(SUBJECT_SETS))
    ap.add_argument("--pool-docs", type=int, default=50_000,
                    help="CPT 가 쓴 문서 수 (본 run 들은 50,000)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    subjects = SUBJECT_SETS[args.subjects]
    items: list = []
    for s in subjects:
        path = DATA / f"{s}-test.csv"
        if not path.exists():
            continue
        for i, it in enumerate(read_items(path)):
            items.append((s, i, grams(it["question"])))
    if not items:
        raise SystemExit(f"{args.subjects} 과목의 test 파일이 없다 — 먼저 받아라")

    # 문항 n-gram 을 모아 두고 **코퍼스를 한 번만** 훑는다. 문서마다 n-gram 을
    # 만들어 집합으로 맞추는 쪽이 싸다 — 반대로 하면 문항 gram 수 x 문서 수가 된다.
    index: dict = {}
    for k, (s, i, gs) in enumerate(items):
        for g in gs:
            index.setdefault(g, []).append(k)

    hit_grams: dict = {}
    n_docs = 0
    with POOL.open(encoding="utf-8") as fh:
        for line in fh:
            if n_docs >= args.pool_docs:
                break
            if not line.strip():
                continue
            n_docs += 1
            text = norm(json.loads(line)["text"])
            for i in range(len(text) - N + 1):
                g = text[i:i + N]
                owners = index.get(g)
                if owners is not None:
                    hit_grams.setdefault(g, set()).update(owners)
            if n_docs % 10_000 == 0:
                print(f"  {n_docs:,}문서  겹친 n-gram {len(hit_grams):,}", flush=True)

    hit_items: dict = {}
    for g, owners in hit_grams.items():
        for k in owners:
            hit_items.setdefault(k, 0)
            hit_items[k] += 1

    cover = {k: c / max(len(items[k][2]), 1) for k, c in hit_items.items()}
    long_enough = {k for k in cover if len(items[k][2]) >= MIN_GRAMS}
    suspect = {k for k, v in cover.items() if v >= SUSPECT and k in long_enough}
    watch = {k for k, v in cover.items()
             if WATCH <= v < SUSPECT and k in long_enough}
    short_high = {k for k, v in cover.items()
                  if v >= SUSPECT and k not in long_enough}

    by_subject: dict = {}
    for k in hit_items:
        by_subject[items[k][0]] = by_subject.get(items[k][0], 0) + 1
    totals: dict = {}
    for s, _, _ in items:
        totals[s] = totals.get(s, 0) + 1

    o: list = []
    w = o.append
    w("# KMMLU 오염 검사 — CPT 문서 풀과의 13-gram 겹침\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/contamination_check.py`")
    w("> 학습 없음, GPU 0시간.\n")
    w("`DATA_SOURCES.md` 의 오염 제거 단계가 구현되지 않아 따로 센다.")
    w("문항의 **질문 본문** 13-gram(공백 제거·NFKC 소문자)이 CPT 풀 문서에")
    w("나타나는지 본다. **겹침이 곧 오염은 아니다** — 흔한 관용구도 걸린다.\n")
    w(f"과목 집합 `{args.subjects}` · 문항 {len(items):,} · CPT 풀 앞 {n_docs:,}문서\n")
    w("## 결과\n")
    w("```")
    w(f"n-gram 이 하나라도 걸린 문항  {len(hit_items):,} / {len(items):,}  "
      f"({len(hit_items) / len(items):.2%})")
    w(f"  그중 덮임 >= {SUSPECT:.0%}           {len(suspect):,}   <- 외웠을 수 있다")
    w(f"  그중 덮임 {WATCH:.0%} ~ {SUSPECT:.0%}         {len(watch):,}")
    w(f"  (문항이 짧아 제외: 13-gram {MIN_GRAMS}개 미만인데 덮임 {SUSPECT:.0%} 이상  "
      f"{len(short_high):,})")
    w(f"걸린 n-gram 종류             {len(hit_grams):,}")
    w("```\n")
    if hit_items and not suspect:
        w(f"**덮임 {SUSPECT:.0%} 를 넘는 문항이 하나도 없다.** 걸린 것은 상용구 쪽이다 —")
        w(f"문항 {len(hit_items):,}개가 n-gram {len(hit_grams):,}종을 나눠 쓰고 있다.\n")
    if by_subject:
        w("| 과목 | 겹친 문항 | 문항 | 비율 |")
        w("|---|---:|---:|---:|")
        for s in sorted(by_subject, key=lambda x: -by_subject[x]):
            w(f"| {s} | {by_subject[s]:,} | {totals[s]:,} | "
              f"{by_subject[s] / totals[s]:.1%} |")
        w("")
        top = sorted(cover.items(), key=lambda kv: -kv[1])[:10]
        w("덮임이 큰 문항 (과목, 행, 덮임, 걸린 gram) — 본문은 싣지 않는다 (CC-BY-ND)\n")
        w("```")
        for k, v in top:
            w(f"{items[k][0]:<40} row {items[k][1]:>4}   {v:6.1%}   "
              f"{hit_items[k]:>3} / {len(items[k][2]):>3} gram")
        w("```\n")
    else:
        w("**겹친 문항이 없다.**\n")
    w("## 읽는 법\n")
    w(f"- **덮임 {SUSPECT:.0%} 이상** 인 문항이 있으면 P3-F 는 그 문항을 뺀 판정도 낸다.")
    w("  둘이 다르면 뺀 쪽을 판정으로 쓴다 (PLAN \"P3-F\")")
    w("- n-gram 이 하나 걸린 것만으로는 아무것도 아니다. 덮임을 본다")
    w(f"- 13-gram 이 {MIN_GRAMS}개 미만인 짧은 문항은 덮임이 쉽게 올라간다 "
      "(4개 중 2개면 50%). 그래서 의심 집계에서 뺀다")
    w("- D3(6과목)는 이미 돌았으므로 **보고만** 한다. 결과를 바꾸지 않는다")
    w("- Qwen 사전학습 데이터의 오염은 이 도구로 알 수 없다. 우리 CPT 풀만 본다")
    w("- 13-gram 은 문자 기준이다. 짧은 수식·단위 표기는 우연 일치가 날 수 있다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(o))
    print(f"\n썼다  {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
