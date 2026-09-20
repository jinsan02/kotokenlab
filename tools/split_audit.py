"""분할이 새는지 manifest 로 확인한다 (학습 없음, 본문을 열지 않는다).

    .conda/python.exe tools/split_audit.py

## 왜

평가 문서가 학습 풀에 있으면 dev BPB 는 기억을 재는 것이 된다. 파이프라인이
문서 단위로 나눴다고 **적혀** 있지만, 2026-09-19 감사가 지적했듯 적힌 것과
실제가 같은지는 따로 확인해야 한다.

## final_test 를 어떻게 다루나

**본문을 열지 않는다.** `data/manifests/final_test.tsv` 의 `sha256` 열만 읽는다.
해시는 문서 내용을 드러내지 않고, 겹침 여부만 답한다. 이것은 RULES 2번이 막는
"최종 테스트 개봉" 이 아니다 — 개봉은 본문을 읽고 점수를 내는 것이다.

그래도 조심스러운 쪽으로: 이 도구는 **개수만** 출력하고 doc_id 나 해시를
찍지 않는다. 겹침이 나오면 그때 사람이 판단한다.

## 무엇을 세나

```
중복       같은 sha256 이 두 split 에 있는가 (정확 일치)
split 내부 한 split 안의 중복 문서
누수 방향  train ∩ dev · train ∩ final_test · dev ∩ final_test
```

**near-dup 은 이 도구가 못 잡는다.** MinHash 는 파이프라인이 split 전에 돌렸고,
그 결과를 다시 검증하려면 본문이 필요하다. 여기서는 정확 일치만 본다 —
그것조차 안 보고 넘어가지 않기 위해서다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFESTS = ROOT / "data" / "manifests"
OUT = ROOT / "reports" / "tables" / "split_audit.md"

FAMILIES = {
    "한국어": ("train", "dev", "final_test"),
    "영어 대조군": ("train_control_english", "dev_control_english",
                "final_test_control_english"),
    "코드 대조군": ("train_control_code", "dev_control_code",
               "final_test_control_code"),
}


def hashes(name: str) -> list:
    """manifest 의 sha256 열만 읽는다. 본문도 doc_id 도 쓰지 않는다."""
    path = MANIFESTS / f"{name}.tsv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [r["sha256"] for r in csv.DictReader(fh, delimiter="\t",
                                                    quoting=csv.QUOTE_NONE)]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="split 누수 감사 (manifest 해시만)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    o: list = []
    w = o.append
    w("# 분할 감사 — 평가 문서가 학습 풀에 있는가\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/split_audit.py`")
    w("> `data/manifests/*.tsv` 의 `sha256` 열만 읽는다. **본문을 열지 않는다** —")
    w("> `final_test` 도 해시만 본다 (RULES 2번의 \"개봉\" 이 아니다).\n")
    w("정확 일치만 본다. near-dup 은 파이프라인이 split 전에 MinHash 로 걸렀고,")
    w("그 결과를 다시 검증하려면 본문이 필요하므로 여기서는 하지 않는다.\n")

    leaks = 0
    dupes = 0
    for family, names in FAMILIES.items():
        w(f"## {family}\n")
        sets = {}
        w("| split | 문서 | 고유 해시 | split 내부 중복 |")
        w("|---|---:|---:|---:|")
        for name in names:
            hs = hashes(name)
            uniq = set(hs)
            sets[name] = uniq
            inner = len(hs) - len(uniq)
            dupes += inner
            w(f"| `{name}` | {len(hs):,} | {len(uniq):,} | "
              + (f"**{inner:,}**" if inner else "0") + " |")
        w("")
        w("| 겹침 | 문서 수 |")
        w("|---|---:|")
        pairs = [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]
        for a, b in pairs:
            n = len(sets.get(a, set()) & sets.get(b, set()))
            leaks += n
            w(f"| `{a}` ∩ `{b}` | " + (f"**{n:,}**" if n else "0") + " |")
        w("")

    w("## 판정\n")
    if leaks == 0 and dupes == 0:
        w("**샌 곳이 없다.** 어느 두 split 에도 같은 문서가 없고, split 안에서도")
        w("정확 중복이 없다. 문서 단위 분할이 설계대로 지켜졌다.\n")
    else:
        w(f"**겹침 {leaks:,}건 · split 내부 중복 {dupes:,}건.** 어느 문서인지는")
        w("여기 찍지 않는다 — 사람이 manifest 를 직접 보고 판단한다.\n")
    w("이 결과가 말하지 않는 것:\n")
    w("- **near-dup.** 문구만 조금 다른 문서는 해시가 다르므로 여기서 안 잡힌다")
    w("- **사전학습 오염.** Qwen 이 이 문서들을 봤는지는 이 도구로 알 수 없다")
    w("- **평가 문항 오염.** KMMLU 쪽은 `contamination.md` 가 따로 센다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(o))
    print(f"\n썼다 {dest}")
    return 0 if leaks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
