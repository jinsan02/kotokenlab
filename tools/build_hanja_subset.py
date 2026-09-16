"""한자 밀집 한국어 부분집합을 만든다 — D1·D2 의 입력 (학습 없음).

    .conda/python.exe tools/build_hanja_subset.py

## 무엇을 만드나

`data/interim/docs/dev.jsonl` 에서 **한자 >= 20자 AND 한자 비율 >= 0.5%** 인
문서만 골라 `data/interim/docs/dev_hanja.jsonl` 에 쓴다. 기준은 2026-09-13
`docs/PLAN.md` "downstream 과 한자 프로브" 에 등록한 그대로다. 여기서 바꾸지
않는다.

등록 시점에 센 값은 **787문서 6.71MB** 다. 다른 값이 나오면 dev.jsonl 이
바뀌었거나 이 스크립트가 등록과 다르게 세는 것이다 — 멈추고 확인한다.

## 왜 파일로 떨어뜨리나

`src/evaluation/bpb.py` 와 `src/evaluation/tokenizer_eval.py` 가 둘 다
`--split <이름>` 으로 `data/interim/docs/<이름>.jsonl` 을 읽는다. 파일만 있으면
D1(tok/byte)과 D2(BPB)가 기존 도구로 그대로 돈다.

`dev.jsonl` 평가를 오염시키지 않는다 — 두 도구 모두 파일명이 정확히
`dev.jsonl` 이거나 `dev_control_` 로 시작하는 것만 dev 로 읽는다.

## 읽지 않는 것

**`dev.jsonl` 하나만 연다.** `final_test*` 는 경로 조립에도 쓰지 않는다
(RULES 2번).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "interim" / "docs"
SRC = DOCS / "dev.jsonl"
DST = DOCS / "dev_hanja.jsonl"

# 2026-09-13 PLAN.md 등록값. 바꾸면 사전 등록이 아니다.
MIN_HAN = 20
MIN_RATIO = 0.005
REGISTERED_DOCS = 787


def is_han(ch: str) -> bool:
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
            or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="한자 밀집 한국어 부분집합")
    ap.add_argument("--force", action="store_true",
                    help="등록 문서 수와 달라도 파일을 쓴다 (쓰지 마라)")
    args = ap.parse_args(argv)

    if "final_test" in str(SRC):          # 방어. 경로가 바뀌어도 여기서 막힌다
        raise SystemExit("final_test 는 읽지 않는다")

    kept: list = []
    n_all = 0
    with SRC.open(encoding="utf-8") as fh:
        for line in fh:
            n_all += 1
            d = json.loads(line)
            t = d.get("text") or ""
            if not t:
                continue
            h = sum(1 for ch in t if is_han(ch))
            if h >= MIN_HAN and h / len(t) >= MIN_RATIO:
                kept.append(line if line.endswith("\n") else line + "\n")

    body = "".join(kept).encode("utf-8")
    n_bytes = sum(len(json.loads(x)["text"].encode("utf-8")) for x in kept)
    print(f"dev.jsonl        {n_all:,}문서")
    print(f"부분집합         {len(kept):,}문서   본문 {n_bytes / 1e6:.2f}MB")
    print(f"기준             한자 >= {MIN_HAN}자 AND 비율 >= {MIN_RATIO:.1%}")
    print(f"등록값           {REGISTERED_DOCS}문서")

    if len(kept) != REGISTERED_DOCS and not args.force:
        print("\n  등록 시점과 문서 수가 다르다. 쓰지 않는다.", file=sys.stderr)
        return 1

    DST.write_bytes(body)
    print(f"\n썼다             {DST.relative_to(ROOT)}")
    print(f"sha256           {hashlib.sha256(body).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
