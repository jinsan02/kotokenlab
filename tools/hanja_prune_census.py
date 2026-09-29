"""T2a 가 지운 토큰 가운데 한자가 얼마인지 센다 (학습 없음, GPU 0시간).

    .conda/python.exe tools/hanja_prune_census.py

## 왜

`docs/PLAN.md` "downstream 과 한자 프로브" 의 숫자(13,310개 · 44.4% · 단일 한자
48.0%)는 2026-09-13 에 손으로 센 값이고, 그 계산 경로가 저장소에 없었다.
2026-09-29 논문 스터디 정정에서 전체 한자 토큰 수(26,060)와 제거율(51.1%)이
더 필요해졌는데, 그것도 일회성 스크립트로 셌다. 인용할 숫자는 다시 돌릴 수
있어야 한다 (RULES 13번).

## 함정 — `csv` 모듈로 읽으면 9행이 사라진다

`prune_<N>.tsv` 와 `token_stats.tsv` 의 `token` 칸은 **JSON 문자열** 이다
(`"Ġ{čĊ"`). 토큰 자체에 `"` 가 든 경우(`"));čĊ` 등)가 있어서, `csv` 의 기본
인용 해석으로 읽으면 여러 줄이 한 행으로 붙는다. 2026-09-29 에 실제로 그렇게
읽어 30,000행을 29,991행으로 잘못 셌다. 저장소의 생성 코드
(`src/tokenizer/substitute.py` `_read_tsv`)는 탭으로 직접 쪼개므로 이 문제가
없다. 이 도구도 그렇게 읽고, 기본 인용 해석으로 읽었을 때의 행 수를 함께 적어
함정을 남긴다.

## 한자의 정의

CJK 통합 한자(U+4E00–9FFF), 확장 A(U+3400–4DBF), 호환 한자(U+F900–FAFF),
보조 평면 확장 B 이후(U+20000–2FFFF). 부수·획·기호 블록은 넣지 않는다.
ByteLevel 토큰을 UTF-8 로 되돌릴 때 음절 경계에서 잘린 바이트는 U+FFFD 로
바뀌므로 한자로 세지 않는다 — 조각난 토큰은 한자 "포함" 이 아니다.

## 판정하지 않는다

기술 집계다. 간체/정자 관측은 PLAN.md 등록대로 **예측 근거로만** 쓰고 결과로
인용하지 않는다. 이 도구는 상위 목록을 보여 줄 뿐 문자 체계를 분류하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.hashing import sha256_file, short  # noqa: E402

STATS = ROOT / "artifacts" / "vocab_stats" / "v1"
TOK = ROOT / "artifacts" / "tokenizers"
OUT = ROOT / "reports" / "tables" / "hanja_prune_census.md"

# Qwen2.5 의 BPE 기본 어휘는 0..151642 다. 그 위는 added token(특수 토큰)이다.
BASE_VOCAB = 151_643

HANJA_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FFFF),
)


def is_hanja(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in HANJA_RANGES)


def has_hanja(s: str) -> bool:
    return any(is_hanja(c) for c in s)


def pure_hanja(s: str) -> bool:
    return bool(s) and all(is_hanja(c) for c in s)


def _byte_decoder() -> dict:
    """GPT-2 ByteLevel 의 bytes_to_unicode 역표. transformers 를 부르지 않으려고 둔다."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_DEC = _byte_decoder()


def surface(field: str) -> str:
    """TSV 의 token 칸(JSON 문자열) -> 사람이 읽는 표면형."""
    tok = json.loads(field)
    raw = bytes(_DEC[c] for c in tok)
    return raw.decode("utf-8", errors="replace")


def read_tsv(path: Path) -> list:
    """탭으로 직접 쪼갠다. csv 의 인용 해석을 쓰지 않는다 (위 '함정')."""
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        for line in fh:
            rows.append(dict(zip(header, line.rstrip("\r\n").split("\t"))))
    return rows


def naive_csv_rows(path: Path) -> int:
    """csv 기본 인용 해석으로 읽었을 때의 행 수. 함정 기록용이다."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(io.StringIO(fh.read()), delimiter="\t"))


def census(stats: list, removed: set) -> dict:
    """기본 어휘의 한자 토큰을 세고 제거 집합과 겹친다."""
    tokens = {}
    for r in stats:
        i = int(r["token_id"])
        if i < BASE_VOCAB:
            tokens[i] = (surface(r["token"]), int(r["count_total"]))

    han = {i for i, (s, _) in tokens.items() if has_hanja(s)}
    han_rm = han & removed

    by_len = {}
    for i in han:
        s = tokens[i][0]
        if not pure_hanja(s):
            continue
        k = len(s) if len(s) <= 4 else 5
        rm, keep = by_len.get(k, (0, 0))
        by_len[k] = (rm + 1, keep) if i in removed else (rm, keep + 1)

    singles = [i for i in han if len(tokens[i][0]) == 1 and pure_hanja(tokens[i][0])]
    s_rm = [i for i in singles if i in removed]
    s_keep = [i for i in singles if i not in removed]

    def top(ids: list, n: int = 10) -> list:
        ids = sorted(ids, key=lambda i: (-tokens[i][1], i))[:n]
        return [(tokens[i][0], tokens[i][1]) for i in ids]

    rm_counts = [tokens[i][1] for i in s_rm]
    return {
        "base": len(tokens),
        "removed": len(removed),
        "hanja": len(han),
        "hanja_removed": len(han_rm),
        "by_len": by_len,
        "single": len(singles),
        "single_removed": len(s_rm),
        "single_rm_median": statistics.median(rm_counts) if rm_counts else None,
        "single_rm_max": max(rm_counts) if rm_counts else None,
        "single_rm_at_max": rm_counts.count(max(rm_counts)) if rm_counts else 0,
        "top_removed": top(s_rm),
        "top_kept": top(s_keep),
    }


def removed_from_id_map(path: Path) -> set:
    """T2a 토크나이저의 id_map.json 에서 실제로 빠진 기본 어휘 id 를 복원한다.

    map 에는 자리가 바뀐 id 만 있다(old -> new). 새 id 는 0 부터 빈틈없이
    이어지므로, 자리가 바뀐 가장 작은 새 id 보다 앞은 old == new 로 남은
    구간이다. 복원한 수가 `num_pruned` 와 다르면 멈춘다.
    """
    m = json.loads(path.read_text(encoding="utf-8"))
    moved = {int(k): int(v) for k, v in m["map"].items()}
    unchanged = set(range(min(moved.values()) if moved else BASE_VOCAB))
    removed = set(range(BASE_VOCAB)) - set(moved) - unchanged
    if len(removed) != int(m["num_pruned"]):
        raise SystemExit(f"id_map 복원 {len(removed)} != num_pruned {m['num_pruned']}")
    return removed


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "NA"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="T2a 제거분의 한자 집계")
    ap.add_argument("--num", type=int, default=30000)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--tokenizer", default="kot2a_v1_n30000")
    args = ap.parse_args(argv)

    stats_dir = ROOT / "artifacts" / "vocab_stats" / args.tag
    prune_path = stats_dir / f"prune_{args.num}.tsv"
    stats_path = stats_dir / "token_stats.tsv"
    idmap_path = TOK / args.tokenizer / "id_map.json"

    prune = read_tsv(prune_path)
    prune_ids = {int(r["token_id"]) for r in prune}
    idmap_removed = removed_from_id_map(idmap_path)
    naive = naive_csv_rows(prune_path)
    c = census(read_tsv(stats_path), prune_ids)

    L: list = []
    w = L.append
    w("# T2a 제거분의 한자 집계")
    w("")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/hanja_prune_census.py`")
    w("> 학습 없음, GPU 0시간. 기술 집계이며 판정하지 않는다.")
    w("")
    w("등록 맥락은 [`docs/PLAN.md`](../../docs/PLAN.md) \"downstream 과 한자 프로브\".")
    w("간체/정자 관측은 그 등록대로 **예측 근거로만** 쓰고 결과로 인용하지 않는다.")
    w("")
    w("## 입력")
    w("")
    w("| 파일 | sha256 |")
    w("|---|---|")
    for p in (prune_path, stats_path, idmap_path):
        w(f"| `{p.relative_to(ROOT).as_posix()}` | `{short(sha256_file(p))}` |")
    w("")
    w("## 제거 집합 대조")
    w("")
    w("```")
    w(f"prune_{args.num}.tsv 행 (탭으로 직접 읽음)     {len(prune):>7,}")
    w(f"  서로 다른 token_id                      {len(prune_ids):>7,}")
    w(f"  기본 어휘(< {BASE_VOCAB:,}) 밖의 id           "
      f"{sum(1 for i in prune_ids if i >= BASE_VOCAB):>7,}")
    w(f"{args.tokenizer}/id_map.json 에서 빠진 id    {len(idmap_removed):>7,}")
    w(f"  두 집합이 같은가                          {'예' if idmap_removed == prune_ids else '아니오'}")
    w("```")
    w("")
    w(f"**함정:** 같은 파일을 `csv` 기본 인용 해석으로 읽으면 **{naive:,}행** 이 된다.")
    w("`token` 칸이 JSON 문자열이고 `\"` 를 품은 토큰이 있어서, 여러 줄이 한 행으로")
    w("붙는다. 2026-09-29 에 이 방식으로 읽어 \"9행이 모자란다\" 고 잘못 보고했다.")
    w("파일도 T2a 도 30,000개가 맞다 — 틀린 것은 읽는 쪽이었다.")
    w("")
    w("## 한자 포함 토큰")
    w("")
    w("```")
    w(f"기본 어휘                         {c['base']:>7,}")
    w(f"한자 포함 토큰                    {c['hanja']:>7,}")
    w(f"  그중 T2a 가 제거                {c['hanja_removed']:>7,}   "
      f"한자 토큰의 {pct(c['hanja_removed'], c['hanja'])}")
    w(f"제거 {c['removed']:,}개 중 한자 포함        {c['hanja_removed']:>7,}   "
      f"제거분의 {pct(c['hanja_removed'], c['removed'])}")
    w("```")
    w("")
    w("### 순수 한자 토큰 — 글자 수별")
    w("")
    w("| 글자 수 | 제거 | 생존 | 제거율 |")
    w("|---:|---:|---:|---:|")
    for k in sorted(c["by_len"]):
        rm, keep = c["by_len"][k]
        w(f"| {k if k < 5 else '5+'} | {rm:,} | {keep:,} | {pct(rm, rm + keep)} |")
    w("")
    w("### 단일 한자")
    w("")
    w("```")
    w(f"단일 한자 토큰    {c['single']:,}")
    w(f"  제거            {c['single_removed']:,}   {pct(c['single_removed'], c['single'])}")
    w(f"  제거분 count_total 중앙값 {c['single_rm_median']:g} · 최대 {c['single_rm_max']}"
      f" (최대값 동률 {c['single_rm_at_max']:,}개)")
    w("```")
    w("")
    w("상위 10개 (count_total 내림차순, 같으면 id 순). 문자 체계는 분류하지 않는다.")
    w("제거 쪽 상위는 최대값 동률 안에서 id 순으로 자른 것이다 — 동률 안의 순서에는")
    w("뜻이 없다. PLAN.md 에 손으로 적은 \"상위\" 목록은 정렬 기준이 기록되지 않았고,")
    w("이 표와 구성이 다르다 (그 목록에는 최대값이 아닌 글자도 들어 있다).")
    w("")
    w("| | 제거 | 생존 |")
    w("|---:|---|---|")
    for n, (a, b) in enumerate(zip(c["top_removed"], c["top_kept"]), 1):
        w(f"| {n} | {a[0]} ({a[1]:,}) | {b[0]} ({b[1]:,}) |")
    w("")
    w("## 한계")
    w("")
    w("- `count_total` 은 `token_stats.tsv` 를 만든 코퍼스 표본의 발화 수다. PLAN.md 의")
    w("  dev 글자 중 한자 비율과는 다른 측정이다")
    w("- 한자 범위는 네 블록이다: U+3400–4DBF · U+4E00–9FFF · U+F900–FAFF ·")
    w("  U+20000–2FFFF. 강희 부수·CJK 획·기호 블록은 세지 않는다")
    w("- 한자 **포함** 은 한 글자라도 있으면 센다. 한글·영문과 섞인 토큰도 들어간다")
    w("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"썼다 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
