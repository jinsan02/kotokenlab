"""저빈도·저중요 토큰 pruning 후보 선정 (스펙 §19).

스펙의 조건은 셋이다.

    Pruning Candidate = 저빈도 + 저의미중요도 + 한국어/영어/code 핵심 아님

여기에 스펙에 없는 네 번째 조건을 더한다. **merge DAG 의 잎이어야 한다.**
근거는 analyze_vocab 의 docstring 에 적었다 — 부품 토큰을 지우면 그 부품으로
만들어지던 토큰이 도달 불가능해지고, 연쇄적으로 번진다.

"저의미중요도" 는 직접 재지 않는다. 대신 **세 코퍼스 모두에서 저빈도** 라는
조작적 정의를 쓴다. 영어나 코드에서 자주 쓰이는 토큰을 한국어 빈도만 보고
지우면, Candidate Gate 의 regression 조건을 스스로 무너뜨리기 때문이다.

T2a 와 T2b 는 **여기서 나온 같은 목록** 을 쓴다. 목록이 다르면 "축소 vs 치환"
이 아니라 서로 다른 두 토크나이저를 비교하는 것이 되어 신규성 주장이 성립하지
않는다.

    .conda/python.exe -m src.tokenizer.prune --tag v1 --report
    .conda/python.exe -m src.tokenizer.prune --tag v1 -n 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TAB = chr(9)
NL = chr(10)


def load_stats(tag: str, root: Path | None = None) -> list:
    """analyze_vocab 이 만든 token_stats.tsv 를 읽는다."""
    base = Path(root or ROOT)
    path = base / "artifacts" / "vocab_stats" / tag / "token_stats.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없다. 먼저 -m src.tokenizer.analyze_vocab 를 돌려라.")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip(NL).split(TAB)
        for line in fh:
            cells = line.rstrip(NL).split(TAB)
            row = dict(zip(header, cells))
            row["token_id"] = int(row["token_id"])
            row["token"] = json.loads(row["token"])
            for k in ("count_total", "count_ko", "count_en", "count_code"):
                row[k] = int(row[k])
            row["is_leaf"] = row["is_leaf"] == "1"
            row["is_protected"] = row["is_protected"] == "1"
            rows.append(row)
    return rows


def live_guard(repo: str, revision: str) -> tuple:
    """토크나이저에서 직접 보호 ID 와 model.vocab 을 읽는다.

    token_stats.tsv 의 is_protected 를 믿지 않는다. 그 컬럼은 만들 때의
    protected.py 판단이 굳어 있어서, 보호 규칙을 고쳐도 따라오지 않는다.
    실제로 added token 8개(`<tool_call>`, `<|fim_middle|>` 등)가 그 틈으로
    후보에 새어 나갔다. 여기서 살아 있는 토크나이저로 다시 확인한다.

    model.vocab 에 없는 토큰도 뺀다. added token 은 merge 로 만들어지지
    않으므로 vocab 수술의 대상이 아니다.
    """
    import json as _json
    from transformers import AutoTokenizer

    from src.tokenizer.protected import protected_token_ids

    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    model_vocab = set(_json.loads(tok.backend_tokenizer.to_str())["model"]["vocab"])
    return protected_token_ids(tok), model_vocab


def eligible(rows: list, en_floor: int, code_floor: int, ko_floor: int,
             guard: tuple | None = None) -> list:
    """다섯 조건을 모두 통과한 토큰을, 총빈도 오름차순으로."""
    prot, model_vocab = guard if guard else (set(), None)
    out = [
        r for r in rows
        if r["is_leaf"]
        and not r["is_protected"]
        and r["token_id"] not in prot
        and (model_vocab is None or r["token"] in model_vocab)
        and r["count_en"] < en_floor
        and r["count_code"] < code_floor
        and r["count_ko"] < ko_floor
    ]
    out.sort(key=lambda r: (r["count_total"], r["token_id"]))
    return out


def report(rows: list, cand: list) -> None:
    total_tokens = sum(r["count_total"] for r in rows)
    n_leaf = sum(1 for r in rows if r["is_leaf"])
    n_prot = sum(1 for r in rows if r["is_protected"])
    print(f"전체 {len(rows):,}  잎 {n_leaf:,}  보호 {n_prot:,}  후보 {len(cand):,}")
    print(f"코퍼스 총 토큰 {total_tokens:,}")
    print(f"{NL}상위 N 을 잘랐을 때 잃는 코퍼스 점유율:")
    print(f"  {'N':>8}  {'누적 출현':>14}  {'점유율':>9}  {'N 번째 빈도':>12}")
    run = 0
    marks = (1000, 2000, 5000, 10000, 20000, 30000, 50000)
    idx = 0
    for i, r in enumerate(cand, 1):
        run += r["count_total"]
        if idx < len(marks) and i == marks[idx]:
            print(f"  {i:>8,}  {run:>14,}  {run / max(total_tokens, 1):>8.4%}"
                  f"  {r['count_total']:>12,}")
            idx += 1
    if cand:
        print(f"  {len(cand):>8,}  {run:>14,}  {run / max(total_tokens, 1):>8.4%}"
              f"  {cand[-1]['count_total']:>12,}   (후보 전체)")


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="pruning 후보 선정")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("-n", "--num", type=int, default=0, help="제거할 토큰 수")
    ap.add_argument("--en-floor", type=int, default=100,
                    help="영어 코퍼스에서 이 횟수 이상이면 영어 핵심으로 보고 남긴다")
    ap.add_argument("--code-floor", type=int, default=100)
    ap.add_argument("--ko-floor", type=int, default=10 ** 12,
                    help="기본은 사실상 무제한 — 한국어 빈도는 순위로만 쓴다")
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--revision", default="060db6499f32faf8b98477b0a26969ef7d8b9987")
    ap.add_argument("--report", action="store_true", help="목록을 쓰지 않고 분포만 본다")
    args = ap.parse_args(argv)

    rows = load_stats(args.tag)
    guard = live_guard(args.repo, args.revision)
    naive = eligible(rows, args.en_floor, args.code_floor, args.ko_floor)
    cand = eligible(rows, args.en_floor, args.code_floor, args.ko_floor, guard)
    if len(naive) != len(cand):
        print(f"[방어] 통계 파일만 믿었으면 {len(naive) - len(cand)}개가 "
              f"후보에 새어 들어갔다 (보호 토큰 또는 added token)")
    report(rows, cand)

    if args.report:
        return 0
    if args.num <= 0:
        print(f"{NL}-n 으로 제거할 개수를 정해라 (분포는 --report).", file=sys.stderr)
        return 2
    if args.num > len(cand):
        print(f"{NL}후보가 {len(cand):,}개뿐이라 {args.num:,}개를 뽑을 수 없다.",
              file=sys.stderr)
        return 2

    chosen = cand[:args.num]
    out = ROOT / "artifacts" / "vocab_stats" / args.tag / f"prune_{args.num}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ("token_id", "token", "count_total", "count_ko", "count_en", "count_code")
    with out.open("w", encoding="utf-8", newline=NL) as fh:
        fh.write(TAB.join(cols) + NL)
        for r in chosen:
            fh.write(TAB.join((
                str(r["token_id"]), json.dumps(r["token"], ensure_ascii=False),
                str(r["count_total"]), str(r["count_ko"]),
                str(r["count_en"]), str(r["count_code"]))) + NL)
    lost = sum(r["count_total"] for r in chosen)
    print(f"{NL}{out}  {len(chosen):,}행")
    print(f"  이 목록을 T2a(제거만) 와 T2b(제거+치환) 가 함께 쓴다")
    print(f"  잃는 출현 {lost:,}건, 최대 빈도 {chosen[-1]['count_total']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
