"""E1 초기화가 FVT 와 정말 같은지 확인한다 (학습 없음, GPU 0시간).

    .conda/python.exe tools/fvt_check.py

## 왜

논문에서 E1(부품 평균)을 **FVT (Gee et al., 2022) baseline** 으로 재기술하면
"초기화 baseline 과 비교하지 않았다" 는 지적 하나가 사라진다. 새 실험 0건이다.

그런데 **정말 같은지** 를 확인하지 않고 이름을 붙이면 안 된다. 두 방법은
"새 토큰을 조각으로 쪼개 임베딩을 평균한다" 까지 같고, **조각을 어디서
가져오는가** 가 다르다.

    FVT   새 토큰의 표면형을 원 토크나이저로 **다시 토큰화** 한 결과
    E1    id_map.json 의 **merge 부품 둘** (left, right)

T2b 는 새 토큰 하나 = merge 하나라 보통 일치한다. 그러나 BPE 가 그 문자열을
다르게 쪼갤 수 있으므로 **구성상 보장되지 않는다.** `src/surgery/init_mean.py`
docstring 이 "다시 토큰화해서 알아내야 하는 불확실성이 없다" 고 적어 둔 것이
바로 이 차이다 — 의도적으로 다른 경로를 택했다.

이 도구가 불일치율을 센다. 결과에 따라 논문 표기가 갈린다.

    100%      "FVT 와 동등하다" 로 쓴다
    그 미만    "FVT 와 동등하되 분해를 merge 계보에서 취한다" + 불일치율 각주

## E2 는 FOCUS 가 아니다 (이 도구가 재지 않는 것)

FOCUS 는 겹치는 어휘에 대해 **보조 임베딩(fasttext) 유사도** 로 희소 볼록결합을
만든다. E2 는 같은 부품 둘을 **코퍼스 빈도의 역수** 로 가중한다. 유사도가
아니라 빈도이므로 "FOCUS 의 단순판" 으로 쓸 수 없다. 이건 코드를 읽으면
바로 갈리는 문제라 측정할 것이 없다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT = ROOT / "reports" / "tables" / "fvt_check.md"
DEFAULT_MAP = ROOT / "artifacts" / "tokenizers" / "kot2b_v2_n30000" / "id_map.json"


def byte_decoder() -> dict:
    """byte-level BPE 문자 -> 바이트. GPT-2 bytes_to_unicode 의 역이다."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def surface(tok: str, dec: dict) -> str | None:
    """토큰 문자열을 실제 텍스트로. 바이트가 UTF-8 로 안 떨어지면 None."""
    try:
        raw = bytes(dec[ch] for ch in tok)
    except KeyError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1 이 FVT 와 같은지 확인")
    ap.add_argument("--id-map", default=str(DEFAULT_MAP))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.id_map).read_text(encoding="utf-8"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["base"], revision=spec["revision"])

    dec = byte_decoder()
    total = match = partial_undecodable = 0
    mismatches: list = []
    for slot, info in spec["map"].items():
        total += 1
        left, right = info["left"], info["right"]
        text = surface(info["token"], dec)
        if text is None:
            # 조각난 UTF-8 — 단일 한글 음절이 3바이트라 흔하다. 재토큰화가
            # 정의되지 않으므로 일치/불일치 어느 쪽으로도 세지 않는다.
            partial_undecodable += 1
            continue
        got = tok.tokenize(text)
        if got == [left, right]:
            match += 1
        elif len(mismatches) < 12:
            mismatches.append((text, [left, right], got))
    comparable = total - partial_undecodable
    rate = match / comparable * 100 if comparable else 0.0
    n_mis = comparable - match

    o: list = []
    w = o.append
    w("# E1 초기화는 FVT 와 같은가\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/fvt_check.py`")
    w("> 학습 없음, GPU 0시간.\n")
    w("FVT (Gee et al., 2022) 는 새 토큰의 표면형을 **원 토크나이저로 다시**")
    w("**토큰화** 해 나온 조각들의 임베딩을 평균한다. E1 은 `id_map.json` 의")
    w("**merge 부품 둘** 을 쓴다. 같은 결과가 나오는지 직접 세었다.\n")
    w(f"대상: `{Path(args.id_map).parent.name}` 의 새 토큰 {total:,}개")
    w(f"기준 토크나이저: `{spec['base']}` @ `{spec['revision'][:12]}`\n")
    w("## 결과\n")
    w("```")
    w(f"새 토큰                {total:7,}")
    w(f"  재토큰화 정의 안 됨   {partial_undecodable:7,}   조각난 UTF-8 (아래 설명)")
    w(f"  비교 가능            {comparable:7,}")
    w(f"    [left, right] 일치 {match:7,}   {rate:.2f}%")
    w(f"    불일치             {n_mis:7,}   {100-rate:.2f}%")
    w("```\n")

    if partial_undecodable:
        w("### \"재토큰화 정의 안 됨\" 이 무엇인가\n")
        w("ByteLevel BPE 의 토큰은 **바이트열** 이지 문자가 아니다. 한글 음절 하나가")
        w("UTF-8 3바이트라, merge 로 만들어진 새 토큰이 음절 경계에 안 맞으면 그")
        w("토큰만으로는 유효한 문자열이 되지 않는다. 그런 토큰은 \"표면형을 다시")
        w("토큰화한다\" 는 연산 자체가 정의되지 않는다.\n")
        w(f"**{partial_undecodable:,}개 ({partial_undecodable/total*100:.1f}%)가 여기 해당한다.**")
        w("FVT 를 이 조건에 그대로 적용할 수 없다는 뜻이고, E1 이 merge 계보를")
        w("쓰기로 한 것이 회피가 아니라 **필요** 였다는 근거다.\n")

    if mismatches:
        w("### 불일치 예시\n")
        w("| 표면형 | E1 이 쓴 부품 | 재토큰화 결과 |")
        w("|---|---|---|")
        for text, exp, got in mismatches:
            w(f"| `{text}` | `{exp}` | `{got}` |")
        w("")

    w("## 논문 표기\n")
    if n_mis == 0 and partial_undecodable == 0:
        w("> E1 은 **FVT 와 동등하다.** 모든 새 토큰에서 merge 부품이 재토큰화")
        w("> 결과와 일치한다.")
    else:
        w("> E1 은 **FVT 와 동등하되, 분해를 재토큰화가 아니라 merge 계보에서**")
        w("> **취한다.** T2b 는 새 토큰 하나 = merge 하나라 두 경로가 대체로")
        w(f"> 일치한다 (비교 가능한 {comparable:,}개 중 {rate:.1f}%). 나머지는")
        w("> 표면형이 유효한 UTF-8 이 아니어서 재토큰화가 정의되지 않거나,")
        w("> BPE 가 다른 분해를 내놓는 경우다.")
    w("")
    w("어느 쪽이든 **\"FVT 를 재현했다\" 가 아니라 \"FVT 와 같은 계열이며 분해")
    w("경로가 다르다\"** 로 쓴다. 각주에 이 표를 건다.\n")
    w("## E2 는 FOCUS 가 아니다\n")
    w("FOCUS 는 겹치는 어휘에 대해 **보조 임베딩 유사도** 로 희소 볼록결합을")
    w("만든다. E2(`src/surgery/init_weighted.py`)는 같은 부품 둘을 **코퍼스 빈도의**")
    w("**역수** 로 가중한다. 유사도가 아니라 빈도다.")
    w("**\"FOCUS 의 단순판\" 으로 쓰면 안 된다** — \"가중 평균 계열, 가중치 출처가")
    w("다름\" 까지만.\n")
    w("## 한계\n")
    w("- 재토큰화는 **토큰 하나를 단독으로** 넣은 결과다. 실제 문맥에서는 앞뒤")
    w("  글자와 함께 다른 분해가 나올 수 있다. FVT 원문도 단독 표면형을 쓴다")
    w("- `add_special_tokens` 없이 `tokenize()` 를 부른다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"썼다  {dest}")
    print(f"  비교 가능 {comparable:,} / {total:,}   일치 {match:,} ({rate:.2f}%)")
    print(f"  재토큰화 정의 안 됨 {partial_undecodable:,} ({partial_undecodable/total*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
