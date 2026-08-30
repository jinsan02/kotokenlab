"""T2a / T2b vocab 수술의 무결성 (스펙 §12, §19).

실제 Qwen 을 받지 않고 작은 BPE 를 손으로 만들어 검사한다. 여기서 잡으려는
것은 모델이 아니라 **규칙의 일관성** 이다.

    vocab    a b c ab abc z
    merges   (a,b)->ab   (ab,c)->abc

    부품     a b ab c          <- 다른 규칙에 쓰인다
    잎       abc z             <- pruning 후보가 될 수 있는 것
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer.substitute import build, _merge_parts  # noqa: E402


def base_json() -> dict:
    return {
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "z": 5},
            "merges": ["a b", "ab c"],
        },
        "added_tokens": [{"id": 5, "content": "z", "special": True}],
    }


def merge_targets(tj: dict) -> set:
    return {"".join(_merge_parts(e)) for e in tj["model"]["merges"]}


def assert_consistent(tj: dict) -> None:
    """모든 merge 규칙의 부품과 결과가 vocab 안에 있어야 한다."""
    vocab = set(tj["model"]["vocab"])
    for entry in tj["model"]["merges"]:
        parts = _merge_parts(entry)
        assert set(parts) <= vocab, f"부품이 vocab 에 없다: {entry}"
        assert "".join(parts) in vocab, f"결과가 vocab 에 없다: {entry}"


def test_잎을_지우면_그_토큰을_만드는_규칙도_사라진다():
    tj, id_map, info = build(base_json(), [(4, "abc")], [], "t2a")
    assert "abc" not in tj["model"]["vocab"]
    assert "abc" not in merge_targets(tj)
    assert info["merges_dropped"] == 1
    assert_consistent(tj)


def test_t2a_는_번호를_빈틈없이_다시_매긴다():
    tj, id_map, info = build(base_json(), [(4, "abc")], [], "t2a")
    ids = sorted(tj["model"]["vocab"].values())
    assert ids == list(range(len(ids))), "ID 에 구멍이 있으면 embedding 행이 낭비된다"
    assert info["vocab_size"] == 5
    # z 는 5 -> 4 로 내려온다. added_tokens 의 id 도 따라와야 한다.
    assert tj["model"]["vocab"]["z"] == 4
    assert tj["added_tokens"][0]["id"] == 4
    assert id_map["5"] == 4


def test_t2b_는_vocab_크기를_유지하고_빈_슬롯에_넣는다():
    donors = [("a", "z", "az")]
    tj, id_map, info = build(base_json(), [(4, "abc")], donors, "t2b")
    assert info["vocab_size"] == 6, "크기 유지가 T2b 의 정의다"
    assert tj["model"]["vocab"]["az"] == 4, "비운 슬롯 그대로 들어가야 한다"
    assert "abc" not in tj["model"]["vocab"]
    assert "az" in merge_targets(tj)
    assert id_map["4"]["left"] == "a" and id_map["4"]["right"] == "z"
    assert_consistent(tj)


def test_t2b_새_merge_는_맨_뒤에_붙는다():
    """더 이른 rank 가 부품을 먼저 가져가면 새 토큰이 안 나온다 —
    순서를 바꾸면 기존 토큰화가 통째로 달라지므로 뒤에 붙인다."""
    donors = [("a", "z", "az")]
    tj, _, _ = build(base_json(), [(4, "abc")], donors, "t2b")
    assert tj["model"]["merges"][-1] == "a z"
    assert tj["model"]["merges"][0] == "a b", "기존 규칙의 순서는 보존된다"


def test_기증자가_모자라면_거부한다():
    with pytest.raises(ValueError, match="기증자가"):
        build(base_json(), [(4, "abc"), (5, "z")], [("a", "b", "ab2")], "t2b")


def test_vocab_에_없는_토큰은_지울_수_없다():
    with pytest.raises(ValueError, match="vocab 에 없는"):
        build(base_json(), [(99, "없는토큰")], [], "t2a")


def test_부품_토큰을_지우면_규칙이_끊긴다():
    """prune.py 가 잎만 고르는 이유. 부품인 ab 를 지우면 (ab,c)->abc 가 깨진다.

    build 는 이것을 막지 않는다 — 후보 선정 단계의 책임이다. 이 테스트는
    그 책임이 어디에 있는지 못 박아 둔다.
    """
    tj, _, _ = build(base_json(), [(3, "ab")], [], "t2a")
    with pytest.raises(AssertionError, match="부품이 vocab 에 없다"):
        assert_consistent(tj)


def test_t2b_는_같은_문자열로_합쳐지는_기증자를_거부한다():
    """("a","bc") 와 ("ab","c") 는 둘 다 "abc" 다. 둘 다 넣으면 뒤엣것이
    앞엣것의 슬롯을 덮어써서 vocab 이 조용히 줄고 크기 유지가 깨진다."""
    tj0 = base_json()
    tj0["model"]["vocab"]["bc"] = 6      # 부품 하나를 더 만들어 둔다
    donors = [("a", "bc", "abc2"), ("ab", "c", "abc2")]
    with pytest.raises(AssertionError, match="크기를 유지"):
        build(tj0, [(4, "abc"), (5, "z")], donors, "t2b")
