"""D3 KMMLU 프로브의 순수 함수. 실제 데이터 없이 합성 문항으로 확인한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.capability import (  # noqa: E402
    build_prompt,
    format_item,
    paired_diff_ci,
    read_items,
    verdict,
)


def _item(q: str, gold: int = 0) -> dict:
    return {"question": q, "choices": ["가", "나", "다", "라"], "gold": gold}


def test_format_item_ends_open_without_answer():
    s = format_item(_item("질문", gold=2), with_answer=False)
    assert s.endswith("정답:")
    assert "A. 가" in s and "D. 라" in s
    assert format_item(_item("질문", gold=2), with_answer=True).endswith("정답: C")


def test_build_prompt_uses_five_shots_when_they_fit():
    shots = [_item(f"예시{i}") for i in range(5)]
    prompt, k = build_prompt(shots, _item("본문"), n_tokens=len, limit=10_000)
    assert k == 5
    assert prompt.index("예시0") < prompt.index("예시4") < prompt.index("본문")
    assert prompt.endswith("정답:")


def test_build_prompt_drops_earliest_shots_first():
    shots = [_item(f"예시{i}" + "x" * 40) for i in range(5)]
    full, _ = build_prompt(shots, _item("본문"), n_tokens=len, limit=10_000)
    prompt, k = build_prompt(shots, _item("본문"), n_tokens=len, limit=len(full) - 10)
    assert k == 4
    assert "예시0" not in prompt and "예시4" in prompt


def test_build_prompt_never_truncates_the_question():
    prompt, k = build_prompt([_item("예시")], _item("본문" * 100), n_tokens=len, limit=5)
    assert k == 0
    assert prompt == format_item(_item("본문" * 100), with_answer=False)


def test_read_items_maps_answer_and_rejects_bad(tmp_path):
    p = tmp_path / "x-test.csv"
    p.write_text("question,answer,A,B,C,D,Category\n질문,4,a,b,c,d,law\n", encoding="utf-8")
    assert read_items(p)[0]["gold"] == 3
    p.write_text("question,answer,A,B,C,D\n질문,5,a,b,c,d\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_items(p)
    p.write_text("question,A,B,C,D\n질문,a,b,c,d\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_items(p)


def test_paired_diff_is_exact_mean_difference():
    a = [1, 1, 0, 1] * 50
    b = [1, 0, 0, 1] * 50
    d, lo, hi = paired_diff_ci(a, b, n_boot=2000)
    assert d == pytest.approx(0.25)
    assert lo <= d <= hi


def test_verdict_rules():
    assert verdict(0.25, -0.01, 0.01) == "측정 불가 — 바닥"
    assert verdict(0.30, -0.015, 0.019) == "같다"
    assert verdict(0.30, -0.05, -0.001) == "T2a 저하"
    assert verdict(0.30, 0.001, 0.05) == "T2a 향상"
    assert verdict(0.30, -0.03, 0.01) == "분해되지 않음"
