"""KMMLU 결과 옆의 오염 검사 범위 한 줄 — 숫자를 표에서 읽는다 (amendment §1)."""

import pytest

from tools.contamination_check import scope_line


def test_scope_line_reads_counts_from_generated_table(tmp_path):
    p = tmp_path / "c.md"
    p.write_text("과목 집합 `d3` · 문항 1,900 · CPT 풀 앞 50,000문서\n"
                 "  그중 덮임 >= 50%           2   <- 외웠을 수 있다\n", encoding="utf-8")
    s = scope_line(p)
    assert "2 / 1,900문항" in s and "앞 50,000문서" in s
    assert "Qwen 사전학습" in s and "의역" in s


def test_scope_line_is_none_without_table_and_stops_on_format_change(tmp_path):
    assert scope_line(tmp_path / "없음.md") is None
    p = tmp_path / "c.md"
    p.write_text("형식이 바뀐 표\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        scope_line(p)
