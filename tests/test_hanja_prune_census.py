"""tools/hanja_prune_census.py — 한자 판별, TSV 읽기, id_map 복원."""

import json

from tools.hanja_prune_census import (
    _DEC,
    census,
    has_hanja,
    naive_csv_rows,
    pure_hanja,
    read_tsv,
    removed_from_id_map,
    surface,
)


_ENC = {b: c for c, b in _DEC.items()}


def _bl_bytes(raw: bytes) -> str:
    """바이트열 -> TSV token 칸(JSON 문자열로 감싼 ByteLevel 형)."""
    return json.dumps("".join(_ENC[b] for b in raw), ensure_ascii=False)


def _bl(s: str) -> str:
    return _bl_bytes(s.encode("utf-8"))


def test_hanja_ranges():
    assert has_hanja("國") and has_hanja("韓국") and has_hanja("𠀀")  # 확장 B
    assert not has_hanja("한국") and not has_hanja("abc") and not has_hanja("⼀")  # 강희 부수
    assert pure_hanja("國無") and not pure_hanja("國a") and not pure_hanja("")


def test_surface_roundtrip_and_broken_bytes():
    assert surface(_bl("國")) == "國"
    # 음절 경계에서 잘린 바이트는 U+FFFD 가 되어 한자로 세지 않는다
    assert not has_hanja(surface(_bl_bytes("國".encode("utf-8")[:2])))


def test_read_tsv_survives_quotes_that_break_csv(tmp_path):
    p = tmp_path / "prune.tsv"
    lines = ["token_id\ttoken\tcount_total"]
    lines += [f'{i}\t{json.dumps(t)}\t0' for i, t in
              enumerate(['a"', 'b', 'c', 'd'])]  # " 로 끝나는 토큰 -> JSON 끝이 \""
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert len(read_tsv(p)) == 4
    assert naive_csv_rows(p) < 4  # 이 함정을 기록하려고 도구가 따로 센다


def test_census_counts(tmp_path):
    stats = [
        {"token_id": "0", "token": _bl("國"), "count_total": "9"},
        {"token_id": "1", "token": _bl("竖"), "count_total": "1"},
        {"token_id": "2", "token": _bl("國家"), "count_total": "3"},
        {"token_id": "3", "token": _bl("한國"), "count_total": "0"},
        {"token_id": "4", "token": _bl("ab"), "count_total": "5"},
        {"token_id": "151643", "token": _bl("國"), "count_total": "0"},  # added, 제외
    ]
    c = census(stats, removed={1, 3, 4})
    assert c["base"] == 5
    assert c["hanja"] == 4 and c["hanja_removed"] == 2
    assert c["single"] == 2 and c["single_removed"] == 1
    assert c["by_len"] == {1: (1, 1), 2: (0, 1)}
    assert c["top_removed"] == [("竖", 1)] and c["top_kept"] == [("國", 9)]


def test_removed_from_id_map(tmp_path):
    # old 0..5, 2·4 를 제거 -> 0,1 은 그대로, 3->2, 5->3
    p = tmp_path / "id_map.json"
    p.write_text(json.dumps({"num_pruned": 2, "map": {"3": 2, "5": 3}}), encoding="utf-8")
    import tools.hanja_prune_census as m
    old = m.BASE_VOCAB
    m.BASE_VOCAB = 6
    try:
        assert removed_from_id_map(p) == {2, 4}
    finally:
        m.BASE_VOCAB = old
