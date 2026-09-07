"""P2 R1~R3 의 행 손상 — 세 종류가 각자 약속한 것만 바꾸는지 본다.

모델을 안 띄운다. damage() 는 numpy 행렬만 다루므로 합성 행렬로 충분하고,
그래야 CI 에서도 돈다.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("damage_rows",
                                               ROOT / "scripts" / "damage_rows.py")
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)


def _fixture(n=64, dim=8, k=10):
    rng = np.random.default_rng(0)
    emb = rng.normal(0.0, 0.02, size=(n, dim)).astype(np.float32)
    # 손상 대상은 2..2+k. 부모는 0 과 1 로 고정해 mean 을 검산할 수 있게 한다
    rows = [{"token_id": str(i), "token": f"t{i}", "count_ko": "1"}
            for i in range(2, 2 + k)]
    parents = {f"t{i}": ("p0", "p1") for i in range(2, 2 + k)}
    vocab = {"p0": 0, "p1": 1}
    return emb, rows, parents, vocab


def test_permute_preserves_the_multiset_exactly():
    """분포도 노름도 그대로여야 한다. 바뀌는 것은 배정뿐이다."""
    emb, rows, parents, vocab = _fixture()
    ids = np.array([int(r["token_id"]) for r in rows])
    before = emb[ids].copy()

    info = dr.damage(emb, rows, "permute", parents, vocab, seed=42)

    after = emb[ids]
    assert info["fixed_points"] == 0, "자기 자리에 남은 행이 있으면 손상이 덜 들어간다"
    assert info["std_before"] == pytest.approx(info["std_after"])
    # 같은 벡터 집합인가 (순서만 다른가)
    assert np.allclose(np.sort(before, axis=0), np.sort(after, axis=0))
    assert not np.allclose(before, after)


def test_mean_is_the_average_of_the_two_parents():
    emb, rows, parents, vocab = _fixture()
    expect = (emb[0] + emb[1]) / 2.0

    dr.damage(emb, rows, "mean", parents, vocab, seed=42)

    for r in rows:
        assert np.allclose(emb[int(r["token_id"])], expect)


def test_random_matches_the_surviving_rows_spread():
    """1차 E0 과 같은 규약 — sigma 를 임의로 정하지 않고 살아남은 행에서 가져온다."""
    emb, rows, parents, vocab = _fixture(n=4096, dim=32, k=512)

    info = dr.damage(emb, rows, "random", parents, vocab, seed=42)

    assert info["std_after"] == pytest.approx(info["ref_std"], rel=0.1)


def test_untouched_rows_stay_untouched():
    for how in dr.HOWS:
        emb, rows, parents, vocab = _fixture()
        keep = emb[40:].copy()
        dr.damage(emb, rows, how, parents, vocab, seed=42)
        assert np.array_equal(emb[40:], keep), f"{how} 가 대상 밖 행을 건드렸다"


def test_row_selection_needs_parents_and_skips_protected():
    stats = [
        {"token_id": "0", "token": "a", "count_ko": "999", "is_protected": "1"},
        {"token_id": "1", "token": "bc", "count_ko": "100", "is_protected": "0"},
        {"token_id": "2", "token": "z", "count_ko": "500", "is_protected": "0"},
        {"token_id": "3", "token": "de", "count_ko": "300", "is_protected": "0"},
    ]
    parents = {"bc": ("b", "c"), "de": ("d", "e"), "a": ("x", "y")}
    id2tok = {0: "a", 1: "bc", 2: "z", 3: "de"}

    picked = dr.pick_rows(stats, parents, id2tok, 2)

    # a 는 protected 라 빠지고, z 는 부모가 없어 빠진다. 남은 둘이 빈도순
    assert [r["token"] for r in picked] == ["de", "bc"]
    with pytest.raises(SystemExit):
        dr.pick_rows(stats, parents, id2tok, 3)


def test_row_selection_rejects_a_stale_stats_file():
    """TSV 와 토크나이저가 어긋나면 조용히 진행하지 않는다."""
    stats = [{"token_id": "1", "token": "bc", "count_ko": "100",
              "is_protected": "0"}]
    parents = {"bc": ("b", "c")}

    with pytest.raises(SystemExit):
        dr.pick_rows(stats, parents, {1: "다른토큰"}, 1)
