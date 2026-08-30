"""embedding 수술의 무결성 (스펙 §20~22).

실제 모델을 받지 않고 작은 행렬로 검사한다. 여기서 잡으려는 것은 가중치가
아니라 **행이 제자리로 갔는지, 빈 자리가 남지 않았는지** 다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.surgery.init_mean import component_ids, init_mean  # noqa: E402
from src.surgery.init_random import init_random, reference_stats  # noqa: E402
from src.surgery.init_weighted import init_weighted  # noqa: E402
from src.surgery.resize import (  # noqa: E402
    assert_filled, id_mapping, padded_vocab_size, rearrange)


class FakeTok:
    def __init__(self, vocab):
        self._v = dict(vocab)

    def get_vocab(self):
        return dict(self._v)


def test_id_매핑은_토큰_문자열로_맞춘다():
    """T2a 의 id_map 은 번호가 바뀐 것만 담는다. 그것만 보면 안 바뀐 행을 놓친다."""
    base = FakeTok({"a": 0, "b": 1, "c": 2, "dead": 3, "z": 4})
    new = FakeTok({"a": 0, "b": 1, "c": 2, "z": 3})       # dead 제거 후 번호 재배치
    m = id_mapping(base, new)
    assert m.tolist() == [0, 1, 2, 4], "z 는 4 -> 3 으로 내려온다"


def test_새로_생긴_자리는_음수로_표시된다():
    base = FakeTok({"a": 0, "b": 1})
    new = FakeTok({"a": 0, "b": 1, "새토큰": 2})
    assert id_mapping(base, new).tolist() == [0, 1, -1]


def test_padded_vocab_size_는_128_배수로_올린다():
    """Qwen 의 실제 행렬은 151,936 이지만 정렬만이면 151,680 이다.
    256칸은 학습 신호를 받은 적 없는 여분이라 T2a 가 물려받지 않는다."""
    assert padded_vocab_size(151665) == 151680
    assert padded_vocab_size(121665) == 121728
    assert padded_vocab_size(128) == 128, "이미 배수면 그대로"


def test_T2b_는_원본_행_수를_그대로_써야_한다():
    """padded_vocab_size 를 T2b 에 쓰면 151,936 -> 151,680 으로 256행이 잘려
    형상 불변이라는 정의가 깨진다. run_surgery 가 mode 로 갈라 쓰는 이유다."""
    assert padded_vocab_size(151665) != 151936


def test_rearrange_는_행을_옮기고_빈_자리를_알려준다():
    old = np.arange(15, dtype=np.float32).reshape(5, 3)
    mapping = np.array([0, 1, 2, 4])                 # 3번 토큰이 사라졌다
    new, todo = rearrange(old, mapping, 4)
    assert np.array_equal(new[3], old[4]), "z 의 행이 따라와야 한다"
    assert todo == []


def test_rearrange_의_빈_자리는_0_이_아니라_NaN_이다():
    """0 으로 두면 초기화를 빠뜨려도 저장이 되고 학습이 조용히 이상해진다."""
    old = np.arange(6, dtype=np.float32).reshape(2, 3)
    new, todo = rearrange(old, np.array([0, -1]), 2)
    assert todo == [1]
    assert np.isnan(new[1]).all()
    with pytest.raises(AssertionError, match="초기화되지 않은 행"):
        assert_filled(new)


def test_패딩_구간도_채워야_할_자리로_잡힌다():
    old = np.arange(6, dtype=np.float32).reshape(2, 3)
    new, todo = rearrange(old, np.array([0, 1]), 4)   # 4칸으로 패딩
    assert todo == [2, 3]


def test_E0_는_살아남은_행의_분포를_따른다():
    """config 의 initializer_range 를 쓰면 새 행만 노름이 어긋나서,
    무작위라서 나쁜 것인지 노름이 안 맞아서 나쁜 것인지 구별할 수 없다."""
    emb = np.full((600, 8), np.nan, dtype=np.float32)
    rng = np.random.default_rng(0)
    emb[:500] = rng.normal(0.0, 0.35, size=(500, 8))
    init_random(emb, list(range(500, 600)), seed=42)
    assert_filled(emb)
    _, std = reference_stats(emb, np.arange(500, 600))
    assert 0.28 < std < 0.42, f"참조 분포(0.35)에서 크게 벗어났다: {std}"


def test_E1_은_부품의_평균이다():
    emb = np.full((3, 2), np.nan, dtype=np.float32)
    src = np.array([[1.0, 2.0], [3.0, 6.0], [0.0, 0.0]], dtype=np.float32)
    emb[0], emb[1] = src[0], src[1]
    init_mean(emb, {2: [0, 1]}, src)
    assert np.allclose(emb[2], [2.0, 4.0])


def test_E1_은_원본에서_읽는다():
    """같은 배열에서 읽고 쓰면 채우는 순서에 따라 결과가 달라진다."""
    src = np.array([[1.0], [3.0], [0.0], [0.0]], dtype=np.float32)
    emb = src.copy()
    emb[2:] = np.nan
    init_mean(emb, {2: [0, 1], 3: [0, 2]}, src)
    assert np.allclose(emb[2], [2.0])
    assert np.allclose(emb[3], [0.5]), "3번이 갓 채운 2번을 참조하면 안 된다"


def test_component_ids_는_부품이_없으면_뺀다():
    id_map = {"5": {"left": "a", "right": "b"}, "6": {"left": "a", "right": "없음"}}
    got = component_ids(id_map, {"a": 0, "b": 1})
    assert got == {5: [0, 1]}


def test_E2_가중치는_합이_1_이라_노름이_보존된다():
    """정규화하지 않으면 초기화 방법의 차이가 아니라 노름의 차이를 재게 된다."""
    src = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    emb = np.full((3, 2), np.nan, dtype=np.float32)
    emb[0], emb[1] = src[0], src[1]
    nbytes = np.array([1, 7, 1])
    freq = np.array([10, 1000, 0])
    for scheme in ("uniform", "length", "freq"):
        init_weighted(emb, {2: [0, 1]}, src, nbytes, freq, scheme=scheme)
        assert np.isclose(emb[2][0], 1.0), f"{scheme}: 가중합이 1 이어야 한다"


def test_E2_역빈도는_흔한_부품의_지배를_막는다():
    """"있습니다." 가 "다." 쪽으로 끌려가지 않게 하는 것이 목적이다."""
    src = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
    emb = np.full((3, 1), np.nan, dtype=np.float32)
    emb[0], emb[1] = src[0], src[1]
    nbytes = np.array([1, 1, 1])
    freq = np.array([10, 100_000, 0])          # 1번이 훨씬 흔하다
    init_weighted(emb, {2: [0, 1]}, src, nbytes, freq, scheme="freq")
    assert emb[2][0] > 0.9, "희소한 0번 쪽으로 실려야 한다"


def test_모르는_가중_방식은_거부한다():
    with pytest.raises(ValueError, match="모르는 가중 방식"):
        init_weighted(np.zeros((1, 1)), {}, np.zeros((1, 1)),
                      np.ones(1, dtype=np.int64), np.ones(1, dtype=np.int64),
                      scheme="semantic")
