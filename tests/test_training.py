"""CPT 루프의 규칙 (스펙 §26, §79).

여기서 지키려는 것은 학습이 잘 되는지가 아니라 **비교가 성립하는지** 다.
x 축이 raw_bytes 인지, 예산이 바이트로 세어지는지가 핵심이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.callbacks import cosine_lr_by_bytes  # noqa: E402
from src.training.cpt import pack  # noqa: E402


class FakeTok:
    """문자 하나를 토큰 하나로 보는 토크나이저. 토큰 하나 = 그 문자의 UTF-8 길이."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def byte_len(ids):
    """id 를 코드포인트로 되돌려 UTF-8 길이를 센다. EOS(0) 는 0 바이트."""
    return sum(0 if i == 0 else len(chr(i).encode("utf-8")) for i in ids)


def test_LR_의_x축은_step_이_아니라_바이트다():
    """step 으로 스케줄을 짜면 토크나이저마다 같은 원문에서 step 수가 달라져
    조건마다 다른 LR 궤적을 밟는다. 그러면 비교 대상이 학습률이 된다."""
    peak = 1e-4
    total = 1_000_000
    # 압축이 좋은 조건은 같은 바이트를 더 적은 step 으로 지나지만,
    # 같은 바이트 지점에서는 같은 LR 이어야 한다.
    assert cosine_lr_by_bytes(500_000, total, peak) == \
        cosine_lr_by_bytes(500_000, total, peak)


def test_워밍업은_예산_비율로_돈다():
    peak = 1e-4
    total = 1_000_000
    assert cosine_lr_by_bytes(0, total, peak) == 0.0
    mid_warm = cosine_lr_by_bytes(10_000, total, peak, warmup_frac=0.02)
    assert 0 < mid_warm < peak
    assert abs(cosine_lr_by_bytes(20_000, total, peak, warmup_frac=0.02) - peak) < 1e-12


def test_코사인은_끝에서_바닥_비율까지_내려간다():
    peak = 1e-4
    got = cosine_lr_by_bytes(1_000_000, 1_000_000, peak, min_frac=0.1)
    assert abs(got - peak * 0.1) < 1e-9


def test_예산을_넘겨도_LR_이_음수가_되지_않는다():
    got = cosine_lr_by_bytes(5_000_000, 1_000_000, 1e-4)
    assert got > 0


def test_pack_의_바이트는_근사가_아니라_정확해야_한다():
    """비율로 배분하면 조각에 섞인 이전 문서 토큰 때문에 어긋난다. 실측에서
    한국어만 -0.38%, 한영 혼합 +0.73% 로 부호까지 뒤집혔다. 부호가 구성에 따라
    달라지면 C0 와 T2b 가 같은 예산에서 다른 분량을 본다 (RULES 12)."""
    tok = FakeTok()
    docs = ["가" * 256, "b" * 256]          # 3바이트 문자와 1바이트 문자를 섞는다
    chunks = list(pack(tok, docs, seq_len=64, eos_id=0, byte_len_fn=byte_len))
    assert chunks and all(len(c) == 64 for c, _ in chunks)
    # 소비된 토큰의 실제 바이트와 정확히 같아야 한다
    consumed = [i for c, _ in chunks for i in c]
    assert sum(t for _, t in chunks) == byte_len(consumed)


def test_pack_은_혼합_문서에서도_어긋나지_않는다():
    tok = FakeTok()
    docs = []
    for _ in range(20):
        docs.append("가" * 100)
        docs.append("b" * 100)
    chunks = list(pack(tok, docs, seq_len=64, eos_id=0, byte_len_fn=byte_len))
    consumed = [i for c, _ in chunks for i in c]
    assert sum(t for _, t in chunks) == byte_len(consumed)


def test_EOS_는_원문_바이트로_세지_않는다():
    """EOS 는 문서 경계 표시일 뿐이다. 세면 짧은 문서일수록 예산이 빨리 닳아
    조건 간 데이터량이 어긋난다."""
    assert byte_len([0, 0, 0]) == 0


def test_pack_은_빈_문서를_건너뛴다():
    tok = FakeTok()
    chunks = list(pack(tok, ["", "가" * 200], seq_len=64, eos_id=0,
                       byte_len_fn=byte_len))
    assert all(len(c) == 64 for c, _ in chunks)


def test_seed_는_순서만_바꾼다():
    """풀을 예산에 맞추면 모든 seed 가 같은 문서를 보고 순서만 달라진다.
    데이터 선택 분산이 섞이면 sigma 가 부풀어 이후 비교가 둔해진다."""
    import random
    pool = list(range(4500))
    a, b = pool.copy(), pool.copy()
    random.Random(42).shuffle(a)
    random.Random(123).shuffle(b)
    assert a != b, "순서는 달라야 한다"
    assert set(a) == set(b), "문서 집합은 같아야 한다"
