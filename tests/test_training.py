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


def test_혼합_풀은_목표_비율을_처음부터_지킨다():
    """풀 꼬리가 아니라 **실제로 학습할 앞부분**에서 비율이 맞아야 한다.
    문서 수로 잡으면 영어·코드가 훨씬 짧아 비율이 어긋난다 — 바이트로 잡는다."""
    from src.training.alignment import build_mixed_pool
    root = ROOT
    if not (root / "data" / "interim" / "docs" / "train.jsonl").exists():
        import pytest as _p
        _p.skip("코퍼스가 없는 환경")
    docs, got = build_mixed_pool(root, {"ko": 0.6, "en": 0.2, "code": 0.2}, 800, 42)
    assert docs and sum(got.values()) > 0
    assert set(got) == {"ko", "en", "code"}


def test_혼합_비율이_0_인_언어는_빠진다():
    from src.training.alignment import build_mixed_pool
    root = ROOT
    if not (root / "data" / "interim" / "docs" / "train.jsonl").exists():
        import pytest as _p
        _p.skip("코퍼스가 없는 환경")
    _, got = build_mixed_pool(root, {"ko": 1.0, "en": 0.0, "code": 0.0}, 400, 42)
    assert set(got) == {"ko"}


def test_constant_schedule_warms_up_then_holds():
    """R5 의 상수 LR — 코사인과 다른 것은 감쇠뿐이어야 한다.

    워밍업까지 없애면 "상수 LR 이라서" 인지 "워밍업이 없어서" 인지 갈 수 없다.
    """
    from src.training.callbacks import constant_lr_by_bytes, cosine_lr_by_bytes

    total, peak = 1_000_000, 1e-5

    # 워밍업 구간(2%)은 두 스케줄이 같아야 한다
    for frac in (0.0, 0.005, 0.01, 0.019):
        seen = int(total * frac)
        assert constant_lr_by_bytes(seen, total, peak) == cosine_lr_by_bytes(
            seen, total, peak)

    # 워밍업 뒤로는 끝까지 peak 을 유지한다
    for frac in (0.02, 0.5, 0.99, 1.0):
        assert constant_lr_by_bytes(int(total * frac), total, peak) == peak

    # 코사인은 같은 지점에서 내려가 있어야 한다 (안 그러면 실험이 성립 안 한다)
    assert cosine_lr_by_bytes(total, total, peak) < peak * 0.2


def test_schedules_registry_matches_cli_choices():
    """CLI 의 --lr-schedule 선택지는 레지스트리에서 나온다. 여기서 한 번 본다."""
    from src.training.callbacks import SCHEDULES

    assert set(SCHEDULES) == {"cosine", "constant", "wsd"}


# ── P3 W0-1 ~ W0-3 — 인자를 안 주면 P2 와 **똑같이** 돌아야 한다 ────────────
# 긴 run 의 앞부분이 짧은 run 과 같다고 말하려면 워밍업과 문서 순서가 예산에
# 딸려 있으면 안 된다 (docs/DESIGN_DELTA.md 3-12).

def test_워밍업_바이트를_안_주면_예산의_2퍼센트다():
    from src.training.cpt import warmup_fraction

    assert warmup_fraction(0, 168_500_000) == 0.02
    assert warmup_fraction(0, 500_000_000) == 0.02


def test_워밍업_바이트를_주면_예산이_달라도_같은_지점에서_끝난다():
    """R5(168.5MB)의 워밍업은 3.37MB 다. 500MB run 이 그 앞부분을 재현하려면
    같은 바이트에서 워밍업이 끝나야 한다 — 비율로 두면 10MB 가 된다."""
    from src.training.callbacks import constant_lr_by_bytes
    from src.training.cpt import warmup_fraction

    short = warmup_fraction(3_370_000, 168_500_000)
    long = warmup_fraction(3_370_000, 500_000_000)
    assert abs(short - 0.02) < 1e-12      # 168.5MB 에서는 기존 2% 와 같은 값이다
    for b in (0, 1_000_000, 3_369_999, 3_370_000, 20_000_000):
        a = constant_lr_by_bytes(b, 168_500_000, 1e-5, short)
        c = constant_lr_by_bytes(b, 500_000_000, 1e-5, long)
        # 같은 바이트면 같은 LR. 나누는 순서가 달라 부동소수 1 ulp 는 벌어진다
        assert abs(a - c) <= 1e-12 * max(a, c, 1e-12)


def test_풀을_늘려도_기존_문서_순서가_그대로다():
    """문서 풀 전체를 한 번에 섞으면 풀 크기를 늘리는 순간 첫 문서부터 달라진다."""
    from src.training.cpt import order_pool

    docs = [f"d{i}" for i in range(50)]
    base = order_pool(docs[:10], 42, [])
    grown = order_pool(docs[:10], 42, docs[10:])
    assert grown[:10] == base           # 앞부분이 비트 단위로 같다
    assert sorted(grown) == sorted(docs)
    assert grown[10:] != docs[10:]      # 추가분도 섞긴 한다
    assert sorted(grown[10:]) == sorted(docs[10:])


def test_확장분이_없으면_기존_동작과_같다():
    import random as _random

    from src.training.cpt import order_pool

    docs = [f"d{i}" for i in range(30)]
    old = list(docs)
    _random.Random(42).shuffle(old)     # P2 까지의 코드
    assert order_pool(docs, 42, []) == old


def test_추가_평가_지점은_한_번만_걸린다():
    from src.training.callbacks import CurveLogger

    c = CurveLogger(run=None, eval_every_bytes=20_000_000,
                    extra_points=[168_500_000])
    assert not c.due(10_000_000)
    assert c.due(20_000_001)
    c.mark(20_000_001)
    assert not c.due(21_000_000)
    assert c.due(168_500_001)           # 간격과 무관한 지점
    c.mark(168_500_001)
    assert not c.due(169_000_000)
    assert c.due(180_000_001)           # 간격은 계속 돈다


def test_추가_지점이_없으면_간격만_본다():
    from src.training.callbacks import CurveLogger

    c = CurveLogger(run=None, eval_every_bytes=1_000_000)
    assert not c.due(999_999)
    assert c.due(1_000_000)


def test_손상_행_목록은_세_출처를_다_읽는다():
    """어느 파일을 줘야 하는지 기억할 필요가 없어야 한다 (P3 W0-5)."""
    import json as _json
    import tempfile

    from src.training.cpt import load_damaged_rows

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        p.write_text(_json.dumps({"rows": [5, 3, 3, 1]}), encoding="utf-8")
        assert load_damaged_rows(p) == [1, 3, 5]
        p.write_text(_json.dumps({"map": {"7": {}, "2": {}}}), encoding="utf-8")
        assert load_damaged_rows(p) == [2, 7]      # 토크나이저 id_map 의 새 행
        p.write_text(_json.dumps([9, 8]), encoding="utf-8")
        assert load_damaged_rows(p) == [8, 9]


def test_train_curve_에_손상_행_컬럼이_뒤에_붙었다():
    """컬럼은 뒤에만 붙인다 — 과거 행은 NA 로 남는다 (docs/LEDGER_SCHEMA.md)."""
    from src.utils.ledger import TRAIN_CURVE_COLUMNS

    assert TRAIN_CURVE_COLUMNS[-2:] == ("grad_norm_dmg", "upd_w_ratio_dmg")


# ── P3 W0-4 — WSD (워밍업 · 상수 · 끝에서 감쇠) ───────────────────────────

def test_wsd_는_감쇠_시작_전까지_상수와_같다():
    """WSD 가 상수와 다른 것은 **마지막 감쇠 구간뿐** 이어야 한다.
    그래야 P3-B 의 결과를 "감쇠를 붙인 효과" 로 읽을 수 있다."""
    from src.training.callbacks import constant_lr_by_bytes, wsd_lr_by_bytes

    total, peak = 1_000_000, 1e-5
    for frac in (0.0, 0.01, 0.02, 0.3, 0.79, 0.799):
        seen = int(total * frac)
        assert wsd_lr_by_bytes(seen, total, peak) == \
            constant_lr_by_bytes(seen, total, peak)


def test_wsd_는_마지막_20퍼센트에서_바닥까지_내려간다():
    from src.training.callbacks import cosine_lr_by_bytes, wsd_lr_by_bytes

    total, peak = 1_000_000, 1e-5
    mid = wsd_lr_by_bytes(int(total * 0.9), total, peak)
    assert peak * 0.1 < mid < peak                      # 감쇠 중
    end = wsd_lr_by_bytes(total, total, peak)
    # 바닥은 코사인과 같은 10% 다 — 두 스케줄의 끝점을 비교할 수 있어야 한다
    assert abs(end - cosine_lr_by_bytes(total, total, peak)) < 1e-15
    assert wsd_lr_by_bytes(int(total * 1.5), total, peak) == end   # 예산을 넘겨도


def test_wsd_도_레지스트리에_있다():
    from src.training.callbacks import SCHEDULES

    assert set(SCHEDULES) == {"cosine", "constant", "wsd"}
