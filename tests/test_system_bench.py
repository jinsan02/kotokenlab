"""시스템 벤치마크의 계산 규칙 (스펙 §49, §50).

GPU 없이 검증할 수 있는 것만 본다 — 분포 통계와 KV cache 산식이다.
지연 자체는 하드웨어가 답하므로 테스트할 것이 없지만, **무엇을 세는가** 는
여기서 못 박아야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import latency, memory  # noqa: E402


class Cfg:
    """Qwen2.5-0.5B 실측값 (experiments/models.tsv)."""
    num_hidden_layers = 24
    num_attention_heads = 14
    num_key_value_heads = 2
    head_dim = 64
    hidden_size = 896


def test_kv_는_attention_head_가_아니라_kv_head_로_센다():
    """GQA 라서 둘이 다르다. attention head(14)로 세면 7배 부풀려진다."""
    assert memory.kv_bytes_per_token(Cfg) == 2 * 24 * 2 * 64 * 2 == 12288


def test_kv_는_토큰_수에_선형이다():
    """Q6 예측 1번의 근거다 — 토큰이 30.2% 줄면 KV 도 정확히 그만큼 준다."""
    a = memory.kv_cache_mb(Cfg, 10_000)
    b = memory.kv_cache_mb(Cfg, 6_980)          # -30.2%
    assert abs(b / a - 0.698) < 1e-9


def test_kv_head_가_없으면_attention_head_로_떨어진다():
    class MHA(Cfg):
        num_key_value_heads = None
    assert memory.kv_bytes_per_token(MHA) == 2 * 24 * 14 * 64 * 2


def test_p95_는_실제_관측값이다():
    """보간하면 관측되지 않은 값이 P95 로 보고된다. 꼬리를 재는 지표에서
    그건 없는 사건을 만드는 것이다."""
    s = latency.stats([float(x) for x in range(1, 101)])
    assert s["p95"] in {float(x) for x in range(1, 101)}
    assert s["p95"] == 95.0


def test_분포를_전부_남긴다():
    s = latency.stats([10.0, 12.0, 11.0, 50.0])
    assert set(s) == {"mean", "median", "std", "p95"}
    assert s["p95"] == 50.0, "꼬리가 P95 에 잡혀야 한다"
    assert s["mean"] < s["p95"], "평균만 보면 꼬리가 숨는다"


def test_표본이_하나면_std_는_0():
    assert latency.stats([7.5])["std"] == 0.0


def test_빈_표본은_거부한다():
    import pytest
    with pytest.raises(ValueError):
        latency.stats([])
