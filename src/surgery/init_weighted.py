"""E2 — 가중 초기화 (스펙 §22).

    E_new = w1 E_정보 + w2 E_처리 + w3 E_기사

스펙이 제시한 가중치 후보는 token length / corpus frequency / character coverage
/ semantic contribution 이다. 앞의 셋은 계산할 수 있고 마지막은 정의가 필요하다.
여기서는 **계산 가능한 것만** 구현하고, 무엇을 골랐는지 원장에 남긴다.

    length     부품의 바이트 길이. 긴 조각이 새 토큰의 의미를 더 많이 진다는 가정
    freq       코퍼스 빈도의 역수. 흔한 조각(조사·어미)이 새 토큰을 지배하는 것을
               막는다. "있습니다." 에서 "다." 쪽으로 끌려가지 않게 하는 것이 목적이다
    uniform    E1 과 같다. 비교용으로 남겨 둔다

freq 가중은 역수를 쓴다. 흔한 조각일수록 그 embedding 이 특정 의미보다 문법적
기능을 담고 있어서, 그대로 평균하면 새 토큰이 전부 비슷한 곳으로 몰린다.

가중치는 항상 합이 1 이 되게 정규화한다. 노름이 부품들의 스케일을 벗어나면
E0 과의 비교에서 "초기화 방법의 차이" 가 아니라 "노름의 차이" 를 재게 된다.
"""

from __future__ import annotations

import numpy as np

SCHEMES = ("length", "freq", "uniform")


def _weights(parts: list, scheme: str, token_bytes: np.ndarray,
             token_freq: np.ndarray) -> np.ndarray:
    if scheme == "uniform":
        w = np.ones(len(parts), dtype=np.float64)
    elif scheme == "length":
        w = token_bytes[parts].astype(np.float64)
    elif scheme == "freq":
        # 역빈도. 0 빈도 부품이 무한대가 되지 않게 1 을 더한다.
        w = 1.0 / (token_freq[parts].astype(np.float64) + 1.0)
    else:
        raise ValueError(f"모르는 가중 방식: {scheme!r} (가능: {SCHEMES})")
    total = w.sum()
    if total <= 0:
        w = np.ones(len(parts), dtype=np.float64)
        total = w.sum()
    return w / total


def init_weighted(emb: np.ndarray, components: dict, source_emb: np.ndarray,
                  token_bytes: np.ndarray, token_freq: np.ndarray,
                  scheme: str = "freq") -> list:
    """가중 평균으로 채운다. 채운 행의 목록을 돌려준다."""
    if scheme not in SCHEMES:
        raise ValueError(f"모르는 가중 방식: {scheme!r} (가능: {SCHEMES})")
    done = []
    for slot, parts in components.items():
        if slot >= emb.shape[0]:
            continue
        w = _weights(parts, scheme, token_bytes, token_freq)
        emb[slot] = (source_emb[parts] * w[:, None]).sum(axis=0)
        done.append(slot)
    return done
