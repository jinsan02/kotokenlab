"""E0 — 무작위 초기화 (스펙 §21).

기준선이다. 새 토큰의 의미를 pretrained embedding 으로부터 **전혀 전달하지
않는다.** E1/E2 가 얼마나 벌었는지는 이 선 대비로만 말할 수 있다.

sigma 를 임의로 정하지 않는다. 살아남은 행들의 실제 표준편차를 쓴다.
config 의 initializer_range(0.02) 는 학습 전 값이라 학습된 embedding 의 분포와
다르다 — 그 값으로 초기화하면 새 행만 노름이 어긋나서, "무작위라서 나쁜 것"
인지 "노름이 안 맞아서 나쁜 것" 인지 구별할 수 없게 된다.
"""

from __future__ import annotations

import numpy as np


def reference_stats(emb: np.ndarray, rows: np.ndarray | None = None) -> tuple:
    """살아남은 행의 (평균, 표준편차). NaN 행은 제외한다."""
    src = emb if rows is None else emb[rows]
    ok = ~np.isnan(src).any(axis=1)
    if not ok.any():
        raise ValueError("참조할 수 있는 행이 없다")
    good = src[ok]
    return float(good.mean()), float(good.std())


def init_random(emb: np.ndarray, todo: list, seed: int = 42) -> np.ndarray:
    """`todo` 행을 살아남은 행과 같은 분포의 난수로 채운다."""
    if not todo:
        return emb
    filled = np.array([i for i in range(emb.shape[0]) if i not in set(todo)])
    mean, std = reference_stats(emb, filled)
    rng = np.random.default_rng(seed)
    emb[np.asarray(todo)] = rng.normal(mean, std,
                                       size=(len(todo), emb.shape[1])).astype(np.float32)
    return emb
