"""시드 고정 (스펙 §60 Reproducibility).

Exploration 단계는 seed=42 하나, Final 단계만 42/123/2026 세 개다 (스펙 §51).
어느 쪽이든 시드는 반드시 원장에 기록된다.
"""

from __future__ import annotations

import os
import random

# 스펙 §51 — 최종 검증용 시드
FINAL_SEEDS: tuple[int, ...] = (42, 123, 2026)
EXPLORATION_SEED: int = 42


def set_seed(seed: int, deterministic: bool = False) -> int:
    """random / numpy / torch / cuda 시드를 한 번에 고정한다.

    deterministic=True 면 cuDNN 결정론 모드까지 켠다. 느려지므로 기본은 꺼둔다.
    학습 속도가 tokens/sec 지표에 들어가기 때문에, 결정론 모드를 켰다면
    그 사실이 run note 에 남아야 한다.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
