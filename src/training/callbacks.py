"""학습 중 기록과 평가 (스펙 §26, §84).

**x 축은 step 이 아니라 raw_bytes 다** ([RULES.md](../../docs/RULES.md) 12b).
토크나이저가 다른 run 을 step 이나 tokens_seen 으로 나란히 놓으면, 토큰을 크게
자르는 쪽이 같은 step 에서 더 많은 원문을 본 것이 되어 공짜로 이긴다. 원문
바이트를 x 축에 두면 그 우회로가 막힌다.

그래서 train_curve.tsv 의 모든 행에 raw_bytes_seen 이 함께 들어간다. 나중에
다시 돌릴 필요가 없도록 처음부터 남긴다.
"""

from __future__ import annotations

import math
import time
from typing import Any


class CurveLogger:
    """평가 지점마다 train_curve 에 한 행. 원장 쓰기는 RunContext 가 한다."""

    def __init__(self, run: Any, eval_every_bytes: int) -> None:
        self.run = run
        self.eval_every_bytes = eval_every_bytes
        self._next_eval = eval_every_bytes
        self._t0 = time.time()

    def due(self, raw_bytes_seen: int) -> bool:
        return raw_bytes_seen >= self._next_eval

    @staticmethod
    def _r(v):
        return round(v, 6) if v is not None else None

    def mark(self, raw_bytes_seen: int) -> None:
        while self._next_eval <= raw_bytes_seen:
            self._next_eval += self.eval_every_bytes

    def log(self, *, step: int, tokens_seen: int, raw_bytes_seen: int,
            train_loss: float, dev_bpb: float | None = None,
            dev_loss: float | None = None, lr: float,
            grad_norm: float | None = None, peak_vram_mb: int | None = None,
            grad_norm_emb: float | None = None,
            grad_norm_attn: float | None = None,
            grad_norm_ffn: float | None = None) -> None:
        elapsed = time.time() - self._t0
        self.run.log(
            "train_curve", step=step, tokens_seen=tokens_seen,
            raw_bytes_seen=raw_bytes_seen,
            train_loss=round(train_loss, 6),
            dev_loss=round(dev_loss, 6) if dev_loss is not None else None,
            dev_bpb=round(dev_bpb, 6) if dev_bpb is not None else None,
            lr=lr, grad_norm=round(grad_norm, 6) if grad_norm is not None else None,
            grad_norm_emb=self._r(grad_norm_emb),
            grad_norm_attn=self._r(grad_norm_attn),
            grad_norm_ffn=self._r(grad_norm_ffn),
            peak_vram_mb=peak_vram_mb,
            tok_per_s=round(tokens_seen / elapsed, 2) if elapsed > 0 else None,
            raw_bytes_per_s=round(raw_bytes_seen / elapsed, 2) if elapsed > 0 else None,
            elapsed_sec=round(elapsed, 2),
        )


def cosine_lr_by_bytes(raw_bytes_seen: int, total_bytes: int, peak_lr: float,
                       warmup_frac: float = 0.02, min_frac: float = 0.1) -> float:
    """LR 스케줄의 x 축도 raw_bytes 다 (RULES 12b).

    step 으로 스케줄을 짜면 토크나이저마다 같은 원문에서 step 수가 달라져,
    조건마다 다른 LR 궤적을 밟게 된다. 그러면 비교 대상이 토크나이저가 아니라
    학습률이 된다.
    """
    if total_bytes <= 0:
        return peak_lr
    frac = min(max(raw_bytes_seen / total_bytes, 0.0), 1.0)
    if frac < warmup_frac:
        return peak_lr * (frac / warmup_frac)
    p = (frac - warmup_frac) / max(1.0 - warmup_frac, 1e-9)
    return peak_lr * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p)))
