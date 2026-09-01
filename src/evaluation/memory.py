"""peak allocated/reserved VRAM, KV cache 추정 (스펙 §44, §76).

두 종류를 구분해서 남긴다.

    peak_alloc_mb     텐서가 실제로 잡은 최대치
    peak_reserved_mb  캐싱 할당자가 OS 에서 받아 쥐고 있는 최대치

reserved 만 보면 할당자의 파편화까지 세고, alloc 만 보면 "실제로 GPU 를 얼마나
차지하는가" 를 놓친다. 둘 다 남긴다 ([RULES.md](../../docs/RULES.md) 9번).

KV cache 는 **이론값** 이다. 실측 peak 에는 가중치와 활성화가 섞여 있어서
"토크나이저가 KV 를 얼마나 아끼는가" 를 분리할 수 없다. 컬럼 이름이
`kv_cache_mb_est` 인 이유다.
"""

from __future__ import annotations

MB = 1024 * 1024


def reset() -> None:
    """이후 측정을 위해 peak 통계를 0 으로 되돌린다."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peaks() -> dict:
    """{peak_alloc_mb, peak_reserved_mb}. CUDA 가 없으면 0."""
    import torch

    if not torch.cuda.is_available():
        return {"peak_alloc_mb": 0.0, "peak_reserved_mb": 0.0}
    torch.cuda.synchronize()
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / MB,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / MB,
    }


def kv_bytes_per_token(config) -> int:
    """토큰 하나가 차지하는 KV cache 바이트.

    layer 마다 K 와 V 를 각각 (n_kv_heads x head_dim) 씩 들고, bf16 이라 2바이트다.
    GQA 라서 n_kv_heads 는 n_attention_heads 와 다르다 — 여기를 attention head
    수로 잘못 세면 Qwen2.5-0.5B 에서 7배 부풀려진다 (kv 2 vs attn 14).
    """
    n_kv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    return 2 * config.num_hidden_layers * n_kv * head_dim * 2


def kv_cache_mb(config, n_tokens: int) -> float:
    """n_tokens 를 담는 KV cache 의 이론 크기 (MB). 토큰 수에 선형이다.

    선형이라는 것이 Q6 의 예측 1번을 검증 가능하게 만든다 — 토크나이저가 토큰을
    30.2% 줄이면 KV 도 정확히 30.2% 줄어야 한다. 안 맞으면 계측이 틀린 것이다.
    """
    return kv_bytes_per_token(config) * n_tokens / MB


def weight_mb(model) -> float:
    """가중치가 차지하는 메모리 (MB). vocab 축소의 효과를 여기서 본다."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / MB
