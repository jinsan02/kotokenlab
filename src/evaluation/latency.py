"""Level 4 — warm-up + cuda synchronize 기반 지연 측정 (스펙 §45, §48, §50).

GPU 는 비동기라 `synchronize()` 없이 잰 시간은 커널이 큐에 들어간 시각일 뿐이다.
warm-up 도 필수다 — 첫 호출에는 커널 자동 선택과 메모리 할당이 섞인다.

**회당 시간을 따로 잰다.** 예전 자원 실측은 N회를 한 번에 재서 평균만 냈는데,
그러면 std 와 P95 를 낼 수 없다. [RULES.md](../../docs/RULES.md) 9번이 요구하는
것은 평균이 아니라 분포다 — 벤치마크에서 평균만 보고하는 것은 꼬리를 숨기는 짓이다.

attention 백엔드는 이 모듈이 직접 정한다 — 경로마다 달라야 하기 때문이다.

    prefill  EFFICIENT + CUDNN 만. 강제 안 하면 MATH 로 폴백해 8,192 토큰에서
             메모리 7.1배, 시간 10.3배가 된다 (resource_probe.md)
    decode   위 둘에 MATH 를 **더한다.** seq_len=1 에서는 두 융합 커널이 모두
             거부한다 — mem_efficient 는 GQA 브로드캐스트를(query 14 heads vs
             key 2 heads), cuDNN 은 길이 1 자체를 지원하지 않아 "No available
             kernel" 로 죽는다. 그리고 decode 에서 MATH 는 해롭지 않다:
             attention 행렬이 1 x N 이라 27,300 토큰에서도 1.5MB 다. prefill 의
             N x N 과 달리 실체화 비용이 없다.

경로마다 다른 백엔드를 쓴다는 사실 자체가 기록돼야 한다 — 그래서 정책을
호출자에게 맡기지 않고 여기서 상수로 못 박는다.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager


def _backends(decode: bool = False) -> list:
    from torch.nn.attention import SDPBackend

    b = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
    if decode:
        b.append(SDPBackend.MATH)      # seq_len=1 은 융합 커널이 전부 거부한다
    return b


@contextmanager
def _kernel(decode: bool = False):
    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(_backends(decode)):
        yield


def stats(samples_ms: list) -> dict:
    """{mean, median, std, p95}. 표본이 1개면 std 는 0 이다."""
    s = sorted(samples_ms)
    n = len(s)
    if n == 0:
        raise ValueError("표본이 없다")
    # 최근접 순위법. 보간하면 실제로 관측되지 않은 값이 P95 로 보고된다.
    p95 = s[min(n - 1, max(0, round(0.95 * n) - 1))]
    return {
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "p95": p95,
    }


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prefill(model, ids, n_warmup: int = 20, n_runs: int = 100) -> dict:
    """prefill 한 번의 비용을 n_runs 회 재서 분포로 돌려준다.

    `logits_to_keep=1` 이 **실제 생성 경로** 다. 다음 토큰 하나만 필요하므로
    LM head 를 마지막 위치에만 적용한다. 이걸 빼면 seq x vocab 로짓 텐서가
    생기는데, vocab 151,936 에서는 그 텐서가 KV cache 보다 훨씬 커져서
    "토크나이저가 메모리를 얼마나 아끼는가" 를 완전히 잘못 재게 된다.
    """
    import torch

    samples = []
    with torch.inference_mode(), _kernel():
        for _ in range(n_warmup):
            model(input_ids=ids, use_cache=True, logits_to_keep=1)
        _sync()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(input_ids=ids, use_cache=True, logits_to_keep=1)
            _sync()
            samples.append((time.perf_counter() - t0) * 1000)
    return stats(samples)


def generate(model, ids, gen_tokens: int = 128,
             n_warmup: int = 5, n_runs: int = 20) -> dict:
    """TTFT 와 decode 처리량.

    `model.generate` 대신 greedy 루프를 직접 돈다. TTFT 를 정확히 끊으려면
    첫 토큰이 나온 순간에 synchronize 해야 하는데, generate 안에서는 그 지점을
    잡을 수 없기 때문이다. 샘플링 없이 argmax 만 쓴다 — 샘플링 방식이 지연에
    섞이면 토크나이저 효과가 아니라 디코딩 전략을 재게 된다.

    반환: ttft(ms 분포), decode_tok_s_mean, total(ms 분포)
    """
    import torch

    ttfts, totals, rates = [], [], []
    with torch.inference_mode():
        for i in range(n_warmup + n_runs):
            measured = i >= n_warmup
            t0 = time.perf_counter()
            with _kernel():                     # prefill — 융합 커널만
                out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                nxt = out.logits[:, -1:, :].argmax(-1)
                past = out.past_key_values
            _sync()
            t_first = time.perf_counter()

            with _kernel(decode=True):          # decode — MATH 를 허용한다
                for _ in range(gen_tokens - 1):
                    out = model(input_ids=nxt, past_key_values=past,
                                use_cache=True)
                    nxt = out.logits[:, -1:, :].argmax(-1)
                    past = out.past_key_values
            _sync()
            t_end = time.perf_counter()

            if measured:
                ttfts.append((t_first - t0) * 1000)
                totals.append((t_end - t0) * 1000)
                decode_s = t_end - t_first
                rates.append((gen_tokens - 1) / decode_s if decode_s > 0 else 0.0)
            del past, out

    return {
        "ttft": stats(ttfts),
        "total": stats(totals),
        "decode_tok_s_mean": statistics.fmean(rates),
    }


def tokenize_ms(tokenizer, text: str, n_warmup: int = 3,
                n_runs: int = 20) -> dict:
    """토큰화 자체의 비용. CPU 작업이라 synchronize 가 필요 없다.

    별도로 재는 이유는, 토큰을 줄이는 토크나이저가 **토큰화 자체는 더 느릴 수
    있기** 때문이다. 치환으로 merge 규칙이 바뀌었으니 확인 없이 가정하지 않는다.
    """
    samples = []
    for _ in range(n_warmup):
        tokenizer(text, add_special_tokens=False)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        tokenizer(text, add_special_tokens=False)
        samples.append((time.perf_counter() - t0) * 1000)
    return stats(samples)
