"""절대 pruning 하면 안 되는 토큰 (검토 C3).

왜 필요한가
    BBPE 는 임의의 바이트열을 표현하기 위해 **256개 바이트 토큰**을 바닥에 깔아둔다.
    이 토큰들은 코퍼스에서 개별 빈도가 낮게 보이기 때문에 스펙 §19 의 pruning
    조건("저빈도 + 저중요")에 그대로 걸린다. 하나라도 지우면 처음 보는 바이트열에서
    토크나이저가 실패한다 — 그리고 그 실패는 학습을 한참 돌린 뒤에야 드러난다.

    스펙 §19 의 pruning 조건에는 이 항목이 없다. 그래서 코드로 못 박는다.

쓰는 법
    from src.tokenizer.protected import protected_token_ids, assert_byte_roundtrip

    keep = protected_token_ids(tok)
    candidates = [i for i in low_frequency_ids if i not in keep]
    ...
    assert_byte_roundtrip(new_tok)   # pruning 후 반드시
"""

from __future__ import annotations

import random
from typing import Any, Iterable


def byte_level_alphabet() -> set:
    """GPT-2/Qwen 계열 byte-level BPE 의 256개 바이트 표현 문자.

    `tokenizers` 가 제공하면 그것을 쓰고, 없으면 동일한 매핑을 직접 만든다.
    """
    try:
        from tokenizers.pre_tokenizers import ByteLevel

        return set(ByteLevel.alphabet())
    except Exception:
        pass

    # GPT-2 의 bytes_to_unicode 와 같은 매핑
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c) for c in cs}


def byte_token_ids(tokenizer: Any) -> set:
    """바이트 하나를 표현하는 토큰의 ID 집합.

    vocab 에 실제로 존재하는 것만 돌려준다 (SentencePiece 계열은 다르게 생겼다).
    """
    vocab = tokenizer.get_vocab()
    return {vocab[ch] for ch in byte_level_alphabet() if ch in vocab}


def special_token_ids(tokenizer: Any) -> set:
    """BOS/EOS/PAD/UNK 와 **모든** added token (스펙 §102).

    `special` 플래그를 믿지 않는다. Qwen2.5 는 added token 22개 중 8개가
    `special=False` 로 들어 있다 — `<tool_call>`, `<|fim_middle|>`,
    `<|file_sep|>` 같은 제어 토큰들이다. 플래그만 보면 이것들이 pruning
    후보로 새어 나가고, 실제로 N=20,000 에서 그렇게 됐다.

    added token 은 정의상 merge 로 만들어지지 않고 예약된 ID 를 차지한다.
    코퍼스 빈도가 0 이어도 지우면 안 되는 것이므로 전부 보호한다.
    """
    ids = {i for i in getattr(tokenizer, "all_special_ids", []) or [] if i is not None}
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            ids.add(v)
    for tid in (getattr(tokenizer, "added_tokens_decoder", {}) or {}):
        ids.add(int(tid))
    return ids


def protected_token_ids(tokenizer: Any, extra: Iterable = ()) -> set:
    """pruning 금지 ID 전체. `extra` 로 프로젝트별 보호 토큰을 더할 수 있다."""
    return byte_token_ids(tokenizer) | special_token_ids(tokenizer) | set(extra)


class ByteRoundtripError(AssertionError):
    """토크나이저가 임의 바이트열을 왕복하지 못한다 — byte fallback 이 깨졌다."""


def assert_byte_roundtrip(tokenizer: Any, n_samples: int = 64, seed: int = 42) -> None:
    """임의 바이트열 왕복 검사. pruning·치환 직후에 반드시 호출한다.

    UTF-8 로 디코드되지 않는 바이트열까지 포함해 시험한다. byte fallback 이
    살아 있으면 어떤 입력이든 원문 그대로 복원되어야 한다.
    """
    rng = random.Random(seed)
    samples = [
        "정보처리기사 필기",
        "The quick brown fox",
        "def f(x): return x**2  # 주석",
        "🇰🇷 이모지와 混合 テキスト",
        "​　 제로폭·전각 공백",
    ]
    for _ in range(n_samples):
        raw = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 24)))
        samples.append(raw.decode("utf-8", errors="replace"))

    failures = []
    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        back = tokenizer.decode(ids, skip_special_tokens=False)
        if back != text:
            failures.append((text, back))
    if failures:
        text, back = failures[0]
        raise ByteRoundtripError(
            f"바이트 왕복 실패 {len(failures)}/{len(samples)}건. "
            f"byte fallback 토큰이 제거되었는지 확인하라.\n"
            f"  입력: {text!r}\n  출력: {back!r}"
        )


def assert_protected_survive(tokenizer: Any, before_protected: set) -> None:
    """pruning 전에 보호 대상이던 ID 가 전부 살아 있는지 확인한다."""
    vocab_ids = set(tokenizer.get_vocab().values())
    lost = sorted(before_protected - vocab_ids)
    if lost:
        raise ByteRoundtripError(
            f"보호 토큰 {len(lost)}개가 사라졌다: {lost[:10]}{' …' if len(lost) > 10 else ''}"
        )
