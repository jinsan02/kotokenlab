"""byte fallback 토큰 보호 (검토 C3).

pruning 이 바이트 토큰을 지우면 처음 보는 입력에서 토크나이저가 조용히 깨진다.
그 실패는 학습을 한참 돌린 뒤에야 드러나므로, 여기서 미리 막는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from src.tokenizer.protected import (  # noqa: E402
    ByteRoundtripError,
    assert_byte_roundtrip,
    assert_protected_survive,
    byte_level_alphabet,
    byte_token_ids,
    protected_token_ids,
    special_token_ids,
)

QWEN = "Qwen/Qwen2.5-0.5B"


def _has_qwen() -> bool:
    cache = ROOT / ".hf_cache" / "hub" / "models--Qwen--Qwen2.5-0.5B"
    return cache.is_dir()


requires_qwen = pytest.mark.skipif(
    not _has_qwen(), reason="Qwen 토크나이저 캐시 없음 (scripts/download_models.py)"
)


# ── 순수 로직 (모델 없이) ─────────────────────────────────────────────────
def test_byte_level_alphabet_has_256_symbols():
    assert len(byte_level_alphabet()) == 256


def test_alphabet_symbols_are_single_characters():
    assert all(len(c) == 1 for c in byte_level_alphabet())


class _FakeTokenizer:
    """vocab 과 special id 만 있는 최소 토크나이저."""

    def __init__(self, vocab, special_ids=()):
        self._vocab = dict(vocab)
        self.all_special_ids = list(special_ids)
        self.bos_token_id = self.eos_token_id = None
        self.pad_token_id = self.unk_token_id = None
        self.added_tokens_decoder = {}

    def get_vocab(self):
        return dict(self._vocab)


def test_protected_includes_bytes_and_specials():
    alphabet = sorted(byte_level_alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["한국어"] = 1000
    vocab["<|endoftext|>"] = 1001
    tok = _FakeTokenizer(vocab, special_ids=[1001])

    keep = protected_token_ids(tok)
    assert len(byte_token_ids(tok)) == 256
    assert 1001 in keep          # special
    assert 1000 not in keep      # 일반 토큰은 pruning 후보가 될 수 있다
    assert len(keep) == 257


def test_extra_protected_tokens_are_honored():
    tok = _FakeTokenizer({"가": 5, "나": 6})
    assert 5 in protected_token_ids(tok, extra=[5])


def test_special_ids_from_attributes():
    tok = _FakeTokenizer({"a": 1})
    tok.bos_token_id, tok.eos_token_id = 7, 8
    assert {7, 8} <= special_token_ids(tok)


def test_assert_protected_survive_detects_loss():
    tok = _FakeTokenizer({"a": 1, "b": 2})
    assert_protected_survive(tok, {1, 2})
    with pytest.raises(ByteRoundtripError, match="사라졌다"):
        assert_protected_survive(tok, {1, 2, 3})


# ── 실제 Qwen 토크나이저 ──────────────────────────────────────────────────
@requires_qwen
def test_qwen_has_full_byte_alphabet():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    assert len(byte_token_ids(tok)) == 256, "Qwen vocab 에 바이트 토큰 256개가 있어야 한다"


@requires_qwen
def test_qwen_byte_roundtrip_passes():
    from transformers import AutoTokenizer

    assert_byte_roundtrip(AutoTokenizer.from_pretrained(QWEN))


@requires_qwen
def test_protected_set_is_small_relative_to_vocab():
    """보호 집합이 vocab 의 극히 일부여야 pruning 여지가 남는다."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN)
    keep = protected_token_ids(tok)
    assert 256 <= len(keep) < 0.01 * len(tok)
