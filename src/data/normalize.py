"""문서 정규화 — dedup 해시의 전제 조건 (스펙 §8).

정규화 **이후에** 해시를 만든다. 공백이나 유니코드 형태 차이 때문에 같은 문서가
다른 해시를 갖으면 exact dedup 이 그냥 통과해버린다.

정규화는 **보수적으로** 한다. 반복 문자 축약처럼 원문을 크게 바꾸는 조작은
여기서 하지 않는다 — 그건 변환이 아니라 판정이므로 quality.py 의 필터 신호로 둔다.
원문을 바꾸면 BPB 의 분모(raw bytes)가 달라져서 "원문 기준 압축률"이라는 지표의
의미가 흔들린다.
"""

from __future__ import annotations

import re
import unicodedata

# 제어문자(줄바꿈·탭 제외)와 폭 없는 문자. 눈에 안 보이면서 해시를 바꾼다.
_INVISIBLE = re.compile(
    "[---"
    "​-‏‪-‮⁠-⁤﻿]"
)

# 유니코드 공백 변종을 일반 공백으로. 전각 공백(U+3000)이 한국어 웹에 흔하다.
_SPACES = re.compile("[   -   　]")

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_SPACES = re.compile(r"[ \t]{2,}")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")

_HANGUL = re.compile("[가-힣ᄀ-ᇿ㄰-㆏]")
_LATIN = re.compile(r"[A-Za-z]")
_HANJA = re.compile("[一-鿿]")
_KANA = re.compile("[぀-ヿ]")
_DIGIT = re.compile(r"[0-9]")


def normalize_text(text: str) -> str:
    """NFC 정규화 + 보이지 않는 문자 제거 + 공백 정리. 내용은 바꾸지 않는다."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_WS.sub("", text)
    text = _MANY_SPACES.sub(" ", text)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()


# ── 문자 구성 ─────────────────────────────────────────────────────────────
def char_stats(text: str) -> dict:
    """문자 종류별 비율. 언어 판정과 도메인 라벨링에 쓴다."""
    n = len(text)
    if n == 0:
        return {"n_chars": 0, "hangul": 0.0, "latin": 0.0,
                "hanja": 0.0, "kana": 0.0, "digit": 0.0}
    return {
        "n_chars": n,
        "hangul": len(_HANGUL.findall(text)) / n,
        "latin": len(_LATIN.findall(text)) / n,
        "hanja": len(_HANJA.findall(text)) / n,
        "kana": len(_KANA.findall(text)) / n,
        "digit": len(_DIGIT.findall(text)) / n,
    }


def hangul_ratio(text: str) -> float:
    return char_stats(text)["hangul"]


def count_eojeol(text: str) -> int:
    """어절 수 = 공백으로 나눈 토막 수 (스펙 §35 tok_per_eojeol 의 분모).

    Noisy Korean 도메인은 띄어쓰기가 깨져 있어 이 값이 불안정하다.
    그래서 fertility 는 형태소 기준으로 따로 잰다 (src/data/morph.py).
    """
    return len(text.split())


def byte_len(text: str) -> int:
    """UTF-8 바이트 수. BPB 의 분모다 (스펙 §37)."""
    return len(text.encode("utf-8"))
