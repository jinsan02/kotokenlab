"""품질 필터 — 무엇을 왜 버렸는지 남긴다 (스펙 §5, docs/DATA_SOURCES.md 3절).

조사할 때 실제로 본 것:

    HPLT kor_Hang 첫 행 = "SAP C-SAC-2107 100%시험패스 공부자료 ..."

자격증 덤프 판매 SEO 스팸이다. 이런 문서는 어휘가 극도로 반복적이라
**토크나이저 merge rule 을 오염시킨다** — 스팸 상용구가 고빈도 토큰이 되어
vocabulary 예산을 잡아먹는다. T3(New BBPE)에 특히 치명적이다.

그리고 FineWeb-2 kor_Hang 에는 한국어가 아닌 문서가 섞여 있다 (스크립트 기준
분류라서). 이건 버리지 않고 Ko-En Mixed 도메인으로 라벨링한다 (domain.py).

필터는 **버린 이유를 반드시 기록한다.** 탈락률이 어디서 나오는지 모르면
필터가 코퍼스를 어떻게 왜곡하는지 알 수 없다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .normalize import char_stats


@dataclass(frozen=True)
class QualityConfig:
    """모든 값이 config_sha256 에 들어간다. 바꾸면 다른 코퍼스다."""

    min_chars: int = 200
    max_chars: int = 200_000
    min_hangul_ratio: float = 0.15          # 이보다 낮으면 한국어 코퍼스에서 제외
    min_latin_ratio: float = 0.0            # 영어 대조군에서만 쓴다
    min_language_score: float = 0.60        # FineWeb-2 의 language_score
    max_line_repeat_ratio: float = 0.30     # 같은 줄의 반복
    min_distinct_ngram_ratio: float = 0.25  # 고유 5-gram 비율 (SEO 스팸은 매우 낮다)
    max_digit_ratio: float = 0.30
    min_mean_line_chars: float = 10.0       # 메뉴·링크 나열 문서 배제
    ngram_n: int = 5
    check_line_repeat: bool = True
    check_ngram: bool = True

    @classmethod
    def for_language(cls, lang: str) -> "QualityConfig":
        """언어별 프로파일. 같은 필터를 그대로 쓰면 대조군이 통째로 날아간다.

        ko    기본값. 한글 비율 하한이 있다.
        en    한글 하한을 끄고 라틴 하한을 건다. 영어 regression 측정용이라
              한국어 문서가 섞이면 안 된다.
        code  반복 검사를 **끈다**. 코드는 import 줄, 닫는 괄호, 보일러플레이트가
              정당하게 반복되므로 한국어 SEO 스팸 기준을 적용하면 멀쩡한 파일이
              전부 탈락한다. 길이 하한도 낮춘다 (짧은 소스 파일이 흔하다).
        """
        if lang == "ko":
            return cls()
        if lang == "en":
            return cls(min_hangul_ratio=0.0, min_latin_ratio=0.50,
                       min_language_score=0.0)
        if lang == "code":
            return cls(min_chars=120, min_hangul_ratio=0.0,
                       min_language_score=0.0, max_digit_ratio=0.60,
                       min_mean_line_chars=3.0,
                       check_line_repeat=False, check_ngram=False)
        raise ValueError(f"알 수 없는 언어 프로파일: {lang!r} (ko|en|code)")


@dataclass
class FilterStats:
    """탈락 사유별 집계. 리포트에 그대로 싣는다."""

    total: int = 0
    kept: int = 0
    reasons: Counter = field(default_factory=Counter)

    def record(self, reason: str | None) -> None:
        self.total += 1
        if reason is None:
            self.kept += 1
        else:
            self.reasons[reason] += 1

    def as_lines(self) -> list:
        out = [f"  전체 {self.total:,}  통과 {self.kept:,} "
               f"({self.kept / max(self.total, 1) * 100:.1f}%)"]
        for reason, n in self.reasons.most_common():
            out.append(f"    {reason:<24} {n:>8,}  ({n / max(self.total, 1) * 100:5.2f}%)")
        return out


def line_repeat_ratio(text: str) -> float:
    """중복된 줄이 차지하는 비율. 게시판 템플릿·네비게이션 반복을 잡는다."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return 0.0
    counts = Counter(lines)
    duplicated = sum(c for c in counts.values() if c > 1)
    return duplicated / len(lines)


def distinct_ngram_ratio(text: str, n: int = 5) -> float:
    """고유 n-gram / 전체 n-gram. **낮을수록** 반복적이다.

    처음에는 "가장 흔한 n-gram 의 비율"을 썼는데 반복을 못 잡았다.
    13글자 상용구가 60번 반복되면 13개의 서로 다른 5-gram 이 각각 60번씩 나오므로
    상위 n-gram 비율은 60/780 = 0.077 에 불과하다. 반복이 여러 gram 에 퍼지기 때문이다.

    고유 비율은 그 반복을 정확히 잡는다: 13/780 = 0.017.
    일반 한국어 문서는 0.8 이상이다.
    """
    compact = "".join(text.split())
    if len(compact) < n * 8:
        return 1.0
    grams = [compact[i:i + n] for i in range(len(compact) - n + 1)]
    return len(set(grams)) / len(grams)


def mean_line_chars(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    return sum(len(ln) for ln in lines) / len(lines)


def check(text: str, cfg: QualityConfig, language_score: float | None = None) -> str | None:
    """통과하면 None, 아니면 탈락 사유 문자열."""
    n = len(text)
    if n < cfg.min_chars:
        return "too_short"
    if n > cfg.max_chars:
        return "too_long"

    if language_score is not None and language_score < cfg.min_language_score:
        return "low_language_score"

    stats = char_stats(text)
    if stats["hangul"] < cfg.min_hangul_ratio:
        return "low_hangul"
    if cfg.min_latin_ratio and stats["latin"] < cfg.min_latin_ratio:
        return "low_latin"
    if cfg.min_latin_ratio and stats["hangul"] > 0.05:
        return "korean_in_english"    # 영어 대조군에 한국어가 섞이면 측정이 흐려진다
    if stats["digit"] > cfg.max_digit_ratio:
        return "digit_heavy"

    if mean_line_chars(text) < cfg.min_mean_line_chars:
        return "short_lines"          # 메뉴·링크 나열
    if cfg.check_line_repeat and line_repeat_ratio(text) > cfg.max_line_repeat_ratio:
        return "repeated_lines"
    if cfg.check_ngram and \
            distinct_ngram_ratio(text, cfg.ngram_n) < cfg.min_distinct_ngram_ratio:
        return "repeated_ngram"       # SEO 스팸

    return None
