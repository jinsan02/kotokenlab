"""형태소 분절 — fertility 의 "정답" 기준 (스펙 §14, 검토 A7).

스펙 §14 는 `정보처리기사` 가 3조각이면 fertility=3, 1조각이면 1 이라고 한다.
그런데 **무엇이 정답 분절인지** 정의되어 있지 않았다. 정의가 없으면
`tokenizer_metrics.tsv` 의 `fertility_mean` 컬럼을 채울 수 없다.

여기서 고정한다:

    fertility = 토크나이저 토큰 수 / 형태소 수     (kiwipiepy 기준)

kiwipiepy 버전은 `env_sha256` 에 포함된다. 분석기가 바뀌면 지표 정의가 바뀌므로
다른 환경으로 취급되고 `RunContext` 가 실행을 막는다.

    >>> morphemes("정보처리기사 필기 시험")
    ['정보', '처리', '기사', '필기', '시험']
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def morphemes(text: str) -> list:
    """형태소 표면형 목록."""
    if not text.strip():
        return []
    return [t.form for t in _kiwi().tokenize(text)]


def count_morphemes(text: str) -> int:
    return len(morphemes(text))


def fertility(n_tokens: int, n_morphemes: int) -> float:
    """토큰/형태소. 1.0 에 가까울수록 의미 단위를 잘 보존한다.

    1.0 미만도 가능하다 — 토크나이저가 여러 형태소를 한 토큰으로 묶은 경우다
    (예: `정보처리기사` 를 통째로). 그건 압축에는 좋지만 형태소 경계를
    무시했다는 뜻이기도 하므로, 압축률과 함께 봐야 한다.
    """
    if n_morphemes <= 0:
        return float("nan")
    return n_tokens / n_morphemes


def analyzer_version() -> str:
    try:
        import kiwipiepy

        return f"kiwipiepy-{kiwipiepy.__version__}"
    except Exception:
        return "NA"
