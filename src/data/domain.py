"""URL 과 문자 구성으로 도메인을 라벨링한다 (스펙 §5, §16).

**왜 필요한가**: FineWeb-2 에는 도메인 라벨이 없다. 대신 `url` 이 있다.
이걸 쓰지 않으면 스펙 §16 의 도메인별 평가가 불가능하고, "한국어 평균은 좋아졌지만
커뮤니티에서는 나빠졌는가" 같은 질문에 답할 수 없다.

**규칙 기반 분류의 오류율을 모른 채 도메인별 결과를 주장하면 안 된다.**
`scripts/audit_domain_rules.py` 로 무작위 표본을 손으로 라벨링해 정확도를 재고
리포트에 적는다. 규칙은 `configs/data/domain_rules.yaml` 에서 버전 관리한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .normalize import char_stats

DOMAINS: tuple[str, ...] = (
    "news", "encyclopedia", "blog", "community",
    "conversational", "technical", "ko_en_mixed", "code",
    "noisy", "web_general",
)


@dataclass
class DomainRules:
    version: str
    rules: dict                      # domain -> {host_suffix: [...], host_contains: [...]}
    fallback: str
    ko_en_mixed: dict
    drop_below: float
    spam: dict
    use_content: bool = True

    @classmethod
    def load(cls, path: Path | str) -> "DomainRules":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=str(raw.get("version", "v0")),
            rules=raw.get("rules", {}) or {},
            fallback=raw.get("fallback", "web_general"),
            ko_en_mixed=raw.get("ko_en_mixed", {}) or {},
            drop_below=float(raw.get("drop_if_hangul_ratio_below", 0.15)),
            spam=raw.get("spam_filter", {}) or {},
            use_content=bool(raw.get("use_content_signals", True)),
        )


def host_of(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    except ValueError:
        return ""
    return host.lower().lstrip(".")


def classify_host(host: str, rules: DomainRules) -> str | None:
    """호스트만으로 판정. 못 정하면 None."""
    if not host:
        return None
    for domain, spec in rules.rules.items():
        for suffix in spec.get("host_suffix", []) or []:
            s = suffix.lower().lstrip(".")
            if host == s or host.endswith("." + s):
                return domain
        for frag in spec.get("host_contains", []) or []:
            if frag.lower() in host:
                return domain
    return None


def latin_share(text: str) -> float:
    """알파벳 문자 중 라틴이 차지하는 비율 = latin / (hangul + latin).

    한글 비율을 그대로 쓰면 안 된다. FineWeb-2 kor_Hang 2,868건을 실측하니
    한글 비율의 중앙값이 0.589, p10=0.431, p90=0.703 이었다 — 공백과 문장부호
    때문에 순수 한국어 문서도 0.4~0.7 에 몰린다. "한글 0.15~0.70" 같은 밴드는
    거의 전부를 한영 혼용으로 잘못 분류한다 (실제로 41% 가 그렇게 됐다).

    latin_share 는 분포가 훨씬 잘 갈린다: 중앙값 0.131, p90 0.390, p95 0.498.
    실제 한영 혼용 문서(ko.urbandictionary.com 표제어 사전)는 0.77 이었다.
    """
    stats = char_stats(text)
    denom = stats["hangul"] + stats["latin"]
    return stats["latin"] / denom if denom > 0 else 0.0


def classify(url: str, text: str, rules: DomainRules) -> tuple:
    """(domain, host) 를 돌려준다.

    **호스트가 명시적으로 매칭되면 그것이 이긴다.** 처음에는 한영 혼용 판정을
    앞에 뒀는데, 4샤드 감사에서 `docs.blackberry.com` 이 technical 이 아니라
    ko_en_mixed 로 분류되는 것을 발견했다. 영어가 섞인 기술문서는 technical
    이기도 하고 ko_en_mixed 이기도 한데 컬럼이 하나뿐이라 하나를 잃는다.
    출처가 확실한 쪽(technical)을 남기는 것이 도메인별 평가에 더 쓸모 있다.

    ko_en_mixed 는 이제 **호스트 규칙에 걸리지 않은 문서**의 내용 기반 라벨이다.
    trade-off: 알려진 호스트의 한영 혼용 문서는 ko_en_mixed 로 잡히지 않는다.
    그 문서들의 압축률을 따로 보려면 latin_share 를 별도 신호로 기록해야 한다.
    """
    host = host_of(url)
    by_host = classify_host(host, rules)
    if by_host:
        return by_host, host

    # 호스트가 아무것도 못 잡으면 본문을 본다 (v5).
    # 블라인드 감사에서 technical 재현율 8%, community/ko_en_mixed 0% 였다.
    # 호스트에 docs. 나 cafe. 가 없으면 규칙이 아무것도 못 잡기 때문이다.
    # 호스트 규칙이 먼저인 이유는 그쪽 정밀도가 더 높기 때문이다 (news 86%).
    if rules.use_content:
        from .content import classify_content

        by_content = classify_content(text)
        if by_content:
            return by_content, host

    threshold = float(rules.ko_en_mixed.get("latin_share_min", 0.35))
    if latin_share(text) >= threshold:
        return "ko_en_mixed", host
    return rules.fallback, host
