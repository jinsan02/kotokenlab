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

    한영 혼용 판정이 호스트 판정보다 **우선**한다. 스펙 §5 의 Ko-En Mixed 는
    출처가 아니라 문서 성격이고, 토크나이저 압축률이 가장 다르게 나오는 구간이다.
    """
    host = host_of(url)
    threshold = float(rules.ko_en_mixed.get("latin_share_min", 0.35))
    if latin_share(text) >= threshold:
        return "ko_en_mixed", host
    return classify_host(host, rules) or rules.fallback, host
