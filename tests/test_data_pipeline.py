"""데이터 파이프라인 — 정규화·필터·도메인·dedup·분할."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import dedup as dd  # noqa: E402
from src.data.domain import DomainRules, classify, host_of, latin_share  # noqa: E402
from src.data.normalize import (  # noqa: E402
    byte_len, char_stats, count_eojeol, normalize_text,
)
from src.data.quality import (  # noqa: E402
    FilterStats, QualityConfig, check, distinct_ngram_ratio, line_repeat_ratio,
)
from src.data.split import SplitConfig, assign  # noqa: E402

RULES = DomainRules.load(ROOT / "configs" / "data" / "domain_rules.yaml")


# ── 정규화 ────────────────────────────────────────────────────────────────
def test_normalize_removes_invisible_and_unifies_space():
    s = "가​나　다\r\n\n\n\n라﻿"
    n = normalize_text(s)
    assert "​" not in n and "﻿" not in n and "　" not in n
    assert "\r" not in n and "\n\n\n" not in n
    assert n == "가나 다\n\n라"


def test_normalize_is_idempotent():
    s = "  정보처리기사​ 필기　시험  \n\n\n\n다음  "
    assert normalize_text(normalize_text(s)) == normalize_text(s)


def test_normalize_strips_null_bytes():
    assert "\x00" not in normalize_text("가\x00나")


def test_byte_len_is_utf8():
    assert byte_len("가") == 3 and byte_len("a") == 1


def test_count_eojeol():
    assert count_eojeol("나는 학교에 간다") == 3


def test_char_stats_ratios_sum_sanely():
    s = char_stats("가나다abc123")
    assert s["hangul"] == pytest.approx(3 / 9)
    assert s["latin"] == pytest.approx(3 / 9)
    assert s["digit"] == pytest.approx(3 / 9)


# ── 품질 필터 ─────────────────────────────────────────────────────────────
def test_filter_rejects_short_and_non_korean():
    cfg = QualityConfig()
    assert check("짧다", cfg) == "too_short"
    assert check("a" * 500, cfg) == "low_hangul"


def test_filter_rejects_seo_spam_pattern():
    """자격증 덤프 스팸: 같은 상용구를 문서 전체에 반복한다."""
    spam = "시험패스 공부자료 최신덤프 " * 60
    assert distinct_ngram_ratio(spam, 5) < 0.05      # 고유 gram 이 거의 없다
    assert check(spam, QualityConfig()) in ("repeated_ngram", "repeated_lines")


def test_filter_rejects_repeated_lines():
    text = ("메뉴 홈 로그인 회원가입 고객센터 안내 페이지입니다\n" * 20)
    assert line_repeat_ratio(text) > 0.9
    assert check(text, QualityConfig()) is not None


def test_filter_accepts_normal_korean():
    text = (
        "일기도는 기상 상태를 지도 위에 나타낸 것이다. 기압과 기온, 바람의 방향과 "
        "세기를 기호로 표시하며 날씨를 예측하는 데 쓰인다.\n"
        "관측 자료를 모아 등압선을 그리면 고기압과 저기압의 위치가 드러난다.\n"
        "기상청은 이 자료를 바탕으로 예보를 만들어 국민에게 알린다.\n"
        "인공위성 영상과 레이더 관측이 더해지면서 예보 정확도가 크게 높아졌다.\n"
        "여름철에는 장마전선의 위치를 파악하는 일이 특히 중요하게 다루어진다.\n"
        "태풍이 접근할 때는 진로와 강도를 예측해 미리 경보를 발령하게 된다.\n"
        "겨울에는 대륙고기압의 세력에 따라 한파의 강도가 달라진다고 알려져 있다.\n"
        "이러한 자료는 농업과 항공, 해운 등 여러 분야에서 널리 활용되고 있다.\n"
    )
    assert distinct_ngram_ratio(text, 5) > 0.8
    assert check(text, QualityConfig()) is None


def test_filter_stats_records_reasons():
    stats = FilterStats()
    stats.record(None)
    stats.record("too_short")
    stats.record("too_short")
    assert stats.total == 3 and stats.kept == 1
    assert stats.reasons["too_short"] == 2
    assert any("too_short" in ln for ln in stats.as_lines())


# ── 도메인 ────────────────────────────────────────────────────────────────
def test_host_of():
    assert host_of("https://ko.wikipedia.org/wiki/x") == "ko.wikipedia.org"
    assert host_of("") == ""


def test_host_suffix_is_not_substring():
    """'news.' 같은 조각이 아무 데나 걸리면 안 된다."""
    assert classify("https://mynewsletter.example.com/x", "한글 " * 200, RULES)[0] \
        != "news"


def test_latin_share_separates_mixed_from_plain_korean():
    plain = "일기도는 기상 상태를 나타낸 지도다 " * 20
    mixed = "Definition: a person who is talented 뜻 재능 " * 20
    assert latin_share(plain) < 0.35
    assert latin_share(mixed) >= 0.35


def test_known_hosts_classify():
    ko = "한국어 본문입니다 " * 30
    for url, want in [
        ("https://ko.wikipedia.org/wiki/x", "encyclopedia"),
        ("https://blog.naver.com/x", "blog"),
        ("https://www.yna.co.kr/x", "news"),
        ("https://docs.python.org/ko/x", "technical"),
        ("https://github.com/a/b", "code"),
        ("https://unknown-site.example/x", "web_general"),
    ]:
        assert classify(url, ko, RULES)[0] == want, url


# ── dedup ─────────────────────────────────────────────────────────────────
def test_exact_dedup_keeps_first():
    keep, removed = dd.exact_dedup([("a", "h1"), ("b", "h1"), ("c", "h2")])
    assert keep == {"a", "c"} and removed == 1


def test_content_sha256_is_stable():
    assert dd.content_sha256("가나다") == dd.content_sha256("가나다")
    assert len(dd.content_sha256("x")) == 64


def test_minhash_signature_similarity():
    cfg = dd.MinHashConfig()
    h = dd.MinHasher(cfg)
    base = "정보처리기사 필기 시험은 소프트웨어 설계와 데이터베이스 구축을 다룬다. " * 6
    same = base
    tweak = base + "추가 문장 하나."
    other = "오늘 날씨가 맑아서 한강 공원에 나가 자전거를 탔다. " * 6
    assert dd.jaccard(h.signature(base), h.signature(same)) == 1.0
    assert dd.jaccard(h.signature(base), h.signature(tweak)) > 0.8
    assert dd.jaccard(h.signature(base), h.signature(other)) < 0.3


def test_near_dedup_collapses_similar_docs():
    cfg = dd.MinHashConfig()
    h = dd.MinHasher(cfg)
    base = "뉴스 재배포 기사 본문입니다. 정부는 오늘 대책을 발표했다. " * 8
    texts = [base, base + " 끝.", "전혀 다른 내용의 블로그 글입니다. 커피가 맛있다. " * 8]
    ids = ["a", "b", "c"]
    sigs = np.stack([h.signature(t) for t in texts])
    keep, removed, clusters = dd.near_dedup(ids, sigs, cfg)
    assert "a" in keep and "c" in keep and "b" not in keep
    assert removed == 1 and clusters == 1


# ── 분할 ──────────────────────────────────────────────────────────────────
def test_split_is_deterministic_and_order_independent():
    cfg = SplitConfig()
    assert assign("doc-1", cfg) == assign("doc-1", cfg)


def test_split_ratios_are_approximately_right():
    cfg = SplitConfig()
    counts = {"train": 0, "dev": 0, "final_test": 0}
    n = 20_000
    for i in range(n):
        counts[assign(f"doc-{i}", cfg)] += 1
    assert counts["train"] / n == pytest.approx(0.90, abs=0.01)
    assert counts["dev"] / n == pytest.approx(0.05, abs=0.005)
    assert counts["final_test"] / n == pytest.approx(0.05, abs=0.005)


def test_split_membership_survives_adding_documents():
    """문서를 더 넣어도 기존 문서의 소속이 바뀌지 않아야 한다."""
    cfg = SplitConfig()
    before = {f"d{i}": assign(f"d{i}", cfg) for i in range(100)}
    after = {f"d{i}": assign(f"d{i}", cfg) for i in range(500)}
    assert all(after[k] == v for k, v in before.items())


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError, match="비율 합"):
        SplitConfig(train=0.9, dev=0.2, final_test=0.05).ratios()


def test_pipeline_driver_compiles():
    """단위 모듈만 통과하고 실행 드라이버가 문법 오류인 회귀를 막는다."""
    path = ROOT / "scripts" / "run_data_pipeline.py"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
