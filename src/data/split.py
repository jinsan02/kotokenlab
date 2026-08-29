"""문서 단위 Train/Dev/Test 분할 (스펙 §6, §9, docs/RULES.md 1번).

**Split first, tokenize later.** 이 분할이 확정되고 manifest_sha256 이 정해지기
전에는 토크나이저를 학습하지 않는다.

문장으로 쪼갠 뒤 무작위로 나누면 같은 문서의 앞뒤가 Train 과 Test 로 갈린다.
그래서 분할 단위는 **문서**다.

분할은 `doc_id` 해시로 결정한다 — 순서에 의존하지 않으므로, 나중에 문서를
더 넣어도 기존 문서의 소속이 바뀌지 않는다. 이게 무작위 셔플보다 중요한 성질이다.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

SPLITS: tuple[str, ...] = ("train", "dev", "final_test")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.90
    dev: float = 0.05
    final_test: float = 0.05
    seed: int = 42

    def ratios(self) -> tuple:
        total = self.train + self.dev + self.final_test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"비율 합이 1이 아니다: {total}")
        return self.train, self.dev, self.final_test


def _bucket(doc_id: str, seed: int) -> float:
    """doc_id -> [0,1) 안정적 위치. 순서·개수와 무관하다."""
    digest = hashlib.blake2b(f"{seed}:{doc_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign(doc_id: str, cfg: SplitConfig) -> str:
    train, dev, _ = cfg.ratios()
    x = _bucket(doc_id, cfg.seed)
    if x < train:
        return "train"
    if x < train + dev:
        return "dev"
    return "final_test"


def assign_stratified(doc_ids_by_domain: dict, cfg: SplitConfig) -> dict:
    """도메인별로 같은 비율을 적용한다 (스펙 §9 Domain Balance).

    해시 기반이라 도메인마다 따로 돌려도 전역 비율이 유지된다. 다만 도메인이
    작으면 편차가 크므로, 호출부에서 도메인별 dev 크기를 확인해야 한다
    (docs/REVIEW.md A5 — dev 가 작으면 BPB 표준오차가 커진다).
    """
    out: dict = {}
    for domain, ids in doc_ids_by_domain.items():
        for doc_id in ids:
            out[doc_id] = assign(doc_id, cfg)
    return out


def summarize(assignment: dict, domains: dict) -> dict:
    """(split, domain) -> 문서 수. 분할이 도메인을 망가뜨리지 않았는지 확인용."""
    table: Counter = Counter()
    for doc_id, split in assignment.items():
        table[(split, domains.get(doc_id, "NA"))] += 1
    return dict(table)
