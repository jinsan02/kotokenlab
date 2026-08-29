"""중복 제거 — Exact(SHA256) + Near(MinHash + LSH) (스펙 §8).

**정규화 이후에** 해시를 만든다. 공백·유니코드 차이 때문에 같은 문서가 다른 해시를
갖으면 exact dedup 이 그냥 통과한다.

Near dedup 이 필요한 이유는 뉴스 재배포, 블로그 복사, 위키 미러 때문이다.
이것들이 Train 과 Dev/Test 양쪽에 남으면 평가가 오염된다 (스펙 §6).

FineWeb-2 의 `minhash_cluster_size` 는 **덤프 단위** dedup 결과다. 우리는 여러
덤프를 섞어 쓰므로 그것으로 대체할 수 없다. 참고 신호로만 쓴다.

외부 의존성 없이 numpy 로 구현한다. 10만 문서 규모에서 수 초면 끝난다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

_MERSENNE = (1 << 61) - 1
_MAX32 = (1 << 32) - 1


def content_sha256(normalized_text: str) -> str:
    """정규화된 본문의 해시. manifest 의 `sha256` 컬럼이자 exact dedup 키."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


# ── Exact ─────────────────────────────────────────────────────────────────
def exact_dedup(items) -> tuple:
    """[(doc_id, sha256)] -> (유지할 doc_id 집합, 제거 수).

    먼저 나온 문서를 남긴다. 입력 순서가 결정론적이면 결과도 결정론적이다.
    """
    seen: set = set()
    keep: set = set()
    removed = 0
    for doc_id, digest in items:
        if digest in seen:
            removed += 1
            continue
        seen.add(digest)
        keep.add(doc_id)
    return keep, removed


# ── MinHash ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MinHashConfig:
    """값이 전부 config_sha256 에 들어간다."""

    num_perm: int = 64          # 시그니처 길이
    bands: int = 16             # LSH 밴드 수 (rows = num_perm // bands = 4)
    shingle: int = 8            # 문자 n-gram 길이
    max_shingles: int = 512     # 문서당 상한 (긴 문서의 비용을 자른다)
    threshold: float = 0.80     # 이 이상이면 중복으로 본다
    seed: int = 42


def _shingle_hashes(text: str, k: int, cap: int) -> np.ndarray:
    """문자 k-gram 을 32비트 해시로. 공백을 접어서 띄어쓰기 흔들림에 둔감하게."""
    compact = " ".join(text.split())
    if len(compact) < k:
        compact = compact.ljust(k, " ")
    grams = {compact[i:i + k] for i in range(len(compact) - k + 1)}
    if not grams:
        return np.zeros(1, dtype=np.uint64)
    hs = np.fromiter(
        (int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=4).digest(), "big")
         for g in grams),
        dtype=np.uint64, count=len(grams),
    )
    if hs.size > cap:
        # 결정론적 샘플링: 해시가 작은 것부터 cap 개 (문서 내용에만 의존)
        hs = np.sort(hs)[:cap]
    return hs


class MinHasher:
    """(a*h + b) mod p 형태의 무작위 순열 num_perm 개."""

    def __init__(self, cfg: MinHashConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.a = rng.integers(1, _MERSENNE, size=cfg.num_perm, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE, size=cfg.num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        hs = _shingle_hashes(text, self.cfg.shingle, self.cfg.max_shingles)
        # (num_perm, n_shingles) 를 만들고 축 1 에서 최소값
        mixed = (self.a[:, None] * hs[None, :] + self.b[:, None]) % _MERSENNE
        return (mixed.min(axis=1) & _MAX32).astype(np.uint32)


def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float((sig_a == sig_b).mean())


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def near_dedup(doc_ids: list, signatures: np.ndarray, cfg: MinHashConfig) -> tuple:
    """LSH 로 후보를 모으고 시그니처 유사도로 확인한다.

    반환 (유지할 doc_id 집합, 제거 수, 군집 수).
    각 군집에서 **입력 순서상 가장 앞선 문서**를 남긴다 — 결정론적이다.
    """
    if len(doc_ids) == 0:
        return set(), 0, 0
    rows = cfg.num_perm // cfg.bands
    uf = _UnionFind()

    for band in range(cfg.bands):
        buckets: dict = {}
        chunk = signatures[:, band * rows:(band + 1) * rows]
        for idx in range(len(doc_ids)):
            key = (band, chunk[idx].tobytes())
            first = buckets.get(key)
            if first is None:
                buckets[key] = idx
            elif jaccard(signatures[idx], signatures[first]) >= cfg.threshold:
                uf.union(first, idx)

    groups: dict = {}
    for idx in range(len(doc_ids)):
        groups.setdefault(uf.find(idx), []).append(idx)

    keep: set = set()
    removed = 0
    clusters = 0
    for members in groups.values():
        members.sort()
        keep.add(doc_ids[members[0]])
        if len(members) > 1:
            clusters += 1
            removed += len(members) - 1
    return keep, removed, clusters
