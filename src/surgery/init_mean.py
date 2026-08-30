"""E1 — 부품 평균 초기화 (스펙 §21).

새 토큰이 기존 토크나이저에서 무엇으로 쪼개지던 것인지 알면, 그 조각들의
embedding 평균으로 시작할 수 있다.

    E_new = (E_정보 + E_처리 + E_기사) / 3

T2b 는 이 조건이 특히 깨끗하다. 새 토큰은 **정확히 두 부품의 결합** 이고
(새 토큰 1개 = merge 규칙 1개), 그 두 부품의 ID 가 id_map.json 에 그대로
적혀 있다. 스펙의 예시처럼 "어떻게 쪼개지는지 다시 토큰화해서 알아내야 하는"
불확실성이 없다.

주의 — tie_word_embeddings 다. 이 벡터는 입력 임베딩이자 출력 로짓 방향이다.
평균은 입력 쪽으로는 자연스럽지만 출력 쪽으로는 두 부품 사이의 어중간한
방향이라, 학습 초기에 두 부품 토큰과 경쟁한다. 그래서 E0 대비 얼마나 나은지를
Pre-CPT BPB 로 재서 확인해야 한다 — 좋을 것이라고 가정하지 않는다.
"""

from __future__ import annotations

import numpy as np


def component_ids(id_map: dict, base_vocab: dict) -> dict:
    """id_map.json 의 map 절을 {new_id: [부품 old_id, ...]} 로 바꾼다.

    부품이 base vocab 에 없으면 그 슬롯은 뺀다 — 평균을 낼 재료가 없으므로
    호출한 쪽이 다른 방법으로 채워야 한다.
    """
    out: dict = {}
    for slot, info in id_map.items():
        left = base_vocab.get(info["left"])
        right = base_vocab.get(info["right"])
        if left is None or right is None:
            continue
        out[int(slot)] = [left, right]
    return out


def init_mean(emb: np.ndarray, components: dict, source_emb: np.ndarray) -> list:
    """부품 평균으로 채운다. 채운 행의 목록을 돌려준다.

    `source_emb` 는 **원본** embedding 이다. 새로 채워지는 행을 참조하지 않도록
    분리한다 — 같은 배열에서 읽고 쓰면 채우는 순서에 따라 결과가 달라진다.
    """
    done = []
    for slot, parts in components.items():
        if slot >= emb.shape[0]:
            continue
        emb[slot] = source_emb[parts].mean(axis=0)
        done.append(slot)
    return done
