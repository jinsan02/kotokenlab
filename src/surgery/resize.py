"""embedding 행 재배치와 크기 조정 (스펙 §20).

토크나이저를 바꾼 직후 바로 CPT 하지 않는다. 그러면 토크나이저 효과 / 초기화
효과 / 정렬 효과 / CPT 효과가 섞여서 무엇이 기여했는지 말할 수 없다. 이 모듈은
그 중 **초기화 직전까지** 를 담당한다 — 살아남은 토큰의 행을 옮기고, 새로 생긴
자리를 표시해서 넘긴다. 채우는 일은 init_*.py 가 한다.

두 조건이 하는 일이 다르다.

    T2a  vocab 이 줄어든다.   행렬을 다시 만들고 살아남은 행만 옮긴다.
         새로 생기는 자리가 없으므로 초기화할 것도 없다.
    T2b  vocab 크기가 같다.   치환된 자리의 행만 새로 채운다. 나머지는 그대로다.

ID 매핑은 `id_map.json` 이 아니라 **두 토크나이저의 vocab 에서 직접 유도한다.**
T2a 의 id_map 은 번호가 바뀐 것만 담고 있어서 그것만 보면 안 바뀐 행을 놓친다.
토큰 문자열을 키로 맞추면 그런 구멍이 없다.

Qwen2.5 는 tie_word_embeddings=True 다. embedding 하나가 입력 임베딩이자 출력
로짓 방향이므로, 행 하나를 고치면 양쪽이 함께 바뀐다. lm_head 를 따로 손대면
안 된다 — 묶여 있는 것을 푸는 순간 파라미터 수가 달라져 비교가 깨진다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PAD_MULTIPLE = 128      # Qwen2.5 의 관례. 151,936 = 128 x 1,187


def id_mapping(base_tok: Any, new_tok: Any) -> np.ndarray:
    """new_id -> old_id 배열. 대응하는 옛 토큰이 없으면 -1.

    길이는 새 토크나이저의 최대 ID + 1 이다. 패딩 구간은 -1 로 남는다.
    """
    base_vocab = base_tok.get_vocab()
    new_vocab = new_tok.get_vocab()
    size = max(new_vocab.values()) + 1
    mapping = np.full(size, -1, dtype=np.int64)
    for token, new_id in new_vocab.items():
        old_id = base_vocab.get(token)
        if old_id is not None:
            mapping[new_id] = old_id
    return mapping


def padded_vocab_size(n_tokens: int, multiple: int = PAD_MULTIPLE) -> int:
    """실제 토큰 수를 128 배수로 올린다.

    Qwen 은 토큰 151,665개에 행렬 151,936칸을 잡아 뒀는데, 이는 정렬만이 아니다
    — 정렬만이면 151,680이면 된다. 256칸을 더 비워 둔 것이다. 그 여분은 학습
    신호를 받은 적이 없는 행이라 T2a 가 물려받을 이유가 없다. **정렬까지만** 한다.

    T2b 에는 이 함수를 쓰지 않는다. 크기 유지가 정의이므로 원본 행 수를 그대로
    써야 하고, 여기서 다시 계산하면 256행을 잘라내 형상이 바뀐다.
    """
    return ((n_tokens + multiple - 1) // multiple) * multiple


def rearrange(old_emb: np.ndarray, mapping: np.ndarray,
              new_size: int) -> tuple:
    """살아남은 행을 새 위치로 옮긴다.

    돌려주는 것은 (새 행렬, 채워야 할 행의 인덱스) 다. 채우는 일은 이 모듈의
    책임이 아니다 — 초기화 방법이 실험 변수이기 때문에 분리해 둔다.

    새 행렬의 미채움 자리는 0 이 아니라 NaN 으로 둔다. 초기화를 빠뜨린 채
    저장하면 학습이 조용히 이상해지는데, NaN 이면 즉시 드러난다.
    """
    if old_emb.ndim != 2:
        raise ValueError(f"embedding 은 2차원이어야 한다: {old_emb.shape}")
    hidden = old_emb.shape[1]
    new_emb = np.full((new_size, hidden), np.nan, dtype=np.float32)

    n = min(len(mapping), new_size)
    src = mapping[:n]
    known = src >= 0
    new_emb[np.arange(n)[known]] = old_emb[src[known]]

    todo = [int(i) for i in range(new_size) if i >= n or not known[i]]
    return new_emb, todo


def assert_filled(emb: np.ndarray, allow_rows: set | None = None) -> None:
    """채우지 않은 행이 남았는지 본다. 패딩 구간만 예외로 허용한다."""
    bad = np.isnan(emb).any(axis=1)
    if allow_rows:
        for i in allow_rows:
            bad[i] = False
    if bad.any():
        idx = np.flatnonzero(bad)
        raise AssertionError(
            f"초기화되지 않은 행 {idx.size:,}개가 남았다: {idx[:10].tolist()}")
