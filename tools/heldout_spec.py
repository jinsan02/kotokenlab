"""Final Test 평가 설정 — PLAN "5주차 동결" D 의 1~5 를 상수로 고정한다.

`docs/PLAN.md` "D. Final Test 선택 규칙 — 2026-10-06 동결 (개봉 전)" 을 코드로 옮긴 것이다.
`src/evaluation/doc_nll.py --split heldout` · `tools/heldout_pairs.py` ·
`tools/heldout_preflight.py` 가 여기서만 읽는다. **개봉 태그 뒤에는 바꾸지 않는다** (D 의 6).

이름에 lockbox 단어를 쓰지 않는다 — 커밋 훅이 경로에 그 단어가 든 파일을 거부한다
(tools/precheck.py). 그래서 이 파일 · run_id · 표는 `heldout` 이라고 부른다.
"""

from __future__ import annotations

from tools import w5_spec as W5

# 1. 대표 조건 — 5주차 재학습 둘만. 다른 체크포인트는 올리지 않는다 (다중 비교를 만들지 않는다)
CHECKPOINTS = {
    "c0": W5.CHECKPOINTS["c0_const"],      # (경로, 학습 run_id)
    "t2b": W5.CHECKPOINTS["t2b_const"],
}

# 3 · 4. 평가 파일과 언어별 원문 예산 — 원문 앞에서부터 고정 순서 (bpb.evaluate)
SPLIT_LABEL = "final_test"          # lm_metrics 의 split 값 (파일 경로가 아니다)
FILES = (
    ("ko", "final_test.jsonl", 20_000_000),
    ("en", "final_test_control_english.jsonl", 2_000_000),
    ("code", "final_test_control_code.jsonl", 2_000_000),
)
SEQ_LEN = 2048
TAG = "heldout"

# 3. 확증 기준: 한국어 d = BPB(T2b) - BPB(C0) 의 문서 paired bootstrap 95% 구간 하한 > 0
# 5. 불확실성: 문서 bootstrap 과 학습 seed 불확실성(dev seed 쌍 SD)을 따로 보고한다
B = 10_000
RNG_SEED = 0
DEV_NAMES = {"c0": "c0_const", "t2b": "t2b_const"}   # w5_spec 의 dev 문서별 평가 이름


def doc_run_id(name: str) -> str:
    """doc_nll.py --split heldout 이 만드는 run_id (make_run_id 와 같은 규칙)."""
    model = CHECKPOINTS[name][0].rsplit("/", 1)[-1]
    return f"eval_docnll_{model}_{TAG}".lower()
