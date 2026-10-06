"""5주차 동결 설정 — 문서 단위 bootstrap · 노출 층화 · Final Test 대표 체크포인트 재학습.

`docs/PLAN.md` "5주차 동결 — 2026-10-06" 을 코드로 옮긴 것이다. `tools/doc_bootstrap.py`
· `tools/exposure_strata.py` · `tools/w5_preflight.py` 가 여기서만 읽는다.
값을 바꾸는 것은 동결을 바꾸는 것이다. PLAN 을 먼저 고친다.
"""

from __future__ import annotations

# ── 문서별 NLL 평가 (src/evaluation/doc_nll.py) ─────────────────────────
EVAL_BYTES = 2_000_000      # CPT 의 --eval-budget 과 같다 — 원장 dev BPB 를 재현해야 한다
EVAL_TAG = "dev2mb"

# 이름 -> (체크포인트 경로, 그것을 만든 학습 run_id)
CHECKPOINTS = {
    "c0_cos": ("artifacts/models/cpt_c0_qwen_main_seed42", "cpt_c0_qwen_main_seed42"),
    "t2a_cos": ("artifacts/models/cpt_t2a_none_main_seed42", "cpt_t2a_none_main_seed42"),
    "t2b_cos": ("artifacts/models/cpt_t2b_mean_main_seed42", "cpt_t2b_mean_main_seed42"),
    "t2b10k_cos": ("artifacts/models/cpt_t2b10k_mean_main_seed42",
                   "cpt_t2b10k_mean_main_seed42"),
    "t2b_eqtok": ("artifacts/models/cpt_t2b_mean_eqtok_seed42", "cpt_t2b_mean_eqtok_seed42"),
    "t2b_upd": ("artifacts/models/cpt_t2b_mean_upd_seed42", "cpt_t2b_mean_upd_seed42"),
    "t2b_warm": ("artifacts/models/cpt_t2b_mean_warm_seed42", "cpt_t2b_mean_warm_seed42"),
    # 아래 둘은 5주차에 재학습해 만든다 (FT_RUNS)
    "c0_const": ("artifacts/models/cpt_c0_qwen_ft_seed42", "cpt_c0_qwen_ft_seed42"),
    "t2b_const": ("artifacts/models/cpt_t2b_mean_ft_seed42", "cpt_t2b_mean_ft_seed42"),
}


def doc_run_id(name: str) -> str:
    """doc_nll.py 가 만드는 run_id (make_run_id 와 같은 규칙)."""
    model = CHECKPOINTS[name][0].rsplit("/", 1)[-1]
    return f"eval_docnll_{model}_{EVAL_TAG}".lower()


# ── 문서 단위 paired bootstrap ──────────────────────────────────────────
B = 10_000
RNG_SEED = 0
# (이름, a, b, 뜻) — 차이는 BPB(a) - BPB(b). b 가 C0 면 그것이 곧 잔차 d 다.
PAIRS = (
    ("잔차 · 1차 코사인 T2b", "t2b_cos", "c0_cos", "Equal-Raw-Data 1차 핵심 격차"),
    ("잔차 · 1차 코사인 T2a", "t2a_cos", "c0_cos", "제거만 한 T2a 의 BPB 순효과"),
    ("잔차 · 1차 코사인 T2b10k", "t2b10k_cos", "c0_cos", "N=10,000"),
    ("잔차 · 1차 등토큰 T2b", "t2b_eqtok", "c0_cos", "같은 토큰 수 (LR 축 다름)"),
    ("잔차 · 상수 LR T2b", "t2b_const", "c0_const", "Final Test 대표 조건의 dev 격차"),
    ("잔차 · 같은 update T2b", "t2b_upd", "c0_const", "3주차 seed42 (C0 는 1 update 차)"),
    ("warm-start - 직접", "t2b_warm", "t2b_const", "4주차 delta 의 부호 반대"),
)

# ── 제거 토큰 노출 층화 (amendment §6 다섯째 줄) ─────────────────────────
STRATA_PAIR = ("t2a_cos", "c0_cos")      # T2a 의 제거가 문서에 따라 다르게 작용했는가
PRUNE_LIST = "artifacts/vocab_stats/v1/prune_30000.tsv"
BASE_TOKENIZER = ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987")
MIN_DOCS_PER_BIN = 30     # 이보다 적은 구간은 수치를 비교하지 않는다
# 구간: 0 / 1..m / >m. m 은 **제거 토큰이 1번 이상 나온 문서들의 출현 수 중앙값**
# (결과를 보기 전에 정한 규칙이고, 값 자체는 데이터에서 계산한다).

# ── Final Test 대표 체크포인트 재학습 (seed 42) ─────────────────────────
# R5 run 과 argv 가 --tag 와 --save 만 다르다.
FT_RUNS = {
    "cpt_c0_qwen_ft_seed42": ("cpt_c0_qwen_r5_seed42",
        "--model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 "
        "--name c0_qwen --budget-bytes 168500000 --pool-docs 50000 --eval-bytes 20000000 "
        "--eval-budget 2000000 --lr-schedule constant --seed 42 --tag ft --save"),
    "cpt_t2b_mean_ft_seed42": ("cpt_t2b_mean_r5_seed42",
        "--model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-bytes 168500000 "
        "--pool-docs 50000 --eval-bytes 20000000 --eval-budget 2000000 --lr-schedule constant "
        "--seed 42 --tag ft --save"),
}
# 재학습이 R5 원장 값을 재현해야 하는 폭 (한국어 dev 최종 BPB). 1차 같은 seed 재학습의
# 9개 지점 최대 편차가 0.000421 이었다 (FINAL_REPORT §8.2). C0 는 회계 수정 이후라
# R5 seed42(1,523 update)보다 1 update 더 돈다.
FT_REPRO_TOL = 0.001
