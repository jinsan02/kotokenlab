"""3주차 같은-update 대조의 동결 설정 — 한 곳에만 둔다.

`docs/PLAN.md` "3주차 run 설정 동결 — 2026-10-02" 를 코드로 옮긴 것이다.
`tools/upd_preflight.py`(돌리기 전)와 `tools/upd_pairs.py`(돌린 뒤)가 둘 다
여기서 읽는다. 두 도구가 설정을 따로 들고 있으면 한쪽만 고쳐져 갈라진다.

이 파일의 값을 바꾸는 것은 **동결을 바꾸는 것** 이다. PLAN 을 먼저 고친다.
"""

from __future__ import annotations

TAG = "upd"
NAME = "t2b_mean"
MODEL = "artifacts/models/kot2b_v2_n30000_mean"
MODEL_ARTIFACT_NAME = "kot2b_v2_n30000_mean"

SEEDS = (42, 123, 2026)
CTRL = "cpt_c0_qwen_r5_seed"          # 짝 통제군 (같은 seed)
OLD_TREAT = "cpt_t2b_mean_r5_seed"    # 기존 168.5MB T2b — 교차 확인용, 판정에 안 쓴다

# PLAN 표. 짝 C0 의 update 수와 같아야 한다 (upd_preflight 가 원장과 대조한다).
UPDATES = {42: 1523, 123: 1524, 2026: 1524}

# 한 update 의 토큰 수 = seq_len x micro_bs x accum. C0 config 와 대조한다.
SEQ_LEN, MICRO_BS, ACCUM = 2048, 2, 8
TOKENS_PER_UPDATE = SEQ_LEN * MICRO_BS * ACCUM

POOL_DOCS = 50_000
POOL_EXTEND_DOCS = 20_000
EVAL_BYTES = 20_000_000
EVAL_BUDGET = 2_000_000
EVAL_AT = 168_500_000                  # d_168 을 재는 지점 (원문 바이트)

# d_168 행을 고를 때 EVAL_AT 위로 허용하는 폭. T2b 한 update 는 원문 약 160KB 다.
# --eval-at 을 빠뜨리면 다음 격자 지점(180MB)이 잡히는데, 그것을 거르려는 폭이다.
EVAL_AT_SLACK = 1_000_000

# 판정 경계 (PLAN). rho 는 seed 평균으로 판정한다.
RHO_PREDICT = 0.25
RHO_HEADLINE = 0.50

# 계획 시간 (PLAN). 시각 검증의 남은 유효시간과 대조한다.
PLANNED_HOURS = 5.3 * 1.3

# 이 동결이 다룬 측정 경로 커밋. C0 seed 2026(0fde665) 이후 src/configs 를 건드린
# 커밋은 이것뿐이어야 한다. 늘어나면 동결 범위 밖이므로 preflight 가 막는다.
ALLOWED_CODE_COMMITS_SINCE_CTRL = ("2b37998",)


def run_id(seed: int) -> str:
    return f"cpt_{NAME}_{TAG}_seed{seed}"


def budget_tokens(seed: int) -> int:
    return UPDATES[seed] * TOKENS_PER_UPDATE


def argv(seed: int) -> list:
    """PLAN 의 명령을 인자 목록으로. 원장 argv 와 문자 그대로 같아야 한다."""
    return [
        "--model", MODEL, "--name", NAME,
        "--budget-tokens", str(budget_tokens(seed)),
        "--pool-docs", str(POOL_DOCS), "--pool-extend-docs", str(POOL_EXTEND_DOCS),
        "--eval-bytes", str(EVAL_BYTES), "--eval-at", str(EVAL_AT),
        "--eval-budget", str(EVAL_BUDGET),
        "--lr-schedule", "constant", "--seed", str(seed), "--tag", TAG, "--save",
    ]


def argv_str(seed: int) -> str:
    """원장 argv 컬럼 형식 (tracking.py 가 `" ".join(sys.argv[1:])` 로 쓴다)."""
    return " ".join(argv(seed))


def command(seed: int) -> str:
    return r"C:\llm_tokenizer\.conda\python.exe -m src.training.cpt " + argv_str(seed)


def classify(rho_mean) -> str:
    """PLAN 의 해석 구간. None 이면 판정하지 않는다."""
    if rho_mean is None:
        return "판정 불가"
    if rho_mean <= RHO_PREDICT:
        return "예측 적중"
    if rho_mean <= RHO_HEADLINE:
        return "부분 설명"
    return "헤드라인 변경"
