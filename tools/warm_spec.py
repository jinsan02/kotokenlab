"""4주차 신규 행 warm-start 의 동결 설정 — 한 곳에만 둔다.

`docs/PLAN.md` "4주차 run 설정 동결 — 2026-10-06" 을 코드로 옮긴 것이다.
`tools/warm_preflight.py`(돌리기 전)와 `tools/warm_pairs.py`(돌린 뒤)가 여기서만 읽는다.
이 파일의 값을 바꾸는 것은 **동결을 바꾸는 것** 이다. PLAN 을 먼저 고친다.
"""

from __future__ import annotations

TAG = "warm"
NAME = "t2b_mean"
MODEL = "artifacts/models/kot2b_v2_n30000_mean"
MODEL_ARTIFACT_NAME = "kot2b_v2_n30000_mean"
# T2b 의 새 행 30,000 — id_map 의 키가 곧 새 토큰 id 다 (cpt.load_damaged_rows 가 읽는다).
WARM_ROWS = "artifacts/tokenizers/kot2b_v2_n30000/id_map.json"
N_WARM_ROWS = 30_000

SEEDS = (42, 123, 2026)
GATE_SEED = 42
DIRECT = "cpt_t2b_mean_r5_seed"     # 비교 대상: 기존 직접 CPT (상수 LR 168.5MB)
CTRL = "cpt_c0_qwen_r5_seed"        # 잔차의 기준: 같은 seed 의 C0 상수 LR

BUDGET_BYTES = 168_500_000
WARM_BYTES = 33_700_000             # 예산의 20% (amendment §6)
POOL_DOCS = 50_000
EVAL_BYTES = 20_000_000
EVAL_BUDGET = 2_000_000
EVAL_AT_SLACK = 1_000_000           # 단계 경계 평가 행을 고르는 폭 (한 update 는 원문 약 160KB)

# gate (PLAN). 잔차 개선 = d_direct - d_warm (양수면 warm 이 낫다).
FLOOR = 0.010

# 직접 CPT 와 비교할 때 compare_runs 에서 허용하는 치명 필드와 그 이유.
# 사전 점검이 계획 config 와 직접 CPT config 를 실제로 대조해, 여기 없는 차이가
# 하나라도 나오면 실패로 막는다.
ALLOW_FIELDS = {
    "warm_rows": "이 실험의 처치다",
    "warm_bytes": "이 실험의 처치다",
    "warmup_bytes": "직접 CPT(7683b83)는 이 필드가 생기기(7c3dfda) 전이다. 기본값 0 이 "
                    "예전 동작(예산의 2%)을 재현함은 2주차에 확인했다",
    "pool_extend_docs": "같은 이유로 기록이 없다. 계획은 0(확장 없음) — 예전 동작과 같다",
}

# 직접 CPT(7683b83) 이후 HEAD 까지 src·configs 를 건드린 커밋과, 각각이 직접 CPT 의
# 계산 경로를 바꾸지 않는다는 근거. 사전 점검이 실제 git log 와 대조해 목록 밖 커밋이
# 있으면 막는다.
REVIEWED_COMMITS = {
    "a9300df": "한자 프로브 입력 도구 (학습 경로 밖)",
    "1105df7": "KMMLU 프로브 (학습 경로 밖)",
    "a478426": "원장 git 계보 기록 방식 (계산 무관)",
    "7c3dfda": "워밍업·풀 순서·평가 지점 인자화. 기본값이 예전 동작을 재현 (2주차 확인)",
    "c9bdcae": "손상 행 계측 — 측정만 한다. 이번 run 은 새 행 계측에 쓴다",
    "5325f39": "WSD 스케줄 추가 (상수 LR 경로 불변)",
    "7af61e3": "이식·과목 인자·오염 검사 (학습 경로 밖)",
    "378b3ce": "BPB docstring 정정 (계산 무관)",
    "2da591e": "예산 꼬리: update 경계에서 멈춤. seed42·123 은 직접 CPT 도 1,063 update "
               "경계에서 끝났고, seed2026 은 0.75 update 의 미반영 꼬리가 있었다",
    "2b37998": "gitinfo 함수 추가 (학습 경로 밖)",
    "61017a3": "GPU 잠금 — 진입·종료에서만 돈다",
    "b9a2144": "reserved 메모리 기록 · 로그 단위 · 체크포인트 해시 칸 (계산 무관)",
    "16b9733": "warm-start 추가. 직접 CPT config 해시가 원장과 같음을 테스트한다",
}


def run_id(seed: int) -> str:
    return f"cpt_{NAME}_{TAG}_seed{seed}"


def argv(seed: int) -> list:
    """PLAN 의 명령. 원장 argv 와 문자 그대로 같아야 한다."""
    return [
        "--model", MODEL, "--name", NAME,
        "--budget-bytes", str(BUDGET_BYTES), "--pool-docs", str(POOL_DOCS),
        "--eval-bytes", str(EVAL_BYTES), "--eval-at", str(WARM_BYTES),
        "--eval-budget", str(EVAL_BUDGET), "--lr-schedule", "constant",
        "--seed", str(seed), "--tag", TAG,
        "--warm-rows", WARM_ROWS, "--warm-bytes", str(WARM_BYTES),
        "--damaged-rows", WARM_ROWS, "--save",
    ]


def argv_str(seed: int) -> str:
    return " ".join(argv(seed))


def command(seed: int) -> str:
    return r"C:\llm_tokenizer\.conda\python.exe -u -m src.training.cpt " + argv_str(seed)


def gate(delta) -> str:
    """seed42 의 잔차 개선으로 나머지 두 seed 를 돌릴지 정한다."""
    if delta is None:
        return "판정 불가"
    return "통과" if delta >= FLOOR else "미달"
