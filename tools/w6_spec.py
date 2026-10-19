"""6주차 동결 설정 — WSD 스케줄 (P3-B, amendment 6주차).

`docs/PLAN.md` "6주차 동결 — 2026-10-19" 를 코드로 옮긴 것이다. `tools/w6_preflight.py`
· `tools/wsd_pairs.py` 가 여기서만 읽는다. 값을 바꾸는 것은 동결을 바꾸는 것이다.
PLAN 을 먼저 고친다.
"""

from __future__ import annotations

from tools import w5_spec as W5

# ── 비교 기준: 5주차 재학습 (상수 LR, seed 42, 체크포인트 저장) ──────────
BASE = {"c0": "cpt_c0_qwen_ft_seed42", "t2b": "cpt_t2b_mean_ft_seed42"}


def _wsd_argv(ft_argv: str) -> str:
    """재학습 argv 에서 스케줄과 태그만 바꾼다. 손으로 옮겨 적지 않는다."""
    out = ft_argv.replace("--lr-schedule constant", "--lr-schedule wsd").replace(
        "--tag ft", "--tag wsd")
    if out.count("--lr-schedule wsd") != 1 or out.count("--tag wsd") != 1:
        raise RuntimeError(f"재학습 argv 형식이 예상과 다르다: {ft_argv}")
    return out


# run_id -> (기준 run_id, argv). 순서대로 하나씩 돈다.
WSD_RUNS = {
    "cpt_c0_qwen_wsd_seed42": (BASE["c0"], _wsd_argv(W5.FT_RUNS[BASE["c0"]][1])),
    "cpt_t2b_mean_wsd_seed42": (BASE["t2b"], _wsd_argv(W5.FT_RUNS[BASE["t2b"]][1])),
}
WSD = {"c0": "cpt_c0_qwen_wsd_seed42", "t2b": "cpt_t2b_mean_wsd_seed42"}

# ── 스케줄 (src/training/callbacks.wsd_lr_by_bytes 의 기본값과 같아야 한다) ─
# config 에는 스케줄 이름만 남는다. 기본값이 바뀌면 같은 이름으로 다른 스케줄이 돈다 —
# tests/test_w6_tools.py 가 이 값과 함수 기본값을 대조한다.
BUDGET = 168_500_000
PEAK_LR = 1e-5
WARMUP_FRAC = 0.02          # --warmup-bytes 를 주지 않으면 예산의 2% (cpt.warmup_fraction)
DECAY_FRAC = 0.2
MIN_FRAC = 0.1
DECAY_START = int(BUDGET * (1 - DECAY_FRAC))   # 134,800,000 원문 바이트

# ── 판정 ────────────────────────────────────────────────────────────────
SIGMA_SEEDS = (42, 123, 2026)
SIGMA_T2B = tuple(f"cpt_t2b_mean_r5_seed{s}" for s in SIGMA_SEEDS)
SIGMA_C0 = tuple(f"cpt_c0_qwen_r5_seed{s}" for s in SIGMA_SEEDS)
FLOOR = 0.005               # BPB 효과 크기 바닥 (P3-B)
REPRO_TOL = 0.001           # 감쇠 전 구간의 run 안 재현 폭 (5주차 FT_REPRO_TOL 과 같다)
FOLLOWUP_SEEDS = (123, 2026)

# ── GPU 시험 실행 (결과로 인용하지 않는다) ─────────────────────────────
SMOKE_RUN = "cpt_t2b_mean_wsdsmoke_seed42"
SMOKE_BUDGET = 3_000_000
SMOKE_ARGV = ("--model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean "
              f"--budget-bytes {SMOKE_BUDGET} --pool-docs 50000 --eval-bytes 1000000 "
              "--eval-at 2700000 --eval-budget 200000 --lr-schedule wsd --seed 42 "
              "--tag wsdsmoke")
SMOKE_LR_RTOL = 1e-3

# ── 문서별 NLL (기술만, 판정에 쓰지 않는다) ─────────────────────────────
CHECKPOINTS = {
    "c0_const": W5.CHECKPOINTS["c0_const"],
    "t2b_const": W5.CHECKPOINTS["t2b_const"],
    "c0_wsd": ("artifacts/models/cpt_c0_qwen_wsd_seed42", "cpt_c0_qwen_wsd_seed42"),
    "t2b_wsd": ("artifacts/models/cpt_t2b_mean_wsd_seed42", "cpt_t2b_mean_wsd_seed42"),
}
DOC_PAIRS = (
    ("WSD - 상수 · T2b", "t2b_wsd", "t2b_const"),
    ("WSD - 상수 · C0", "c0_wsd", "c0_const"),
    ("잔차 · WSD", "t2b_wsd", "c0_wsd"),
    ("잔차 · 상수 (5주차와 같다)", "t2b_const", "c0_const"),
)


def doc_run_id(name: str) -> str:
    """doc_nll.py 가 만드는 run_id (w5_spec.doc_run_id 와 같은 규칙)."""
    model = CHECKPOINTS[name][0].rsplit("/", 1)[-1]
    return f"eval_docnll_{model}_{W5.EVAL_TAG}".lower()
