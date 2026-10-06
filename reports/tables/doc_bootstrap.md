# 문서 단위 paired bootstrap — dev BPB 차이의 구간

> **이 표는 코드가 적는다.** `.conda/python.exe tools/doc_bootstrap.py`
> 동결 설정 `tools/w5_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md) "5주차 동결".
> dev 원문 언어별 2MB, 복원 추출 10,000회 (seed 0), 95% 백분위 구간.

차이 = BPB(a) − BPB(b). b 가 C0 면 잔차 d 다. **문서 표본의 불확실성만 담는다** —
학습 seed 의 불확실성은 seed 쌍 표가 따로 낸다.

## 체크포인트 평가

| 이름 | 체크포인트 | 평가 run | 상태 |
|---|---|---|---|
| c0_const | `artifacts/models/cpt_c0_qwen_ft_seed42` | `eval_docnll_cpt_c0_qwen_ft_seed42_dev2mb` | 아직 없음 |
| c0_cos | `artifacts/models/cpt_c0_qwen_main_seed42` | `eval_docnll_cpt_c0_qwen_main_seed42_dev2mb` | 통과 |
| t2a_cos | `artifacts/models/cpt_t2a_none_main_seed42` | `eval_docnll_cpt_t2a_none_main_seed42_dev2mb` | 통과 |
| t2b10k_cos | `artifacts/models/cpt_t2b10k_mean_main_seed42` | `eval_docnll_cpt_t2b10k_mean_main_seed42_dev2mb` | 통과 |
| t2b_const | `artifacts/models/cpt_t2b_mean_ft_seed42` | `eval_docnll_cpt_t2b_mean_ft_seed42_dev2mb` | 아직 없음 |
| t2b_cos | `artifacts/models/cpt_t2b_mean_main_seed42` | `eval_docnll_cpt_t2b_mean_main_seed42_dev2mb` | 통과 |
| t2b_eqtok | `artifacts/models/cpt_t2b_mean_eqtok_seed42` | `eval_docnll_cpt_t2b_mean_eqtok_seed42_dev2mb` | 통과 |
| t2b_upd | `artifacts/models/cpt_t2b_mean_upd_seed42` | `eval_docnll_cpt_t2b_mean_upd_seed42_dev2mb` | 통과 |
| t2b_warm | `artifacts/models/cpt_t2b_mean_warm_seed42` | `eval_docnll_cpt_t2b_mean_warm_seed42_dev2mb` | 통과 |

## 한국어

| 비교 | a | b | 문서 | BPB(a) | BPB(b) | 차이 | 95% 구간 |
|---|---|---|---:|---:|---:|---:|---|
| 잔차 · 1차 코사인 T2b | t2b_cos | c0_cos | 434 | 1.567095 | 1.137540 | +0.429554 | [+0.419735, +0.440484] |
| 잔차 · 1차 코사인 T2a | t2a_cos | c0_cos | 434 | 1.136817 | 1.137540 | -0.000723 | [-0.000801, -0.000642] |
| 잔차 · 1차 코사인 T2b10k | t2b10k_cos | c0_cos | 434 | 1.497272 | 1.137540 | +0.359732 | [+0.352662, +0.367587] |
| 잔차 · 1차 등토큰 T2b | t2b_eqtok | c0_cos | 434 | 1.532183 | 1.137540 | +0.394642 | [+0.385514, +0.404739] |
| 잔차 · 상수 LR T2b | t2b_const | c0_const | — | — | — | — | 계산 안 함 (t2b_const: 아직 없음 · c0_const: 아직 없음) |
| 잔차 · 같은 update T2b | t2b_upd | c0_const | — | — | — | — | 계산 안 함 (t2b_upd: 통과 · c0_const: 아직 없음) |
| warm-start - 직접 | t2b_warm | t2b_const | — | — | — | — | 계산 안 함 (t2b_warm: 통과 · t2b_const: 아직 없음) |

## 영어

| 비교 | a | b | 문서 | BPB(a) | BPB(b) | 차이 | 95% 구간 |
|---|---|---|---:|---:|---:|---:|---|
| 잔차 · 1차 코사인 T2b | t2b_cos | c0_cos | 472 | 0.821234 | 0.813282 | +0.007951 | [+0.007523, +0.008395] |
| 잔차 · 1차 코사인 T2a | t2a_cos | c0_cos | 472 | 0.813393 | 0.813282 | +0.000111 | [-0.000008, +0.000266] |
| 잔차 · 1차 코사인 T2b10k | t2b10k_cos | c0_cos | 472 | 0.818725 | 0.813282 | +0.005442 | [+0.005101, +0.005783] |
| 잔차 · 1차 등토큰 T2b | t2b_eqtok | c0_cos | 472 | 0.822307 | 0.813282 | +0.009024 | [+0.008556, +0.009496] |
| 잔차 · 상수 LR T2b | t2b_const | c0_const | — | — | — | — | 계산 안 함 (t2b_const: 아직 없음 · c0_const: 아직 없음) |
| 잔차 · 같은 update T2b | t2b_upd | c0_const | — | — | — | — | 계산 안 함 (t2b_upd: 통과 · c0_const: 아직 없음) |
| warm-start - 직접 | t2b_warm | t2b_const | — | — | — | — | 계산 안 함 (t2b_warm: 통과 · t2b_const: 아직 없음) |

## 코드

| 비교 | a | b | 문서 | BPB(a) | BPB(b) | 차이 | 95% 구간 |
|---|---|---|---:|---:|---:|---:|---|
| 잔차 · 1차 코사인 T2b | t2b_cos | c0_cos | 389 | 0.448528 | 0.440420 | +0.008108 | [+0.006757, +0.009924] |
| 잔차 · 1차 코사인 T2a | t2a_cos | c0_cos | 389 | 0.441564 | 0.440420 | +0.001144 | [+0.000456, +0.002109] |
| 잔차 · 1차 코사인 T2b10k | t2b10k_cos | c0_cos | 389 | 0.445057 | 0.440420 | +0.004637 | [+0.003950, +0.005554] |
| 잔차 · 1차 등토큰 T2b | t2b_eqtok | c0_cos | 389 | 0.450057 | 0.440420 | +0.009637 | [+0.008141, +0.011622] |
| 잔차 · 상수 LR T2b | t2b_const | c0_const | — | — | — | — | 계산 안 함 (t2b_const: 아직 없음 · c0_const: 아직 없음) |
| 잔차 · 같은 update T2b | t2b_upd | c0_const | — | — | — | — | 계산 안 함 (t2b_upd: 통과 · c0_const: 아직 없음) |
| warm-start - 직접 | t2b_warm | t2b_const | — | — | — | — | 계산 안 함 (t2b_warm: 통과 · t2b_const: 아직 없음) |

## 비교의 뜻

- **잔차 · 1차 코사인 T2b** — Equal-Raw-Data 1차 핵심 격차
- **잔차 · 1차 코사인 T2a** — 제거만 한 T2a 의 BPB 순효과
- **잔차 · 1차 코사인 T2b10k** — N=10,000
- **잔차 · 1차 등토큰 T2b** — 같은 토큰 수 (LR 축 다름)
- **잔차 · 상수 LR T2b** — Final Test 대표 조건의 dev 격차
- **잔차 · 같은 update T2b** — 3주차 seed42 (C0 는 1 update 차)
- **warm-start - 직접** — 4주차 delta 의 부호 반대

## 한계

- dev 문서는 1차부터 여러 번 본 집합이다 — 낙관 편향이 있을 수 있다 (Final Test 를 따로 둔 이유)
- 고정 token-context 평가다. 같은 원문 byte-context 비교가 아니다 (amendment §2)
- 1차 코사인 run 들은 seed 42 하나이고, 상수 LR 대표 조건도 seed 42 재학습 하나다
