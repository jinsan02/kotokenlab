# 3주차 사전 점검 — 같은 update 대조

> **이 표는 코드가 적는다.** `.conda/python.exe tools/upd_preflight.py`
> 동결 설정은 `tools/upd_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md)
> "3주차 run 설정 동결". GPU 0시간. 판정하지 않는다 — 돌려도 되는지만 본다.

HEAD `3f50e83` · 풀 `data/interim/docs/train.jsonl` sha256 `89bd4269999b`

| 점검 | 상태 | 내용 |
|---|---|---|
| 측정 코드가 깨끗한가 | **통과** | src·configs 에 커밋 안 된 변경 없음 |
| 짝 C0 seed42 update 수 | **통과** | 1,523 (관찰 토큰에서 내림 (회계 수정 이전)) vs PLAN 1,523 |
| C0 seed42 이후 측정 경로 커밋 | **통과** | 10개 — 회계 수정 이전 run. PLAN 이 09-20 커밋별 확인으로 허용 |
| 짝 C0 seed42 공유 플래그 | **통과** | pool-docs · eval-bytes · eval-budget · lr-schedule 같음 |
| 짝 C0 seed42 update 크기 | **통과** | seq_len·micro_bs·accum = (2048, 2, 8) -> 32,768 토큰/update |
| 짝 C0 seed123 update 수 | **통과** | 1,524 (원장 updates) vs PLAN 1,524 |
| C0 seed123 이후 측정 경로 커밋 | **통과** | 동결이 다룬 것뿐 (2b37998) |
| 짝 C0 seed123 공유 플래그 | **통과** | pool-docs · eval-bytes · eval-budget · lr-schedule 같음 |
| 짝 C0 seed123 update 크기 | **통과** | seq_len·micro_bs·accum = (2048, 2, 8) -> 32,768 토큰/update |
| 짝 C0 seed2026 update 수 | **통과** | 1,524 (원장 updates) vs PLAN 1,524 |
| C0 seed2026 이후 측정 경로 커밋 | **통과** | 동결이 다룬 것뿐 (2b37998) |
| 짝 C0 seed2026 공유 플래그 | **통과** | pool-docs · eval-bytes · eval-budget · lr-schedule 같음 |
| 짝 C0 seed2026 update 크기 | **통과** | seq_len·micro_bs·accum = (2048, 2, 8) -> 32,768 토큰/update |
| 새 run_id 가 비어 있는가 | **통과** | cpt_t2b_mean_upd_seed42, cpt_t2b_mean_upd_seed123, cpt_t2b_mean_upd_seed2026 |
| 기존 T2b seed42 와 같은 모델 | **통과** | artifacts/models/kot2b_v2_n30000_mean |
| 기존 T2b seed123 와 같은 모델 | **통과** | artifacts/models/kot2b_v2_n30000_mean |
| 기존 T2b seed2026 와 같은 모델 | **통과** | artifacts/models/kot2b_v2_n30000_mean |
| 모델 체크포인트 해시 | **통과** | 2904fc79fb76 vs 등록 2904fc79fb76 |
| 환경 등록 | **통과** | env_sha256 등록됨 |
| 시각 검증 | **통과** | 남은 유효시간 24.0h / 계획 6.9h |
| 디스크 여유 | **통과** | 1573GB (체크포인트 3 x 약 1GB) |
| GPU 여유 | **통과** | 15,158MB 비어 있음 (필요 약 12,000MB). 다른 학습이 돌고 있으면 시작하지 않는다 |
| 재현 cpt_t2b_mean_r5_seed42 | **통과** | 토큰 34,832,384 -> 바이트 168,502,276 vs 원장 168,502,276 |
| 재현 cpt_t2b_mean_r5_seed123 | **통과** | 토큰 34,832,384 -> 바이트 168,503,536 vs 원장 168,503,536 |
| 재현 cpt_t2b_mean_r5_seed2026 | **통과** | 토큰 34,824,192 -> 바이트 168,505,066 vs 원장 168,505,066 |
| 재현 cpt_c0_qwen_r5_seed42 | **통과** | 토큰 49,922,048 -> 바이트 168,505,891 vs 원장 168,505,891 |
| 재현 cpt_c0_qwen_r5_seed123 | **통과** | 토큰 49,938,432 -> 바이트 168,575,610 vs 원장 168,575,610 |
| 재현 cpt_c0_qwen_r5_seed2026 | **통과** | 토큰 49,938,432 -> 바이트 168,568,340 vs 원장 168,568,340 |
| 풀 여유 seed42 | **통과** | 최대 58,310,656 토큰 / 예산 49,905,664 (여유 16.8%) |
| d_168 지점 seed42 | **통과** | update 1,063 · 원문 168,502,276 바이트 |
| 풀 여유 seed123 | **통과** | 최대 58,310,656 토큰 / 예산 49,938,432 (여유 16.8%) |
| d_168 지점 seed123 | **통과** | update 1,063 · 원문 168,503,536 바이트 |
| 풀 여유 seed2026 | **통과** | 최대 58,310,656 토큰 / 예산 49,938,432 (여유 16.8%) |
| d_168 지점 seed2026 | **통과** | update 1,063 · 원문 168,545,685 바이트 |

## 예측 (`upd_preflight.json`)

| seed | update | 예산 토큰 | 종료 원문 바이트 | d_168 update | d_168 원문 바이트 | 기존 r5 종료 |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 1,523 | 49,905,664 | 241,398,542 | 1,063 | 168,502,276 | 1,063 update · 168,502,276 |
| 123 | 1,524 | 49,938,432 | 241,545,084 | 1,063 | 168,503,536 | 1,063 update · 168,503,536 |
| 2026 | 1,524 | 49,938,432 | 241,521,009 | 1,063 | 168,545,685 | 1,062 update · 168,505,066 |

반복 바이트는 0 이다 — 풀(50,000 + 20,000문서)이 예산을 반복 없이 채우고,
`cpt.py` 에는 풀을 되감는 경로가 없다 (모자라면 멈춘다).

## 실행 명령 (seed 42 부터, 한 번에 하나)

```
C:\llm_tokenizer\.conda\python.exe -m src.training.cpt --model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-tokens 49905664 --pool-docs 50000 --pool-extend-docs 20000 --eval-bytes 20000000 --eval-at 168500000 --eval-budget 2000000 --lr-schedule constant --seed 42 --tag upd --save
C:\llm_tokenizer\.conda\python.exe -m src.training.cpt --model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-tokens 49938432 --pool-docs 50000 --pool-extend-docs 20000 --eval-bytes 20000000 --eval-at 168500000 --eval-budget 2000000 --lr-schedule constant --seed 123 --tag upd --save
C:\llm_tokenizer\.conda\python.exe -m src.training.cpt --model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-tokens 49938432 --pool-docs 50000 --pool-extend-docs 20000 --eval-bytes 20000000 --eval-at 168500000 --eval-budget 2000000 --lr-schedule constant --seed 2026 --tag upd --save
```

## 한계

- 재현은 원문 바이트 회계와 데이터 순서를 검증한다. GPU 연산·난수는 보지 않는다
- 기존 r5 T2b 는 회계 수정 이전 코드라 종료가 update 경계가 아닐 수 있다 —
  표의 "기존 r5 종료" update 수는 관찰 토큰을 내린 값이다
