# 4주차 사전 점검 — 신규 행 warm-start

> **이 표는 코드가 적는다.** `.conda/python.exe tools/warm_preflight.py`
> 동결 설정은 `tools/warm_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md)
> "4주차 run 설정 동결". GPU 0시간. 판정하지 않는다 — 돌려도 되는지만 본다.

HEAD `16b9733` · 풀 sha256 `89bd4269999b`

| 점검 | 상태 | 내용 |
|---|---|---|
| 측정 코드가 깨끗한가 | **통과** | src·configs 에 커밋 안 된 변경 없음 |
| 돌고 있는 학습 프로세스 | **통과** | 없음 |
| GPU 잠금 | **통과** | 잠금 파일 없음 |
| run_id seed42 | **통과** | cpt_t2b_mean_warm_seed42 — 비어 있음 |
| run_id seed123 | **통과** | cpt_t2b_mean_warm_seed123 — 비어 있음 |
| run_id seed2026 | **통과** | cpt_t2b_mean_warm_seed2026 — 비어 있음 |
| 비교 대상 seed42 | **통과** | 직접 CPT 있음 · C0 있음 |
| 계획 config vs 직접 CPT seed42 | **통과** | 허용한 차이만: pool_extend_docs, warm_bytes, warm_rows, warmup_bytes |
| 직접 CPT seed42 공유 플래그 | **통과** | 모델·예산·풀·평가·스케줄 같음 |
| 비교 대상 seed123 | **통과** | 직접 CPT 있음 · C0 있음 |
| 계획 config vs 직접 CPT seed123 | **통과** | 허용한 차이만: pool_extend_docs, warm_bytes, warm_rows, warmup_bytes |
| 직접 CPT seed123 공유 플래그 | **통과** | 모델·예산·풀·평가·스케줄 같음 |
| 비교 대상 seed2026 | **통과** | 직접 CPT 있음 · C0 있음 |
| 계획 config vs 직접 CPT seed2026 | **통과** | 허용한 차이만: pool_extend_docs, warm_bytes, warm_rows, warmup_bytes |
| 직접 CPT seed2026 공유 플래그 | **통과** | 모델·예산·풀·평가·스케줄 같음 |
| 직접 CPT 이후 측정 경로 커밋 | **통과** | 13개 모두 근거 있음 |
| 새 행 목록 | **통과** | 30,000행 · 최대 id 151,642 < vocab 151,936 |
| 모델 체크포인트 해시 | **통과** | 2904fc79fb76 vs 등록 2904fc79fb76 |
| 환경 등록 | **통과** | 등록됨 |
| 시각 검증 | **실패** | 24시간 안의 성공 기록 없음 — tools/check_clock.py --record |
| 디스크 여유 | **통과** | 1568GB |
| GPU 여유 | **통과** | 14,881MB 비어 있음 (필요 약 12,000MB) |
| 재현 cpt_t2b_mean_r5_seed42 | **통과** | 토큰 34,832,384 -> 바이트 168,502,276 vs 원장 168,502,276 |
| 재현 cpt_t2b_mean_r5_seed123 | **통과** | 토큰 34,832,384 -> 바이트 168,503,536 vs 원장 168,503,536 |
| 재현 cpt_t2b_mean_r5_seed2026 | **통과** | 토큰 34,824,192 -> 바이트 168,505,066 vs 원장 168,505,066 |
| 예측 seed42 | **통과** | 2단계 전환 update 213 (33,796,179 B) · 종료 update 1,063 (168,502,276 B) |
| 예측 seed123 | **통과** | 2단계 전환 update 213 (33,792,802 B) · 종료 update 1,063 (168,503,536 B) |
| 예측 seed2026 | **통과** | 2단계 전환 update 214 (33,846,388 B) · 종료 update 1,063 (168,545,685 B) |

## 예측 (`warm_preflight.json`)

| seed | 2단계 전환 update | 전환 원문 바이트 | 종료 update | 종료 원문 바이트 |
|---:|---:|---:|---:|---:|
| 42 | 213 | 33,796,179 | 1,063 | 168,502,276 |
| 123 | 213 | 33,792,802 | 1,063 | 168,503,536 |
| 2026 | 214 | 33,846,388 | 1,063 | 168,545,685 |

## 실행 명령 (gate seed 부터, 한 번에 하나)

```
C:\llm_tokenizer\.conda\python.exe -u -m src.training.cpt --model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-bytes 168500000 --pool-docs 50000 --eval-bytes 20000000 --eval-at 33700000 --eval-budget 2000000 --lr-schedule constant --seed 42 --tag warm --warm-rows artifacts/tokenizers/kot2b_v2_n30000/id_map.json --warm-bytes 33700000 --damaged-rows artifacts/tokenizers/kot2b_v2_n30000/id_map.json --save
C:\llm_tokenizer\.conda\python.exe -u -m src.training.cpt --model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-bytes 168500000 --pool-docs 50000 --eval-bytes 20000000 --eval-at 33700000 --eval-budget 2000000 --lr-schedule constant --seed 123 --tag warm --warm-rows artifacts/tokenizers/kot2b_v2_n30000/id_map.json --warm-bytes 33700000 --damaged-rows artifacts/tokenizers/kot2b_v2_n30000/id_map.json --save
C:\llm_tokenizer\.conda\python.exe -u -m src.training.cpt --model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-bytes 168500000 --pool-docs 50000 --eval-bytes 20000000 --eval-at 33700000 --eval-budget 2000000 --lr-schedule constant --seed 2026 --tag warm --warm-rows artifacts/tokenizers/kot2b_v2_n30000/id_map.json --warm-bytes 33700000 --damaged-rows artifacts/tokenizers/kot2b_v2_n30000/id_map.json --save
```

seed 123 · 2026 은 seed 42 가 gate 를 통과할 때만 돌린다 (PLAN).

## 한계

- 재현은 원문 바이트 회계와 데이터 순서를 검증한다. GPU 연산·난수는 보지 않는다
- 1단계에서 옛 행이 그대로인지는 실제 run 이 단계 전환 때 비트 단위로 확인한다
