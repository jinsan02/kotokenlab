# 핵심 주장 표 — 3차(amendment) 종료 시점

> **이 표는 코드가 적는다.** `.conda/python.exe tools/claims.py` — 수치는 원장과 문서별
> NLL 파일에서 각 도구의 함수로 다시 냈다. 손으로 고치지 마라.

한국어 품질 지표는 고정 token-context dev BPB 다 (amendment §2). 잔차 d = BPB(T2b) − BPB(C0).
불확실성은 종류가 다르면 섞지 않는다.

## 주장

| | 주장 | 수치 | 통제축 | 불확실성 | 상태 | 근거 |
|---|---|---|---|---|---|---|
| C1 | T2b 는 한국어 토큰 수를 줄인다 (크기 보존 치환 N=30,000) | dev tok/byte 0.2963 -> 0.2068 (-30.2%) | 같은 원문 · 토크나이저만 | 전수 측정 (표본 오차 없음) | **dev 측정** | tokenizer_metrics (tok_bench_v2) · heldout.md 압축률 |
| C2 | 같은 원문을 넣으면 T2b 의 prefill 이 짧다 | prefill -41.3% ~ -34.7% (4개 길이) | raw_prompt — 같은 원문, 토큰 수 다름, **품질 미매칭** | 측정 반복(n_runs) 안의 분산만, 배치 1 · GPU 한 대 | **dev 측정** | system_bench.md |
| C3 | 같은 원문 168.5MB CPT 후 T2b 의 한국어 dev BPB 가 C0 보다 나쁘다 (잔차 d) | d 평균 +0.345679 · seed SD 0.000235 (3쌍) | 같은 원문 168.5MB · 상수 LR · 같은 seed 짝 (compute 는 다르다) | 학습 seed (n=3) | **dev 재현** | seed_pairs.md |
| C4 | 사전에 고른 seed 42 대표 쌍에서, 처음 보는 한국어 문서에도 양의 잔차가 있다 (Final Test) | d +0.348637 [+0.345985, +0.351245] (5,067문서) | C3 과 같다 · 대표 체크포인트 seed 42 둘 | 문서 표본 (seed 는 C3 의 SD) | **Final Test 확증** | heldout.md |
| C5 | 1차 격차의 일부는 코사인 스케줄의 몫이다 — '65% 벽' 은 식별되지 않았다 | d 코사인 +0.429555 -> 상수 +0.345639 (-0.083916) | 같은 원문 · 스케줄만 (seed 42) | 학습 seed (seed 42 하나) | **dev 측정** | cpt_main.md · seed_pairs.md · amendment §3 |
| C6 | 시험한 update 보정(같은 1,523 update)만으로는 격차가 해소되지 않았다 | 같은 update 에서 d +0.321189 · 줄어든 몫 rho 7.2% (seed 별 평균, 원인의 기여율이 아니다) | 같은 optimizer update 수 (compute proxy, FLOPs 동일 아님) | 학습 seed (n=3) | **dev 판정 (예측 적중)** | upd_pairs.md |
| C7 | 같은 총예산 안에서 새 행을 먼저 학습시키는 배분(첫 20%)은 직접 CPT 보다 낫지 않았다 | delta -0.011709 (gate +0.010 미달) | 총 원문 168.5MB (준비 비용 포함) | 학습 seed (seed 42 하나) | **dev 판정 (예측 적중)** | warm_pairs.md |
| C8 | 시험한 WSD(마지막 20% 감쇠)는 상수 LR 보다 낫지 않다 | T2b +0.010403 · 잔차 +0.009184 (경계 0.005, 둘 다 악화) | 같은 원문 168.5MB · 스케줄만 | 학습 seed (seed 42 하나, σ 는 상수 LR 에서 빌림) | **dev 판정 (예측 빗나감)** | wsd_pairs.md |
| C9 | 제거만 한 T2a 는 시험한 aggregate 지표에서 저하가 검출되지 않았다 | 한국어 dev BPB -0.064% · KMMLU 6과목 -0.21%p [-1.05%p, +0.58%p] | 같은 원문 · 코사인 seed 42 · vocab 축소만 | 문항 표본 (KMMLU) · seed 하나. 오염 검사 범위: 우리 CPT 풀(앞 50,000문서)과 질문 본문의 문자 13-gram(공백 제거 · NFKC 소문자) 겹침만 — 덮임 50% 이상 2 / 1,900문항 (`contamination.md`). Qwen 사전학습 · 의역 · 문자 13개보다 짧은 겹침은 검사하지 않았다. | **dev 측정 ("무손상" 은 철회)** | cpt_main.md · hanja_probe.md |

## 철회한 주장 (본문은 각 문서에 보존, amendment §3)

| | 이전 주장 | 지금 말할 수 있는 것 |
|---|---|---|
| W1 | 회복률 65% 는 보편적 벽이다 | 코사인 스케줄의 유한 종점이었다 (C5). 점근선은 미식별 |
| W2 | R1 이 치환 특유 현상을 입증했다 | 행 수 · 노출 · 토큰화가 함께 달랐다 — 두 구성의 궤적이 달랐을 뿐 |
| W3 | tie 는 원인이 아니다 | 17.5MB · 그 설정에서 5%p 급 효과가 없었을 뿐 (paired +0.444%p) |
| W4 | CPT 행 이식은 완벽한 초기화 상한이다 | 몸통-행 상호적응을 깨는 compatibility stress test 다 |
| W5 | Common Crawl 중복은 C0 에 유리하다 | 겹침의 크기와 방향을 재지 않았다 |
| W6 | E1 은 FVT 와 동등하다 | standalone 표면이 있는 25,821개에서만 일치 |
| W7 | 168.5MB 는 같은 compute 다 | 같은 원문량이다. 토큰 · update 가 다르다 (C6 이 따로 잰다) |

## 읽는 법

- **Final Test 확증** 은 C4 하나뿐이다. 나머지는 dev 결과다 — dev 는 여러 번 본 집합이다
- 시스템 수치(C2)는 품질이 나쁜 모델의 비용이다. 품질을 맞춘 서비스 개선이 아니다
- compute 를 맞춘 비교는 C6 의 update 수 proxy 뿐이다 (FLOPs 는 재지 않았다)
