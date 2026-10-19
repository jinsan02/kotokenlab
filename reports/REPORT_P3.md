# KoTokenLab — 3차 보고서 (외부 비판 감사 뒤 amendment, 2026-09-19 ~ 10-19)

> 이 문서는 서술이다. **수치의 정본은 코드가 쓴 표다** — 핵심 주장은
> [`tables/core_claims.md`](tables/core_claims.md) (`tools/claims.py`), 주차별 판정은 아래에
> 링크한 각 표, 사전 등록과 판정 원문은 [`docs/PLAN.md`](../docs/PLAN.md) 의 "2~6주차 run
> 설정 동결" 과 "Final Test 개봉" 절이다. 1차 결과는 [`FINAL_REPORT.md`](FINAL_REPORT.md)
> (역사적 정본, 소급 수정하지 않는다).

## 1. 요약

**Qwen2.5-0.5B 의 한국어 어휘 30,000개를 크기 보존 치환(T2b)하면 한국어 토큰이 30% 줄지만,
같은 원문 168.5MB 로 CPT 한 뒤에도 한국어 BPB 가 원본(C0)보다 약 0.35 나쁘다.** 이 격차는
seed 3쌍에서 재현됐고(SD 0.000235), 한 번도 쓰지 않은 Final Test 5,067문서에서
d +0.348637, 95% 문서 구간 [+0.345985, +0.351245] 로 **사전 등록 기준(하한 > 0)을 넘었다.**

격차의 원인으로 의심한 것 셋은 이 설정에서 격차를 설명하지 못했다.

| 대안 설명 | 대조 | 결과 |
|---|---|---|
| T2b 가 update 를 덜 받았다 | 같은 1,523 update 까지 학습 | 잔차의 약 93% 가 남는다 (rho 7.2%) |
| 새 행이 몸통보다 늦게 배운다 | 첫 20% 동안 새 행만 학습 (warm-start) | 직접 CPT 보다 0.0117 나쁘다 |
| 끝에서 학습률을 내려야 한다 | 마지막 20% 감쇠 (WSD) | 상수 LR 보다 T2b +0.0104 · 잔차 +0.0092 나쁘다 |

그 대가로 산 것은 시스템 비용이다 — 같은 원문의 prefill 이 34.7~41.3% 짧다. **다만 이것은
품질이 나쁜 모델의 비용이지 품질을 맞춘 서비스 개선이 아니다** (통제축 `raw_prompt`).

이 프로젝트가 방어하는 기여는 amendment 의 문장 그대로다: 한국어 소형 모델에서 vocabulary
크기를 고정한 prune+replace 가 압축 · 시스템 비용과 품질 사이에 만드는 교환비를, 동일 원문량과
동일 update 예산으로 나누어 측정하고, 제한된 CPT 에서의 회복 곡선과 그 처방 시도를 보고한 증거.

## 2. 무엇이 바뀌었나 — amendment

2026-09-19 의 외부 비판 감사([`docs/CRITICAL_REVIEW_AMENDMENT_2026-09-19.md`](../docs/CRITICAL_REVIEW_AMENDMENT_2026-09-19.md))가
1차 · 2차의 몇몇 주장을 거둬들이고 3차 계획을 7주로 다시 짰다.

- **같은 원문 ≠ 같은 compute.** 168.5MB 비교는 같은 원문량이고 토큰 · update 가 다르다.
  compute 쪽은 3주차가 update 수 proxy 로 따로 쟀다 (FLOPs 는 재지 않았다)
- **회복률 R 은 보조 지표다.** 절대 격차 d = BPB(T2b) − BPB(C0) 가 1차 지표다
- **2σ 는 결정 휴리스틱이지 95% CI 가 아니다.** 불확실성은 종류(학습 seed · 문서 표본 · 문항
  표본)를 나누어 따로 보고한다
- 거둬들인 주장 일곱은 [`core_claims.md`](tables/core_claims.md) 의 W1~W7 에 있다 — "65% 벽",
  "R1 은 치환 특유", "tie 는 원인이 아니다", "이식 = 초기화 상한", "CC 중복은 C0 에 유리",
  "E1 = FVT", "168.5MB = 같은 compute"

## 3. 결과

### 3.1 주차별

| 주 | 질문 | 판정 | 표 |
|---|---|---|---|
| 1 | 학습 회계 · BPB 경계 · split 감사 | 회계 수정(`2da591e`), split 간 문서 해시 겹침 0 | [`split_audit`](tables/split_audit.md) · [`scored_bytes`](tables/scored_bytes.md) |
| 2 | 같은 원문 격차가 seed 42 의 우연인가 | 아니다 — seed 3쌍 재현 | [`seed_pairs`](tables/seed_pairs.md) |
| 3 | 격차가 update 부족 때문인가 | 주로 아니다 — 예측 적중 | [`upd_pairs`](tables/upd_pairs.md) |
| 4 | 신규 행 warm-start 가 나은가 | 아니다 — gate 미달, 예측 적중 | [`warm_pairs`](tables/warm_pairs.md) |
| 5 | 문서 표본 구간 · 노출 층화 · 문구 · Final Test 규칙 | 규칙 고정 | [`doc_bootstrap`](tables/doc_bootstrap.md) · [`exposure_strata`](tables/exposure_strata.md) |
| 6 | 마지막 감쇠(WSD)가 나은가 | 아니다 — 두 판정 악화, **예측 빗나감** | [`wsd_pairs`](tables/wsd_pairs.md) |
| FT | 격차가 본 적 없는 문서에서도 성립하는가 | **확증** — 예측 적중 | [`heldout`](tables/heldout.md) |

### 3.2 Final Test

규칙은 개봉 13일 전(5주차 D, `f20aaeb`)에 고정했고, 도구는 결과 전에 만들었다(`e59d515`).
개봉 커밋에 `eval-freeze-v1` · `final-test-opened` 태그를 달았다(`2faaf79`). 대표 조건 둘
(상수 LR C0 · T2b 재학습, seed 42)만 평가했다 — 다중 비교를 만들지 않기 위해서다.

- **한국어**: d +0.348637 [+0.345985, +0.351245], 5,067문서 · 원문 20.0MB. 확증
- dev 의 같은 쌍은 +0.345564 — **본 적 없는 문서에서 격차가 조금 더 크다.** dev 를 여러 번
  본 것이 격차를 과장했다는 증거는 없다. 두 모델의 절대 BPB 는 Final Test 쪽이 둘 다 낮다
- 영어 d +0.008230 · 코드 +0.009895 — T2b 가 두 대조 축에서도 조금 나쁘다 (기술만)
- 압축: 평가한 그 문서들에서 한국어 tok/byte −30.27%, 영어 −0.01%, 코드 +0.22% —
  dev 의 −30.2% 와 같다
- 학습 seed 불확실성은 위 구간에 없다. dev seed 쌍 SD 0.000235 를 따로 적는다

### 3.3 처방 쪽 결과 (3 · 4 · 6주차)

세 대조는 모두 "같은 예산 안에서 무엇을 바꾸면 격차가 줄어드는가" 를 물었고 셋 다 아니라고
답했다. 6주차가 특히 예상과 달랐다 — 감쇠 구간에서 학습률이 약 2e-6 아래로 내려가자 WSD 는
거의 배우지 않았고, 상수 LR 은 끝까지 내려갔다. 손해는 아직 빨리 배우는 T2b 가 C0 의 약
8배였다. 우리의 읽기는 "peak 1e-5 에서 이 예산의 끝은 잡음이 아니라 학습량에 묶여 있다" 이고,
결과를 보고 만든 설명이라 **탐색적** 이다. 이 예산에서 어휘를 교체한 CPT 의 처방은 **상수
LR** 로 남는다 (1차 코사인 대비 seed 42 에서 격차 −0.083916).

### 3.4 T2a (제거만)

시험한 aggregate 지표에서 저하가 검출되지 않았다 — 한국어 dev BPB −0.064%, KMMLU 6과목
−0.21%p [−1.05%p, +0.58%p]. 노출 층화(5주차)에서 제거 토큰이 나온 dev 문서는 434개 중
4개뿐이었고, 나머지 430문서에서의 작은 이득은 softmax 경쟁 항목 축소와 부합한다 (탐색적).
"무손상" 이라는 말은 쓰지 않는다. KMMLU 의 오염 검사 범위는 우리 CPT 풀과의 13-gram 겹침
뿐이다.

## 4. 예측 기록 — 맞은 것과 빗나간 것

결과를 보기 전에 PLAN 에 적은 예측이다. 빗나간 것을 지우지 않는다.

| 주 | 예측 | 결과 |
|---|---|---|
| 3 | rho ≤ 25% (격차 대부분이 update 부족이 아니다) | 적중 (7.2%) |
| 4 | gate 미달, 방향도 음수 | 적중 (−0.0117) |
| 6 | T2b 절대 BPB 개선 | **빗나감** (악화 +0.0104) |
| 6 | 잔차 구별 불가 | **빗나감** (악화 +0.0092) |
| 6 | R 71% 이상 (보조) | 적중 (71.66%) |
| FT | 확증 (구간 하한 > 0) | 적중 |

1 · 2차에서 우리가 세운 가설 세 개(노름 보정 · 정렬 예열 · 노출 부족)도 우리가 반증했다
([`FINAL_REPORT.md`](FINAL_REPORT.md) §4). 실수와 그 교정은 [`docs/MISTAKES.md`](../docs/MISTAKES.md).

## 5. 한계

- **0.5B 한 모델, 168.5MB 한 예산.** 1.5B 는 수술 손상만 쟀고 CPT 는 하지 않았다. 500MB 연장도
  하지 않았다. 점근선은 식별되지 않았다
- **seed.** 핵심 격차(2 · 3주차)는 3 seed, 처방 대조(4 · 6주차)와 Final Test 체크포인트는 seed 42
  하나다. 6주차 경계의 σ 는 상수 LR 3 seed 에서 빌렸다
- **지표.** 고정 token-context BPB 다. 같은 원문 byte-context 비교는 하지 않았다 (amendment §2)
- **compute.** 같은 update 수는 proxy 다. FLOPs · 실제 vocab softmax 비용은 맞추지 않았다
- **시스템.** 배치 1 · RTX 5070 Ti 한 대 · dev 를 이어 붙인 프롬프트. 품질을 맞춘 서비스 수준
  지연은 재지 않았다. decode 는 잡음이 커 인용하지 않는다
- **데이터.** CPT 코퍼스와 Qwen 사전학습의 겹침과 그 방향을 모른다. 한국어 내부 도메인 라벨은
  정확도 ~55% 라 도메인별 결과를 보고하지 않는다
- **처방 탐색의 폭.** warm-start 는 한 설정(20%, 같은 학습률), WSD 는 한 설정(20%, 바닥 10%,
  peak 1e-5)만 시험했다
- **비결정성.** 같은 설정의 T2b 재학습이 학습 중반에 최대 약 0.001 갈렸다 (6주차 재현 검사가
  폭 ±0.001 에 근접). 종점은 ±0.0002 안에서 재현됐다

## 6. 선행 연구 대비 위치

자세한 비교는 [`docs/RELATED_WORK.md`](../docs/RELATED_WORK.md) 와 amendment §5.

- **어휘 수정 자체의 신규성은 없다.** continued BPE · leaf pruning (Purason et al.), 부품 평균
  초기화와 2단계 적응 (Smith et al.), 한국어 어휘 확장 (EEVE) 은 이미 있다. 우리 쪽은 **크기를
  고정한 치환** 을 아주 작은 CPT 예산에서 잰 것이다
- **대규모 토크나이저 연구** (Dagan et al.) 는 50B+ 토큰 규모다. 여기는 수천만 토큰의 미회복
  구간이다
- **ZeTT · FOCUS 를 돌리지 않았다.** 초기화 우열은 주장하지 않는다
- **스케줄** 의 민감도는 constant + cooldown 계열 (Hägele et al.) 의 재현 사례이지 새 기법이
  아니다. 이 예산에서는 그 cooldown 이 오히려 손해였다는 것이 우리 관측이다
- 우리가 기여로 남기는 것은 같은 파이프라인 안에서 **압축 · 시스템 · 품질** 을 함께 재고, 같은
  원문과 같은 update 두 축으로 나누고, 처방 시도를 사전 등록으로 반증한 기록이다

## 7. 미확인 목록 (amendment §10 갱신)

확인 전까지 수치나 결론을 만들지 않는다.

이번에 채운 것
- 문서 단위 BPB 구간 (5주차, `doc_bootstrap`)
- 168.5MB 에서 C0 의 seed 분산과 paired 격차 분산 (2주차, `seed_pairs`)
- 격차가 본 적 없는 문서에서도 성립하는가 (Final Test)

남은 것
- Qwen 사전학습과 CPT 문서의 실제 겹침과 편향 방향
- 과거 run 의 정확한 누적 학습률
- 같은 원문 byte-context BPB
- 품질을 맞춘 서비스 수준 TTFT · throughput
- layer 단위 표현 변화와 어느 파라미터가 적응하는가 (스펙 Q13 · Q14 — 임베딩 기울기만 쟀다)
- 다른 peak · 감쇠 길이의 WSD, 다른 1단계 길이의 warm-start
- 1.5B CPT, 500MB 연장, 정식 ZeTT · FOCUS 비교

## 8. 재현

```
.conda/python.exe tools/claims.py          # core_claims.md
.conda/python.exe tools/heldout_pairs.py   # heldout.md (평가 run 이 있어야 한다)
.conda/python.exe tools/wsd_pairs.py       # 그 밖의 주차 표도 같은 방식
.conda/python.exe tools/overview.py --bundle
```

모든 표는 원장(`experiments/*.tsv`)과 `experiments/runs/<run_id>/` 의 문서별 파일에서
다시 만든다. 학습 run 의 argv 는 원장에 있고, 주차마다 동결 설정은 `tools/*_spec.py` 다.
