# 인수인계 — 현재 상태와 다음 순서

최종 갱신 2026-08-31. 규칙은 [`RULES.md`](RULES.md), 범위와 종료 조건은
[`PLAN.md`](PLAN.md), 프롬프트는 [`PROMPTS.md`](PROMPTS.md),
도메인 라벨 신뢰 범위는 [`DOMAIN_LABELS.md`](DOMAIN_LABELS.md).

**스펙과 다르게 한 것은 [`DESIGN_DELTA.md`](DESIGN_DELTA.md) 에 모여 있다.**
Gate 이원화, 도메인 2분류, 인접쌍 채굴, pretokenizer 경계 함정, 노름 보정 반증,
조건별 sigma — 스펙만 읽고 코드를 고치면 이 결정들을 되돌리게 된다.

---

## 한 줄 상태

**Level 1 정식 측정까지 완료. Step 3 T2a/T2b 도 끝났다.**
코퍼스는 `phase1-tokenizer-freeze` 로 얼렸고, T2b n=30,000 이 한국어 -28.4% 로
Candidate Gate 를 통과했다. 다음은 Step 4 embedding surgery 다. 한국어 118.7만 문서(정제 4.73GB)와
영어·코드 대조군을 만들었고 `manifest_sha256` 을 [`PLAN.md`](PLAN.md) 에 고정했다.
남은 것은 그 코퍼스로 Level 1 을 다시 재고 얼리는 일이다.

```
Step 0   저장소 · 원장 · 훅 · CI · 환경 · 자원 실측              완료
Step 1   한국어 파이프라인 (규칙 v4, 호스트 상한 400)            완료 (파일럿)
Step 1b  영어 · 코드 대조군                                     완료 (파일럿)
Step 1c  도메인 라벨 블라인드 감사                               완료
Step 2   Level 1 벤치마크 + Gate 임계값 확정                     완료
Step 2b  전체 규모 코퍼스 v1 (한국어 + 영어·코드 대조군)          완료
Step 2c  Level 1 정식 측정 (v1 dev 259MB)                        완료
Step 3   T2a / T2b 수술 + N 스윕 -> N=30,000 확정               완료
Step 4   embedding surgery — E1 부품 평균 채택                  완료
         ────────────────────────────────────────────────
현재     Step 5 Embedding Alignment + 노이즈 플로어 3 seed
금지     phase1-tokenizer-freeze 이후 코퍼스를 바꾸지 않는다
```

---

## 측정된 것

### Level 1 — 1층: 언어 간 (라벨이 출처 데이터셋이라 신뢰 가능)

| 구분 | 문자 수 | Qwen2.5 | HCX-SEED | A.X-4.0-Light |
|---|---:|---:|---:|---:|
| 한국어 | 103,657,654 | 0.6834 | −24.9% | **−39.4%** |
| 영어 | 11,549,402 | 0.2177 | +0.0% | **+7.3%** |
| 코드 | 8,350,350 | 0.2842 | **+17.6%** | **+22.5%** |

한국어 압축을 얻는 만큼 코드에서 20~27% 를 잃는다. 스펙 §16 이 경고한 trade-off 가
실측으로 확인됐다.

### Level 1 — 2층: 한국어 내부 (참고. 세분화는 보고하지 않는다)

| 구분 | 문자 수 | Qwen2.5 | A.X |
|---|---:|---:|---:|
| news | 16,136,240 | 0.7086 | −40.5% |
| 기타(한국어) | 87,521,414 | 0.6788 | −39.2% |

**차이가 1.3%p 다.** 반면 언어 간은 62%p 갈린다. 라벨을 못 믿는 축과 결과가
갈리는 축이 서로 다르므로, 세분화를 포기해도 잃는 정보가 거의 없다.

### 도메인 라벨 신뢰도

```
블라인드 감사 150건        54.7%
  news 정밀도              81%
  technical 재현율          8%
  community / ko_en_mixed   0%
```

경위는 [`DOMAIN_LABELS.md`](DOMAIN_LABELS.md).
**한국어 내부 세분화 수치를 단독으로 인용하지 마라.**

### 자원 실측 (RTX 5070 Ti 16GB)

```
학습 운영 설정      seq 2048 / micro_bs 2 / AdamW 8bit -> 13.3GB, 9,089 tok/s
50M 토큰 CPT        약 1.6시간.  코어 전체 GPU 예산 약 13시간
prefill 32,768 tok  0.5B 2,718MB / 1,310ms    1.5B 6,808MB / 2,972ms
```

전체 표: [`reports/tables/resource_probe.md`](../reports/tables/resource_probe.md)

---

## 다음에 할 일

### 1. 전체 규모 실행 — 완료 (2026-08-30)

만들어진 것 (계보와 한계는 [`PLAN.md`](PLAN.md) "고정된 계보"):

```
data_v1_seed42              한국어 1,186,892문서  정제 4,735MB  79분
data_controlen_v1_seed42    영어      51,960문서  정제  247MB   5분
data_controlcode_v1_seed42  코드      30,216문서  정제  187MB   2분
```

**dev 기준은 세분화 도메인이 아니라 보고 버킷 4개로 본다.** 한국어 news 38.2MB /
기타 201.2MB / 영어 11.6MB / 코드 8.3MB 로 전부 5MB 를 넘겼다. encyclopedia
0.4MB 등이 미달이지만 보고 대상이 아니다 ([`DOMAIN_LABELS.md`](DOMAIN_LABELS.md)).

**필터 통과율은 79.2% 로 기준(90~92%)을 밑돈다.** 사유가 `host_cap` 하나에 몰려
있고(15.79%, 파일럿 4.3%), 상한이 비율이 아니라 절대 건수라 규모가 커질수록 더
빡빡해지기 때문이다. 위키백과가 6,652건에서 400건으로 잘렸다 — 코퍼스의 약 1.2%.
v1 은 이대로 가되 백과 문체 과소대표를 전제로 해석한다.

대조군을 다시 돌릴 일이 있으면 `--max-bytes` 를 반드시 올려라. 기본값 120MB 가
`--max-docs` 보다 먼저 걸려서 조용히 절반 규모가 나온다. 처음 돌렸을 때
코드 dev 가 4.64MB 로 미달했던 원인이다.

### 2. Level 1 정식 측정 — 완료 (2026-08-30, run `tok_bench_v1`)

v1 dev 259.2MB / 63,961문서를 토크나이저 3종으로 234초에 쟀다. 수치는 위
"측정된 것" 표가 정식 값이고, 코퍼스는 `phase1-tokenizer-freeze` 로 얼렸다.
Candidate Gate 는 hcx_seed / ax_4_0_light 둘 다 하드 상한 이내라 GPU 진행이지만,
둘 다 코드 악화가 보고 기준 10% 를 넘겼다 (+17.6% / +22.5%).

### 3. Step 3 — T2a / T2b 완료 (2026-08-30)

전체 표는 [`reports/tables/t2_sweep.md`](../reports/tables/t2_sweep.md).

```
                한국어      영어     코드    vocab
T2a    n=30,000  +0.0%    +0.0%   +0.1%   121,643  (embedding -26.9M)
T2b v2 n=30,000  -30.2%   -0.0%   -0.8%   151,643  (유지)
```

기증자 채굴은 v2 를 쓴다. v1 은 pretokenizer 세그먼트 경계를 확인하지 않아
전체 인접쌍의 43.8% 가 발화 불가능한 쌍이었고, 슬롯 활성률이 59.0% 에
그쳤다 (v2 는 92.9%).

**T2a 가 못한 것이 아니라 사는 물건이 다르다.** 제거 대상이 코퍼스에서 거의
안 쓰이므로(3만개 = 13.6억 토큰 중 0.0035%) 지우기만 해서는 토큰화가 바뀔
이유가 없다. T2a 가 사는 것은 파라미터 26.9M(embedding 의 19.7%) 감소이고,
T2b 가 사는 것은 압축 -28.4% 다. 압축은 **빈자리를 채워야** 나온다.

외부 참조 대비: HCX 는 한국어 -24.9% 에 코드 +17.6%, A.X 는 -39.4% 에 코드
+22.5% 다. T2b 는 HCX 보다 한국어가 좋으면서 코드 손해가 없다. v2 는 n=10,000
만으로도 HCX 를 넘는다 (-25.2%).

N 은 CPU 스윕으로 골랐다. 한계이득이 25.2 -> 3.5 -> 1.5%p 로 반감하고, 더 밀면
제거 비용이 빠르게 오르며(50,000 에서 10배) T2a 코드가 +0.8% 로 악화되기 시작한다.

**CPT 가 쓸 토크나이저는 `artifacts/tokenizers/kot2b_v2_n30000` 이다.**
대조군은 `kot2a_v1_n30000` 이고 둘은 같은 `prune_30000.tsv` 를 쓴다.
v1 계열 T2b 는 결함이 있으니 쓰지 마라.

### 4. Step 4 — embedding surgery 완료 (2026-08-30)

전체 표는 [`reports/tables/precpt_bpb.md`](../reports/tables/precpt_bpb.md).
학습을 전혀 하지 않은 Pre-CPT BPB 라 **초기화 효과만** 보인다.

```
                     파라미터   한국어    영어    코드
기준선 (수술 없음)      494.0M   1.1569  0.8115  0.4387
T2a 제거만             467.0M   1.1559  0.8116  0.4398   <- 사실상 무손실
T2b E0 무작위          494.0M   5.0148  1.1246  0.6695
T2b E1 부품 평균       494.0M   2.3803  0.8126  0.4413   <- 채택
T2b E1 + 노름 보정     494.0M   3.0017  0.9448  0.5470
T2b E2 역빈도 가중     494.0M   2.4282  0.8145  0.4525
```

**T2a 는 공짜다.** embedding 27.1M(5.5%)을 능력 손실 없이 줄인다. Level 1 에서
압축 이득이 0 이었던 것과 같은 사실의 다른 면이다.

**T2b 는 수술 직후에는 못 쓴다.** 가장 좋은 초기화도 기준선의 2.06배다. 스펙 §20 이
수술 직후 CPT 로 가지 말라고 한 이유가 수치로 확인됐다. 다만 그 간격을 메우는
수단은 정렬이 **아니다** — 정렬은 손해 없이 제 일을 하는 lr 이 없어서 뺐다
([`DESIGN_DELTA.md`](DESIGN_DELTA.md) 1-5). CPT 가 메운다.

**노름 보정은 해로웠다 — 가설이 반증됐다.** 평균은 노름이 줄어드니 E0 처럼 분포를
맞춰 주면 공정해질 줄 알았는데 2.3803 -> 3.0017 로 나빠졌다. tie_word_embeddings
라서 그 벡터가 출력 로짓 방향이기도 하고, 크기를 키우면 학습 안 된 새 토큰을
모델이 더 자신 있게 예측한다. 영어가 0.8115 -> 0.9448 로 함께 나빠진 것이
증거다 — softmax 분모를 통해 모든 문맥을 침범한 것이다. 작은 노름은 보호 장치였다.

산출물: `artifacts/models/kot2b_v2_n30000_mean` (본안),
`artifacts/models/kot2a_v1_n30000_none` (대조군).

### 5. Step 5 — 노이즈 플로어와 정렬 폐기 (완료)

노이즈 플로어(seed 42/123/2026)를 조건마다 쟀다. sigma 는 조건별로 21배 다르다 —
하나로 퉁칠 수 없다.

정렬은 3라운드 탐침 끝에 **뺐다.** 손해를 안 내면서 T2b 를 고치는 lr 이 없다
([`DESIGN_DELTA.md`](DESIGN_DELTA.md) 1-5). Pre-CPT 2.38 과 기준선 1.16 사이의
간격은 CPT 가 메운다 — 회복하는지 자체가 결과다.

### 6. Step 6 — 본 CPT 3조건 168.5MB (완료, 2026-08-31)

전체 표: [`reports/tables/cpt_main.md`](../reports/tables/cpt_main.md).

```
            학습 전     학습 후    vs C0      판정
  C0  ko    1.156880   1.137540      -       기준
  T2a ko    1.155912   1.136817   -0.064%   개선 (7.2 sigma)
  T2b ko    2.380297   1.567095  +37.76%    악화 (348 sigma)
```

**Q4 는 부정, Q5 는 긍정으로 닫혔다.** 압축은 약속대로 됐다 — 같은 원문을
C0 는 49.9M, T2b 는 34.8M 토큰으로 처리했다 (비율 0.6977). 못 따라온 것은
언어모델이다.

진단 시도: 새 토큰 30,000개의 중앙값 발화가 168.5MB 전체에서 143회였다.
여기서 "노출 부족" 이라고 봤는데 **Phase 4 에서 반증됐다** (아래 7번).

산출물: `artifacts/models/cpt_{c0_qwen,t2a_none,t2b_mean}_main_seed42`
(artifacts.tsv 에 등록됨).

### 7. Phase 4 — 등토큰 예산과 N 스윕 (완료, 2026-09-01)

전체 표: [`../reports/tables/phase4.md`](../reports/tables/phase4.md).

```
168.5MB Equal-Raw-Data       ko          토큰        압축
  C0                      1.137540   49,922,048      —
  T2a                     1.136817   49,922,048    +0.0%
  T2b N=30,000            1.567095   34,832,384   -30.2%
  T2b N=10,000            1.497272   37,351,424   -25.2%

등토큰 (49,922,048 토큰 = 원문 241.4MB)
  T2b N=30,000            1.532183
```

**사전 등록 예측이 두 실험 모두 맞았다** — 격차가 줄되 C0 를 넘지 못했다.
"조건부 긍정" 분기는 발동하지 않았고 N=20,000 은 돌리지 않는다.

**노출 가설은 반증됐다.** Phase 3 의 진단이 틀렸다.

```
              중앙값 발화    회복률
N=30,000         143회       65.4%
N=10,000         623회       65.9%    <- 노출 4.4배, 회복률 +0.5%p
N=30,000 등토큰  143회       68.2%    <- 데이터 43% 증가, +2.8%p
```

토큰당 노출은 거의 안 듣고 데이터 총량은 조금 듣는다. CPT 가 초기 손상의
약 65% 를 회복하고 멈추는 이유는 **아직 모른다.** `tie_word_embeddings` 가
유력한 후보지만(노름 보정·정렬이 반증된 것과 같은 구조를 가리킨다) 검증하지
않았다 — 가설로만 다뤄라.

N=10,000 의 sigma 도 쟀다: 한국어 0.001166 (N=30,000 의 0.001234 와 거의 같다).

### 8. 다음 — Q6 시스템 벤치마크

닫힌 것: Q1~Q5. 남은 것은 **Q6** 와 Level 3 능력 평가다. 압축 -25~30% 가
prefill·TTFT·KV cache 에서 얼마로 나타나는가 — **T2b 가 유일하게 이길 수 있는
축이다.**

설계와 예측 4개는 [`PLAN.md`](PLAN.md) "Q6 시스템 벤치마크" 절에 사전 등록했다.
핵심은 예측 2번이다: prefill 이 초선형이라 **토큰 30% 감소가 시간 33~42% 감소로
증폭돼야 한다.** 확인되면 "압축의 값어치는 품질이 아니라 시스템에 있다" 는
조건부 권고가 선다.

**완료 (2026-09-02).** 전체 표: [`../reports/tables/system_bench.md`](../reports/tables/system_bench.md).

```
같은 한국어 원문      토큰Δ     prefillΔ    증폭
  5,000자            -31.1%    -34.7%     1.12x
 40,000자            -30.6%    -41.3%     1.35x

KV cache  토큰과 정확히 선형 (-30.2 ~ -31.1%)
TTFT      40,000자에서 728ms -> 428ms
peak VRAM 길이와 함께 -4.5% -> -17.7%
T2a       메모리 51.8MB, prefill 시간은 C0 와 같다
```

**사전 등록 예측 4개가 전부 맞았다.** 널 대조군 둘(equal_tokens 의 T2b vs C0,
raw_prompt 의 T2a vs C0)도 통과했다 — 계측기 바닥은 약 2% 다.

**Q1~Q6 가 전부 닫혔다.** PLAN 의 종료 조건이 채워졌다.

스모크에서 **버그를 하나 잡았다.** decode 단계(seq_len=1)에서 강제한 두 융합
커널이 모두 거부해 `No available kernel` 로 죽었다 — mem_efficient 는 GQA
브로드캐스트를, cuDNN 은 길이 1 자체를 지원하지 않는다. decode 에만 MATH 를
더하도록 고쳤고 [`RULES.md`](RULES.md) 9번에 예외로 명시했다. 우회가 아니라
조건이 다르기 때문이다 — decode 의 attention 행렬은 1 x n 이라 27,300 토큰에서도
1.5MB 다.

### 9. Phase 6 — 유효 범위 (완료, 2026-09-02)

전체 표: [`../reports/tables/phase6.md`](../reports/tables/phase6.md).

```
A 스케일   1.5B 손상 배율 2.324 (0.5B 는 2.058)  — 예측 틀림. 더 나빠진다
C 기울기   임베딩 기울기가 안 죽는다 (attn 의 1.57배) — 병목은 최적화가 아니다
E 배치     동시 1.40배 (예측 적중), 처리량 1.605배 (예측 틀림 — 더 크다)
           배치는 처리량을 사지 않는다. 시퀀스 하나가 이미 GPU 를 포화시킨다
```

C 는 덤으로 **같은 시드 재현성** 을 처음 쟀다 — 편차 0.000421 로 시드 간
sigma 의 34%. sigma 기준이 과소평가가 아니었다.

**최종 보고서: [`../reports/FINAL_REPORT.md`](../reports/FINAL_REPORT.md).**

### 10. 다음 — 1차는 끝났다. 2차로 무엇을 할 것인가

PLAN 의 종료 조건("여섯 질문에 모두 답이 붙으면 프로젝트는 끝난다")이 채워졌다.
이제 남은 것은 **선택** 이고, 셋 다 새 사전 등록이 필요하다.

**A. Level 3 능력 평가** (스펙 §40). BPB 는 language modeling 이지 능력이
아니다. T2b 의 품질 손실이 실제 과제에서 얼마인지는 아직 모른다. `capability.py`
가 스텁이고 벤치마크 선정부터 해야 한다.

**B. N < 10,000.** N 곡선이 강하게 오목하다 — N=0 은 정의상 C0(1.1375)인데
N=10,000 에서 이미 1.4973 이다. 첫 1만 개가 손상의 84% 를 낸다. 압축도 같은
모양이라(첫 1만 개가 -25.2%) 그 사이 구간에 데이터가 없다. **Q6 결과가 이
질문의 값을 올렸다** — 압축이 시스템에서 값을 하는 것이 확인됐으므로,
"품질 손실을 줄이면서 압축을 얼마나 남길 수 있나" 가 실용적 물음이 됐다.

**C. 65% 벽의 정체.** CPT 가 초기 손상의 65% 만 회복하고 멈추는 이유.
`tie_word_embeddings` 를 끊고 같은 실험을 돌리면 가릴 수 있지만 통제축을
바꾸는 일이라 별도 설계가 필요하다.

**Final Test 는 아직 열지 않았다.** 열려면 별도 결정과 `final-test-opened`
태그가 필요하다 (스펙 §10).

시작 전에 `nvidia-smi` 로 GPU 여유를 확인하고 사용자에게 알린다. 장시간 run 은
`tools/watch_run.py` 로 감시를 건다.

---

**참고 — N 곡선이 강하게 오목하다.** N=0 은 정의상 C0(1.1375)
인데 N=10,000 에서 이미 1.4973 이다. 첫 1만 개가 손상의 84% 를 낸다. 압축도
같은 모양이라(첫 1만 개가 -25.2%) N<10,000 구간에 데이터가 없다. 사전 등록에
없던 질문이므로 사후에 끼워 넣지 않았다 — 하려면 새로 등록하고 시작하라.

### 하지 않아도 되는 것

도메인 라벨 재감사. 이미 3회 했고 결론이 나왔다. 규칙을 고쳤다면
`scripts/eval_domain_rules.py` 로 **추가 라벨링 없이** 채점하면 된다.

---

## 반드시 지킬 것

1. `data/final_test/` 는 읽거나 커밋하지 않는다. 마지막 1회만 개봉한다.
2. 다른 토크나이저 비교는 PPL 이 아니라 BPB 로 한다.
3. HCX/A.X 는 external reference 다. 인과 효과는 같은 Qwen backbone 에서만.
4. attention 은 `EFFICIENT_ATTENTION + CUDNN_ATTENTION` 을 강제한다.
   강제하지 않으면 8,192 토큰에서 메모리 7.1배, 시간 10.3배가 된다.
5. byte fallback 256개와 special token 은 pruning 하지 않는다.
6. 첫 CPT 는 동일 config seed 42/123/2026 으로 노이즈 플로어부터.
7. 원장은 append-only. 실패 run 도 지우지 않는다.
8. `record` 커밋에 코드·설정을 섞지 않는다.
9. 실험 전 `tools/check_clock.py --record` 를 실행한다.
10. **해시를 손으로 적지 않는다.** 원장에서 읽어온다 — 훅이 대조해서 거부한다.
11. **도메인별 수치를 라벨 정확도 없이 인용하지 않는다.** 한국어 내부는 ~55% 다.
12. 라벨링 화면으로 정확도를 잴 때는 **반드시 `--blind`**.

전체는 [`RULES.md`](RULES.md) 17개 항목이 유일한 기준이다.

---

## 알아두면 시간 아끼는 것

- python 은 항상 `C:\llm_tokenizer\.conda\python.exe` 절대경로
- 원장 시각은 UTC 다. 로컬로 보려면 `tools/ledger_tail.py`
- 원장 테이블 10종 + manifest. 컬럼을 추가했으면 `tools/migrate_ledger.py`
- CI 는 `pytest numpy pyyaml tokenizers` 만 설치한다. 새 테스트가 서드파티를
  쓰면 `.github/workflows/ci.yml` 에 추가해야 한다
- **파이썬 문자열 치환으로 코드를 수정할 때 `\n` 이스케이프가 조용히 어긋난다.**
  실제로 네 번 당했다. 치환 후 반드시 `grep` 이나 `assert` 로 적용 여부를 확인하라
- 감사 TSV 를 브라우저에서 받으면 `&` 가 `&amp;` 로 온다. `html.unescape` 후 적용
- 라벨링 화면은 `python -m http.server` 로 띄운다 (`.claude/launch.json` 의 `labeler`).
  `file://` 로 열면 localStorage 가 막히는 브라우저가 있다

---

## 아직 안 만든 것

| 파일 | 용도 |
|---|---|
| `src/tokenizer/train.py` `prune.py` `substitute.py` | T2a / T2b 학습 |
| `src/surgery/*.py` | embedding resize, E0/E1 초기화 |
| `src/training/cpt.py` `callbacks.py` | CPT 루프 |
| `src/evaluation/bpb.py` | Level 2 BPB |
| `src/evaluation/latency.py` `memory.py` | Level 4 |

`src/` 의 스텁에는 각각 스펙 절 번호가 docstring 에 적혀 있다.

---

## 세션 시작

```bash
cd C:\llm_tokenizer
git log --oneline -20
git status
C:\llm_tokenizer\.conda\python.exe tools\ledger_tail.py
C:\llm_tokenizer\.conda\python.exe -m src.utils.env --check
C:\llm_tokenizer\.conda\python.exe tools\check_clock.py --record
C:\llm_tokenizer\.conda\python.exe -m pytest tests/ -q
C:\llm_tokenizer\.conda\python.exe tools\validate_ledger.py
```

앞선 대화를 기억한다고 가정하지 말고, 커밋과 `experiments/` 원장만 현재 상태로 믿는다.
