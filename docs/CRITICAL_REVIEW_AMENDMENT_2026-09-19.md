# 외부 비판 검토와 P3 실행 amendment — 2026-09-19

> 이 문서는 2026-09-19 현재 코드·원장·사전 등록을 감사한 결과다. 과거
> [`PLAN.md`](PLAN.md)의 등록과 기존 결과를 고치지 않는다. **아직 실행하지 않은
> P3의 해석과 실행 순서만 이 문서가 대체한다.** 새 장기 GPU 실험과
> `data/final_test/` 열람은 하지 않았다.
>
> 수치 표는 [`tools/methodology_audit.py`](../tools/methodology_audit.py)가 원장에서
> 생성한 [`reports/tables/methodology_audit.md`](../reports/tables/methodology_audit.md)를
> 정본으로 삼는다. 아래 표의 정성 판정은 이 날짜의 연구 판단이다.

## 결론부터

프로젝트의 1차 결과는 폐기하지 않는다. 다만 논문의 중심은 "회복률 65%의
보편적 벽"이나 개별 tokenizer 기법의 신규성이 아니다.

> **방어 가능한 중심 기여 후보:** 한국어 소형 모델에서 vocabulary 크기를 고정한
> prune+replace가 압축·시스템 비용과 품질 사이에 만드는 교환비를, 동일 원문량과
> 동일 update/token 예산으로 나누어 측정하고 제한된 CPT에서의 회복 곡선을
> 보고한 증거.

현재 가장 큰 결손은 동일 원문량 결과의 대응 seed와 동일 compute proxy 대조다.
P3는 이 둘을 먼저 채운다. 500MB 점근선, R 불변성, "완벽한 초기화 상한"은 핵심
범위에서 뺀다.

## 1. 검토 판정표

| 지적 | 판정 | 저장소 근거 | 결론 영향 | 조치 |
|---|---|---|---|---|
| 168.5MB는 동일 compute가 아니다 | **수용** | `cpt_*_main_seed42`; C0 49,922,048토큰/1,523 update, T2b30k 34,832,384/1,063, T2b10k 37,351,424/1,139 | Equal-Raw-Data 결론은 유지, compute 결론은 철회 | 동일 update/token 대조 추가 |
| 마지막 batch 회계가 정확한가 | **수용 — 추가 발견** | [`cpt.py`](../src/training/cpt.py)의 budget break가 accumulation 중간에도 발생. C0 16,384토큰, T2b10k 28,672토큰은 gradient만 누적되고 step되지 않았지만 `tokens_seen/raw_bytes_seen`에 포함 | 원장 학습량은 한 update 미만 꼬리를 과대계상. 기존 큰 BPB 격차는 무효화하지 않지만 정밀한 예산 주장 제한 | 다음 run 전에 applied/observed counters 분리와 완전 update 경계 중단 |
| R은 나쁜 B0를 보상할 수 있다 | **수용** | `R=(B0-Bf)/(B0-Cf)` 자체의 성질, [`RULES.md` 14b](RULES.md) | R 불변성·기전·초기화 우열 주장을 철회 | B0/Bf/Cf와 `Bf-Cf` 우선, R은 보조 |
| BPB가 같은 원문 위치를 채점하지 않을 수 있다 | **부분 수용** | [`bpb.py`](../src/evaluation/bpb.py): 같은 문서를 읽지만 2,048 token 비중첩 창의 첫 token을 제외 | 고정 token-context 배포 비교로는 유효. 동일 byte-context likelihood는 미측정 | scored byte와 경계 손실 진단, 필요 시 dev 보조평가 |
| 큰 토큰이면 token PPL이 자동으로 유리하다는 설명 | **수용** | `log2(PPL_token)=BPB/(tokens/byte)` | 수치가 아니라 예측 단위가 달라 직접 비교 불가라고 정정 | RULES·evaluator·개요 수정 |
| R1이 치환 특유 현상을 입증했다 | **수용** | `cpt_dmg_mean_k40_r1_seed42` vs `cpt_kot2b_v2_n30000_mean_r1base_seed42`; 행 수·노출·토큰화가 다름 | 인과 귀속 철회 | "시험한 aggregate-damage matched 구성의 궤적이 다름"으로 축소 |
| 노출과 손상량을 동시에 맞출 수 없다 | **부분 수용** | `reports/tables/p3_gates.md`, 시험한 rank-band와 K<=30,000에서 실패 | 일반적 불가능성은 아님 | 탐색 범위를 문장에 붙임 |
| Q7이 tying을 일반적으로 배제하지 못한다 | **수용** | 17.5MB×3 seed, +136,134,656 params(+27.6%), input/output gradient 분리 | "원인이 아니다" 철회 | "해당 예산·설정에서 5%p급 효과 없음"으로 축소 |
| Q7 forward 동일성은 학습 동역학 동일성이 아니다 | **수용** | `SPEC_P2.md` §9, untie 후 별도 `lm_head` gradient | 장기 외삽 금지 | 파라미터·gradient 차이를 한계에 명시 |
| 상수 LR 개선은 인정하되 65% 벽은 식별하지 않았다 | **수용** | R5 curve는 종점에서도 하강, cosine 최종 LR 약 1e-6, constant 1e-5 | "벽" 철회. 유한 168.5MB 종점 개선만 주장 | 절대 gap 0.429555→0.345543, −0.084012 BPB 보고 |
| 누적 LR을 비교할 수 있는가 | **판단 불가** | peak LR·warmup·마지막 LR은 기록, historical per-step 누적 LR은 미기록 | 누적 LR 수치를 만들지 않음 | 향후 scheduler가 step별 LR/누적 LR 기록 |
| 2σ는 95% CI가 아니다 | **수용** | [`noise_floor.py`](../tools/noise_floor.py): n=3 seed final BPB의 표본 SD | 과거 등록 판정은 보존하되 통계적 유의/동등성 언어 제한 | RULES 14c 추가 |
| T2a KMMLU CI가 학습 seed 불확실성도 나타낸다 | **기각** | `hanja_probe.py`는 같은 문항의 paired bootstrap, 고정 checkpoint | 문항 표본 불확실성만 나타냄 | seed 불확실성과 분리 보고 |
| P3-E 이식은 완벽한 초기화 상한/ZeTT 대체다 | **수용** | `transplant_rows.py`는 CPT 중 몸통과 상호적응한 행만 원본 몸통에 이식 | 상한·ZeTT 대체 주장 철회 | compatibility stress test로만 재정의, 핵심에서 제외 |
| 신규 토큰 종류 수가 곧 충분한 학습 노출이다 | **수용** | N30k: 20MB 표본 0회 11.8%, 환산 <100회 39.7%, 중앙값 143; N10k 중앙값 623 | N 선택 해석 수정 | 노출 분포를 필수 공변량으로 보고 |
| split 사이 near-duplicate가 남았고 tokenizer가 dev를 학습했다 | **기각/확인** | `run_data_pipeline.py`는 split 전에 전역 exact/near dedup; `analyze_vocab.py`, `train.py` 기본·실행은 train split | 현재 확인 범위에서는 이 경로의 누수 없음 | 파이프라인 테스트 유지, 해시 audit만 추가 |
| Common Crawl 계열이면 C0에 유리한 중복이다 | **수용** | 실제 Qwen pretraining 문서와 overlap을 측정하지 않음 | 편향 방향 주장 철회 | "중복률·방향 미상"으로 보고 |
| KMMLU contamination 검사가 충분하다 | **수용** | `contamination_check.py`는 우리 CPT pool의 13-gram overlap만 탐지 | Qwen 사전학습, 의역, 짧은 overlap은 미검사 | 탐지 범위를 결과 옆에 표기 |
| 한자 밀집도가 제거 토큰 영향량이다 | **수용** | D2 subset은 문자 한자 밀도 기준 | causal exposure로 읽지 않음 | 원본 tokenizer의 제거 대상 token count로 dev 층화 |
| T2a 품질 유지가 지식 손상 부재를 뜻한다 | **수용** | vocab 제거는 softmax 경쟁 항목도 줄여 재정규화 이득 가능 | "무손상" 철회 | "시험한 aggregate 지표에서 순효과 미검출"로 축소 |
| E1=FVT가 전 토큰에 일반적으로 성립한다 | **부분 수용** | `fvt_check.py`: standalone 재토큰화 가능한 25,821개만 100%; 4,179개는 정의 불가 | 검증 범위 밖 동등성 철회 | 범위를 항상 병기 |
| cosine similarity/distance를 수식에서 혼용했다 | **기각(구현), 부분 수용(용어)** | `fvt_extension.py` 계산은 cosine similarity로 일관. 일부 prose가 distance를 일반어로 사용 | 수치 오류는 찾지 못함 | 수식 주변 용어를 similarity로 통일 |
| 시스템 수치가 같은 품질 비교다 | **수용** | `raw_prompt`는 같은 원문/다른 토큰/품질 불일치, `equal_tokens`는 같은 토큰/다른 원문 | 품질-매칭 서비스 개선으로 읽지 않음 | 각 표에 통제축 태그 |
| 영어·코드 압축 유지가 품질 보존이다 | **수용** | `tokenizer_metrics.tsv`와 CPT BPB는 다른 측정 | 압축과 품질 결론 분리 | 양쪽을 별도 표로 유지 |

## 2. 학습·평가 구현 감사

### Equal-Raw-Data가 실제로 통제하는 것

- 같은 50,000개 train 문서 풀을 seed로 섞고, tokenizer별 정확한 token-byte 합으로
  약 168.5MB 지점까지 진행한다.
- 기본은 `seq_len=2048`, `micro_bs=2`, `accum=8`, bf16, gradient checkpointing,
  AdamW8bit, weight decay 0.1, clip norm 1.0이다.
- Hugging Face causal LM loss의 token-position mean을 microbatch마다 `accum`으로
  나누어 backward한다. 별도 byte weighting은 없다.
- cosine 진행률과 중단 기준은 main run에서 raw bytes다. equal-token run은 token
  진행률이다.
- packing은 문서를 EOS로 이어 fixed-token chunk를 만들며 마지막 `seq_len` 미만
  잔여는 버린다. 예산을 넘긴 마지막 chunk 묶음을 잘라 정확히 맞추지는 않는다.
- 기존 코드에서는 accumulation 중간에 budget을 넘으면 남은 gradient를 step하지
  않고 종료하면서 그 token/byte를 원장에는 포함했다. 정본 수치는 생성 표 §1이다.

따라서 168.5MB 비교가 답하는 것은 **같은 train 원문량을 각 tokenizer가 얼마나
효율적으로 흡수하고 어떤 최종 품질에 도달하는가**다. 같은 optimizer update,
같은 token 수, 같은 FLOPs를 답하지 않는다.

### 동일 compute 대조가 가르는 것

고정할 proxy는 `seq_len=2048`, micro/effective batch, optimizer, scheduler, **완전한
optimizer update 수 1,523**이다. 이는 같은 49,905,664 applied tokens를 만들지만
vocab softmax 비용·실제 FLOPs까지 완전히 같다는 뜻은 아니다. profiler FLOPs를
기록하기 전에는 **compute-matched proxy**라고 부른다.

T2b는 같은 update를 채우기 위해 더 많은 원문을 본다. 70,000문서 풀에서 먼저
without-replacement로 진행하고, 부족할 때만 같은 고정 순서로 반복하며 unique bytes와
repeated bytes를 따로 기록한다. 이 비교는 "Equal-Raw-Data에서 T2b가 update를 덜 받은
효과"와 "같은 update에서도 남는 tokenizer/initialization 격차"를 나눈다. 같은 정보량
비교가 아니므로 raw bytes도 함께 보고한다.

### BPB가 답하는 질문

현재 evaluator는 같은 JSONL 문서를 같은 순서로 읽고 원문 byte budget을 넘긴 마지막
문서까지 포함한다. 문서마다 별도 tokenization하며 BOS/EOS/padding과 문서 간 문맥은
없다. 2,048 token 비중첩 창마다 첫 token은 NLL과 byte 분모에서 함께 제외한다.
최종값은 문서 BPB 평균이 아니라 `총 NLL / 총 scored bytes`다.

따라서 동일 문서 집합이지만 tokenizer별 창 경계와 제외 byte 위치가 다르다. 이는
**고정 token context에서 실제 배포될 모델의 likelihood** 질문에는 적합하다. 같은
원문 byte prefix에 정확히 같은 문맥을 주는 평가는 별도 진단이다. round-trip은
tokenizer 생성 시 검사되며 evaluator가 매번 재검증하지는 않는다.

## 3. 주장 수정표

| 기존 문장/요지 | 문제 | 방어 가능한 대체 문장 |
|---|---|---|
| "168.5MB 같은 예산"이 곧 같은 compute | 토큰·update가 다름 | "동일 원문량 비교이며 compute는 다르다." |
| "큰 토큰이면 token PPL이 자동으로 유리" | 일반적으로 성립하지 않는 방향성 설명 | "예측 단위가 달라 직접 비교할 수 없고 `log2(PPL)=BPB/(tokens/byte)`다." |
| "R이 비슷하므로 공통 회복 메커니즘" | R은 B0/Cf에 민감 | "R은 등록된 기술 지표이며 절대 `Bf-Cf`가 우선이다." |
| "R1은 치환 특유 현상을 입증" | 행 수·노출·토큰화 교락 | "시험한 두 aggregate-damage matched 구성의 회복 궤적이 달랐다." |
| "tie가 원인이 아니다" | 17.5MB와 특정 optimizer/config에 한정 | "해당 설정에서 untie 효과의 paired 평균은 +0.444%p이고 실용 경계 5%p보다 작았다." |
| "65% 벽은 cosine의 산물" | 벽 자체를 유한 종점으로 추정 | "constant LR은 168.5MB residual gap을 0.084012 BPB 줄였다; 점근선은 미식별이다." |
| "P3-A 500MB로 포화를 판정" | 유한 범위로 무한 한계 판정 불가 | "500MB는 고정 조건의 곡선 연장과 구간별 한계 개선량만 잰다." |
| "CPT 행 이식은 완벽한 초기화 상한" | 몸통-행 co-adaptation 파괴 | "다른 몸통에 대한 CPT 행의 compatibility stress test다." |
| "T2a는 품질 손실 없이 지식을 보존" | softmax 재정규화가 손실을 상쇄할 수 있음 | "시험한 BPB·KMMLU aggregate 순효과에서 저하를 검출하지 못했다." |
| "E1은 FVT와 동등" | 13.9%는 standalone UTF-8 surface가 없어 FVT 미정의 | "FVT가 정의되는 25,821개에서 일치하며 나머지는 merge-genealogy 확장이다." |
| "Common Crawl 중복이 C0에 유리" | 실제 overlap·방향 미측정 | "중복 가능성은 있으나 크기와 방향은 알 수 없다." |
| "같은 문맥 창에 44% 더 많은 글" | 평균 환산이며 품질 미매칭 | "해당 dev 원문의 평균 tokenization rate에서 같은 token limit에 44% 더 많은 문자를 담는 환산값이다." |
| "영어·코드 regression 없음" | compression과 CPT quality는 별개 | "compression 변화와 영어·코드 BPB 변화를 각각 보고한다." |

## 4. 통계 amendment

- `σ_BPB`는 17.5MB 동일 config의 seed 3개 final BPB 표본 SD다. paired-difference
  SD도 평균의 SE도 아니다. `2σ`는 등록된 결정 휴리스틱이지 95% CI가 아니다.
- Q7은 같은 seed의 tied/untied를 pair로 다시 계산하면 평균 `+0.444%p`, paired
  SD `0.011%p`, 탐색적 t(df=2) 95% CI `[+0.417,+0.471]%p`다. margin ±5%p
  안이지만 n=3 정규성 가정의 사후 CI이므로 장기 예산의 형식적 동등성으로
  일반화하지 않는다.
- T2a KMMLU의 `−0.21%p`, paired item-bootstrap 95% CI
  `[-1.05,+0.58]%p`는 등록 margin ±2%p 안이다. 이는 **그 checkpoint와 1,900문항의
  비열등/동등 판정**이지 training-seed 불확실성이나 지식 손상 부재 증명이 아니다.
- BPB에는 현재 document-paired bootstrap이 없다. 추가한다면 문서별 NLL과 scored
  bytes를 pair로 resample하고 각 bootstrap에서 `sum(NLL)/sum(bytes)`를 다시 계산한다.
- checkpoint 곡선과 P3 gate를 본 뒤 만든 설명은 탐색적이다. 등록된 endpoint 판정과
  분리하고 다중 비교 보정을 사후에 꾸미지 않는다.

## 5. 선행 연구 대비 위치

| 선행 연구 | 이미 알려진 것 | KoTokenLab에서 남는 범위 |
|---|---|---|
| [Purason et al.](https://arxiv.org/abs/2512.03989) | continued BPE와 leaf-based pruning을 이용한 controlled vocabulary modification | leaf pruning/continued merge 자체의 신규성 없음. 고정 크기 prune+replace의 한국어 저예산 trade-off |
| [Dagan et al.](https://arxiv.org/abs/2402.01035) | tokenizer 크기·regex·데이터가 속도, 문맥, 메모리, downstream에 영향; 대규모 specialization | 50B+가 아닌 0.5B 모델·수천만 token 규모의 미회복 구간 |
| [Dobler & de Melo](https://arxiv.org/abs/2408.15793) | tight-budget tokenizer swapping은 효율적이며 품질 향상은 언어별로 보장되지 않음 | 한국어·고정 vocab에서 동일 원문/동일 update 축 분리 |
| [EEVE](https://arxiv.org/abs/2402.14714) | 한국어 vocab expansion, parameter freezing, subword initialization, 약 2B token 적응 | expansion이 아닌 크기 보존 치환과 훨씬 작은 CPT 예산 |
| [ZeTT](https://arxiv.org/abs/2405.07883) | hypernetwork로 임의 tokenizer embedding을 zero-shot 예측, <1B token으로 잔차 회복 | ZeTT를 실행하지 않았으므로 초기화 SOTA 주장 없음 |
| [FOCUS](https://aclanthology.org/2023.emnlp-main.829/) | overlap token과 auxiliary semantic space를 이용한 sparsemax 초기화 | E2는 frequency weighting이라 FOCUS가 아님; 직접 우월성 주장 없음 |
| [Hägele et al.](https://arxiv.org/abs/2405.18392) | constant LR+cooldown으로 reusable scaling trajectory | R5의 scheduler 민감도는 재현 사례이지 scheduler 기법의 신규성 아님 |
| [Smith et al.](https://arxiv.org/abs/2607.15232) | continued merges, source-subtoken mean, embedding-only→full-model 2단계 적응 | 신규 행 warm-start baseline을 비교할 근거; 평균 초기화 자체 신규성 없음 |

## 6. 최소 추가 실험표

| 구분할 대안 설명 | 처치군 / 통제군 | 예산·평가 | seed·불확실성 | 실행·중단 기준 | 결과와 무관하게 보고 |
|---|---|---|---|---|---|
| Equal-Raw-Data 핵심 결과가 seed42 우연인가 | N30k constant T2b 기존 3 seed / constant C0 seed123·2026 추가, seed42 재사용 | 168.5MB raw, 동일 pool/order rule; B0/Bf/Cf와 gap | paired seed 3; mean gap, SD, paired t CI는 기술적 | accounting fix와 config diff 통과 전 실행 금지 | 실제 bytes/tokens/updates, tail, wall/VRAM |
| 원문량 결과의 격차가 적은 update 때문인가 | N30k T2b constant 1,523 update / 위 C0 constant 1,523 update | 49,905,664 applied tokens; dev BPB, raw bytes와 unique/repeat bytes | 3 seed; paired gap | seed42에서 NaN/OOM/config 불일치면 중단, 효과 방향으로 seed 수를 줄이지 않음 | FLOP equality가 아닌 proxy임을 명시 |
| 직접 CPT보다 신규 행 warm-start가 나은가 | 첫 33.7MB는 신규 행만 gradient 허용, 뒤 134.8MB full CPT / 168.5MB direct full CPT | 총 raw 168.5MB에 준비 비용 포함; 절대 BPB·gap 우선 | seed42 gate, gap 개선이 prereg practical floor를 넘을 때만 2 seed 추가 | tied matrix의 old-row gradient가 0인지 test; budget 밖 prep 금지 | stage별 bytes/tokens/updates, old/new row update norm |
| WSD가 constant endpoint를 개선하는가 | WSD C0/T2b / constant C0/T2b | 총 168.5MB 안에서 마지막 20%=33.7MB cooldown; 추가 데이터로 숨기지 않음 | seed42 쌍; practical floor 통과 시 T2b만이 아니라 C0도 대응 seed | 앞 실험 지연 시 후순위 | peak·마지막·누적 LR, cooldown 비용, 절대 gap |
| 한자 결과가 제거-token exposure와 관련 있는가 | dev 문서를 원본 tokenizer의 제거 대상 token count 0/low/high로 층화 | GPU 0; 같은 checkpoints, 문서별 NLL/bytes 재집계 | document paired bootstrap, exploratory | bin당 문서가 사전 최소수 미달이면 수치 비교 안 함 | 문자 한자 밀도와 별도 축임을 명시 |

## 7. P3 A~G 재분류

| P3 | 판정 | 이유와 새 정의 |
|---|---|---|
| **A 500MB** | **후순위** | 점근선/포화 검정은 취소. 핵심 대조가 끝난 뒤에만 유한 168.5→500MB 곡선 연장과 구간별 개선량으로 실행 |
| **B WSD** | **수정·유지** | 실용 scheduler 질문은 유효. cooldown 33.7MB를 총예산 안의 비용으로 세고 절대 gap 우선 |
| **C R 불변성** | **취소** | R 자체가 B0/Cf에 민감해 기전 질문에 부적합. N=10k는 기존 절대 gap 기술로 충분 |
| **D R1 분해** | **후순위** | permute는 손상 종류만 가르지만 핵심 논문 결론에 직접 필요하지 않음. "치환 특유" 증명으로 쓰지 않음 |
| **E 이식 상한** | **취소/재정의** | 상한·ZeTT 대체는 취소. 나중에 compatibility stress test로만 가능. 대신 신규 행 warm-start가 우선 |
| **F KMMLU 39과목** | **후순위** | 0.5B 계기의 감도와 contamination 범위가 제한적. 핵심 seed/compute 대조 후, Final Test 규칙 동결 전에 결정 |
| **G 1.5B** | **후순위** | VRAM 탐침은 통과했지만 두 모델 크기보다 통제축·seed가 먼저. 7주 핵심 범위 밖 |

## 8. 수정된 7주 일정

원장 wall time은 T2b constant 168.5MB 약 1.20h, C0 constant 약 1.62h다.
데스크톱 GPU 공유로 wall time이 오염된 구간이 있어 아래는 **±30% 운영 범위**로
잡는다. 학습은 동시에 하나만 실행한다.

| 주 | 작업 | GPU 예상 | 완료 기준 |
|---|---|---:|---|
| **1** | budget tail 회계 수정·테스트, BPB scored-byte 진단, split/near-dup/train-only audit, amendment 동결 | 0h | 테스트·ledger validator 통과, 새 run config freeze |
| **2** | N30k constant C0 seed123·2026; 기존 T2b 3 seed와 paired 핵심 재현 | 3.2h ±30% | B0/Bf/Cf/gap·tokens·updates 자동 표 |
| **3** | N30k 동일 1,523-update T2b constant 3 seed | 5.3h ±30% | raw/unique/repeat bytes 포함 compute-proxy 표 |
| **4** | 신규 행 20% warm-start seed42; gate 통과 시 seed123·2026 | 1.2h, 조건부 +2.4h | 준비 비용 포함 direct-CPT 비교 |
| **5** | 문서-paired BPB 진단, 제거-token exposure 층화, contamination·system 축 문구 확정 | 0h | exploratory/confirmatory 구분과 Final Test 선택 규칙 동결 |
| **6** | 여유가 있으면 WSD C0/T2b seed42, 아니면 분석·집필 | 2.8h ±30% | cooldown 포함 절대 gap; 미실행도 계획 실패가 아님 |
| **7** | 표 재생성, 반증·한계·related work 집필, Final Test 개봉 여부를 별도 결정 | 0h | core claim 표와 미확인 목록 완료 |

기본 GPU 합계는 약 9.7h, warm-start 반복과 WSD까지 모두 발동하면 약 14.9h다.
500MB, P3-D/F/G는 이 7주 계획의 성공 조건이 아니다.

## 9. Final Test 계획

Week 5가 끝날 때 다음을 커밋으로 고정한 뒤에만 별도 개봉 결정을 한다.

1. 대표 조건: C0 constant, T2b direct constant, gate를 통과한 adaptation 하나
2. checkpoint 선택 규칙: dev 절대 BPB와 사전 지정 budget endpoint
3. 1차 outcome: 한국어 total NLL/total scored bytes BPB와 residual gap
4. 부차 outcome: 영어·코드 BPB, compression, system raw-prompt 지표
5. 문서 bootstrap과 seed 불확실성을 별도 보고
6. `final-test-opened` 태그 이후 조건·margin·checkpoint 변경 금지

개봉은 이 amendment가 자동 승인하지 않는다. 사용자가 별도로 결정한다.

## 10. 실제 변경과 남은 미확인

이번 감사에서 바꾼다:

- [`RULES.md`](RULES.md): token PPL 관계, R 보고 우선순위, 2σ 의미
- [`bpb.py`](../src/evaluation/bpb.py): 평가 단위·경계·round-trip 한계 docstring
- [`overview.py`](../tools/overview.py): 과도한 R1/Q7/R5/P3/novelty 문구와 CC 방향
- [`SPEC_P3.md`](SPEC_P3.md), [`SCHEDULE_P3.md`](SCHEDULE_P3.md),
  [`PLAN.md`](PLAN.md): 원 등록 보존 + 이 amendment가 향후 실행을 대체한다는 표지
- [`FINAL_REPORT.md`](../reports/FINAL_REPORT.md): 1차 결과를 소급 변경하지 않고
  해석 정정 링크
- [`methodology_audit.py`](../tools/methodology_audit.py): 수치 재계산 경로

남아 있는 미확인:

- historical run에서 optimizer update에 실제 반영된 raw bytes의 정확한 값
- Qwen 사전학습과 CPT 문서의 실제 overlap 및 편향 방향
- historical cosine/constant의 정확한 누적 LR
- 동일 byte-context BPB와 document-paired BPB CI
- 168.5MB에서 C0의 seed 분산과 paired gap 분산
- 품질을 맞춘 서비스 수준 TTFT/throughput
- 1.5B, 500MB, 정식 ZeTT/FOCUS 비교

이 항목은 확인 전까지 수치나 결론을 만들지 않는다.
