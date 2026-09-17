# KoTokenLab

**Korean Tokenizer Surgery & LLM Adaptation on a Single RTX 5070 Ti 16GB**

> **1차 종료** (2026-08-29 → 09-02, 태그 `p1-closed`). 사전 등록 질문 Q1~Q6 와
> 유효 범위 Phase 6 이 모두 닫혔다. **결과 전체:
> [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)**
>
> **2차 종료** (2026-09-17). R1·Q7 부정, R5 긍정 — **"65% 회복 정체" 는 코사인
> LR 스케줄의 산물이었다** (상수 LR 에서 72.42%). 한자 프로브 D1~D3 은 T2a 가
> C0 와 구별되지 않았다. 설계 [`docs/SPEC_P2.md`](docs/SPEC_P2.md) ·
> 일정과 결과 [`docs/SCHEDULE_P2.md`](docs/SCHEDULE_P2.md)
>
> **3차 설계 완료, 실행 전** (2026-09-17). [`docs/SPEC_P3.md`](docs/SPEC_P3.md) ·
> [`docs/SCHEDULE_P3.md`](docs/SCHEDULE_P3.md) · 사전 등록 [`docs/PLAN.md`](docs/PLAN.md) "P3 확장"

```
Qwen2.5-0.5B
        ↓
Vocabulary Surgery
        ↓
Continued Pretraining
        ↓
Ko-Qwen
```

> Embedding Alignment 단계는 스펙에 있었으나 **폐기했다.** 손해를 안 내면서
> 제 일을 하는 lr 이 없다는 것을 3라운드 탐침으로 확인했다
> ([`docs/DESIGN_DELTA.md`](docs/DESIGN_DELTA.md) 1-5).

한국어 특화 토크나이저가 LLM 의 언어 모델링 품질, 학습 효율, 시퀀스 길이,
Attention 연산량, KV Cache, 추론 지연에 미치는 영향을 **통제된 실험**으로 분석한다.

"한국어 성능이 올라갔다"를 보이는 것이 목표가 아니다.
`Tokenizer → Embedding → Transformer → Training → Evaluation → GPU Systems` 를
하나의 재현 가능한 파이프라인으로 연결하는 것이 목표다.

전체 연구 설계: [`docs/SPEC_KoTokenLab.md`](docs/SPEC_KoTokenLab.md)

---

## 결과

**1차 종료 (2026-09-02) — 압축과 추론 효율은 성공, 언어모델 품질은 악화.**

두 결과를 함께 읽어야 한다. 하나만 떼면 프로젝트를 잘못 이해하게 된다.

Qwen2.5-0.5B 에 T2b(크기 보존 치환, n=30,000) 수술을 하고 세 조건에 **같은
원문 168.5MB** 로 통제된 CPT 를 돌렸다.

**① 성공한 축 — 압축과 추론 효율**

| 지표 | 값 | 출처 |
|---|---:|---|
| 한국어 토큰 수 | **−30.2%** | [t2_sweep.md](reports/tables/t2_sweep.md) |
| 같은 원문의 토큰 | 49.9M → 34.8M (비 0.6977) | [cpt_main.md](reports/tables/cpt_main.md) |
| prefill (5,000~40,000자 전 구간) | **−34.7% ~ −41.3%** | [system_bench.md](reports/tables/system_bench.md) |
| └ 40,000자 | 729.5 → 428.5ms | 같음 |
| └ 5,000자 | 51.4 → 33.6ms | 같음 |
| TTFT (40,000자) | **728 → 428ms** | 같음 |
| KV cache | **−30%** (327.3 → 227.1MB) | 같음 |
| peak VRAM (40,000자) | −17.7% (2,264 → 1,864MB) | 같음 |
| 같은 문맥 창의 원문량 | **+44.1%** (환산값) | 같음 |
| 동시 처리 배치 | 20 → **28** (1.40배) | [phase6.md](reports/tables/phase6.md) |
| 처리량 | 3.85 → **6.18 seq/s** (1.605배) | 같음 |

**② 악화한 축 — 언어모델 품질**

| 지표 | 값 | 출처 |
|---|---:|---|
| 한국어 BPB | **+37.8%** (1.5671 vs C0 1.1375, 348σ) | [cpt_main.md](reports/tables/cpt_main.md) |
| 영어 | +0.98% | 같음 |
| 코드 | +1.84% | 같음 |
| 등토큰 조건에서도 (C0 와 같은 토큰수) | +34.7% (1.5322) | [phase4.md](reports/tables/phase4.md) |

> **이 프로젝트는 한국어 품질 개선에 실패했다.** 압축이 잘 됐다는 사실이
> 품질 개선을 뜻하지 않는다. 두 표를 함께 보지 않으면 결론이 뒤집힌다.

<details><summary><b>시스템 수치의 측정 조건과 한계</b> — 인용 전에 반드시 본다</summary>

```
하드웨어    단일 RTX 5070 Ti 16GB (sm_120) · torch 2.7.1+cu128 · Windows
            증폭 계수는 이 GPU 의 연산·대역폭 비율에 딸려 있다
실행 횟수   prefill  warm-up 20 · 측정 100회
            생성 경로 warm-up  5 · 측정  20회
배치 실험   warm-up 5 · 측정 20회 · 시작 여유 VRAM 14,045MB (두 조건 동일)
입력 분포   dev 문서를 이어 붙인 한국어. 실제 대화·지시문의 길이 분포와 다르다
실행 순서   C0 -> T2a -> T2b 고정. 무작위화하지 않았다
            16,384 토큰에서 C0 가 2% 튀었으나 단독 재측정에서 재현되지 않았다
배치 크기   1 (배치 실험 제외). 단일 요청 기준이다
decode      **미검증.** total_ms_std/mean 이 5.7~17.5% 이고 배치별로 단조롭지
            않다. decode_tok_s 는 인용하지 않는다
문맥 용량   +44.1% 는 raw_prompt 실측률에서 **환산** 한 값이다
```

계측기는 **널 대조군 둘** 로 검증했다 — 차이가 나면 안 되는 자리에서
0.1~0.5%. 그래서 2% 미만의 차이는 이 표에서 주장하지 않는다.

수치 하나하나의 run_id · 설정 해시 · 로그 경로는
[`docs/results_provenance.md`](docs/results_provenance.md).
</details>

**토크나이저 수술은 설계대로 작동했다. 못 따라온 것은 언어모델이다.**
T2b 는 수술 직후 BPB 2.3803 에서 출발해 1.5671 까지 왔지만 C0 는 1.1375 다 —
메워야 할 1.2428 중 65.4% 를 메우고 **0.4296 을 남겼다.**

**병목이 무엇인지는 아직 모른다.** 처음에는 노출 부족으로 봤다 — 새 토큰
30,000개의 중앙값 발화가 168.5MB 전체에서 143회뿐이었다. 그래서 새 토큰을
1/3 로 줄여(N=10,000) 노출을 4.4배(중앙값 623회)로 올려 봤는데 **회복률이
65.4% -> 65.9% 로 거의 그대로였다.** **단순 노출량 부족 가설은 지지되지
않았다**
([`reports/tables/phase4.md`](reports/tables/phase4.md)).

**T2b 30k 고정 원문 조건 대비** 원문·토큰을 각 43.3% 늘려 C0 와 토큰수를
맞춘 등토큰 조건에서는 회복률이 68.2% 로 오른다. 즉 예산에는 반응하지만
기울기가 완만하다. CPT 가 초기 손상의 약 65% 를 회복하고 멈추는 이유는
아직 가리지 못했다. `tie_word_embeddings` 가 유력한 후보다 — 각 행이 입력
표현이자 출력 로짓 방향이라 한 벡터가 두 역할을 동시에 해내야 한다. 노름
보정과 정렬이 반증된 것도 같은 구조를 가리킨다. **가설이고 아직 검증하지
않았다.** 초기화 문제는 아니다 (E0/E1/E2 에서 부품 평균이 이미 최선이었다).

한편 **T2a(제거만, 크기 축소)는 거의 공짜다.** embedding 27.1M(5.5%)을 줄이고
한국어 -0.064%, 영어 구별 불가, 코드 +0.26% 다. "수술을 받았는가" 가 아니라
**"수술로 손상됐는가"** 가 결과를 가른다.

아직 열려 있는 문 두 개 — 같은 **토큰수** 를 주는 조건(스펙 §32~33)과 더 작은
N. [`scripts/run_phase4.sh`](scripts/run_phase4.sh) 가 둘 다 잰다. 예측은
[`docs/PLAN.md`](docs/PLAN.md) 에 미리 박아 뒀다.

### 토크나이저만 놓고 본 참조점 (Level 1, `phase1-tokenizer-freeze`)

```
한국어  Qwen 0.6834 tok/char   HCX -24.9%   A.X -39.4%   T2b v2 -30.2%
영어    Qwen 기준               HCX  +0.0%   A.X  +7.3%   T2b v2  -0.0%
코드    Qwen 기준               HCX +17.6%   A.X +22.5%   T2b v2  -0.8%
```

한국어 압축을 얻는 만큼 코드에서 잃는 것이 보통인데 T2b v2 는 코드 손실이
거의 없다. 도메인 라벨 신뢰 범위는
[`docs/DOMAIN_LABELS.md`](docs/DOMAIN_LABELS.md).

### 유효 범위 (Phase 6) — 1차에서 **관측한** 것

전부 0.5B 와 1.5B 에서 직접 잰 값이다. 결과표: [phase6.md](reports/tables/phase6.md)

```
스케일   1.5B 에서 손상 배율이 2.058 -> 2.324배로 오히려 커졌다.
         수술 후 상태는 스케일 무관인데(-3.1%) 기준선만 좋아진다(-14.2%) —
         큰 모델일수록 잃을 것이 많다.
         **Pre-CPT 손상만 쟀다. 1.5B 의 CPT 회복력은 측정하지 않았다.**
기울기   임베딩 기울기가 끝까지 죽지 않는다 (attn 의 1.57배).
         BPB 가 멎은 뒤에도 기울기 신호는 남아 있다는 관측이다.
         무엇이 병목인지는 이것만으로 정해지지 않는다.
배치     동시 처리 1.40배, 처리량 1.605배. 배치를 20~28배 늘려도
         처리량은 +1~2% 다 — 시퀀스 하나가 이미 GPU 를 포화시킨다.
```

### 7B 이상 — **측정하지 않았다. 구조 추론만 있다**

관측과 추론을 섞지 않기 위해 절을 나눈다. 아래는 `experiments/models.tsv` 의
config 값이지 우리가 잰 성능이 아니다.

| | tie_word_embeddings | 임베딩 비중 | 이 프로젝트에서 |
|---|---|---:|---|
| Qwen2.5-0.5B | 1 | 27.6% | **측정함** (품질 · 시스템 · 기울기) |
| Qwen2.5-1.5B | 1 | 15.1% | **측정함** (Pre-CPT 손상 · 추론만) |
| A.X-4.0-Light (7B) | **0** | 5.1% | **미측정** — 16GB 에 안 들어간다 |
| A.X-4.0 (71B) | **0** | 1.2% | **미측정** |

**7B 급은 tie 가 끊겨 있다.** 우리 기전 가설이 tie 구조에 기대므로, 7B 에서
같은 결과가 나올지는 **알 수 없다.** 좋아질 수도 나빠질 수도 있다.

그래서 7B 로 바로 가면 스케일과 tie 를 동시에 바꾸는 셈이 되어 **차이가 나와도
원인을 가릴 수 없다.** 0.5B 에서 tie 만 끊는 실험이 먼저다 — 2차의 R4.

> 위 표의 손상·회복 추세를 7B 로 외삽하지 마라. 이 저장소에 그 근거는 없다.

### 2차에서 **답할 계획인** 것 (아직 안 돌렸다)

| | 질문 | 상태 |
|---|---|---|
| R1 | 토크나이저를 안 바꾸고 임베딩 행만 망가뜨려도 65% 가 나오는가 | **부정** — 72.50% vs 치환 50.79% (record `2e97bd4`) |
| R2 | 손상 종류에 따라 다른가 | R1 부정으로 전제 상실 -> P3-D 로 재설계 |
| R3 | 손상 규모 K 에 따라 다른가 | 위와 같다 |
| R4 | **tie 를 끊으면 달라지는가** | **구별 불가** — ΔR +0.44%p, 효과 크기 바닥 5%p 미달 (record `36c8b5f`) |
| R5 | 상수 LR 이면 65% 를 넘는가 | **긍정** — 72.42%, sigma_R 0.026%p (record `64556e2`) |
| D1~D3 | T2a 의 한자 토큰 제거가 흔적을 남기는가 | **아니다** — 세 측정 모두 예측 적중 (record `855c07b`) |

설계와 예측은 [`docs/SPEC_P2.md`](docs/SPEC_P2.md), 일정과 결과는
[`docs/SCHEDULE_P2.md`](docs/SCHEDULE_P2.md), 판정표는 `reports/tables/`.

**1차의 절대값은 전부 코사인 스케줄 조건부로 읽는다** — C0 도 상수 LR 에서 더 나았다.
3차는 이어지는 질문(상수 LR 포화 · 불변성 · R1 격차 · 과제 비용 · WSD 처방)을 묻는다 —
[`docs/SPEC_P3.md`](docs/SPEC_P3.md).

### 세 번 가설을 세웠고 세 번 반증했다

| 가설 | 결과 |
|---|---|
| 노름 보정이 초기화를 공정하게 만든다 | 한국어 2.3803 → **3.0017**. 영어까지 함께 나빠졌다 |
| Embedding Alignment 로 예열한다 | 손해 없이 제 일을 하는 lr 이 **없다.** 단계를 폐기 |
| 노출이 부족해서 못 배웠다 | 노출 4.4배에도 회복률 65.4% → **65.9%** |

셋 다 우리가 세운 가설이고 우리가 무너뜨렸다. 셋 다 `tie_word_embeddings`
구조를 가리키는데, **아직 검증하지 않았다** — 그것이 2차의 본안이다.

> The final test set was never used for tokenizer design, hyperparameter tuning,
> model selection, or checkpoint selection.

---

## 시작하기

```bash
git clone <this repo> C:/llm_tokenizer
cd C:/llm_tokenizer
git config core.hooksPath .githooks          # 훅 연결 — 클론 직후 반드시
```

환경 구성은 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). 요약:

```bash
C:/Miniconda3/Scripts/conda.exe create -p ./.conda python=3.11 -y
./.conda/python.exe -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
./.conda/python.exe -m pip install -r env/requirements-lock.txt
./.conda/python.exe -m src.utils.env --check   # 환경이 등록되어 있는지
./.conda/python.exe -m pytest tests/ -q
```

---

## 이 저장소가 지키는 것

실험 결과는 **전부 TSV 원장**에 들어가고, 커밋은 **훅이 검사**한다.
사람이 기억해서 지키는 규칙은 지켜지지 않는다는 전제로 만들었다.

| | |
|---|---|
| **하드룰 17개** | [`docs/RULES.md`](docs/RULES.md) — 단일 진실 공급원 |
| **커밋 규칙** | [`docs/COMMIT_CONVENTION.md`](docs/COMMIT_CONVENTION.md) — `record` / `fix` / `upgrade` … |
| **원장 스키마** | [`docs/LEDGER_SCHEMA.md`](docs/LEDGER_SCHEMA.md) — TSV 10종 + manifest |
| **작업 흐름** | [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — 실험 수명주기, 개발 순서 |
| **환경** | [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — 가상환경도 버전 관리한다 |
| **데이터셋** | [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) — 무엇을 쓰고 왜 그것인가 |
| **선행 연구** | [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) — 이미 답이 나온 것과 열려 있는 것 |
| **계획** | [`docs/PLAN.md`](docs/PLAN.md) — 범위·일정·사전 등록 질문 |
| **인수인계** | [`docs/HANDOFF.md`](docs/HANDOFF.md) — 현재 상태와 다음 할 일 |
| **프롬프트** | [`docs/PROMPTS.md`](docs/PROMPTS.md) — 에이전트에 붙여 넣는 지시문 |
| **도메인 라벨** | [`docs/DOMAIN_LABELS.md`](docs/DOMAIN_LABELS.md) — 라벨 검증 기록과 신뢰 범위 |
| **검토** | [`docs/REVIEW.md`](docs/REVIEW.md) — 결함·헛점과 보강 우선순위 |
| **스펙과의 차이** | [`docs/DESIGN_DELTA.md`](docs/DESIGN_DELTA.md) — 다르게 한 것과 **그 이유.** 반증된 가설이 여기 있다 |
| **2차 설계** | [`docs/SPEC_P2.md`](docs/SPEC_P2.md) — 손상된 임베딩의 회복 한계 |
| **2차 일정** | [`docs/SCHEDULE_P2.md`](docs/SCHEDULE_P2.md) — 일정과 결과, 종료 |
| **3차 설계** | [`docs/SPEC_P3.md`](docs/SPEC_P3.md) — 스케줄 이후의 질문 |
| **3차 일정** | [`docs/SCHEDULE_P3.md`](docs/SCHEDULE_P3.md) — W0 부터 |

### 결과물

| | |
|---|---|
| **최종 보고서** | [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) — 1차 전체 |
| **보고표 9종** | [`reports/tables/`](reports/tables/) — 측정마다 하나. 각 문서 끝에 **한계** 절이 있다 |
| **원장** | `experiments/*.tsv` — 모든 숫자의 출처. 239행, append-only |

특히:

- **주장할 수 있는 것만 주장한다** — 한국어 내부 도메인 라벨은 감사 정확도
  ~55% 라 세분화를 보고하지 않는다 ([`docs/DOMAIN_LABELS.md`](docs/DOMAIN_LABELS.md))
- **Split first, tokenize later** — 문서 단위 분할이 토크나이저 학습보다 먼저다
- **Final Test 는 마지막까지 열지 않는다** — 훅이 `final_test` 경로를 하드 차단한다
- **BPB 로 비교한다** — 토크나이저가 다르면 token-level PPL 은 비교 대상이 아니다
- **원장은 append-only** — 이미 쓴 행은 고치지 않는다. 실패한 run 도 남긴다

에이전트로 작업한다면: [`CLAUDE.md`](CLAUDE.md) (Claude Code) /
[`AGENTS.md`](AGENTS.md) (Codex).

---

## 구조

```
configs/      실험 설정 (tokenizer / cpt / evaluation)
data/         raw·interim·final_test 는 git 제외, manifests/*.tsv 만 커밋
src/
  data/       정규화 · dedup · 문서 단위 split
  tokenizer/  Substitute(T2b) 채굴 · pruning(T2a) · vocab/merge DAG 분석
  surgery/    embedding resize · 초기화 E0/E1/E2 · distillation(스텁)
  training/   CPT · alignment(폐기, 재현용으로 보존)
  evaluation/ bpb · token_exposure · latency · memory · capability(스텁)
  utils/      seed · hashing · env · ledger · run tracking
tools/        원장 검사 · git hook 본체 · 장시간 run 감시
experiments/  TSV 원장 + runs/<run_id>/
reports/      FINAL_REPORT.md · tables/ · figures/
```

`alignment.py` 와 `align` phase 는 **폐기했지만 지운다 하지 않았다** — 그
측정을 재현할 수 있어야 왜 뺐는지가 남는다. 되살리기 전에
[`reports/tables/alignment_probe.md`](reports/tables/alignment_probe.md) 를 읽어라.

외부 모델 원본은 `experiments/models.tsv`, 프로젝트가 만든 토크나이저·체크포인트·
리포트는 `experiments/artifacts.tsv`에 기록한다. 실험 시각은
`experiments/clock_checks.tsv`의 외부 HTTPS 대조 결과와 연결된다.

---

## 하드웨어

```
GPU     NVIDIA GeForce RTX 5070 Ti · 16GB · sm_120
torch   2.7.1+cu128
```

16GB 제약 아래 모델 역할을 나눈다 (스펙 §2):

| 규모 | 역할 |
|---|---|
| Qwen2.5-0.5B | 핵심 실험 모델 — Full CPT · tokenizer surgery · ablation |
| Qwen2.5-1.5B | scale validation — Pre-CPT 손상 측정 완료. **전면 CPT 는 16GB 에 안 들어간다** (추정 21.6GB) |
| HyperCLOVA X SEED 0.5B | 한국어 특화 external baseline |
| A.X 4.0 | Qwen 기반 한국어 adaptation 산업 사례 (4-bit 추론 · 토크나이저 분석) |

HCX / A.X 는 **참조**이지 인과 실험이 아니다 ([`docs/RULES.md`](docs/RULES.md) 4번).

---

## 이 저장소가 실제로 잡아낸 것

규율이 값을 한 자리들이다. 하나라도 안 잡혔으면 보고서에 틀린 숫자가 들어갔다.

- **BPB 분모가 한국어에서 2배였다** — ByteLevel 토큰의 바이트 길이는
  `len(token)` 이지 `len(token.encode())` 가 아니다
- **`pack()` 의 바이트 귀속이 구성에 따라 부호가 뒤집혔다** — 한국어만 −0.38%,
  한영 혼합 +0.73%. Equal-Raw-Data 통제축이 조용히 깨지는 경로였다
- **CPT 문서 풀이 예산보다 작았다** — 103.8MB 로 168.5MB 를 채우려 했다.
  그대로 걸었으면 죽었다
- **훅과 CI 가 다른 트리를 봤다** — 로컬은 통과하고 CI 만 빨개졌다.
  훅이 인덱스를 읽도록 고쳤다
- **배치 한계를 `allocated` 로 쟀다** — `reserved` 가 맞다. allocated 로 재면
  이미 성능이 39% 무너진 지점을 "들어간다" 고 보고하게 된다
- **OOM 을 정지 조건으로 쓴 설계가 Windows 에서 성립하지 않았다** — WDDM
  드라이버가 시스템 RAM 으로 흘려 죽지 않고 느려지기만 한다

실패한 run 도 원장에 남아 있다 (`fail` 9건 · `abort` 5건). 지우지 않는 이유는
무엇이 왜 깨졌는지가 규칙의 근거이기 때문이다.
