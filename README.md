# KoTokenLab

**Korean Tokenizer Surgery & LLM Adaptation on a Single RTX 5070 Ti 16GB**

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

**결과 전체: [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)** (1차 종료, 2026-09-02)

전체 연구 설계: [`docs/SPEC_KoTokenLab.md`](docs/SPEC_KoTokenLab.md)

---

## 결과

**2026-08-31 현재 — 압축은 됐고, 품질은 안 따라왔다.**

Qwen2.5-0.5B 에 T2b(크기 보존 치환, n=30,000) 수술을 하고 세 조건에 **같은
원문 168.5MB** 로 통제된 CPT 를 돌린 결과다. 전체 표:
[`reports/tables/cpt_main.md`](reports/tables/cpt_main.md).

```
한국어 토큰 수        -30.2%    tok/char, v1 dev (약속대로 압축됨)
같은 원문의 토큰      49.9M -> 34.8M  (비율 0.6977, 예측 0.696 과 일치)

한국어 BPB           +37.8%    C0 대비 **악화** (1.5671 vs 1.1375, 348 sigma)
영어 regression       +0.98%
코드 regression       +1.84%

Prefill / TTFT / VRAM   미측정 (Step 12)
```

**토크나이저 수술은 설계대로 작동했다. 못 따라온 것은 언어모델이다.**
T2b 는 수술 직후 BPB 2.3803 에서 출발해 1.5671 까지 왔지만 C0 는 1.1375 다 —
메워야 할 1.2428 중 65.4% 를 메우고 0.4296 을 남겼다.

**병목이 무엇인지는 아직 모른다.** 처음에는 노출 부족으로 봤다 — 새 토큰
30,000개의 중앙값 발화가 168.5MB 전체에서 143회뿐이었다. 그래서 새 토큰을
1/3 로 줄여(N=10,000) 노출을 4.4배(중앙값 623회)로 올려 봤는데 **회복률이
65.4% -> 65.9% 로 거의 그대로였다.** 노출 가설은 반증됐다
([`reports/tables/phase4.md`](reports/tables/phase4.md)).

데이터를 43% 더 주면(등토큰) 회복률이 68.2% 로 오른다. 즉 예산에는 반응하지만
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
configs/      실험 설정 (tokenizer / alignment / cpt / evaluation)
data/         raw·interim·final_test 는 git 제외, manifests/*.tsv 만 커밋
src/
  data/       정규화 · dedup · 문서 단위 split
  tokenizer/  Extend · Substitute · New BBPE 학습, vocab 분석
  surgery/    embedding resize · 초기화 4종 · distillation
  training/   embedding alignment · CPT
  evaluation/ Level 1~4 (intrinsic / BPB / capability / system)
  utils/      seed · hashing · env · ledger · run tracking
tools/        원장 검사, git hook 본체
experiments/  TSV 원장 + runs/<run_id>/
reports/      figures · tables
```

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
| Qwen2.5-1.5B | scale validation — embedding alignment · LoRA/QLoRA |
| HyperCLOVA X SEED 0.5B | 한국어 특화 external baseline |
| A.X 4.0 | Qwen 기반 한국어 adaptation 산업 사례 (4-bit 추론 · 토크나이저 분석) |

HCX / A.X 는 **참조**이지 인과 실험이 아니다 ([`docs/RULES.md`](docs/RULES.md) 4번).
