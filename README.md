# KoTokenLab

**Korean Tokenizer Surgery & LLM Adaptation on a Single RTX 5070 Ti 16GB**

```
Qwen2.5-0.5B
        ↓
Vocabulary Surgery
        ↓
Embedding Alignment
        ↓
Continued Pretraining
        ↓
Ko-Qwen
```

한국어 특화 토크나이저가 LLM 의 언어 모델링 품질, 학습 효율, 시퀀스 길이,
Attention 연산량, KV Cache, 추론 지연에 미치는 영향을 **통제된 실험**으로 분석한다.

"한국어 성능이 올라갔다"를 보이는 것이 목표가 아니다.
`Tokenizer → Embedding → Transformer → Training → Evaluation → GPU Systems` 를
하나의 재현 가능한 파이프라인으로 연결하는 것이 목표다.

전체 연구 설계: [`docs/SPEC_KoTokenLab.md`](docs/SPEC_KoTokenLab.md)

---

## 결과

> 아래 표는 정식 데이터와 자체 토크나이저가 고정된 뒤 채운다. 현재는 Step 1
> 소규모 데이터 관통과 Step 2 Level 1 파일럿까지 완료한 **검증 단계**다.

```
Korean Tokens        -XX.X%
P95 Sequence Length  -XX.X%
Korean BPB           -X.X%
English Regression   +X.X%
Code Regression      +X.X%

Prefill Latency      -XX.X%
TTFT                 -XX.X%
Peak VRAM            -XX.X%
```

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
| **원장 스키마** | [`docs/LEDGER_SCHEMA.md`](docs/LEDGER_SCHEMA.md) — TSV 8종 |
| **작업 흐름** | [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — 실험 수명주기, 개발 순서 |
| **환경** | [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — 가상환경도 버전 관리한다 |
| **데이터셋** | [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) — 무엇을 쓰고 왜 그것인가 |
| **선행 연구** | [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) — 이미 답이 나온 것과 열려 있는 것 |
| **계획** | [`docs/PLAN.md`](docs/PLAN.md) — 범위·일정·사전 등록 질문 |
| **인수인계** | [`docs/HANDOFF.md`](docs/HANDOFF.md) — 현재 상태와 다음 할 일 |
| **프롬프트** | [`docs/PROMPTS.md`](docs/PROMPTS.md) — 에이전트에 붙여 넣는 지시문 |
| **검토** | [`docs/REVIEW.md`](docs/REVIEW.md) — 결함·헛점과 보강 우선순위 |

특히:

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
