# 하드룰 — 이 저장소의 단일 진실 공급원

> **규칙은 이 파일에만 적는다.** `CLAUDE.md`, `AGENTS.md`, `README.md` 는 이 문서를
> 가리키기만 한다. 규칙을 바꾸려면 여기를 고치고 `docs(docs):` 로 커밋한다.
> 다른 파일에 규칙을 복사하면 두 문서가 갈라지고, 갈라진 규칙은 규칙이 아니다.

원본 연구 설계는 [`SPEC_KoTokenLab.md`](SPEC_KoTokenLab.md) (스펙). 아래 12개는 스펙
§122 의 핵심 원칙을 **집행 가능한 형태**로 옮긴 것이다. 각 항목에 "무엇이 이것을
강제하는가"를 붙였다. 강제 장치가 없는 규칙은 지켜지지 않는다.

---

## 1. Split first, tokenize later

Train/Dev/Test 분할은 **문서 단위로**, 토크나이저 학습보다 **먼저** 한다.
문장 단위로 쪼갠 뒤 무작위 분할하면 같은 문서의 앞뒤가 Train 과 Test 에 나뉘어 들어간다.

순서: `정규화 → Exact Dedup → Near Dedup → 문서 단위 분할 → 청킹`

- **강제**: `src/data/split.py` 가 manifest 를 먼저 요구한다. manifest 없이 토크나이저를
  학습할 수 없다. 원장의 모든 run 행에 `manifest_sha256` 이 들어간다.

## 2. Final Test 는 마지막까지 열지 않는다

Final Test 는 토크나이저 vocabulary 설계, merge rule, pruning threshold,
하이퍼파라미터, learning rate, checkpoint 선택, 모델 구조 선택 중
**어디에도** 쓰지 않는다. checkpoint 선택은 Dev BPB 로 한다.

- **강제**: `.gitignore` 가 `data/final_test/` 를 제외하고, `tools/precheck.py` 가
  경로에 `final_test` 가 들어간 **모든** 스테이지 파일을 거부한다.
  실제로 개봉하는 순간은 `final-test-opened` 태그로 커밋에 못 박는다.
- README 에 다음 문장을 유지한다:
  *The final test set was never used for tokenizer design, hyperparameter tuning,
  model selection, or checkpoint selection.*

## 3. 서로 다른 토크나이저의 PPL 을 직접 비교하지 않는다

토큰 단위가 다르면 token-level perplexity 는 비교 대상이 아니다.
교차 비교는 **BPB (bits per byte)** 로 한다. `BPB = TotalNLL / (ln2 × RawBytes)`

token-level PPL 은 **같은 토크나이저 내부의 학습 곡선 확인용**으로만 쓴다.

- **강제**: `lm_metrics.tsv` 에 `bpb` 는 필수 컬럼, `token_ppl` 은 선택 컬럼이다.
  `n_bytes`(원문 바이트)가 같은 행에 있어서 언제든 재계산할 수 있다.

## 4. Qwen ↔ HCX/A.X 의 점수 차이를 tokenizer 효과라고 부르지 않는다

backbone, 사전학습 데이터, 학습 토큰 수, post-training 이 전부 다르다.
HCX / A.X 는 **External Reference / Industry Baseline** 이지 인과 실험이 아니다.

허용되는 주장: *"HCX 토크나이저는 본 corpus 에서 Qwen 대비 한국어 compression 이 X% 높았다"*

## 5. 토크나이저의 인과 효과는 같은 Qwen backbone 에서만 검증한다

`Qwen Original` vs `Qwen + Korean Tokenizer`. 이것만이 인과 실험이다.

## 6. Original Tokenizer + 동일 CPT 통제군을 반드시 둔다

`Original pretrained Qwen` vs `KoTokenizer + CPT` 는 불공정하다.
후자만 추가 학습을 했기 때문이다. 비교는 항상
`Original Tokenizer + Same CPT` vs `Korean Tokenizer + Same CPT`.

- **강제**: CPT ablation 은 C0(Original) 없이 시작하지 않는다. C0 의 `run_id` 가
  `LEDGER.tsv` 에 `status=ok` 로 있어야 C1~C3 결과를 기록한다.

## 7. Equal Raw Data 와 Equal Token Budget 을 둘 다 실험한다

같은 원문만 보면 compute 가 다르고, 같은 토큰 예산만 보면 본 정보량이 다르다.
둘 다 있어야 trade-off 가 보인다.

- **강제**: 원장의 모든 학습 행이 `tokens_seen` 과 `raw_bytes_seen` 을 함께 갖는다.
  학습 곡선의 x축은 `step` 이 아니라 이 둘이다.

## 8. Tokenizer Surgery / Embedding Alignment / CPT 를 분리한다

한 번에 하면 네 가지 효과(토크나이저·초기화·정렬·CPT)를 분리할 수 없다.
Embedding 초기화 직후 **CPT 없이** 평가하는 Pre-CPT 지점을 반드시 남긴다.

- **강제**: `phase` 컬럼이 `surgery`/`align`/`cpt` 를 구분하고, run 은 phase 하나만 담는다.

## 9. GPU 벤치마크는 warm-up + `torch.cuda.synchronize()` 를 쓴다

GPU 는 비동기 실행이라 동기화 없이 잰 시간은 의미가 없다.
warm-up 20~30회, 측정 100회. mean / median / std / P95 를 모두 기록한다.
peak VRAM 은 `max_memory_allocated` 와 `max_memory_reserved` 를 둘 다 남긴다.

- **강제**: `system_bench.tsv` 에 `n_warmup`, `n_runs`, `*_p95`, `total_ms_std`,
  `peak_alloc_mb`, `peak_reserved_mb` 가 컬럼으로 있다. 비워두면 눈에 띈다.

## 10. Exploration 은 1 seed, Final 만 multi-seed

탐색 단계는 `seed=42` 하나. 상위 1~2개 후보만 `42 / 123 / 2026` 으로 반복하고
`1.103 ± 0.008` 형태로 보고한다. 단일 GPU 에서 처음부터 3 seed 는 낭비다.

## 11. 모든 실험을 hash 와 version 으로 추적한다

결과에서 원본까지 끊기지 않아야 한다:
`결과 → config → tokenizer version → dataset manifest → raw data → git commit`

- **강제**: 원장의 모든 행에 `git_commit` 이 자동으로 들어간다
  (`src/utils/ledger.py` 가 채운다). run 행은 `config_sha256`, `tokenizer_sha256`,
  `manifest_sha256`, `env_sha256` 를 갖는다. `tools/validate_ledger.py` 가
  메트릭 행의 `run_id` 가 `LEDGER.tsv` 에 실재하는지 확인한다.

## 12. 0.5B 에서 깊게, 1.5B 에서 scale validation

0.5B(Qwen2.5-0.5B Base)는 full-parameter CPT 와 ablation 의 무대다.
1.5B 는 0.5B 에서 이미 결론 난 항목을 반복하지 않고, `Original vs Best KoTokenizer`
하나만 Embedding Alignment + LoRA/QLoRA 로 확인한다. (16GB VRAM 제약)

---

## 13. 기록 규칙 (aimers 에서 배운 것)

이전 프로젝트(`C:/aimers`)에서 실제로 있었던 일이 `src/train_gbdt2.py` 주석에 남아 있다:

> 30개 실험을 돌리고 로그를 하나도 갱신하지 않았다. 그래서 이미 실패로 끝난 축을
> "안 해본 축"이라 부르며 다시 큐에 걸었다. **사람이 적는 단계를 없앤다.**

그래서:

- 실험 결과는 **전부 TSV 원장**에 들어간다. 마크다운 실험일지를 손으로 쓰지 않는다.
- 원장은 **append-only**. 이미 쓴 행은 고치지 않는다. 정정도 새 행이다.
- **실패한 run 도 기록한다** (`status=fail`). 실패 기록이 없으면 같은 실패를 반복한다.
- 기록은 코드가 한다. `RunContext` 로 감싸면 자동으로 남는다.

## 14. 평가 지표는 실험 전에 고정한다

본 실험을 시작하기 전에 evaluation metric 을 확정하고 `eval-freeze-v1` 태그를 찍는다.
"이 지표가 더 좋게 나오네"라는 이유로 중간에 지표를 추가·변경하지 않는다.
필요한 변경은 `eval-freeze-v2` 처럼 버전으로 관리한다.

## 15. 훅을 우회하지 않는다

`git commit --no-verify` 를 쓰지 않는다. 훅이 막으면 그 커밋에 문제가 있는 것이다.
훅 자체가 틀렸다면 훅을 고치고 `fix(infra):` 로 커밋한다.

---

## 관련 문서

| 문서 | 내용 |
|---|---|
| [`COMMIT_CONVENTION.md`](COMMIT_CONVENTION.md) | 커밋 타입·트레일러·브랜치·태그 |
| [`LEDGER_SCHEMA.md`](LEDGER_SCHEMA.md) | TSV 원장 전체 스키마 |
| [`WORKFLOW.md`](WORKFLOW.md) | 실험 1건의 수명주기 |
| [`DATA_SOURCES.md`](DATA_SOURCES.md) | 데이터셋 조사와 Step 1 적용 계획 |
| [`RELATED_WORK.md`](RELATED_WORK.md) | 선행 연구와 이 프로젝트의 차별점 |
| [`REVIEW.md`](REVIEW.md) | 설계·인프라 검토와 보강 항목 |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | 가상환경과 그 버전 관리 |
| [`SPEC_KoTokenLab.md`](SPEC_KoTokenLab.md) | 원본 연구 설계 (변경하지 않는다) |
