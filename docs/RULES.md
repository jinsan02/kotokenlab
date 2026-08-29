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

**이 규칙은 시스템 지표에도 그대로 적용된다.** 예를 들어 KV cache 는
`experiments/models.tsv` 기준 Qwen2.5-0.5B 가 12,288 B/token, HCX-SEED 가
98,304 B/token 으로 **8배** 차이 난다. GQA 설정(KV head 2개 vs 8개, head_dim 64 vs 128)이
다르기 때문이다. 두 모델의 VRAM 을 비교하면 그 차이의 대부분은 아키텍처지
토크나이저가 아니다.

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

### 그리고 attention 백엔드를 반드시 강제한다

`attn_implementation="sdpa"` 만 주면 이 환경의 디스패처는 **MATH 로 폴백한다.**
FLASH 는 이 Windows torch 빌드에 컴파일되어 있지 않다 (`No available kernel`).

```
8,192 토큰 prefill (Qwen2.5-0.5B)
  기본 디스패처              9,561 MB / 1,892 ms
  mem_efficient+cudnn 강제   1,430 MB /   183 ms      메모리 7.1배, 시간 10.3배
```

MATH 는 `n×n` attention 행렬을 실제로 만든다. 강제하지 않으면 16k 토큰 이상은
전부 OOM 이고, 측정된 지연은 커널 비효율이지 토크나이저 효과가 아니다.

- **강제**: 학습·평가·벤치마크의 모든 forward 를 아래로 감싼다.

  ```python
  from torch.nn.attention import SDPBackend, sdpa_kernel
  EFFICIENT_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
  with sdpa_kernel(EFFICIENT_SDPA):
      ...
  ```

- `attn_backend` 가 `env_sha256` 에 포함된다. 백엔드가 달라지면 다른 환경이다.
- 토크나이저 비교는 **같은 백엔드에서** 한다. 구현이 섞이면 지연 차이의 원인을 알 수 없다.

## 10. Exploration 은 1 seed, Final 만 multi-seed — 단, 노이즈 플로어를 먼저 잰다

탐색 단계는 `seed=42` 하나. 상위 1~2개 후보만 `42 / 123 / 2026` 으로 반복하고
`1.103 ± 0.008` 형태로 보고한다. 단일 GPU 에서 처음부터 3 seed 는 낭비다.

**그러나 단일 시드 비교는 노이즈 플로어를 알기 전까지 해석할 수 없다.**
seed 분산을 모르면 "BPB 1.207 vs 1.198" 이 의미 있는 차이인지 알 수 없고,
§17 Candidate Gate 가 노이즈로 후보를 탈락시킬 수 있다.

- **강제**: 가장 먼저 하는 CPT 실험은 **동일 config 를 seed 42/123/2026 으로 3회**
  돌려 `σ_BPB` 를 재는 것이다. 이 run 들의 `run_id` 는 `noise_*` 로 시작한다.
- 이후 모든 비교는 **`Δ > 2σ` 일 때만 "차이 있음"** 으로 보고한다.
  그 미만은 결과 표에 **"구별 불가"** 로 적는다. 유리한 쪽으로 반올림하지 않는다.
- 비용은 Stage 1 예산의 3배(≈15M 토큰)로, 잘못된 후보 선택 한 번보다 싸다.

## 11. 모든 실험을 hash 와 version 으로 추적한다

결과에서 원본까지 끊기지 않아야 한다:
`결과 → config → tokenizer version → dataset manifest → raw data → git commit`

- **강제**: 원장의 모든 행에 `git_commit` 이 자동으로 들어간다
  (`src/utils/ledger.py` 가 채운다). run 행은 `config_sha256`, `tokenizer_sha256`,
  `manifest_sha256`, `env_sha256` 를 갖는다. `tools/validate_ledger.py` 가
  메트릭 행의 `run_id` 가 `LEDGER.tsv` 에 실재하는지 확인한다.

## 12. 0.5B 에서 깊게, 1.5B 에서 scale validation — 결론에 스케일을 명시한다

0.5B(Qwen2.5-0.5B Base)는 full-parameter CPT 와 ablation 의 무대다.
1.5B 는 0.5B 에서 이미 결론 난 항목을 반복하지 않고, `Original vs Best KoTokenizer`
하나만 Embedding Alignment + LoRA/QLoRA 로 확인한다. (16GB VRAM 제약)

**임베딩 비중이 스케일마다 크게 다르다** (`experiments/models.tsv` 실측):

```
Qwen2.5-0.5B  27.6%      Qwen2.5-1.5B  15.1%      A.X-4.0-Light(7B)  5.1%
```

토크나이저 수술이 건드리는 파라미터의 비중이 5배 차이 난다. 따라서
**0.5B 에서 잰 "얼마나 무너지고 얼마나 회복되는가" 는 큰 모델보다 체계적으로
과대 측정된다.** 1.5B 검증은 확인이 아니라 **보정**이다.

- **강제**: `LEDGER.tsv` 에 `embedding_share` 컬럼이 있다. 결과 표와 결론 문장에
  스케일을 반드시 병기한다. 스케일 외삽을 주장하지 않는다.

## 12b. 학습 스케줄의 x축은 원문 바이트다

같은 원문을 학습해도 토크나이저가 다르면 토큰 수가 다르고 → **optimizer step 수가
다르다.** 스케줄을 step 기준으로 정의하면 두 run 이 서로 다른 warmup 비율과 decay
곡선을 받아, §31 Equal Raw Data 가 재려는 것이 오염된다.

- **강제**: config 의 `training.schedule_axis` 를 `raw_bytes` 로 둔다
  (`configs/cpt/example.yaml` 참조). 다르게 하려면 config 에 명시해서
  `config_sha256` 에 들어가게 한다.

## 12c. byte fallback 토큰은 절대 pruning 하지 않는다

BBPE 는 임의 바이트열을 표현하기 위해 **256개 바이트 토큰**을 바닥에 깔아둔다.
개별 빈도가 낮아 §19 의 pruning 조건에 그대로 걸리는데, 하나라도 지우면 처음 보는
입력에서 토크나이저가 실패한다 — 그리고 그 실패는 한참 뒤에야 드러난다.

- **강제**: `src/tokenizer/protected.py` 의 `protected_token_ids()` 를 pruning 후보에서
  제외하고, pruning·치환 직후 `assert_byte_roundtrip()` 을 호출한다.
  `tests/test_protected_tokens.py` 가 이를 검사한다.

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
| [`PLAN.md`](PLAN.md) | 범위·일정·사전 등록 질문 (시작과 끝) |
| [`REVIEW.md`](REVIEW.md) | 설계·인프라 검토와 보강 항목 |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | 가상환경과 그 버전 관리 |
| [`SPEC_KoTokenLab.md`](SPEC_KoTokenLab.md) | 원본 연구 설계 (변경하지 않는다) |
