# 실험 원장 스키마

정의는 [`src/utils/ledger.py`](../src/utils/ledger.py) 에 있고, 이 문서는 그 설명이다.
컬럼을 바꾸면 **양쪽을 함께** 고친다. 검사는 [`tools/validate_ledger.py`](../tools/validate_ledger.py).

## 공통 규약

- 탭 구분, 헤더 1행, **UTF-8(BOM 없음)**, **LF**
- 결측은 빈칸이 아니라 **`NA`**
- 필드 안의 탭/개행은 리터럴 두 글자(`\t`, `\n`)로 이스케이프
- **append-only** — 이미 쓴 행은 고치지 않는다. 정정도 새 행이다
- 컬럼은 **뒤에만** 추가한다. 중간 삽입·순서 변경 금지 (기존 파일이 깨진다).
  컬럼을 추가했으면 기존 파일의 헤더가 낡으므로 **한 번 마이그레이션**한다:

  ```bash
  .conda/python.exe tools/migrate_ledger.py --dry-run   # 무엇이 바뀌는지 먼저
  .conda/python.exe tools/migrate_ledger.py             # 헤더 교체 + 기존 행 NA 패딩
  ```

  값은 절대 바꾸지 않는다. 뒤에 `NA` 를 덧대기만 하고 `.bak` 을 남긴다.
  헤더 순서가 바뀐 경우는 자동으로 손대지 않고 보고만 한다
- `ts_utc`, `git_commit`, `git_dirty` 는 생략하면 **자동으로 채워진다**
- 새 run 의 `clock_check_sha256` 는 24시간 이내의 성공한 시간 검증에서 자동으로 채운다
- `.gitattributes` 가 `*.tsv` 에 `merge=union` 을 걸어둬서, 서로 다른 브랜치에서
  붙인 행들은 충돌 없이 합쳐진다

## 파일

| 파일 | 행 하나의 단위 | 스펙 |
|---|---|---|
| `experiments/LEDGER.tsv` | run 의 상태 전이 (start / ok / fail) | §58 |
| `experiments/tokenizer_metrics.tsv` | (토크나이저 × split × domain) | §16, §35 |
| `experiments/lm_metrics.tsv` | (run × checkpoint × split × domain) | §37~39 |
| `experiments/capability.tsv` | (run × checkpoint × benchmark × metric) | §40, §52 |
| `experiments/system_bench.tsv` | (모델 × 입력조건 × mode) | §49, §50 |
| `experiments/train_curve.tsv` | 평가 지점 하나 | §26, §84 |
| `experiments/models.tsv` | 내려받은 외부 모델 하나 | §58 |
| `experiments/clock_checks.tsv` | 외부 시각 대조 한 번 | §59 |
| `experiments/artifacts.tsv` | 프로젝트가 만든 산출물 하나 | §59 |
| `data/manifests/<split>.tsv` | 문서 하나 | §7 |
| `env/ENV_SNAPSHOT.tsv` | 환경 변경 하나 | §60 |

---

## `LEDGER.tsv` — run 마스터

다른 모든 테이블의 `run_id` 는 여기에 있어야 한다 (참조 무결성).
run 하나가 `start` 행 하나와 `ok`/`fail` 행 하나를 남긴다. `start` 행을 고치지 않고
새 행을 붙이는 이유는, 죽은 run 의 흔적을 지우지 않기 위해서다.

| 컬럼 | 설명 |
|---|---|
| `ts_utc` | ISO-8601 UTC (`2026-08-29T11:22:33Z`). 로컬 타임존은 쓰지 않는다 |
| `run_id` | 스펙 §61 명명 규칙 (아래 참조) |
| `phase` | `data` `tok` `surgery` `align` `cpt` `eval` `sys` |
| `status` | `start` `ok` `fail` `abort` |
| `tokenizer_version` | `qwen_original`, `koext_v1`, `kosub_v3`, `konew_v1` … |
| `vocab_size` | 정수 |
| `tokenizer_sha256` | 토크나이저 산출물 디렉토리 해시 |
| `model` | `qwen2.5-0.5b`, `qwen2.5-1.5b`, `hcx-seed-0.5b` … |
| `init_method` | `original` `random` `mean` `weighted` `distill` (스펙 §21~23) |
| `seed` | 정수 |
| `target_tokens` | 계획된 학습 토큰 예산 |
| `tokens_seen` | 실제 학습한 토큰 수 |
| `raw_bytes_seen` | 실제 학습한 **원문 바이트 수** — 토크나이저가 다른 run 을 비교하는 축 |
| `wall_sec` | 벽시계 초 |
| `peak_vram_mb` | `torch.cuda.max_memory_allocated()` |
| `git_commit` | HEAD 전체 해시 (자동) |
| `git_dirty` | `1` 이면 커밋되지 않은 변경이 있는 상태에서 돌았다는 뜻 (자동) |
| `config_sha256` | config dict 의 정규화 JSON 해시 |
| `manifest_sha256` | 사용한 dataset manifest 해시 |
| `env_sha256` | 환경 스냅샷 해시 → `ENV_SNAPSHOT.tsv` 로 연결 |
| `argv` | 명령행 인자 |
| `note` | 자유 메모. 실패 시 예외 메시지가 자동으로 들어간다 |
| `model_revision` | HF snapshot 해시. `models.tsv` 로 연결된다 (검토 D1) |
| `embedding_share` | 임베딩이 전체 파라미터에서 차지하는 비율. 결론의 유효 범위 (검토 A2) |
| `clock_check_sha256` | 실행 전 시간 검증 해시. `clock_checks.tsv` 로 연결 |

### `run_id` 명명 (스펙 §61)

```
<phase>_<tokenizer>_<init>_<budget>_seed<seed>
```

```
tok_qwen_original_v1
tok_kosub_v1
align_kosub_mean_5m_seed42
cpt_original_mean_50m_seed42
cpt_kosub_mean_50m_seed42
sys_kosub_rawprompt_10k_v1
```

이름만 보고 어떤 실험인지 알 수 있어야 한다.
생성은 `src.utils.tracking.make_run_id()`.

---

## `tokenizer_metrics.tsv` — Level 1 intrinsic

GPU 를 쓰기 전, CPU 단계에서 후보를 거르는 데 쓴다 (스펙 §17 Candidate Gate).
**domain 별로 한 행씩** 쓴다. 전체 평균 하나만 남기면 "한국어 -30%, code +17%"를 놓친다.

`split` `domain` `n_docs` `n_chars` `n_bytes` `n_eojeol` `n_tokens`
`tok_per_char` `tok_per_byte` `bytes_per_tok` `tok_per_eojeol` `fertility_mean`
`p50_len` `p90_len` `p95_len` `p99_len` `max_len`

P95/P99 를 남기는 이유는 평균이 아니라 꼬리가 context overflow 와 긴 입력 지연을
만들기 때문이다 (스펙 §15).

## `lm_metrics.tsv` — Level 2 language modeling

`checkpoint` `tokens_seen` `raw_bytes_seen` `split` `domain` `n_bytes`
`total_nll` `bpb` `bpc` `token_ppl`

- `bpb` 가 **필수**. 교차 토크나이저 비교는 이것으로만 한다 ([RULES.md](RULES.md) 3번)
- `token_ppl` 은 같은 토크나이저 내부 학습 곡선용
- `n_bytes` 와 `total_nll` 이 함께 있으므로 BPB 를 언제든 재계산·검증할 수 있다

## `capability.tsv` — Level 3

`checkpoint` `benchmark` `lang` `n_items` `n_shot` `metric` `value` `ci_lo` `ci_hi`

long format 이다. 지표 하나가 한 행. bootstrap CI 는 문제 단위로 구한다 (스펙 §52).

## `system_bench.tsv` — Level 4 GPU 시스템

`mode` `raw_chars` `raw_bytes` `input_tokens` `gen_tokens` `n_warmup` `n_runs`
`tokenize_ms_mean` `prefill_ms_mean` `prefill_ms_p95` `ttft_ms_mean` `ttft_ms_p95`
`decode_tok_s_mean` `total_ms_mean` `total_ms_std`
`kv_cache_mb_est` `peak_alloc_mb` `peak_reserved_mb`

- `mode` = `raw_prompt` (같은 원문, 스펙 §46) 또는 `equal_tokens` (같은 토큰 수, §47).
  **둘 다 재야** 차이가 compression 때문이라고 말할 수 있다
- `kv_cache_mb_est` 는 **추정값**, `peak_*` 는 **실측값**. 섞지 않는다 (스펙 §77)
- 총 지연 하나만 적지 않는다. tokenize / prefill / decode 를 나눈다 (스펙 §48)

## `train_curve.tsv` — 학습 곡선

`step` `tokens_seen` `raw_bytes_seen` `train_loss` `dev_loss` `dev_bpb` `lr`
`grad_norm` `grad_norm_emb` `grad_norm_attn` `grad_norm_ffn`
`peak_vram_mb` `tok_per_s` `raw_bytes_per_s` `elapsed_sec`

- x축은 `step` 이 아니라 `tokens_seen` / `raw_bytes_seen` 이다 (스펙 §26)
- 모듈별 gradient norm 은 "어느 모듈에 적응 압력이 걸리는가"에 답한다 (스펙 §72)
- `raw_bytes_per_s` 는 토크나이저가 다른 run 사이에서 공정한 처리량 지표다 (스펙 §79)

## `models.tsv` — 외부 모델 레지스트리

`name` `repo_id` `revision` `role` `scope` `vocab_size` `tokenizer_len`
`hidden_size` `n_layers` `n_heads` `n_kv_heads` `head_dim`
`embedding_params` `total_params` `embedding_share`
`tie_word_embeddings` `kv_bytes_per_token` `files_mb`

`scripts/download_models.py` 가 자동으로 채운다. 두 가지 이유로 존재한다:

1. **`revision`** — HF 저장소는 갱신된다. `model: qwen2.5-0.5b` 만으로는 나중에
   같은 것을 다시 가져올 수 없다 (스펙 §58 `model.revision`).
2. **`embedding_share`, `kv_bytes_per_token`, `tie_word_embeddings`** — 결과 해석에
   직접 쓰이는 구조 사실이다. 매번 config 를 다시 뜯지 않도록 한 곳에 못 박는다.

현재 값 (실측):

| name | revision | emb share | KV B/tok | tied |
|---|---|---:|---:|---:|
| Qwen2.5-0.5B | `060db6499f32` | 27.6% | 12,288 | 1 |
| Qwen2.5-1.5B | `8faed761d45a` | 15.1% | 28,672 | 1 |
| HyperCLOVAX-SEED-0.5B | `3da5046fb019` | 21.9% | 98,304 | 1 |
| A.X-4.0-Light | `ba21c20ea1b3` | 5.1% | 57,344 | 0 |
| A.X-4.0 | `a3393fd67bf0` | 1.2% | 327,680 | 0 |

## `clock_checks.tsv` — 호스트 시각 검증

`clock_check_sha256` `status` `server` `server_utc` `local_midpoint_utc`
`offset_ms` `rtt_ms` `windows_source` `method` `git_commit` `note`

Windows Time 원본만 믿지 않고 캐시되지 않은 HTTPS 응답의 `Date`와 요청 왕복 구간의
중간값을 비교한다. HTTP Date는 초 단위이므로 기본 통과 기준은 절대 오차 5초 이하다.
성공 기록은 24시간 동안 유효하며 `RunContext`가 새 run에 해시를 자동 연결한다.

```bash
C:/llm_tokenizer/.conda/python.exe tools/check_clock.py --record
```

이 명령은 시계를 바꾸지 않는다. `windows_source=local_cmos_clock`이면 관리자 권한의
Windows 시간 동기화가 별도로 필요하지만, 해당 시점의 실제 오차는 HTTPS로 검증된다.

## `artifacts.tsv` — 로컬 산출물 레지스트리

`artifact_id` `run_id` `kind` `name` `path` `artifact_sha256` `size_bytes`
`model_revision` `tokenizer_version` `manifest_sha256` `git_commit` `note`

외부에서 받은 모델의 revision·구조는 `models.tsv`에 남긴다. 이 표는 프로젝트가 만든
토크나이저, 체크포인트, adapter, 매니페스트 요약, 표·그림·리포트의 내용 해시와
생성 run을 연결한다. 같은 artifact ID는 두 번 기록하지 않으며 `final_test` 경로는
개봉 태그 전까지 등록을 거부한다.

```bash
C:/llm_tokenizer/.conda/python.exe tools/register_artifact.py reports/table.tsv \
  --kind table --run-id cpt_kosub_mean_50m_seed42
```

## `data/manifests/<split>.tsv` — 문서 manifest

`doc_id` `source` `domain` `date` `language` `sha256` `split` `char_count` `byte_count`

프로젝트 전체 데이터의 기준점이다 (스펙 §7). 문서 단위 분할이 여기서 확정된다.

## `env/ENV_SNAPSHOT.tsv` — 환경 이력

`env_sha256` `python` `torch` `cuda` `cudnn` `transformers` `tokenizers` `datasets`
`accelerate` `bitsandbytes` `peft` `numpy` `driver` `gpu_name` `vram_mb` `change_note`

현재 환경의 `env_sha256` 가 이 파일에 없으면 `RunContext` 가 실행을 거부한다.
자세한 내용은 [ENVIRONMENT.md](ENVIRONMENT.md).

---

## 검사기가 잡는 것

`tools/validate_ledger.py` (pre-commit 과 CI 가 호출):

- CRLF · 헤더 불일치 · 컬럼 수 · 빈 필드 · sha256/git_commit/ts_utc 형식
- **참조 무결성** — 메트릭 행의 `run_id` 가 `LEDGER.tsv` 에 실재하는가
- run 의 `clock_check_sha256` 가 `clock_checks.tsv` 에 실재하는가
- 산출물의 `run_id` 가 `LEDGER.tsv` 에 실재하는가
- **죽은 run** — `status=start` 만 있고 종료 행이 없는 run (검토 D2).
  OOM·정전으로 죽은 run 이 성공한 것처럼 보이면 안 된다.
  의도적으로 중단했다면 `status=abort` 행을 직접 붙인다.
- **중복 행** — `merge=union` 이 같은 행을 두 번 남긴 경우 (검토 D3)

## 쓰는 법

```python
from src.utils.tracking import RunContext, make_run_id

run_id = make_run_id("cpt", "kosub", "mean", "50m", seed=42)

with RunContext(run_id, phase="cpt", config=cfg, seed=42,
                model="qwen2.5-0.5b", tokenizer_version="kosub_v3",
                init_method="mean", target_tokens=50_000_000) as run:
    ...
    run.log("train_curve", step=1000, tokens_seen=1_000_000,
            raw_bytes_seen=3_100_000, train_loss=2.41, dev_bpb=1.207)
    run.log("lm_metrics", split="dev", domain="news",
            n_bytes=8_412_003, total_nll=7.1e6, bpb=1.207)
    run.tokens_seen = 50_000_000
    run.raw_bytes_seen = 155_000_000
```

`run_id`, `config_sha256`, `git_commit`, `ts_utc` 는 자동으로 붙는다.
사람이 적는 단계를 없앤다.

## 읽는 법

```bash
# 최근 run 10개
tail -n 10 experiments/LEDGER.tsv | cut -f1,2,4,12,13

# 도메인별 BPB
awk -F'\t' 'NR==1 || $2=="cpt_kosub_mean_50m_seed42"' experiments/lm_metrics.tsv | column -t -s$'\t'
```

```python
import pandas as pd
df = pd.read_csv("experiments/lm_metrics.tsv", sep="\t", na_values=["NA"])
```
