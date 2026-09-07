# 결과 계보 — 수치 · 설정 · 로그 · 결과표

> **이 문서의 목적**: 보고서에 인용된 수치 하나를 짚으면, 그것을 낸 run 과
> 설정과 로그와 결과표까지 한 줄로 따라갈 수 있게 한다.
>
> 여기 있는 것은 전부 `experiments/` 원장에서 읽었다. **손으로 옮겨 적은
> 해시는 없다** — 훅이 원장과 대조해서 거부한다 ([RULES.md](RULES.md) 13번).

## 증거 등급

| 등급 | 뜻 |
|---|---|
| **확정** | 원장에 값이 있고 run_id · 설정 해시 · 결과표가 전부 연결된다 |
| **환산** | 원장 값에서 산술로 유도했다. 유도식을 함께 적는다 |
| **조건부** | 값은 실측이나 환경(여유 VRAM 등)에 딸려 있다 |
| **증거 부족** | 측정하지 않았거나 잡음이 커서 읽을 수 없다. **인용 금지** |
| **오염** | 측정은 됐으나 통제되지 않은 요인이 섞였다. **인용 금지** |

## 파일 가용성

| 표기 | 뜻 |
|---|---|
| (git) | 저장소에 커밋돼 있다 |
| **로컬 전용** | `.gitignore` 로 제외됐다. 이 기기에만 있다 |

`artifacts/logs/*.log` 와 `artifacts/models/*` 는 전부 **로컬 전용** 이다.
모델은 sha256 만 `experiments/artifacts.tsv` 에 등록된다.

## 읽는 법

| 열 | 의미 |
|---|---|
| `run_id` | 원장의 기본키. `experiments/LEDGER.tsv` 에서 `grep` 하면 나온다 |
| `config_sha256` | 그 run 의 설정 해시(앞 16자). 전문은 원장에 |
| `git_commit` | 실행 시점의 커밋(앞 7자) |
| 설정 | `experiments/runs/<run_id>/config.json` — 전체 인자 |
| 환경 | `experiments/runs/<run_id>/env.json` — 패키지·드라이버·GPU |
| 로그 | `artifacts/logs/` — **git 제외.** 로컬에만 있다 |
| 결과표 | `reports/tables/` — 해석과 한계가 붙은 문서 |

모든 run 의 `env_sha256` 는 `8e2eafe5…` 로 같다. 실험 기간 내내 환경이
바뀌지 않았다는 뜻이고, 바뀌었다면 `env/ENV_SNAPSHOT.tsv` 에 행이 늘고
`RunContext` 가 실행을 거부한다.

---

## 1. 품질 — 본 CPT

### 1.1 비교 기준을 먼저 못 박는다

BPB 비교는 **네 조건이 서로 다른 예산 축** 위에 있다. 섞으면 안 된다.

| 조건 | 원문 | 토큰 | 예산 축 |
|---|---:|---:|---|
| C0 원본 | 168,505,891 B | 49,922,048 | 기준 |
| T2b 30k **고정 원문** | 168,502,276 B | 34,832,384 | C0 와 **같은 원문** (차 0.0021%) |
| T2b 10k **고정 원문** | 168,500,000 B 목표 | 37,351,424 | C0 와 같은 원문 |
| T2b 30k **등토큰** | 241,390,978 B | 49,922,048 | C0 와 **같은 토큰수**(정확히 일치) |

> **"토큰 예산 43% 증가" 는 T2b 30k 고정 원문 조건 대비다.**
> 등토큰 조건은 그 조건보다 원문 +43.3% · 토큰 +43.3% 를 쓰며, 그 결과
> **C0 와 토큰수가 정확히 같아진다**(49,922,048). C0 대비 증가가 아니다.

### 1.2 핵심 수치

| 항목 | 값 |
|---|---|
| **metric** | 한국어 BPB (`TotalNLL ÷ (ln2 × 원문바이트)`), dev split, 평가 예산 언어당 2MB |
| **조건** | C0 원본 · Equal-Raw-Data 168.5MB · seq 2048 · micro_bs 2 · accum 8 · lr 1e-05 · AdamW8bit · bf16 · gradient checkpointing · attention `EFFICIENT+CUDNN` 강제 · seed 42 |
| **원본 값** | **1.137540** |
| **비교 기준** | 이것이 기준선이다 |
| **run ID** | `cpt_c0_qwen_main_seed42` |
| **config SHA-256** | `c16f813e2213bacb…` (전문은 `LEDGER.tsv`) |
| **git commit** | `a6dc77c` |
| **설정 파일** | `experiments/runs/cpt_c0_qwen_main_seed42/config.json` (git) |
| **로그** | `artifacts/logs/phase3_cpt.log` — **로컬 전용** |
| **결과표** | [cpt_main.md](../reports/tables/cpt_main.md) |
| **측정 조건** | 평가는 `checkpoint=final`, `domain=ko` 행. 학습 전 지점은 `pre_cpt`/`step0` |
| **증거 등급** | **확정** |

| 항목 | 값 |
|---|---|
| **metric** | 한국어 BPB (동일 정의) |
| **조건** | T2a 제거만(vocab 121,728) · 나머지 C0 와 동일 |
| **원본 값** | **1.136817** |
| **비교 기준** | C0 1.137540 대비 **−0.064%**, 7.2σ (σ=0.000100, T2a 조건) → **개선** |
| **run ID** | `cpt_t2a_none_main_seed42` |
| **config SHA-256** | `c762a84588af654c…` |
| **git commit** | `a6dc77c` |
| **설정 파일** | `experiments/runs/cpt_t2a_none_main_seed42/config.json` (git) |
| **로그** | `artifacts/logs/phase3_cpt.log` — **로컬 전용** |
| **결과표** | [cpt_main.md](../reports/tables/cpt_main.md) |
| **측정 조건** | 위와 동일 |
| **증거 등급** | **확정** |

| 항목 | 값 |
|---|---|
| **metric** | 한국어 BPB (동일 정의) |
| **조건** | T2b 치환 n=30,000 · E1 부품 평균 초기화 · **고정 원문 168.5MB** |
| **원본 값** | **1.567095** |
| **비교 기준** | C0 1.137540 대비 **+37.8%** · Δ=0.429555 |
| **σ 와 계산식** | σ = max(C0 0.000058, T2b 0.001234) = **0.001234** (큰 쪽 σ 규칙). Δ/σ = 0.429555 / 0.001234 = **348.1σ** → 악화 |
| **run ID** | `cpt_t2b_mean_main_seed42` |
| **config SHA-256** | `0c9c7394fa43089f…` |
| **git commit** | `a6dc77c` |
| **설정 파일** | `experiments/runs/cpt_t2b_mean_main_seed42/config.json` (git) |
| **로그** | `artifacts/logs/phase3_cpt.log` — **로컬 전용** |
| **결과표** | [cpt_main.md](../reports/tables/cpt_main.md) · σ 출처 [noise_floor.md](../reports/tables/noise_floor.md) |
| **측정 조건** | 수술 직후 2.380297 에서 출발. 회복률 65.4% |
| **증거 등급** | **확정** |

| 항목 | 값 |
|---|---|
| **metric** | 한국어 BPB (동일 정의) |
| **조건** | T2b 30k · **등토큰 49,922,048 토큰** (원문 241,390,978 B) |
| **원본 값** | **1.532183** |
| **비교 기준** | ① C0 1.137540 대비 **+34.7%** ② **T2b 30k 고정 원문 1.567095 대비 −2.2%** — 그 조건보다 원문·토큰을 각 43.3% 더 썼다 |
| **σ 와 계산식** | σ = 0.001234 (T2b 조건). C0 와의 Δ=0.394643 → 320σ 악화 |
| **run ID** | `cpt_t2b_mean_eqtok_seed42` |
| **config SHA-256** | `b5309b1ab4e7bf24…` |
| **git commit** | `9ded083` |
| **설정 파일** | `experiments/runs/cpt_t2b_mean_eqtok_seed42/config.json` (git) |
| **로그** | `artifacts/logs/phase4.log` — **로컬 전용** |
| **결과표** | [phase4.md](../reports/tables/phase4.md) |
| **측정 조건** | `--budget-tokens 49922048` · LR 스케줄도 토큰 기준(`cosine_by_tokens`) · 풀 70,000 문서(C0 는 50,000)라 문서 순서가 다르다 |
| **증거 등급** | **확정** (문서 집합 차이는 결과표 한계 절에 기재) |

| 항목 | 값 |
|---|---|
| **metric** | 한국어 BPB (동일 정의) |
| **조건** | T2b 치환 **n=10,000** · 고정 원문 168.5MB · 나머지 30k 와 동일 |
| **원본 값** | **1.497272** |
| **비교 기준** | C0 1.137540 대비 **+31.6%**, σ=0.001166 (N=10k 조건 실측) → 308σ 악화 |
| **run ID** | `cpt_t2b10k_mean_main_seed42` |
| **config SHA-256** | `43a54b48b1863462…` |
| **git commit** | `fcec1fc` |
| **설정 파일** | `experiments/runs/cpt_t2b10k_mean_main_seed42/config.json` (git) |
| **로그** | `artifacts/logs/phase4.log` — **로컬 전용** |
| **결과표** | [phase4.md](../reports/tables/phase4.md) |
| **측정 조건** | 예산·풀·시드가 30k 와 동일해 한 표에 놓을 수 있다 |
| **증거 등급** | **확정** |

> **교차 검증**: T2b 수술 직후 값 **2.380297** 이 두 경로에서 독립적으로
> 나왔다 — 별도 평가 run `eval_bpb_t2b_mean_main0`(`checkpoint=pre_cpt`)와
> CPT 내부 측정 `cpt_t2b_mean_grad_seed42`(`checkpoint=step0`).
> 소수 여섯 자리까지 일치한다.

## 2. 시스템 — Q6 (배치 1, prefill 중심)

| 수치 | run_id | config_sha256 | git_commit | 결과표 |
|---|---|---|---|---|
| C0 prefill (40,000자) **729.5ms** | `sys_c0_qwen_v1` | `e7641cab14546379` | `de67701` | [system_bench.md](../reports/tables/system_bench.md) |
| T2a prefill (40,000자) **727.8ms** | `sys_t2a_none_v1` | `d16b19abd7bcc3a3` | `de67701` | 같음 |
| T2b prefill (40,000자) **428.5ms** | `sys_t2b_mean_v1` | `925ccb892313638f` | `de67701` | 같음 |

**측정 설정** (`config.json` 실측):

```
prefill   warm-up 20회 · 측정 100회
생성 경로 warm-up  5회 · 측정  20회   <- 여기가 decode 가 안 읽히는 이유
원문 길이 5,000 / 10,000 / 20,000 / 40,000자
등토큰    2,048 / 4,096 / 8,192 / 16,384 토큰
attention prefill = efficient+cudnn   decode = efficient+cudnn+math
```

로그: `artifacts/logs/phase5.log`
**지표 원본**: `experiments/system_bench.tsv`, `mode=raw_prompt` / `equal_tokens`

### 재현성 확인 run

| | run_id | config_sha256 | 왜 |
|---|---|---|---|
| C0 16,384 토큰 재측정 | `sys_c0_qwen_ordercheck` | 원장 참조 | 본 측정의 2% 이탈이 재현되는지. **재현되지 않았다** |

---

## 3. 시스템 — 배치 처리량 (Q6-E)

| 수치 | run_id | config_sha256 | git_commit | 결과표 |
|---|---|---|---|---|
| C0 최대 배치 **20**, 처리량 **3.85 seq/s** | `sys_c0_qwen_batchv1` | `038e3e6a2d9e5134` | `ff38779` | [phase6.md](../reports/tables/phase6.md) |
| T2b 최대 배치 **28**, 처리량 **6.18 seq/s** | `sys_t2b_mean_batchv1` | `65319abc8a914538` | `ff38779` | 같음 |

**측정 설정**: 원문 20,000자 · warm-up 5회 · 측정 20회 ·
배치 `1,2,4,8,12,16,18,20,22,24,26,28,30,32,36,40,48` ·
**시작 여유 VRAM 14,045MB** (두 run 동일).

로그: `artifacts/logs/phase6e_v2.log`

> **중단된 선행 run 이 있다.** `sys_c0_qwen_batchv1` 의 첫 시도는 OOM 을 정지
> 조건으로 삼았다가 중단했다 — Windows WDDM 이 시스템 RAM 으로 흘려 죽지
> 않았기 때문이다. 원장에 `abort` 행과 사유가 남아 있다. 지우지 않았다.

---

## 4. 스케일 — Pre-CPT 손상만 (Phase 6-A)

| 수치 | run_id | config_sha256 | git_commit | 결과표 |
|---|---|---|---|---|
| 1.5B 기준선 **0.992076** | `eval_bpb_qwen15b_base_scale15b` | `f8a0540043da37f0` | `fc7f3eb` | [phase6.md](../reports/tables/phase6.md) |
| 1.5B 수술 후 **2.305960** | `eval_bpb_t2b_mean_1p5b_scale15b` | `bb474cd3aec2b10f` | `fc7f3eb` | 같음 |

수술 산출물: `surgery_kot2b_v2_n30000_mean1p5b_seed42` →
`artifacts/models/kot2b_v2_n30000_mean_1p5b` (`experiments/artifacts.tsv` 에 sha256 등록)

**주의**: 이 두 값은 **학습 없이** 잰 것이다. 1.5B 의 CPT 회복력은 측정하지
않았다 — 16GB 에 전면 CPT 가 안 들어간다(추정 21.6GB, `SPEC_P2.md` §2).

---

## 5. 토크나이저 — Level 1

| 수치 | run_id | config_sha256 | git_commit | 결과표 |
|---|---|---|---|---|
| 한국어 −30.2%, 코드 −0.8% (T2b v2 n=30k) | `tok_bench_v1` | `4da24d976f3675f5` | `9860424` | [t2_sweep.md](../reports/tables/t2_sweep.md) |

**지표 원본**: `experiments/tokenizer_metrics.tsv`
코퍼스는 `phase1-tokenizer-freeze` 태그로 고정 — 이후 데이터가 바뀌면
앞의 모든 수치가 무효가 되므로 먼저 얼렸다.

---

## 6. 판정의 근거 — 조건별 σ

| 조건 | σ (한국어) | run_id |
|---|---:|---|
| C0 | 0.000058 | `cpt_c0_qwen_noise2_seed{42,123,2026}` |
| T2a | 0.000100 | `cpt_t2a_none_noise_seed{42,123,2026}` |
| T2b (N=30k) | 0.001234 | `cpt_t2b_mean_noise_seed{42,123,2026}` |
| T2b (N=10k) | 0.001166 | `cpt_t2b10k_mean_noise_seed{42,123,2026}` |

결과표: [noise_floor.md](../reports/tables/noise_floor.md)

**이 σ 는 17.5MB 예산에서 쟀고 본 실험은 168.5MB 다.** 하한으로 다뤄야 한다 —
그 사실이 각 결과표의 한계 절에 적혀 있다.

같은 시드 재실행 분산은 별도로 쟀다 (`cpt_t2b_mean_grad_seed42`,
`999ba55f53c06596`): 9개 지점 최대 편차 **0.000421** 로 시드 간 σ 의 34% 다.

---

## 7. 확인되지 않은 값

인용하면 안 되는 것들이다.

| 항목 | 상태 |
|---|---|
| decode 처리량 (`decode_tok_s_mean`) | **증거 부족** — `total_ms_std/mean` 이 5.7~17.5% 이고 배치별로 단조롭지 않다. 생성 경로 측정이 20회뿐이다 |
| 1.5B CPT 회복률 | **증거 부족** — 측정하지 않았다 |
| 7B 이상의 손상·회복 | **증거 부족** — 하드웨어가 안 된다. 구조상 `tie_word_embeddings=0` 이라는 것만 `models.tsv` 로 확인됨 |
| 능력 평가 (Level 3) | **증거 부족** — `capability.py` 가 스텁이다 |
| 학습 wall-clock 기반 처리량 | **오염됨** — 데스크톱이 GPU 를 함께 썼다. T2a 는 구간 처리량이 26~52 s/MB 로 2배 흔들렸다. 속도 주장에 쓰지 않는다 |
| 배치 최대값의 절대 크기 | **조건부** — 시작 여유 VRAM(14,045MB)에 딸려 있다. 두 조건을 연달아 재서 **비(1.40배)는 튼튼하지만** 절대값은 환경에 따라 변한다 |
| 문맥 용량 +44.1% | **환산값** — `raw_prompt` 40,000자 실측률을 16,384 토큰에 적용했다. 직접 기록은 이후 run 부터 |

---

## 8. 원장을 직접 확인하는 법

```bash
# 어떤 수치의 출처를 찾을 때
grep "cpt_t2b_mean_main_seed42" experiments/LEDGER.tsv
cat experiments/runs/cpt_t2b_mean_main_seed42/config.json

# 지표 원본
grep "cpt_t2b_mean_main_seed42" experiments/lm_metrics.tsv

# 전체 무결성
.conda/python.exe tools/validate_ledger.py
```

원장은 append-only 다. 실패한 run 도 지우지 않는다 (`fail` 9건 · `abort` 5건).
정정도 새 행으로 남긴다.
