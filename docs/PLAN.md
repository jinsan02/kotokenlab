# 프로젝트 계획 — 시작과 끝

확정일 2026-08-29. 이 문서가 **무엇을 하고 무엇을 하지 않는지**를 정의한다.
범위를 넓히려면 이 파일을 고치고 `docs(docs):` 로 커밋한다. 조용히 늘리지 않는다.

---

## 확정된 결정

| 항목 | 결정 |
|---|---|
| **범위** | 최소 코어 (아래 참조). T1/T3, E2/E3, A0~A5, layer-wise 분석은 **제외** |
| **시작점** | 소규모 관통 — 200MB 로 파이프라인 전체를 한 번 통과시킨 뒤 확장 |
| **종료 조건** | 사전 등록 질문 6개에 **σ 기준**으로 답이 나오면 종료. negative result 도 완성 |
| **산출물** | README 결과 표 + 재현 가능한 저장소 · arXiv 스타일 리포트 PDF · HuggingFace 공개 |
| **tied embedding** | tied 유지. untie 하지 않고 **한계로 기술** |
| **fertility 기준** | kiwipiepy 0.23.2 형태소. 버전은 `env_sha256` 에 고정 |
| **시스템 벤치마크** | Qwen2.5-0.5B + 1.5B, 입력 5,000~40,000자 |
| **가용 시간** | 주 35시간 이상 |

---

## 코어 — 이것만으로 완결된 이야기

```
Step 1  데이터 파이프라인      정규화 → dedup → 도메인 라벨 → 오염제거 → manifest → 문서단위 split
Step 2  Level 1 벤치마크       Qwen / HCX / A.X / Ours 를 같은 코퍼스에서 (도메인별)
Step 3  토크나이저 T0/T2a/T2b  T0 원본 · T2a vocab 축소 · T2b 크기유지 치환
Step 4  Surgery + E0/E1        resize(tied 유지), random vs mean 초기화, Pre-CPT 평가
Step 5  노이즈 플로어          동일 config 3 seed 로 σ_BPB 측정      ← 다른 모든 비교의 전제
Step 6  0.5B CPT               C0 통제군 필수, Equal Raw Data, 50M
Step 7  시스템 벤치마크         0.5B + 1.5B, 5k~40k자, raw_prompt / equal_tokens
Step 8  Level 3 + 최종 정리     capability, Final Test 1회, 표·그림, 리포트
```

### 명시적으로 하지 않는 것

`T1 Extend` · `T3 New BBPE` · `E2 Weighted` · `E3 Distillation` ·
`A0~A5 파라미터 범위 ablation` · `layer-wise CKA / gradient norm` ·
`Equal Token Budget` · `1.5B CPT`(추론 벤치마크만 함) · `3 seed 최종 반복`(노이즈 플로어에만 사용)

시간이 남으면 우선순위: **T1 → layer-wise 분석 → Equal Token Budget**.

---

## 사전 등록 질문 (pre-registration)

**실험 전에 고정한다.** 각 질문은 아래 셋 중 하나로 답한다.

```
개선     Δ > 2σ 로 좋아짐
구별 불가 |Δ| ≤ 2σ          ← 이것도 답이다. 유리한 쪽으로 반올림하지 않는다
악화     Δ > 2σ 로 나빠짐
```

σ 는 Step 5 에서 잰 노이즈 플로어다. 여섯 질문에 모두 답이 붙으면 프로젝트는 끝난다.

| # | 질문 | 지표 | 답이 나오는 시점 |
|---|---|---|---|
| **Q1** | 한국어 토큰 수를 얼마나 줄이는가? 뉴스에만 나타나는가, 커뮤니티·기술·noisy 에도 나타나는가? | `tok_per_char` 도메인별 | Step 2 |
| **Q2** | 영어와 코드에 어떤 regression 이 생기는가? Candidate Gate(영어 ≤5%, 코드 ≤10%)를 통과하는가? | `tok_per_char` 영어·코드 | Step 2 |
| **Q3** | 토크나이저 교체 직후 모델이 얼마나 무너지는가? mean 초기화가 random 대비 얼마나 덜 무너지는가? | Pre-CPT Korean BPB | Step 4 |
| **Q4** | 동일 CPT 후에도 원본 토크나이저보다 language modeling 이 좋아지는가? | Korean/English/Code BPB, C0 대비 | Step 6 |
| **Q5** | vocab 을 **줄이는 것**(T2a)과 **크기를 유지한 채 치환하는 것**(T2b)은 다른 결과를 내는가? | Korean BPB + 영어 regression | Step 6 |
| **Q6** | 실제 GPU 에서 prefill / TTFT / VRAM / KV cache 가 얼마나 개선되는가? 이론 감소율과 실측의 괴리는? | `system_bench.tsv` 전체 | Step 7 |

### 결론이 이렇게 나와도 완성이다

> 한국어 토큰을 31.2% 줄였으나, 통제된 CPT 후 Korean BPB 는 노이즈 플로어(σ=0.008)
> 이내에서 **구별되지 않았다**. 0.5B 규모에서 compression 의 이득은 품질이 아니라
> **지연과 메모리에만** 나타났다.

이건 실패가 아니라 답이다. 사후에 지표를 바꿔 "개선"을 만들어내지 않는다
([RULES.md](RULES.md) 14번).

---

## 자원 실측 — 계획의 근거

`scripts/probe_resources.py` 로 이 기계에서 직접 측정했다
(전체 표: [`reports/tables/resource_probe.md`](../reports/tables/resource_probe.md)).

```
GPU  RTX 5070 Ti 16,303 MB    CPU  24 logical cores    RAM  31 GB
attention  mem_efficient+cudnn  (FLASH 는 이 torch 빌드에 없음)
```

### 학습 (Qwen2.5-0.5B full CPT, bf16, gradient checkpointing)

| seq | micro_bs | optimizer | VRAM reserved | 여유 | tok/s |
|---:|---:|---|---:|---:|---:|
| 1024 | 1 | AdamW | 5,990 MB | 10,313 MB | 6,292 |
| 1024 | 1 | **AdamW 8bit** | **5,300 MB** | **11,003 MB** | 6,287 |
| 2048 | 1 | AdamW 8bit | 7,716 MB | 8,587 MB | 7,489 |
| 2048 | 2 | AdamW 8bit | 13,326 MB | 2,977 MB | **9,089** |
| 4096 | 1 | AdamW 8bit | 13,326 MB | 2,977 MB | 8,795 |
| 8192 | 1 | AdamW 8bit | 23,132 MB | **초과** | 3,692 |

- **운영 설정: seq 2048, micro_bs 2, AdamW 8bit** — 13.3GB 로 16GB 안에 들어가고
  처리량이 가장 높다. seq 8192 는 VRAM 을 넘겨 호스트 메모리로 흘러 절반 속도가 된다
- 8bit optimizer 가 690MB 를 아끼면서 속도는 같다. 켜고 시작한다
- RAM 은 학습 중 1.7~1.9GB. 31GB 중 문제 없음

### GPU 시간 예산 — 병목이 아니다

9,000 tok/s 기준 **50M 토큰 CPT 한 번이 약 1.6시간**이다.

```
노이즈 플로어  3 seed × 5M      ≈ 0.5h
탐색 Stage 1~2                  ≈ 5h
본 실험 C0 / T2a / T2b × 50M    ≈ 5h
시스템 벤치마크                  ≈ 2h
                                ─────
                                 ≈ 13h
```

**GPU 는 전혀 병목이 아니다.** 12주 계획의 병목은 **데이터 전처리와 코드 작성**이다.
그래서 Step 1~2 에 절반을 배정한다.

### 추론 prefill (생성 경로, `logits_to_keep=1`)

| 입력 토큰 | 한국어 원문 | 0.5B VRAM | 0.5B ms | 1.5B VRAM | 1.5B ms |
|---:|---:|---:|---:|---:|---:|
| 1,024 | ~1,553자 | 1,066 MB | 23 | 3,246 MB | 58 |
| 4,096 | ~6,215자 | 1,240 MB | 81 | 3,534 MB | 219 |
| 8,192 | ~12,430자 | 1,430 MB | 183 | 3,928 MB | 506 |
| 16,384 | ~24,861자 | 1,810 MB | 472 | 4,980 MB | 1,182 |
| 26,000 | ~39,453자 | 2,384 MB | 918 | 6,066 MB | 2,157 |
| 32,768 | ~49,723자 | 2,718 MB | 1,310 | 6,808 MB | 2,972 |

40,000자(≈26,000 토큰)까지 두 모델 모두 여유롭다. 검토 B2 가 요구한 길이 확장이
**가능하다** — 단, 아래 조건에서만.

### 반드시 지켜야 하는 것 — attention 백엔드

`attn_implementation="sdpa"` 만 주면 이 환경의 디스패처는 **MATH 로 폴백한다.**

| 입력 토큰 | 기본 (MATH) | 강제 (mem_efficient+cudnn) | 차이 |
|---:|---:|---:|---|
| 4,096 | 3,184 MB / 590 ms | 1,240 MB / 81 ms | 메모리 2.7배, 시간 7.4배 |
| 8,192 | 9,561 MB / 1,892 ms | 1,430 MB / 183 ms | **메모리 7.1배, 시간 10.3배** |

MATH 백엔드는 `n×n` attention 행렬을 실제로 만든다. 이걸 모르고 쟀으면
16k 이상은 전부 OOM 이고, 측정된 지연은 커널 비효율을 잰 것이지 토크나이저 효과가
아니게 된다. **Level 4 전체가 무의미해질 뻔했다.**

→ [RULES.md](RULES.md) 9번에 규칙으로 넣었고, `attn_backend` 를 `env_sha256` 에 포함시켰다.

---

## 일정 (주 35시간, 12주)

| 주 | 내용 | 산출 |
|---|---|---|
| **1** | 소규모 관통 — FineWeb-2 1샤드(200MB)로 정규화→dedup→manifest→split→Level 1 까지 끝까지 | 파이프라인이 실제로 돈다는 증거, 첫 원장 행 |
| **2** | Level 1 벤치마크 프레임워크 완성 (도메인별, fertility 포함), URL 도메인 규칙 정확도 검증(수동 200건) | `tokenizer_metrics.tsv` |
| **3–4** | 전체 규모 데이터 — raw 10~15GB → 정제 3~5GB, 오염 제거, 문서 단위 split | `data/manifests/*.tsv`, `manifest_sha256` 확정 |
| **5** | Level 1 정식 측정 + Candidate Gate → **Q1, Q2 답** | `phase1-tokenizer-freeze` 태그 |
| **6** | T2a/T2b 토크나이저 학습, surgery(resize, tied 유지, byte fallback 보호), E0/E1, Pre-CPT 평가 → **Q3 답** | 토크나이저 산출물 + `tok(tok):` 커밋 |
| **7** | 평가 파이프라인 고정 → `eval-freeze-v1`. 학습 루프 안정화. **노이즈 플로어 3 seed** | σ_BPB 확정 |
| **8–9** | 0.5B CPT — C0 / T2a / T2b × 50M, Equal Raw Data → **Q4, Q5 답** | `lm_metrics.tsv`, `train_curve.tsv` |
| **10** | 시스템 벤치마크 0.5B + 1.5B, 5k~40k자, raw_prompt / equal_tokens → **Q6 답** | `system_bench.tsv` |
| **11** | Level 3 capability + **Final Test 1회 개봉** (`final-test-opened` 태그) + 표·그림 | `capability.tsv`, `reports/` |
| **12** | arXiv 스타일 리포트, README 결과 표, HuggingFace 공개 | 최종 산출물 |

Final Test 는 **11주차에 딱 한 번** 연다. 그 전까지 checkpoint 선택과 후보 선정은
전부 Dev BPB 로 한다 ([RULES.md](RULES.md) 2번).

---

## 지금 상태

```
Step 0  저장소 · 원장 · 훅 · CI · 문서 · 환경          완료
        모델 5종 확보 (models.tsv 에 revision 기록)     완료
        자원 실측 · attention 백엔드 함정 발견          완료
Step 1  소규모 관통                                    완료
Step 2  Level 1 파일럿                                 완료
        ─────────────────────────────────────────────
현재    전체 규모 전 검증
        도메인 규칙 정확도 · 영어/코드 대조군 · dev 하한 확정이 다음
```

이 단계의 결과는 파이프라인 재현성과 측정 코드 검증용이다. 정식 manifest 와
Candidate Gate 가 고정되기 전에는 토크나이저 학습·surgery·CPT 를 시작하지 않는다.
