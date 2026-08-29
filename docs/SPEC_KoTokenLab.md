# KoTokenLab
## Korean Tokenizer Surgery & LLM Adaptation on a Single RTX 5070 Ti 16GB

> **Goal:** 한국어 특화 토크나이저가 LLM의 언어 모델링 품질, 학습 효율, 시퀀스 길이, Attention 연산량, KV Cache, 추론 지연에 미치는 영향을 **통제된 실험**으로 분석한다.  
> 단순히 "한국어 성능이 올라갔다"를 보여주는 것이 아니라, **Tokenizer → Embedding → Transformer → Training → Evaluation → GPU Systems**를 하나의 재현 가능한 파이프라인으로 연결하는 것을 목표로 한다.

---

# 1. 프로젝트 핵심 질문

## RQ1. Tokenizer Efficiency

Qwen / HyperCLOVA X / A.X / Custom Korean Tokenizer는 같은 한국어를 얼마나 다르게 토큰화하는가?

측정 항목:

- Tokens / sentence
- Tokens / character
- Tokens / byte
- Bytes / token
- Tokens / eojeol
- Fragmentation
- P50 / P90 / P95 / P99 sequence length
- English / Code compression degradation

---

## RQ2. Tokenizer Surgery

기존 pretrained LLM의 tokenizer를 교체하면 왜 모델 성능이 무너지는가?

핵심 원인:

```text
Old Token ID
    ↓
Old Embedding Semantics

Tokenizer Replacement

Same / New Token ID
    ↓
Different Token Semantics

=> Tokenizer 의미와 Embedding 의미 불일치
```

이를 정량적으로 분석한다.

---

## RQ3. Embedding Alignment

새로운 한국어 토큰의 embedding을 어떻게 초기화해야 pretrained knowledge를 가장 잘 보존할 수 있는가?

비교:

```text
Random Initialization
Mean Initialization
Frequency-weighted Initialization
Context-aware / Distillation Alignment
```

---

## RQ4. Continued Pretraining

Tokenizer를 바꾼 모델이 CPT를 통해 얼마나 빠르게 회복되는가?

비교:

```text
Original Tokenizer + CPT
Korean Extended Tokenizer + CPT
Korean Substitution Tokenizer + CPT
Korean New BBPE + CPT
```

---

## RQ5. System Efficiency

한국어 token 수 감소가 실제 GPU 시스템 성능에 어느 정도 영향을 주는가?

측정:

```text
Input tokens
Prefill latency
TTFT
Decode throughput
KV Cache
Peak VRAM
Tokens/sec
Raw bytes/sec
```

---

# 2. Hardware Constraint

## Main Environment

```text
GPU  : NVIDIA RTX 5070 Ti
VRAM : 16GB
```

16GB VRAM이라는 제약을 고려하여 모델의 역할을 분리한다.

| Model Scale | 역할 | 권장 작업 |
|---|---|---|
| ~0.5B | 핵심 연구 모델 | Full CPT / Tokenizer Surgery / Ablation |
| ~1.5B | Scale-up 검증 | Embedding Alignment / PEFT / QLoRA |
| ~4B | 시스템 및 확장 실험 | 4-bit inference / QLoRA |
| ~7B | 산업 모델 비교 | 4-bit inference / tokenizer benchmark |
| 14B+ | 구조/논문 분석 | 학습 대상 제외 |

---

# 3. 모델 구성

## Main Experimental Backbone

### Qwen2.5-0.5B Base

용도:

```text
Full-parameter CPT
Tokenizer replacement
Embedding surgery
Layer freeze ablation
Attention-only / FFN-only adaptation
Multiple-seed experiments
```

이 프로젝트의 **과학 실험용 메인 모델**이다.

---

## Scale Validation

### Qwen2.5-1.5B Base

용도:

```text
Best tokenizer 재현
Embedding alignment
LoRA / QLoRA
0.5B에서 얻은 결론의 scale-up validation
```

---

## External Comparison

### HyperCLOVA X SEED 0.5B

역할:

```text
한국어 특화 소형 LLM 비교
Qwen과 비슷한 scale에서 tokenizer 효율 비교
한국어 특화 모델의 실제 설계 참고
```

### A.X 4.0

역할:

```text
Qwen 기반 Korean adaptation의 산업 사례
Tokenizer Aligning → CPT 흐름 참고
7B 모델은 4-bit inference / tokenizer 분석 중심
```

---

# 4. 전체 Experimental Pipeline

```text
Raw Corpus
    │
    ▼
Data Cleaning
    │
    ├─ Exact Dedup
    ├─ Near Dedup
    ├─ Language Filtering
    └─ Quality Filtering
    │
    ▼
Document-level Split
    │
    ├─────────┬─────────┐
    ▼         ▼         ▼
  Train      Dev      Final Test
    │
    ▼
Tokenizer Training
    │
    ├─ T0 Original Qwen
    ├─ T1 Ko-Extended
    ├─ T2 Ko-Substitution
    └─ T3 Ko-New BBPE
    │
    ▼
Tokenizer Intrinsic Evaluation
    │
    ▼
Candidate Gate
    │
    ▼
Embedding Surgery
    │
    ├─ Random
    ├─ Mean
    ├─ Weighted
    └─ Distillation
    │
    ▼
Embedding Alignment
    │
    ▼
0.5B Full CPT
    │
    ▼
Dev Evaluation
    │
    ▼
Top Candidates
    │
    ├─ Seed 42
    ├─ Seed 123
    └─ Seed 2026
    │
    ▼
Final Test
    │
    ▼
1.5B Scale Validation
    │
    ▼
HCX / A.X External Comparison
    │
    ▼
System Benchmark
```

---

# 5. Data Pipeline

## 5.1 Corpus Composition

한국어 하나의 도메인에 과도하게 편향되지 않도록 corpus를 분리한다.

```text
Korean Corpus

├── News
├── Encyclopedia
├── Blog
├── Community
├── Conversational
├── Technical
├── Korean-English Mixed
├── Programming
└── Noisy Korean
    ├── Typo
    ├── Spacing Error
    └── Slang
```

여기서 중요한 건 **학습 corpus와 평가 corpus를 동일한 방식으로 섞지 않는 것**이다.

예를 들어 최종 validation에서는:

```text
News
Community
Technical
Conversational
Ko-En Mixed
Code
Noisy Korean
```

각 domain을 따로 평가한다.

---

# 6. Data Leakage Prevention

## 가장 중요한 원칙

**Train / Dev / Test split을 tokenizer 학습보다 먼저 수행한다.**

### 잘못된 방식

```text
Raw Documents
    ↓
Sentence Chunking
    ↓
Random Split
```

이 경우 동일 문서의 일부가 Train/Test 양쪽에 존재할 수 있다.

---

## 권장 방식

```text
Raw Documents
    ↓
Normalize
    ↓
Exact Dedup
    ↓
Near Dedup
    ↓
Document-level Split
    ↓
Train / Dev / Test
    ↓
Chunking
```

즉:

> **Split first, tokenize later.**

---

# 7. Dataset Manifest

각 문서를 metadata와 함께 관리한다.

```text
doc_id
source
domain
date
language
sha256
split
char_count
byte_count
```

예시:

```json
{
  "doc_id": "news_00001234",
  "source": "news",
  "domain": "news",
  "language": "ko",
  "sha256": "...",
  "split": "train",
  "char_count": 2381,
  "byte_count": 6137
}
```

이 manifest를 프로젝트 전체 데이터 기준점으로 사용한다.

---

# 8. Deduplication

## Exact Dedup

```text
Normalize
    ↓
SHA256
    ↓
Duplicate Removal
```

공백이나 Unicode normalization 차이 때문에 동일 문서가 다른 hash를 갖지 않도록 normalize 이후 hash를 만든다.

---

## Near Duplicate

다음 방식 중 하나를 고려한다.

```text
MinHash
LSH
n-gram Jaccard Similarity
```

목표:

```text
동일 뉴스 재배포
블로그 복사
위키 mirror
유사 문서
```

가 Train/Test 양쪽에 존재하는 것을 최소화한다.

---

# 9. Split Strategy

예:

```text
Train : 90%
Dev   : 5%
Test  : 5%
```

비율 자체보다 중요한 것은:

```text
Document-level Split
+
Domain Balance
+
Near Duplicate 제거
```

다.

---

# 10. Final Test Policy

Final Test는 다음 용도로 사용하지 않는다.

```text
Tokenizer vocabulary 설계
Merge rule 수정
Pruning threshold 선택
Hyperparameter tuning
Learning rate 선택
Checkpoint selection
Model architecture 선택
```

최종 README에도 명시한다.

> The final test set was never used for tokenizer design, hyperparameter tuning, model selection, or checkpoint selection.

---

# 11. Tokenizer Experiments

## T0 — Original

```text
Qwen Original Tokenizer
```

Control group.

---

## T1 — Korean Extension

```text
Qwen Vocabulary
      +
Additional Korean Tokens
```

장점:

```text
구현이 상대적으로 단순
기존 vocabulary 보존
기존 multilingual 능력 보존 가능성
```

단점:

```text
Vocabulary 증가
Embedding parameter 증가
LM head 증가 가능
Memory 증가
```

---

# 12. T2 — Korean Substitution

```text
Low-frequency Tokens Pruning
        +
Korean Tokens Substitution
```

핵심:

```text
Vocabulary Size ≈ Constant
```

기존 tokenizer에서 거의 사용되지 않는 token을 제거하고 그 자리를 한국어 고효율 token으로 바꾼다.

연구 질문:

> Vocabulary 크기를 늘리지 않고 한국어 compression을 향상할 수 있는가?

---

# 13. T3 — New Korean-heavy BBPE

동일한 Train corpus를 이용하여 완전히 새 tokenizer를 만든다.

목적:

```text
기존 Qwen vocab 구조에서 벗어난 upper-bound 탐색
BBPE merge rule 분석
Korean-heavy tokenizer의 극단적인 compression 확인
```

단점:

```text
기존 embedding과의 alignment 난이도가 가장 높음
pretrained tokenizer와 의미 공간 차이가 큼
```

---

# 14. Tokenizer Intrinsic Evaluation

모델을 학습하기 전에 CPU 단계에서 tokenizer 후보를 평가한다.

## Compression Metrics

```text
tokens / sentence
tokens / character
tokens / byte
bytes / token
tokens / eojeol
```

---

## Fragmentation

예:

```text
정보처리기사
```

Tokenizer A:

```text
정보 / 처리 / 기사
fertility = 3
```

Tokenizer B:

```text
정보처리기사
fertility = 1
```

단순 token count보다 **한국어 의미 단위가 얼마나 잘 유지되는지**도 분석한다.

---

# 15. Sequence Length Distribution

평균만 보면 안 된다.

측정:

```text
P50
P90
P95
P99
MAX
```

예:

```text
              Qwen     KoTokenizer

P50           312       228
P90           817       588
P95          1032       711
P99          1804      1250
```

P95/P99가 중요한 이유는 실제 inference context overflow와 긴 입력 latency에 직접 영향을 주기 때문이다.

---

# 16. Domain-wise Tokenizer Evaluation

| Domain | Tok/Char | Byte/Tok | P95 Length |
|---|---:|---:|---:|
| News | | | |
| Community | | | |
| Technical | | | |
| Ko-En Mixed | | | |
| Code | | | |
| Noisy Korean | | | |

전체 평균 하나만 기록하지 않는다.

예를 들어:

```text
한국어 -30%
영어 +2%
Code +17%
```

라면 code 성능 저하가 너무 크다는 걸 확인할 수 있다.

---

# 17. Candidate Gate

GPU 비용을 줄이기 위해 Tokenizer와 Model Training 사이에 **Gate**를 둔다.

예:

```text
Korean Tok/Char improvement >= 15%

AND

English degradation <= 5%

AND

Code degradation <= 5~10%
```

조건을 만족하는 tokenizer만 다음 단계로 이동한다.

```text
10 Tokenizer Candidates
       ↓
CPU Benchmark
       ↓
Candidate Gate
       ↓
2~3 Candidates
       ↓
GPU Experiments
```

---

# 18. Vocabulary Analysis

각 tokenizer의 vocabulary를 분석한다.

```text
Vocabulary
├── Korean
├── Latin
├── Chinese
├── Japanese
├── Number
├── Code Symbol
├── Whitespace
└── Byte-related Tokens
```

추가 분석:

```text
Train corpus에서 사용되지 않는 token
Low-frequency token
High-frequency token
한국어를 과도하게 fragmentation하는 token
English에서 중요한 token
Code에서 중요한 token
```

---

# 19. Token Frequency Distribution

각 token의 corpus 사용 빈도를 계산한다.

예:

```text
Top 1%
Top 10%
Median
Bottom 10%
Unused
```

이를 이용해서:

```text
Pruning Candidate
=
Low-frequency
+
Low semantic importance
+
한국어/영어/code 핵심 token 아님
```

조건으로 pruning한다.

---

# 20. Embedding Surgery

Tokenizer 교체 직후 바로 CPT하지 않는다.

다음 단계를 분리한다.

```text
Tokenizer Surgery
       ↓
Embedding Initialization
       ↓
Embedding Alignment
       ↓
Continued Pretraining
```

왜냐하면 바로 CPT를 해버리면:

```text
Tokenizer 효과
Embedding Init 효과
Alignment 효과
CPT 효과
```

를 분리할 수 없기 때문이다.

---

# 21. Embedding Initialization Ablation

새 token:

```text
정보처리기사
```

기존 tokenizer에서:

```text
정보
처리
기사
```

로 분해된다고 가정한다.

---

## E0 — Random Initialization

```text
E_new ~ Normal(0, σ)
```

Baseline.

새 token 의미를 pretrained embedding으로부터 전혀 전달하지 않는다.

---

## E1 — Mean Initialization

\[
E_{new} =
\frac{
E_{\text{정보}} +
E_{\text{처리}} +
E_{\text{기사}}
}{3}
\]

기존 token embedding의 평균으로 초기화한다.

---

# 22. E2 — Weighted Initialization

\[
E_{new}
=
w_1E_{\text{정보}}
+
w_2E_{\text{처리}}
+
w_3E_{\text{기사}}
\]

가중치 후보:

```text
Token length
Corpus frequency
Character coverage
Semantic contribution
```

---

# 23. E3 — Context-aware / Distillation

같은 raw text에 대해 기존 모델과 새 tokenizer 모델의 representation을 최대한 비슷하게 만든다.

예:

\[
L_{align} =
MSE(H_{original}, H_{new})
\]

또는:

```text
Hidden State Matching
Logit Distillation
KL Divergence
Embedding Distillation
```

을 사용할 수 있다.

---

# 24. Pre-CPT Evaluation

Embedding 초기화 직후 바로 평가한다.

목적:

> Tokenizer replacement 자체가 pretrained model을 얼마나 손상시키는가?

예:

| Init | Korean BPB ↓ | English BPB ↓ |
|---|---:|---:|
| Original | | |
| Random | | |
| Mean | | |
| Weighted | | |
| Distillation | | |

이 단계에서는 **CPT를 수행하지 않는다.**

따라서 embedding initialization 자체의 효과를 분리해서 볼 수 있다.

---

# 25. Embedding Alignment

첫 Alignment 단계에서는 Transformer backbone을 freeze한다.

```text
Embedding       TRAIN
    ↓
Transformer     FROZEN
    ↓
LM Head         TRAIN / TIED
```

실험 budget:

```text
1M tokens
5M tokens
10M tokens
20M tokens
```

각 checkpoint에서 Dev evaluation.

---

# 26. Training Curve X-axis

단순 `step` 사용을 지양한다.

예:

```text
step = 1000
```

은 tokenizer마다 실제 처리 정보량이 다르기 때문에 비교하기 어렵다.

추천:

```text
Training Tokens Seen
```

그리고 동시에:

```text
Raw Bytes Seen
```

도 기록한다.

그래프:

```text
Validation BPB
 ↑
 │\
 │ \
 │  \__
 │     \____
 │
 └──────────────────→ Raw Bytes Seen
```

---

# 27. Main CPT

## Qwen2.5-0.5B Base

5070 Ti 16GB에서 가장 중요한 실험 모델.

초기 configuration 방향:

```yaml
precision: bf16

gradient_checkpointing: true

optimizer: 8bit_adamw

sequence_length: 1024

micro_batch_size: 1

gradient_accumulation_steps: configurable
```

안정화된 이후:

```text
1024
 ↓
2048
```

sequence length 확장을 시도한다.

---

# 28. 왜 0.5B를 Full CPT하는가

이 프로젝트의 연구 질문은:

> Tokenizer가 바뀌었을 때 pretrained Transformer 전체가 어떻게 적응하는가?

이다.

따라서 LoRA만 사용하면:

```text
Tokenizer 변화
+
LoRA 제약
```

이라는 변수가 하나 더 추가된다.

0.5B에서는 가능하면 full parameter CPT를 수행해 **순수한 tokenizer adaptation 현상**을 본다.

---

# 29. Main CPT Ablation

| Run | Tokenizer | Init | Training |
|---|---|---|---|
| C0 | Qwen Original | Original | CPT |
| C1 | Ko-Extend | Mean | CPT |
| C2 | Ko-Substitute | Mean | CPT |
| C3 | Ko-New | Mean | CPT |

나머지는 최대한 동일하게 한다.

```text
Raw corpus
Data order
Seed
Optimizer
Learning Rate
Scheduler
Max sequence length
Batch token budget
Training budget
```

---

# 30. 가장 중요한 Control

잘못된 비교:

```text
Original pretrained Qwen

VS

KoTokenizer + CPT
```

후자만 추가 학습했기 때문에 공정하지 않다.

---

## 올바른 비교

```text
Original Tokenizer
+
Same CPT

VS

Korean Tokenizer
+
Same CPT
```

그래야 tokenizer 변경 효과를 분리할 수 있다.

---

# 31. Equal Raw Data Experiment

두 모델이 **동일한 원문**을 학습한다.

예:

```text
Raw Korean Corpus = 동일
```

하지만:

```text
Qwen tokenizer
→ 300M tokens

KoTokenizer
→ 210M tokens
```

일 수 있다.

연구 질문:

> 같은 정보량을 학습할 때 더 효율적인 tokenizer가 language modeling에 이득을 주는가?

---

# 32. Equal Token Budget Experiment

이번에는:

```text
Qwen
100M tokens

KoTokenizer
100M tokens
```

으로 맞춘다.

KoTokenizer가 더 효율적이라면 동일 100M token으로 더 많은 raw text를 볼 수 있다.

연구 질문:

> 동일 Transformer token-compute budget에서 Korean tokenizer가 더 많은 정보를 학습함으로써 이점을 얻는가?

---

# 33. 두 실험을 모두 해야 하는 이유

Tokenizer efficiency를 평가할 때:

```text
Same Raw Data
```

만 보면 compute가 다르고,

```text
Same Token Budget
```

만 보면 본 raw information이 다르다.

따라서 둘 다 해야 trade-off가 명확해진다.

---

# 34. Validation Pipeline

평가를 4계층으로 분리한다.

```text
Level 1
Tokenizer Intrinsic

Level 2
Language Modeling

Level 3
Capability

Level 4
System Efficiency
```

---

# 35. Level 1 — Tokenizer

측정:

```text
Tokens / char
Tokens / byte
Tokens / eojeol
Bytes / token

Fragmentation

P50
P90
P95
P99

English degradation
Code degradation
```

---

# 36. Level 2 — Language Modeling

## Perplexity 문제

Tokenizer A:

```text
나는 / 학교에 / 간다
= 3 tokens
```

Tokenizer B:

```text
나 / 는 / 학교 / 에 / 간 / 다
= 6 tokens
```

두 tokenizer의 token 단위가 다르기 때문에:

```text
PPL_A
vs
PPL_B
```

를 직접 비교하는 것은 공정하지 않다.

---

# 37. Bits Per Byte

Cross-tokenizer 핵심 metric:

\[
BPB =
\frac{\text{Total NLL}}
{\ln(2)\times \text{Raw Bytes}}
\]

즉 원문 byte 기준으로 language modeling quality를 normalize한다.

추가로:

```text
BPC
Byte-normalized NLL
```

도 활용할 수 있다.

---

# 38. Language Modeling Result

| Model | Korean BPB ↓ | English BPB ↓ | Code BPB ↓ |
|---|---:|---:|---:|
| Original | | | |
| Ko-Extend | | | |
| Ko-Sub | | | |
| Ko-New | | | |

Token-level PPL은 **동일 tokenizer 내부 학습 곡선 확인용**으로만 사용한다.

---

# 39. Domain별 BPB

예:

| Domain | Original | KoTokenizer | Δ |
|---|---:|---:|---:|
| News | | | |
| Community | | | |
| Technical | | | |
| Conversation | | | |
| English | | | |
| Code | | | |

이걸 통해:

> 한국어 평균은 좋아졌지만 특정 domain은 오히려 나빠졌는가?

를 확인한다.

---

# 40. Level 3 — Capability

CPT Base model에서는 instruction following 평가보다:

```text
Log-likelihood
Multiple Choice
Few-shot Completion
Reading Comprehension
Commonsense
```

형태의 benchmark가 적합하다.

---

## Evaluation Groups

```text
Korean
├── KMMLU-style
├── QA
├── Commonsense
└── Reading Comprehension

English
├── MMLU-style
└── General QA

Code
└── Completion
```

---

# 41. 평가에서 중요한 것

절대 점수뿐 아니라:

```text
Korean Δ
English Δ
Code Δ
```

를 본다.

예:

```text
Korean +3.1

English -0.6

Code -0.4
```

이면 한국어 특화 과정에서 general capability regression이 작다고 주장할 수 있다.

---

# 42. SFT는 완전히 별도 Phase

```text
Tokenizer
    ↓
Embedding Alignment
    ↓
CPT
    ↓

====== Research Result Fixed ======

    ↓
SFT
    ↓
Instruction Following
```

CPT 전에 SFT를 섞으면:

```text
Tokenizer 효과
CPT 효과
SFT dataset 효과
```

를 분리하기 어려워진다.

---

# 43. Level 4 — GPU System Evaluation

실제 **RTX 5070 Ti 16GB**에서 측정한다.

측정할 핵심 지표:

```text
VRAM
Tokenization latency
Prefill latency
TTFT
Decode tok/s
Total latency
KV Cache
```

---

# 44. VRAM Measurement

예:

```python
torch.cuda.reset_peak_memory_stats()

# inference / training

peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

가능하면:

```text
Allocated
Reserved
```

를 둘 다 기록한다.

---

# 45. Latency Measurement

GPU는 asynchronous execution을 하므로 반드시 synchronization한다.

```text
Warm-up
    ↓
torch.cuda.synchronize()
    ↓
Start Timer
    ↓
Inference
    ↓
torch.cuda.synchronize()
    ↓
Stop Timer
```

그렇지 않으면 실제 GPU 작업이 끝나기 전에 Python timer가 종료될 수 있다.

---

# 46. Raw Prompt Benchmark

동일한 **원문**을 입력한다.

예:

```text
10,000 Korean characters
```

Tokenizer 결과:

```text
Qwen
→ 7,200 tokens

KoTokenizer
→ 5,000 tokens
```

측정:

```text
Tokenization Time
Input Tokens
Prefill Latency
TTFT
Peak VRAM
KV Cache
```

이 실험은 **실제 사용자 관점**의 tokenizer 효율 비교다.

---

# 47. Equal Token Length Control

이번에는 양쪽 모두:

```text
4096 tokens
```

로 맞춘다.

목적:

> 모델 kernel 자체가 동일한 조건에서 동일하게 동작하는가?

를 확인한다.

이 비교를 둬야 Raw Prompt Benchmark의 차이가 **실제로 tokenizer compression 때문**이라고 더 강하게 주장할 수 있다.

---

# 48. Prefill / Decode 분리

Generation latency를:

```text
Total Latency
```

하나만 기록하면 안 된다.

```text
Total Generation
       │
       ├── Tokenization
       │
       ├── Prefill
       │
       └── Decode
```

로 분리한다.

Tokenizer compression은 특히:

```text
Prefill
Context Length
KV Cache
```

에 큰 영향을 미친다.

---

# 49. System Result Table

| Model | Input Tokens ↓ | TTFT ↓ | Prefill ↓ | Decode tok/s ↑ | Peak VRAM ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen | | | | | |
| KoQwen | | | | | |

---

# 50. 반복 측정

Latency를 한 번만 측정하지 않는다.

추천:

```text
Warm-up Runs
20~30

Measured Runs
100
```

기록:

```text
Mean
Median
Standard Deviation
P95
```

---

# 51. Seed Strategy

모든 실험을 처음부터 3 seeds로 하면 single GPU에서는 비용이 너무 크다.

따라서:

## Exploration Phase

```text
seed = 42
```

모든 후보를 1회 평가한다.

---

## Final Phase

상위 1~2개 후보만:

```text
seed = 42
seed = 123
seed = 2026
```

으로 다시 수행한다.

최종 결과:

```text
Korean BPB
1.103 ± 0.008
```

형태로 기록한다.

---

# 52. Confidence Interval

Capability benchmark에서는 가능하면 Bootstrap CI를 추가한다.

예:

```text
KMMLU Accuracy
48.3%

95% CI
[47.1, 49.5]
```

문제 단위 bootstrap으로 구현할 수 있다.

---

# 53. Checkpoint Selection

예:

```text
checkpoint_1M
checkpoint_5M
checkpoint_10M
checkpoint_20M
checkpoint_50M
```

checkpoint 선택은:

```text
Dev BPB
```

를 사용한다.

Final Test 점수를 보고 checkpoint를 고르면 안 된다.

---

# 54. 1.5B Scale Validation

0.5B 실험을 1.5B에서 모두 반복할 필요는 없다.

0.5B에서:

```text
Best Tokenizer
Best Embedding Init
Best Alignment Method
```

를 선택하고,

1.5B에서는:

```text
Original
vs
Best KoTokenizer
```

만 비교한다.

추천 backbone:

```text
Qwen2.5-1.5B Base
```

---

# 55. 1.5B에서는 왜 Full CPT가 아닌가

5070 Ti 16GB에서 1.5B full training은 0.5B보다 VRAM과 시간 부담이 상당히 커진다.

따라서:

```text
Embedding Alignment
+
LoRA / QLoRA
```

중심으로 실험한다.

목표는:

> 0.5B에서 발견한 tokenizer 효과가 더 큰 모델에서도 방향성이 유지되는가?

를 보는 것이다.

---

# 56. External Validation

HCX / A.X와 직접 성능 비교를 할 때 주의한다.

이 모델들은:

```text
Architecture
Pretraining Data
Training Tokens
Optimization
Model Scale
Post-training
```

이 모두 다를 수 있다.

따라서:

```text
Qwen Original
vs
Our Ko-Qwen
```

은 **인과 실험**.

반면:

```text
HCX
A.X
```

는 **External Reference / Industry Baseline**이다.

---

# 57. 실험 주장 예시

잘못된 주장:

> HCX가 Qwen보다 한국어 점수가 높으므로 HCX tokenizer가 더 좋다.

이건 성립하지 않는다.

---

## 더 적절한 주장

> HCX tokenizer는 본 corpus에서 Qwen 대비 한국어 token compression이 X% 높았다.

그리고:

> 동일한 Qwen backbone에서 Korean tokenizer를 적용했을 때 compression과 BPB가 각각 X%, Y% 변화했다.

이렇게 나눠야 한다.

---

# 58. Experiment Tracking

각 run마다 config를 저장한다.

```yaml
run_id: koqwen_sub_mean_50m_seed42

model:
  name: qwen2.5-0.5b
  revision: MODEL_COMMIT_HASH

tokenizer:
  version: kosub_v3
  vocab_size: 00000
  sha256: TOKENIZER_HASH

data:
  manifest_sha256: DATA_HASH

training:
  seed: 42
  learning_rate: 0.0001
  sequence_length: 1024
  target_tokens: 50000000
  precision: bf16

hardware:
  gpu: RTX 5070 Ti
  vram_gb: 16

software:
  python: ...
  pytorch: ...
  transformers: ...
  cuda: ...

git:
  commit: ...
```

---

# 59. Data Lineage

각 결과에서 원본까지 추적 가능해야 한다.

```text
Result
  ↓
Experiment Config
  ↓
Tokenizer Version
  ↓
Dataset Manifest
  ↓
Raw Dataset
  ↓
Git Commit
```

저장:

```text
Dataset Manifest SHA256
Tokenizer SHA256
Configuration SHA256
Git Commit Hash
Model Revision
```

---

# 60. Reproducibility

Seed:

```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

그리고 환경 정보를 저장한다.

```text
CUDA version
PyTorch version
Transformers version
Tokenizers version
bitsandbytes version
Flash Attention version
GPU driver
```

---

# 61. 실험 이름 규칙

예:

```text
tok_qwen_original_v1

tok_kosub_v1

align_kosub_mean_5m_seed42

cpt_original_50m_seed42

cpt_kosub_50m_seed42

sys_kosub_rawprompt_10k_v1
```

이름만 보고도 어떤 실험인지 알 수 있게 한다.

---

# 62. 최소 실험 세트

## Phase 1 — Tokenizer

```text
T0 Original
T1 Extend
T2 Substitute
T3 New
```

CPU benchmark.

---

# 63. Phase 2 — Embedding

Tokenizer 단계의 top 후보로:

```text
E0 Random
E1 Mean
E2 Weighted
```

를 먼저 수행한다.

Distillation은 구현 비용이 높기 때문에 이후 추가.

---

# 64. Phase 3 — 0.5B CPT

```text
C0 Original
C1 Extend
C2 Substitute
C3 New
```

동일 training budget.

---

# 65. Phase 4 — Final Candidates

Top 1~2개:

```text
50M+ tokens
×
3 seeds
```

로 최종 검증.

---

# 66. Phase 5 — Scale Validation

```text
Qwen2.5-1.5B

Original
VS
Best Korean Tokenizer
```

---

# 67. Phase 6 — External Comparison

```text
HCX SEED
A.X
```

와:

```text
Tokenizer compression
Vocabulary
Sequence length
Inference
Architecture
```

를 비교한다.

---

# 68. 추가 Deep Ablation

0.5B를 활용하면 더 깊은 실험이 가능하다.

## Parameter Update Scope

```text
A0 Full CPT

A1 Embedding Only

A2 Embedding + Attention

A3 Embedding + FFN

A4 Attention Only

A5 FFN Only
```

연구 질문:

> Tokenizer 변경에 적응할 때 Transformer의 어느 모듈이 가장 많이 수정되어야 하는가?

---

# 69. 왜 이 실험이 중요한가

새 tokenizer가 들어오면 가장 먼저:

```text
Embedding
```

의 의미가 바뀐다.

하지만 이 변화가 Transformer 깊은 layer까지 전파될 가능성이 있다.

예를 들어:

```text
Embedding mismatch
      ↓
Attention input distribution 변화
      ↓
Q/K/V distribution 변화
      ↓
Attention pattern 변화
      ↓
MLP activation 변화
      ↓
Final logits 변화
```

이를 직접 측정할 수 있다.

---

# 70. Layer-wise Representation Analysis

Tokenizer 교체 전후 hidden representation을 layer별로 비교한다.

```text
Embedding
Layer 0
Layer 4
Layer 8
...
Final Layer
```

측정:

```text
Cosine Similarity
Hidden-state MSE
CKA
Representation Drift
```

연구 질문:

> Tokenizer mismatch의 영향은 embedding에서 가장 큰가, 아니면 deeper layers까지 증폭되는가?

---

# 71. Representation Recovery

Alignment/CPT가 진행될수록:

```text
Original Representation

vs

Tokenizer Surgery Representation
```

의 거리가 어떻게 바뀌는지 볼 수 있다.

예:

```text
CKA Similarity
↑
│          ______
│        /
│      /
│____/
│
└────────────────→ CPT Tokens
```

이런 그래프는 포폴에 상당히 좋다.

---

# 72. Gradient Analysis

Tokenizer surgery 이후:

```text
Embedding Gradient Norm
Attention Gradient Norm
FFN Gradient Norm
Layer-wise Gradient Norm
```

을 기록한다.

연구 질문:

> 어떤 module에 가장 큰 adaptation pressure가 발생하는가?

---

# 73. Gradient 결과 예

```text
Embedding      ███████████
Layer 0 Attn   ███████
Layer 0 FFN    ████

...

Layer 20 Attn  ██
Layer 20 FFN   ██
```

처럼 시각화할 수 있다.

이걸 통해:

> Tokenizer replacement가 low layer에 더 큰 영향을 미친다.

같은 가설을 검증할 수 있다.

단, 결과를 먼저 가정하면 안 되고 실제 실험으로 확인한다.

---

# 74. Attention Analysis

같은 raw sentence를:

```text
Original Tokenizer
KoTokenizer
```

로 넣었을 때 attention length 자체가 다르다.

예:

```text
Original
24 tokens

KoTokenizer
16 tokens
```

Attention matrix는:

```text
24 × 24

vs

16 × 16
```

이 된다.

---

# 75. Attention FLOPs

Self-Attention의 주요 attention score 계산은 sequence length \(n\)에 대해 대략:

\[
O(n^2d)
\]

영향을 받는다.

Tokenizer compression으로:

```text
n_original = 1000

n_korean = 700
```

이라면 attention matrix 영역은:

```text
1,000,000

vs

490,000
```

으로 줄어든다.

다만 실제 전체 Transformer FLOPs가 정확히 같은 비율로 줄어드는 것은 아니다.

FFN 계산은 sequence length에 선형으로 비례하고, kernel overhead 등도 있기 때문이다.

따라서 **실측 latency**를 함께 제시해야 한다.

---

# 76. KV Cache

Autoregressive inference에서 KV cache는 대략:

```text
Layers
×
Sequence Length
×
KV Heads
×
Head Dimension
×
Data Type
```

에 비례한다.

따라서 model architecture가 동일하면 tokenizer compression으로 sequence length가 줄어들 때 KV cache도 거의 선형적으로 줄어든다.

---

# 77. 이론값과 실측값을 구분

결과에는:

```text
Theoretical Attention Reduction

Estimated KV Cache

Measured VRAM

Measured Prefill Latency
```

를 따로 표시한다.

이게 중요하다.

이론상 50% 줄었다고 latency가 반드시 50% 줄지는 않기 때문이다.

---

# 78. Training Efficiency도 측정

Tokenizer가 효율적이면 동일 raw corpus의 token 수가 줄어든다.

그래서 training에서도:

```text
tokens/sec
raw chars/sec
raw bytes/sec
```

를 모두 기록한다.

특히 `tokens/sec`만 보면 tokenizer가 다른 모델끼리 공정하지 않을 수 있다.

---

# 79. Raw Bytes/sec

예:

```text
Qwen
85,000 tok/s
220 KB raw text/s

KoTokenizer
82,000 tok/s
305 KB raw text/s
```

이라면 token throughput은 비슷하거나 조금 낮아도 **실제로 처리하는 원문 정보량은 더 높을 수 있다.**

이게 tokenizer 연구에서 좋은 systems metric이다.

---

# 80. Training Budget 관리

처음부터 50M~100M tokens를 모든 후보에 사용하지 않는다.

### Stage 1

```text
1M~5M tokens
```

목적:

```text
코드 버그 검증
Loss 정상 감소 확인
VRAM 확인
Alignment 방법 후보 제거
```

### Stage 2

```text
10M~20M
```

목적:

```text
후보 간 trend 확인
```

### Stage 3

```text
50M+
```

최종 후보만 수행.

---

# 81. Fail-fast Strategy

예를 들어 새 tokenizer가:

```text
Korean compression +3%
English degradation +15%
```

라면 GPU 학습으로 보내지 않는다.

또 embedding alignment에서:

```text
Mean
Weighted
Random
```

이 1M token부터 차이가 거의 없다면 과도한 추가 ablation을 줄일 수 있다.

---

# 82. Evaluation Frequency

너무 자주 validation하면 학습 효율이 떨어진다.

예:

```text
Every X training tokens
```

기준으로 한다.

예를 들면:

```text
1M
2M
5M
10M
20M
50M
```

처럼 중요한 지점에서 평가한다.

---

# 83. Checkpoint 저장

모든 step을 저장하지 않는다.

예:

```text
checkpoint_1m
checkpoint_5m
checkpoint_10m
checkpoint_20m
checkpoint_50m
best_dev
```

정도로 관리한다.

---

# 84. 실험 로그

각 checkpoint마다:

```text
Train Loss
Dev Loss
Dev BPB
LR
Gradient Norm
Peak VRAM
Tokens Seen
Raw Bytes Seen
Elapsed Time
```

를 저장한다.

---

# 85. Visualization

최종적으로 최소 다음 그래프를 만든다.

```text
1. Domain별 Tokens/Character
2. Sequence Length Distribution
3. Vocab Frequency Distribution
4. Alignment BPB Curve
5. CPT BPB Curve
6. Korean vs English Regression
7. Layer-wise Representation Similarity
8. Layer-wise Gradient Norm
9. Prefill Latency vs Raw Input Length
10. KV Cache vs Raw Input Length
11. Peak VRAM
12. Raw Bytes/sec
```

---

# 86. 대표 그래프 1

## Tokenizer Compression

```text
Tokens / Character ↓

Qwen       ██████████
HCX        ███████
A.X        ███████
Ours       ██████
```

---

# 87. 대표 그래프 2

## Alignment Recovery

```text
Korean BPB ↓

3.0 | Random
    |\
2.5 | \
    |  \
2.0 |   \____
    |
1.5 | Mean
    | \______
    |
1.0 +────────────────────
      0   1M   5M   10M
```

---

# 88. 대표 그래프 3

## Sequence Length Distribution

```text
CDF
 ↑
 │                 KoTokenizer
 │              ___/
 │           __/
 │        __/
 │     __/       Qwen
 │  __/       __/
 │_/_______ _/____________→ Tokens
```

---

# 89. 대표 그래프 4

## Performance Trade-off

```text
                Korean BPB
                    ↓
        Better  ←──────→ Worse

English
Regression
  ↑

  |
  |               ● Ko-New
  |
  |       ● Ko-Sub
  |
  | ● Original
  |
```

한국어 최적화와 multilingual degradation의 Pareto frontier를 볼 수도 있다.

---

# 90. Expected Final Tables

## Table 1 — Tokenizer

| Tokenizer | Vocab | Tok/Char ↓ | Byte/Tok ↑ | P95 Len ↓ |
|---|---:|---:|---:|---:|
| Qwen | | | | |
| HCX | | | | |
| A.X | | | | |
| Ours | | | | |

---

# 91. Table 2 — Embedding Alignment

| Init | 0M BPB | 1M | 5M | 10M |
|---|---:|---:|---:|---:|
| Random | | | | |
| Mean | | | | |
| Weighted | | | | |
| Distillation | | | | |

---

# 92. Table 3 — CPT

| Tokenizer | Korean BPB ↓ | English BPB ↓ | Code BPB ↓ |
|---|---:|---:|---:|
| Original | | | |
| Extend | | | |
| Substitute | | | |
| New | | | |

---

# 93. Table 4 — Capability

| Model | Korean ↑ | English ↑ | Code ↑ |
|---|---:|---:|---:|
| Original | | | |
| Ko-Qwen | | | |

---

# 94. Table 5 — System Efficiency

| Model | Tokens ↓ | TTFT ↓ | Prefill ↓ | VRAM ↓ | tok/s ↑ |
|---|---:|---:|---:|---:|---:|
| Original | | | | | |
| Ko-Qwen | | | | | |

---

# 95. Table 6 — Ablation

| Pruning | Korean Merge | Mean Init | CPT | BPB ↓ |
|---|---|---|---|---:|
| ❌ | ❌ | ❌ | ✅ | |
| ✅ | ❌ | ❌ | ✅ | |
| ✅ | ✅ | ❌ | ✅ | |
| ✅ | ✅ | ✅ | ✅ | |

---

# 96. Repository Structure

```text
kotokenlab/

├── configs/
│   ├── tokenizer/
│   ├── alignment/
│   ├── cpt/
│   └── evaluation/
│
├── data/
│   ├── manifests/
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── normalize.py
│   │   ├── dedup.py
│   │   ├── split.py
│   │   └── preprocessing.py
│   │
│   ├── tokenizer/
│   │   ├── train.py
│   │   ├── analyze_vocab.py
│   │   ├── prune.py
│   │   └── substitute.py
│   │
│   ├── surgery/
│   │   ├── resize.py
│   │   ├── init_random.py
│   │   ├── init_mean.py
│   │   ├── init_weighted.py
│   │   └── distillation.py
│   │
│   ├── training/
│   │   ├── alignment.py
│   │   ├── cpt.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── tokenizer_eval.py
│   │   ├── bpb.py
│   │   ├── capability.py
│   │   ├── representation.py
│   │   ├── latency.py
│   │   └── memory.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── hashing.py
│       └── tracking.py
│
├── tests/
│
├── experiments/
│   ├── tokenizer/
│   ├── alignment/
│   ├── cpt/
│   └── system/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── scripts/
│
├── README.md
└── pyproject.toml
```

---

# 97. Development Order

## Step 1 — Data Pipeline

제일 먼저:

```text
Raw Corpus
↓
Normalize
↓
Dedup
↓
Manifest
↓
Split
```

을 완성한다.

**모델 학습보다 이걸 먼저 해야 한다.**

---

# 98. Step 2 — Tokenizer Benchmark Framework

한 명령으로:

```text
Qwen
HCX
A.X
Custom
```

tokenizer를 같은 dataset에서 평가할 수 있도록 한다.

예:

```bash
python -m src.tokenizer_eval \
    --config configs/evaluation/tokenizer.yaml
```

---

# 99. Step 3 — Tokenizer Trainer

구현:

```text
Extend
Substitute
New BBPE
```

이때 tokenizer 자체를 versioning한다.

예:

```text
kosub_v1
kosub_v2
kosub_v3
```

---

# 100. Step 4 — Candidate Gate

Tokenizer 후보를 GPU에 보내기 전에:

```text
Compression
Fragmentation
English Regression
Code Regression
```

으로 거른다.

---

# 101. Step 5 — Qwen 0.5B Tokenizer Surgery

구현해야 하는 핵심:

```text
Tokenizer replacement
Embedding resize
LM head 처리
Weight tying 확인
Special token 보존
Token ID mapping
Checkpoint 저장/로드
```

---

# 102. Special Token 검증

새 tokenizer에서도:

```text
BOS
EOS
PAD
UNK
Chat Template Token
Special Control Tokens
```

의 ID와 의미가 깨지지 않도록 해야 한다.

실수하기 쉬운 부분이다.

---

# 103. Token ID Mapping

특히 substitution에서는:

```text
Old Token ID
→
New Token
```

mapping table을 명시적으로 저장한다.

예:

```json
{
  "token_id": 48322,
  "old_token": "...",
  "new_token": "정보처리"
}
```

그래야 embedding initialization을 재현할 수 있다.

---

# 104. Step 6 — Embedding Init Ablation

먼저:

```text
Random
Mean
Weighted
```

세 개만 구현한다.

이걸로 baseline을 만든 뒤 Distillation을 추가한다.

---

# 105. Step 7 — Embedding Alignment

Transformer freeze 상태에서:

```text
Embedding
LM Head
```

위주로 적응.

여기서 alignment loss가 제대로 감소하는지 확인한다.

---

# 106. Step 8 — 0.5B Full CPT

이제 전체 parameter를 unfreeze.

```text
Embedding
Attention
FFN
Norm
```

모두 학습.

---

# 107. Step 9 — Evaluation Pipeline Freeze

본 실험 전에 evaluation metric을 고정한다.

중간에:

```text
이 지표가 결과가 더 좋게 나오네
```

라는 이유로 metric을 계속 추가/변경하면 실험 신뢰도가 떨어진다.

필요한 metric 변경은 version으로 관리한다.

---

# 108. Step 10 — Final Multi-seed

Top 후보만:

```text
42
123
2026
```

로 반복.

---

# 109. Step 11 — 1.5B Validation

0.5B에서 이미 결론 난 항목을 전부 반복하지 않는다.

핵심 결과만 scale-up.

---

# 110. Step 12 — System Benchmark

최종 모델을 RTX 5070 Ti에서:

```text
Raw Prompt
Equal Tokens
Short Context
Medium Context
Long Context
```

로 평가한다.

---

# 111. 시스템 benchmark 입력

예:

```text
Raw Korean characters

500
1,000
2,000
5,000
10,000
```

각 길이에서:

```text
token count
TTFT
prefill
VRAM
```

을 측정한다.

---

# 112. Portfolio Story

최종 프로젝트 이야기는 다음 흐름으로 만든다.

```text
Problem

한국어는 multilingual tokenizer에서
상대적으로 많은 token을 소비할 수 있다.

        ↓

Hypothesis

한국어 특화 tokenizer를 사용하면
sequence length와 inference cost를 줄일 수 있다.

        ↓

New Problem

하지만 pretrained tokenizer를 교체하면
token semantics와 embedding이 불일치한다.

        ↓

Solution

Vocabulary Surgery
+
Embedding Alignment
+
Continued Pretraining

        ↓

Validation

Language Modeling
+
Capability
+
System Efficiency
+
Cross-domain Regression

        ↓

Result

한국어 compression 개선이
quality / compute / latency에
어떤 trade-off를 발생시키는지 정량적으로 분석
```

---

# 113. 최종 README 상단 예시

```text
KoTokenLab
Korean Tokenizer Surgery on RTX 5070 Ti 16GB

Qwen2.5-0.5B
        ↓
Vocabulary Surgery
        ↓
Embedding Alignment
        ↓
Continued Pretraining
        ↓
Ko-Qwen

----------------------------------

Korean Tokens        -XX.X%
P95 Sequence Length  -XX.X%
Korean BPB           -X.X%
English Regression   +X.X%
Code Regression      +X.X%

Prefill Latency      -XX.X%
TTFT                 -XX.X%
Peak VRAM            -XX.X%
```

---

# 114. 포폴에서 가장 중요한 결과

최고 benchmark 하나보다 이런 결과가 좋다.

예:

> Korean tokenizer reduced token count by **31.2%** on the held-out Korean corpus.

> Direct tokenizer replacement significantly degraded pretrained language modeling performance.

> Mean-based embedding initialization reduced initial BPB degradation by **X% compared with random initialization**.

> Following controlled CPT, the Korean tokenizer achieved **X% lower Korean BPB** while keeping English and code degradation below **Y%**.

> On an RTX 5070 Ti, the shorter tokenized context reduced measured prefill latency by **Z%** on equal raw Korean prompts.

이 스토리가 완성되면 된다.

---

# 115. 이 프로젝트에서 공부할 Tokenizer 개념

```text
BPE
BBPE
Vocabulary
Merge Rule
Unicode
UTF-8
Token Fertility
Compression
Byte Fallback
Special Tokens
```

---

# 116. Embedding

```text
Token Embedding
Vocabulary Resize
Embedding Initialization
LM Head
Weight Tying
Semantic Alignment
Embedding Distillation
```

---

# 117. Transformer

```text
Self Attention
Q / K / V
GQA
RoPE
RMSNorm
SwiGLU
Residual Connection
Attention Mask
Causal LM
```

---

# 118. Training

```text
Continued Pretraining
Causal LM Loss
AdamW
Learning Rate Scheduler
Gradient Accumulation
Gradient Checkpointing
BF16
Catastrophic Forgetting
```

---

# 119. PEFT

```text
LoRA
QLoRA
NF4
Quantization
Adapter
```

---

# 120. Systems

```text
Attention Complexity
Flash Attention
KV Cache
Memory Bandwidth
VRAM
Prefill
Decode
TTFT
Throughput
```

---

# 121. Evaluation

```text
Perplexity
BPB
BPC
Bootstrap CI
Multi-seed
Ablation
Equal-data Experiment
Equal-compute Experiment
Regression Evaluation
```

---

# 122. 핵심 원칙

**1. Split first, tokenize later.**

Train/Test split 이전에 tokenizer 학습을 하면 안 된다.

**2. Final Test는 마지막까지 보지 않는다.**

Tokenizer 설계와 checkpoint 선택에 사용하지 않는다.

**3. 다른 tokenizer의 PPL을 그대로 비교하지 않는다.**

BPB/BPC를 중심으로 비교한다.

**4. Qwen ↔ HCX/A.X의 성능 차이를 tokenizer 효과라고 주장하지 않는다.**

Backbone과 학습 데이터가 다르다.

**5. Tokenizer의 인과 효과는 동일 Qwen backbone으로 검증한다.**

```text
Qwen Original
vs
Qwen + Korean Tokenizer
```

**6. Original Tokenizer + CPT control을 반드시 둔다.**

그래야 추가 학습 효과와 tokenizer 효과를 분리할 수 있다.

**7. Equal Raw Data와 Equal Token Budget을 모두 실험한다.**

두 실험의 의미가 다르다.

**8. Tokenizer Surgery / Embedding Alignment / CPT를 분리한다.**

각 단계의 효과를 독립적으로 분석한다.

**9. GPU benchmark는 warm-up + CUDA synchronize를 사용한다.**

**10. Exploration은 1 seed, Final은 multiple seeds.**

**11. 모든 실험을 hash/version으로 추적한다.**

**12. 0.5B에서 깊게, 1.5B에서 scale validation한다.**

---

# 123. 프로젝트의 최종 질문

프로젝트가 끝났을 때 아래 질문에 **실험 데이터로 답할 수 있어야 한다.**

### Q1

한국어 tokenizer를 더 효율적으로 만들면 실제 token 수가 얼마나 줄어드는가?

### Q2

그 효과가 뉴스에만 나타나는가, 아니면 community / technical / noisy Korean에도 나타나는가?

### Q3

영어와 코드에는 어떤 regression이 발생하는가?

### Q4

Pretrained tokenizer를 교체했을 때 모델 성능은 왜 무너지는가?

### Q5

새 token의 embedding을 random으로 초기화하는 것과 기존 token embedding으로 초기화하는 것은 얼마나 차이가 나는가?

### Q6

Embedding alignment만으로 pretrained knowledge를 얼마나 회복할 수 있는가?

### Q7

Full CPT 이후에도 original tokenizer보다 language modeling quality가 좋아지는가?

### Q8

동일 raw data 기준에서도 이득이 있는가?

### Q9

동일 token compute budget 기준에서도 이득이 있는가?

### Q10

한국어 token 감소가 실제 Attention workload에 어떤 영향을 미치는가?

### Q11

KV cache가 실제로 얼마나 줄어드는가?

### Q12

RTX 5070 Ti에서 실제 TTFT / Prefill / VRAM이 얼마나 개선되는가?

### Q13

Tokenizer surgery 후 어떤 Transformer layer가 가장 크게 변하는가?

### Q14

Embedding / Attention / FFN 중 어떤 parameter가 tokenizer 변경에 가장 크게 적응하는가?

---

# 124. 최종 목표

이 프로젝트의 목표는 단순히:

> **"한국어 성능이 좋은 모델 하나를 만들었다."**

가 아니다.

목표는:

> **"Pretrained LLM의 tokenizer를 변경했을 때 발생하는 representation mismatch를 분석하고, embedding alignment와 continued pretraining을 통해 이를 복구했으며, tokenizer compression의 효과를 language modeling quality와 실제 GPU inference efficiency 양쪽에서 통제된 실험으로 검증했다."**

라고 설명할 수 있는 포트폴리오를 만드는 것이다.

---

## 최종 구조

```text
Tokenizer
   ↓
Embedding
   ↓
Transformer
   ↓
Continued Pretraining
   ↓
Language Modeling
   ↓
Capability
   ↓
GPU Systems
```

를 하나의 프로젝트에서 끝까지 연결한다.

특히 **RTX 5070 Ti 16GB** 환경에서는:

```text
Qwen2.5-0.5B
→ 깊은 Full CPT / Ablation

Qwen2.5-1.5B
→ Scale Validation

HCX SEED
→ Korean-specialized External Baseline

A.X
→ Qwen → Korean Adaptation 산업 사례
```

로 역할을 분리한다.

이렇게 가면 단순한 **"LLM 파인튜닝 포폴"**보다 훨씬 깊게 보여줄 수 있고, 면접에서도 토크나이저에서 시작해서 **Embedding → Attention → KV Cache → CPT → 평가 설계 → GPU inference**까지 자연스럽게 설명할 수 있는 프로젝트가 된다.