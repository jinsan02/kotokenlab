# 선행 연구

조사 시점 2026-08-29. 이 프로젝트가 **이미 답이 나온 문제에 GPU 를 쓰지 않도록**,
그리고 **무엇이 실제로 새로운지** 분명히 하려고 정리했다.

결론부터: **핵심 아이디어(한국어 토크나이저 교체 + 임베딩 초기화 + CPT)는 이미
여러 번 다뤄졌다.** 남는 차별점은 세 가지이며 5절에 정리했다.

---

## 1. 한국어에 직접 겹치는 연구

### EEVE — Efficient and Effective Vocabulary Expansion
[arXiv:2402.14714](https://arxiv.org/html/2402.14714v1)

영어 중심 LLM(SOLAR-10.7B, Phi-2)에 한국어 vocabulary 를 확장하고, **파라미터 동결 +
subword 기반 초기화**를 단계적으로 적용한다.

> **이 프로젝트의 T1(Extend) + §25 Embedding Alignment 와 골격이 거의 같다.**
> 가장 가까운 선행 연구다. E0/E1/E2 초기화 비교와 "임베딩만 먼저 학습" 전략은
> 여기서 이미 검증됐다고 봐야 한다.
>
> **2026-08-31 갱신** — T1 은 폐기했고 Embedding Alignment 도 폐기했다. 우리
> 환경(Qwen2.5-0.5B, `tie_word_embeddings=True`, 임베딩 비중 27.6%)에서는
> "임베딩만 먼저 학습" 이 손해를 안 내면서 제 일을 하는 lr 이 없었다
> ([`DESIGN_DELTA.md`](DESIGN_DELTA.md) 1-5). EEVE 는 10.7B 급이고 임베딩
> 비중이 훨씬 작다 — **결론이 갈리는 것이 스케일 때문인지 tie 때문인지는
> 우리 데이터로 가릴 수 없다.** 재현 실패로 주장하지 말고 조건 차이를 명시한다.

### Optimizing Korean-Centric LLMs via Token Pruning
[arXiv:2604.16235](https://arxiv.org/pdf/2604.16235) (2026-04, Hoyeol Kim · Hyeonwoo Kim)

Qwen3 / Gemma-3 / Llama-3 / Aya 에 대해 vocabulary 를 Original / En-Ko / En-Ko-Zh
세 가지로 잘라내고 벤치마크한다. 언어 혼동 감소, 한국어 기계번역 성능 향상,
**추론 지연은 소폭 개선**을 보고했다.

> **T2(Substitution)와 정면으로 겹친다.** 다만 결정적 차이가 있다:
> 이들은 **잘라내기만 하고 vocab 크기를 줄인다.** 우리 T2 는 **vocab 크기를 유지한 채
> 그 자리를 한국어 고효율 토큰으로 치환**한다 (스펙 §12). 연구 질문 자체가 다르다 —
> "임베딩 파라미터를 줄이는가" vs "같은 파라미터로 압축률을 높이는가".
> 이 대비를 결과에 명시적으로 써야 한다.

### RedWhale — An Adapted Korean LLM Through Efficient Continual Pretraining
[arXiv:2408.11294](https://arxiv.org/pdf/2408.11294)

한국어 특화 토크나이저 + 사전학습 가중치 초기화 + 효율적 CPT.

### HanjaBridge
[arXiv:2507.10920](https://arxiv.org/pdf/2507.10920)

한자 정보를 주입해 한국어 의미 모호성을 해소. 토크나이저보다는 표현 쪽이지만,
한국어 CPT 의 최근 흐름으로 참고.

### Qwen-Tokenizer-Pruner (코드)
[github.com/KaihuaTang/Qwen-Tokenizer-Pruner](https://github.com/KaihuaTang/Qwen-Tokenizer-Pruner)

Qwen 의 151,936 vocab 을 잘라내는 실제 구현. **T2 구현 시 재사용 후보다.**
Qwen 토크나이저는 BPE 라서 단어만 추가해서는 확장되지 않고 중간 merge rule 이
필요하다는 점도 여기와 [Qwen 공식 문서](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md)에
정리되어 있다 — T1(Extend) 구현에서 가장 흔한 함정이다.

---

## 2. 임베딩 초기화 — 스펙 §21~23 에 대응

| 방법 | 논문 | 스펙 대응 |
|---|---|---|
| **FVT** (Fast Vocabulary Transfer) | 분해 토큰 임베딩 평균 | **E1 Mean** 과 동일 |
| **FOCUS** | 겹치는 토큰들의 **희소 선형결합**으로 새 토큰 표현 | **E2 Weighted** 의 정교한 버전 |
| **Learned Embedding Propagation** | [arXiv:2412.21140](https://arxiv.org/pdf/2412.21140) (러시아어) | E3 계열 |
| **ALM / Cross-Tokenizer Distillation** | [arXiv:2503.20083](https://arxiv.org/pdf/2503.20083) | **E3 Distillation** |
| **OMP 이식 (학습 불필요)** | [arXiv:2506.06607](https://arxiv.org/pdf/2506.06607) | E2/E3 사이 |

**시사점**: `random < mean/FVT` 는 여러 논문이 반복 확인했다. E0 를 baseline 으로
두되 **Stage 1(1~5M 토큰)에서 빨리 통과시키고**, 절약한 예산을 FOCUS 수준의 E2 와
실제로 열려 있는 질문(4절)에 쓴다.

---

## 3. 토크나이저 교체 자체를 다루는 흐름

### Zero-Shot Tokenizer Transfer (ZeTT)
[arXiv:2405.07883](https://arxiv.org/abs/2405.07883) ·
[github.com/bminixhofer/zett](https://github.com/bminixhofer/zett)

**하이퍼네트워크가 토크나이저를 입력받아 임베딩을 예측한다.** 학습 없이 토크나이저를
교체하면서 원 모델 성능에 근접하고 시퀀스 길이를 크게 줄인다. 사전학습된
하이퍼네트워크가 HuggingFace 에 공개되어 있다.

> 이 프로젝트의 §20~25 전체(surgery → init → alignment)를 우회하는 접근이다.
> **정면으로 경쟁하지 말고 강한 baseline 으로 넣는 편이 낫다** — "우리 E1/E2 가
> ZeTT 대비 어디에 있는가"를 보이면 결과가 훨씬 설득력 있다.
>
> **2026-08-31 갱신** — 이 비교의 무게가 커졌다. 우리 E1(부품 평균)은 통제된
> CPT 후에도 C0 를 따라잡지 못했다(한국어 BPB 1.5671 vs 1.1375). 하이퍼네트워크
> 초기화가 그 간격을 메우는지가 곧 "초기화로 될 일인가, 노출의 문제인가" 에
> 대한 외부 증거가 된다. 우리 진단은 노출 쪽이다 — 새 토큰 중앙값 발화 143회
> ([`../reports/tables/cpt_main.md`](../reports/tables/cpt_main.md)).

### Teaching Old Tokenizers New Words
[arXiv:2512.03989](https://arxiv.org/html/2512.03989v2)

### AdaptiVocab
[arXiv:2503.19693](https://arxiv.org/pdf/2503.19693) — 도메인 특화 경량 vocab 적응

### Achieving Tokenizer Flexibility through Heuristics
[arXiv:2505.09738](https://arxiv.org/pdf/2505.09738)

### 언어별 사례
- [Bielik v3 (폴란드어) — 토크나이저 최적화](https://arxiv.org/pdf/2604.10799)
- [BrahmicTokenizer-131K (인도어 계열)](https://arxiv.org/pdf/2605.29379)
- [Exploring Design Choices for Building Language-Specific LLMs](https://arxiv.org/pdf/2406.14670)

---

## 4. 이 프로젝트가 실제로 새로운 지점

선행 연구를 훑고 나면 **뭐가 이미 답이 나왔고 뭐가 안 나왔는지**가 갈린다.

### 이미 답이 나온 것 — 예산을 적게 쓴다

- random 초기화는 나쁘다, mean/FVT 가 낫다 → **Stage 1 에서 확인만 하고 넘어간다**
- 토크나이저 확장은 CPT 없이는 무너진다 → 재확인 대상이지 발견 대상이 아니다
- 한국어 특화 토크나이저는 토큰 수를 크게 줄인다 → A.X 가 이미 보여준다 (§6 사전 점검)

### 아직 열려 있는 것 — 여기에 집중한다

**(1) 품질 개선과 실측 시스템 이득을 같은 실험에서 잇는 연구가 드물다.**
대부분 BPB / 벤치마크까지만 보고한다. Token Pruning 논문조차 지연은 "modest gains"
정도로만 언급한다. 스펙 §43~50 의 Level 4 — warm-up + `cuda.synchronize()` 로
prefill / TTFT / decode 를 분리 측정하고, KV cache **추정값과 실측 VRAM 을 구분해서**
제시하는 것(§77) — 이 조합이 실제 차별점이다.

**(2) Equal Raw Data 와 Equal Token Budget 을 둘 다 돌리는 연구가 거의 없다** (§31~33).
같은 원문만 보면 compute 가 다르고, 같은 토큰 예산만 보면 본 정보량이 다르다.
둘 다 있어야 trade-off 가 나온다.

**(3) "왜 무너지는가"를 layer 단위로 여는 분석이 부족하다** (§70~73).
토크나이저 교체 후 layer-wise representation drift(CKA)와 모듈별 gradient norm 을
CPT 진행에 따라 추적하는 작업. `train_curve.tsv` 에 `grad_norm_emb / attn / ffn` 을
처음부터 넣어둔 이유다.

**(4) vocab 크기를 유지한 치환(T2)** 은 Token Pruning 계열(크기 축소)과 다른 질문이다.

### 반면 주의할 것

- **0.5B 규모의 결론이 큰 모델로 일반화된다고 주장하면 안 된다.** 스펙 §54 의
  1.5B scale validation 이 있어야 "방향성이 유지된다" 정도까지 말할 수 있다.
- **ZeTT 를 baseline 에 넣지 않으면** "왜 하이퍼네트워크 대신 CPT 를 했나"라는
  질문에 답할 수 없다.
- Token Pruning 논문(2026-04)이 **Qwen3** 를 쓴다. 우리는 Qwen2.5 다. 백본이 다르니
  숫자를 직접 비교하지 않는다 ([RULES.md](RULES.md) 4번과 같은 이유).

---

## 5. 포지셔닝 한 문장

> 한국어 토크나이저 교체의 **인과 효과**를 동일 Qwen2.5 백본 위에서
> `Original + CPT` 통제군과 함께 분리 측정하고, 그 효과를
> **언어 모델링 품질(BPB) · 도메인별 regression · 실측 GPU 추론 효율** 세 축에서
> 동시에, 그리고 **동일 원문 / 동일 토큰 예산 두 조건 모두에서** 정량화한다.

선행 연구는 이 축들을 각각 따로 다뤘다. 하나의 통제된 파이프라인으로 잇는 것이
이 프로젝트의 기여다.

---

## 6. 읽는 순서 (구현 전)

1. **EEVE** — T1 + alignment 의 기준선 (둘 다 우리는 폐기했다. 조건 차이를 확인할 것)
2. **Qwen tokenization_note + Qwen-Tokenizer-Pruner** — T1/T2 구현 함정
3. **Token Pruning (2604.16235)** — T2 와의 차별점 확정용
4. **ZeTT** — baseline 으로 넣을지 결정
5. **FOCUS / OMP** — E2 를 어디까지 정교하게 할지 결정
