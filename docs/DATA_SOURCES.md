# 데이터셋 조사와 적용 계획

스펙 §5(Corpus Composition)와 §97(Step 1)을 실제 데이터셋에 매핑한 결과다.
조사 시점 2026-08-29, HuggingFace Hub API 로 직접 확인했다.

---

## 1. 결론 — 채택

| 도메인 (스펙 §5) | 데이터셋 | 게이팅 | 라이선스 | 역할 |
|---|---|---|---|---|
| **웹 일반 (주력)** | `HuggingFaceFW/fineweb-2` `kor_Hang` | 없음 | ODC-By 1.0 | Train/Dev 의 대부분 |
| **웹 보강 (교차검증)** | `HPLT/HPLT2.0_cleaned` `kor_Hang` | 없음 | CC0 (문서별) | 다른 크롤 소스 → near-dup 검증 |
| **백과** | `wikimedia/wikipedia` `20231101.ko` | 없음 | CC BY-SA | 정제된 문어체 |
| **뉴스** | `daekeun-ml/naver-news-summarization-ko` | 없음 | — | 도메인 라벨이 확실한 소규모 뉴스 |
| **기술/교육** | `maywell/korean_textbooks` | 없음 | — | 합성 데이터임에 유의 |
| **커뮤니티/웹텍스트** | `HAERAE-HUB/KOREAN-WEBTEXT` | 없음 | — | 구어체·커뮤니티 |
| **대화체** | `jojo0217/korean_safe_conversation`, `beomi/KoAlpaca-v1.1a` | 없음 | — | SFT 셋이지만 구어체 소스로 활용 |
| **영어 대조군** | `HuggingFaceFW/fineweb-edu` (`sample-10BT`) | 없음 | ODC-By | 영어 regression 측정 (§16) |
| **코드** | `codeparrot/github-code-clean` | 없음 | — | 코드 regression 측정 (§16) |
| **Noisy 한국어** | 직접 구축 — 커뮤니티 텍스트에서 오타·띄어쓰기 오류·은어 필터 | — | — | 스펙 §5 의 Noisy Korean |

**게이팅 때문에 제외**: `uonlp/CulturaX`(auto-gated), `oscar-corpus/OSCAR-2301`(manual-gated),
`bigcode/the-stack-smol`·`starcoderdata`(auto-gated). 토큰 승인이 필요해 파이프라인
재현성이 떨어진다. 위 조합만으로 인증 없이 전 과정이 돌아간다.

**AI Hub / 모두의 말뭉치**: 한국 실명 인증 + 신청 절차가 필요해 스크립트로 재현할 수
없다. 필요해지면 수동으로 추가하되, manifest 에 `source` 를 남겨 분리 가능하게 한다.

### 대조군 — 실제로 쓰는 것 (구현 완료)

`scripts/run_control_pipeline.py` 가 한국어 파이프라인과 **같은 manifest 스키마,
같은 `doc_id` 해시 분할**로 만든다. 다른 것은 품질 필터 프로파일뿐이다
(`QualityConfig.for_language`).

| 언어 | 출처 | 파케이 경로 | 필터 차이 |
|---|---|---|---|
| 영어 | `HuggingFaceFW/fineweb-edu` | `sample/10BT/` | 한글 하한 off, 라틴 하한 0.50, 한글 5% 초과 배제 |
| 코드 | `codeparrot/github-code-clean` | `data/train-` | **반복 검사 off**, 길이 하한 120자 |

코드에서 반복 검사를 끄는 이유는 `import` 줄과 보일러플레이트가 정당하게
반복되기 때문이다. 한국어 SEO 스팸 기준을 그대로 적용하면 멀쩡한 소스가 전부
탈락한다.

**이 라벨은 규칙이 아니라 출처다.** 그래서 도메인 라벨 정확도 문제
([DOMAIN_LABELS.md](DOMAIN_LABELS.md))의 영향을 받지 않는다. 스펙 §17 의
영어·코드 regression 판정은 여기서 이뤄진다.

파일럿 규모: 영어 7,996문서(dev 1.62MB), 코드 7,407문서(dev 2.10MB).
목표는 도메인당 dev 5MB 이며 전체 규모 실행에서 맞춘다.

### 호스트 상한 (구현 완료)

`spam_filter.max_docs_per_host: 400` 을 파이프라인이 강제한다. 상한이 없으면
한 사이트의 문체가 토크나이저 통계를 끌고 간다 — 4샤드 조사에서 tripadvisor 가
6.66% 를 차지했고, 상한 적용 후 2.20% 로 떨어지며 870건(4.35%)이 제외됐다.

### Level 3 평가 (스펙 §40)

| 언어 | 벤치마크 |
|---|---|
| 한국어 | `HAERAE-HUB/KMMLU`, `HAERAE-HUB/KMMLU-HARD`, `HAERAE-HUB/HAE_RAE_BENCH_1.1`, `KorQuAD/squad_kor_v1`, `klue/klue` |
| 영어 | `cais/mmlu` |

전부 게이팅 없음. base 모델이므로 log-likelihood / multiple-choice 형식으로 쓴다.

---

## 2. FineWeb-2 를 주력으로 쓰는 이유

`HuggingFaceFW/fineweb-2` 의 `kor_Hang` 서브셋은 파케이 26개(train 25)로 나뉘어 있고,
행 스키마가 이 프로젝트에 정확히 맞는다:

```
text  id  dump  url  date  file_path
language  language_score  language_script  minhash_cluster_size  top_langs
```

- **`url` 이 있다** → 스펙 §5 의 도메인 분류를 URL 호스트로 만들 수 있다.
  FineWeb-2 자체에는 도메인 라벨이 없으므로, 이걸 쓰지 않으면 §16 도메인별 평가가
  불가능하다. **가장 중요한 이유다.**
- **`id` 가 문서 단위 UUID** → 스펙 §6 의 문서 단위 분할이 자연스럽다
- **`minhash_cluster_size`** → 중복 정도가 이미 계산되어 있다. 우리 near-dedup 의
  1차 필터로 쓸 수 있다 (하지만 대체하지는 않는다 — 아래 3절)
- **`language_score`** → 언어 오분류 문서를 거르는 임계값

HPLT 2.0 은 서로 다른 크롤 소스라서, 두 코퍼스 사이의 near-duplicate 비율 자체가
**dedup 파이프라인이 제대로 도는지 검증하는 테스트**가 된다.

---

## 3. 조사하면서 실제로 확인한 함정

### 3.1 한국어 웹 크롤은 자격증 SEO 스팸이 심하다

`HPLT/HPLT2.0_cleaned` `kor_Hang` 의 첫 행이 이랬다:

```
SAP C-SAC-2107 100%시험패스 공부자료 다른 사람들이 모두 취득하고 있는
자격증에 관심도 없는 분은 치열한 경쟁속에서 살아남기 어렵습니다, ...
```

덤프 시험문제 판매 스팸이다. 이런 문서는 어휘가 극도로 반복적이라 **토크나이저
merge rule 을 오염시킨다** — 스팸 상용구가 고빈도 토큰으로 학습되어 vocabulary
예산을 잡아먹는다. T3(New BBPE)에서 특히 치명적이다.

대응: HPLT 의 `doc_scores` / `filter` 필드로 1차 필터 후, 문서 내 n-gram 반복률과
호스트별 문서 수 상한을 추가로 건다.

### 3.2 FineWeb-2 `kor_Hang` 에 한국어가 아닌 문서가 섞여 있다

첫 행이 영어 비둘기 사진 페이지였다. 스크립트 기준 분류라서 한글이 일부만 있어도
포함된다. `language_score` 임계값과 **한글 문자 비율**을 직접 계산해 거른다.

한영 혼용 문서는 버리지 않는다. 다만 **도메인 라벨로 쓰지 않는다** — 블라인드
감사에서 `ko_en_mixed` 는 정밀도·재현율 모두 0% 였다
([DOMAIN_LABELS.md](DOMAIN_LABELS.md)). 대신 manifest 에 `latin_share` 와
`hangul_ratio` 를 **연속값**으로 남겨서 사후에 임의의 구간으로 문서군을 나눈다.
도메인 라벨이 아니라 문서 속성이므로 규칙 정확도 문제를 우회한다.

### 3.3 벤치마크 누출

KMMLU 는 한국 자격시험 문제 기반이고, KorQuAD 는 한국어 위키 기반이다.
**둘 다 웹 크롤과 위키 코퍼스에 원문이 존재할 가능성이 높다.**

스펙 §6 은 Train/Test 누출만 다루지만, 여기에 하나 더 필요하다:

> Level 3 벤치마크 문항과 학습 코퍼스 사이의 n-gram 중복 검사(decontamination)를
> Step 1 에 포함한다.

이걸 안 하면 §41 의 `Korean Δ` 가 tokenizer 효과가 아니라 암기 효과가 된다.

### 3.4 minhash_cluster_size 는 우리 dedup 을 대체하지 않는다

FineWeb-2 의 dedup 은 **덤프 단위**다. 서로 다른 CC 덤프에 같은 문서가 있으면
남아 있다. 우리는 여러 덤프를 섞어 쓰므로 exact + near dedup 을 그대로 돌려야 한다
([RULES.md](RULES.md) 1번).

---

## 4. 규모 산정

한국어 문서에서 대략:

```
1 문자 ≈ 3 바이트 (UTF-8)
Qwen2.5 기준 1 문자 ≈ 0.66 토큰   →  1 토큰 ≈ 4.5 바이트
```

| 목적 | 필요 토큰 | 필요 원문 |
|---|---:|---:|
| Stage 1 스모크 (§80) | 1~5M | ~25 MB |
| Stage 2 추세 확인 | 10~20M | ~100 MB |
| Stage 3 최종 후보 | 50M+ | ~250 MB |
| 토크나이저 학습 (T1~T3) | — | 1~2 GB |
| Dev + Final Test | — | ~200 MB |

**초기 목표: 정제 후 3~5GB.** dedup·필터링 탈락률을 감안해 raw 는 10~15GB 정도
받는다. FineWeb-2 `kor_Hang` 파케이 몇 개면 충분하므로 전체를 받지 않는다.

디스크는 1.6TB 여유가 있어 제약이 아니다. 제약은 **전처리 시간**이다.

---

## 5. Step 1 실행 순서

```
1. 수집       fineweb-2 kor_Hang 일부 샤드 + HPLT 일부 + 위키 + 뉴스/기술/대화
                 → data/raw/  (git 제외)
2. 도메인 라벨  url 호스트 규칙으로 news/blog/community/technical/ko-en/code 분류
                 → 규칙은 configs/data/domain_rules.yaml 에 버전 관리
3. 정규화      Unicode NFC, 공백·제어문자, 반복 문자 축약
4. 품질 필터   한글 비율, language_score, 문서 길이, n-gram 반복률,
                 호스트별 문서 수 상한, HPLT doc_scores
5. Exact dedup  정규화 후 SHA256
6. Near dedup   MinHash + LSH (fineweb-2 minhash_cluster_size 는 참고용)
7. 오염 제거    KMMLU / KorQuAD / HAE-RAE 문항과 n-gram 중복 문서 제거
8. Manifest    data/manifests/{train,dev,final_test}.tsv
                 doc_id source domain date language sha256 split
                 char_count byte_count latin_share hangul_ratio
9. 문서 단위 분할  90 / 5 / 5, doc_id 해시 기반 (순서·개수와 무관)
10. 청킹        분할 이후에만
```

manifest 본체는 커밋하지 않는다 (전체 규모에서 수백 MB). 커밋되는 것은
`data/manifests/SUMMARY.tsv` 와 LEDGER 의 `manifest_sha256` 이다.

**8번까지 끝나고 manifest_sha256 이 확정되기 전에는 토크나이저를 학습하지 않는다**
([RULES.md](RULES.md) 1번).

7번(오염 제거)은 스펙에 없던 항목이다. 3.3 에서 확인한 위험 때문에 추가했다.

---

## 6. 확보한 모델·토크나이저

`scripts/download_models.py` 로 받았다. 캐시는 `.hf_cache/` (git 제외).

| 모델 | 범위 | 크기 | 역할 |
|---|---|---:|---|
| `Qwen/Qwen2.5-0.5B` | 전체 | 953 MB | 핵심 실험 모델 (Full CPT) |
| `Qwen/Qwen2.5-1.5B` | 전체 | 2955 MB | scale validation |
| `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` | 전체 | 1088 MB | 한국어 특화 external baseline |
| `skt/A.X-4.0-Light` | 토크나이저만 | 12 MB | 산업 사례 external reference |
| `skt/A.X-4.0` | 토크나이저만 | 12 MB | 상동 |

스펙 §3 이 지정한 **HyperCLOVAX-SEED-Text-Base-0.5B 는 공개되어 있지 않다.**
`naver-hyperclovax` 는 Instruct 0.5B / 1.5B 만 공개 중이다. 토크나이저는 동일 계열이고,
HCX 는 애초에 인과 실험이 아니라 external reference 이므로(스펙 §56,
[RULES.md](RULES.md) 4번) Instruct 로 대체했다. **BPB 비교에 쓸 때는 instruction-tuned
모델이라는 점을 결과에 명시한다.**

### 사전 점검 (정식 측정 아님)

한국어 한 문장(82자 / 204바이트)에 대한 스모크 체크다. 코퍼스가 없으므로
원장에 기록하지 않았다. Level 1 정식 측정은 Step 2 에서 한다.

| tokenizer | vocab | ko tokens | tok/char | bytes/tok | en tokens | code tokens |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 | 151,665 | 54 | 0.659 | 3.78 | 16 | 25 |
| HCX-SEED | 110,524 | 42 | 0.512 | 4.86 | 16 | 32 |
| A.X-4.0 | 102,402 | 27 | 0.329 | 7.56 | 16 | 32 |

A.X 가 이 문장에서 Qwen 대비 한국어 토큰을 **절반**으로 줄이면서 영어는 동일하고
코드는 28% 늘었다. 스펙 §16 이 경고한 trade-off 가 실제로 관측된다는 신호다.
`Ours` 가 겨냥할 상한선이 대략 어디인지도 보여준다.

---

## 7. 라이선스

`data/` 는 git 에서 제외되므로 코퍼스를 재배포하지 않는다. 다만 파생물(토크나이저,
모델 가중치)을 공개할 경우 ODC-By(FineWeb-2)의 출처 표기와 CC BY-SA(위키)의
동일조건변경허락 조항을 확인해야 한다. manifest 의 `source` 컬럼이 그때 근거가 된다.
