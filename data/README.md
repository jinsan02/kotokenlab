# data/

| 디렉토리 | 내용 | git |
|---|---|---|
| `raw/` | 원본 corpus | 제외 |
| `interim/` | 정규화·dedup 중간 산출물 | 제외 |
| `manifests/` | 문서 단위 manifest TSV | **커밋** |
| `final_test/` | 최종 평가 세트 | 제외 + **훅이 하드 차단** |

## 원칙

**Split first, tokenize later.** 문서 단위 분할을 토크나이저 학습보다 먼저 한다.

```
Raw Documents → Normalize → Exact Dedup → Near Dedup
              → Document-level Split → Train / Dev / Test → Chunking
```

문장으로 쪼갠 뒤 무작위 분할하면 같은 문서의 앞뒤가 Train 과 Test 로 나뉜다.

## final_test/

이 디렉토리는 프로젝트 마지막까지 열지 않는다. 토크나이저 vocabulary 설계,
merge rule, pruning threshold, 하이퍼파라미터, learning rate, checkpoint 선택,
모델 구조 선택 어디에도 쓰지 않는다. checkpoint 선택은 Dev BPB 로 한다.

`tools/precheck.py` 가 경로에 `final_test` 가 들어간 모든 스테이지 파일을 거부한다.
실제로 개봉하는 시점은 `final-test-opened` 태그로 못 박는다.

## manifest 스키마

`doc_id  source  domain  date  language  sha256  split  char_count  byte_count`

전체 설명: `docs/LEDGER_SCHEMA.md`
