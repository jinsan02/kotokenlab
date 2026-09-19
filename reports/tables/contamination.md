# KMMLU 오염 검사 — CPT 문서 풀과의 13-gram 겹침

> **이 표는 코드가 적는다.** `.conda/python.exe tools/contamination_check.py`
> 학습 없음, GPU 0시간.

`DATA_SOURCES.md` 의 오염 제거 단계가 구현되지 않아 따로 센다.
문항의 **질문 본문** 13-gram(공백 제거·NFKC 소문자)이 CPT 풀 문서에
나타나는지 본다. **겹침이 곧 오염은 아니다** — 흔한 관용구도 걸린다.

과목 집합 `d3` · 문항 1,900 · CPT 풀 앞 50,000문서

## 결과

```
n-gram 이 하나라도 걸린 문항  838 / 1,900  (44.11%)
  그중 덮임 >= 50%           2   <- 외웠을 수 있다
  그중 덮임 20% ~ 50%         92
  (문항이 짧아 제외: 13-gram 10개 미만인데 덮임 50% 이상  15)
걸린 n-gram 종류             174
```

| 과목 | 겹친 문항 | 문항 | 비율 |
|---|---:|---:|---:|
| Law | 343 | 1,000 | 34.3% |
| Political-Science-and-Sociology | 152 | 300 | 50.7% |
| Criminal-Law | 143 | 200 | 71.5% |
| Taxation | 119 | 200 | 59.5% |
| Patent | 60 | 100 | 60.0% |
| Health | 21 | 100 | 21.0% |

덮임이 큰 문항 (과목, 행, 덮임, 걸린 gram) — 본문은 싣지 않는다 (CC-BY-ND)

```
Criminal-Law                             row  126    66.7%     4 /   6 gram
Political-Science-and-Sociology          row  187    56.0%    14 /  25 gram
Political-Science-and-Sociology          row  164    51.5%    17 /  33 gram
Political-Science-and-Sociology          row  215    50.0%     4 /   8 gram
Criminal-Law                             row   15    50.0%     2 /   4 gram
Criminal-Law                             row   18    50.0%     2 /   4 gram
Criminal-Law                             row   30    50.0%     2 /   4 gram
Criminal-Law                             row   32    50.0%     2 /   4 gram
Criminal-Law                             row   39    50.0%     2 /   4 gram
Criminal-Law                             row   78    50.0%     2 /   4 gram
```

## 읽는 법

- **덮임 50% 이상** 인 문항이 있으면 P3-F 는 그 문항을 뺀 판정도 낸다.
  둘이 다르면 뺀 쪽을 판정으로 쓴다 (PLAN "P3-F")
- n-gram 이 하나 걸린 것만으로는 아무것도 아니다. 덮임을 본다
- 13-gram 이 10개 미만인 짧은 문항은 덮임이 쉽게 올라간다 (4개 중 2개면 50%). 그래서 의심 집계에서 뺀다
- D3(6과목)는 이미 돌았으므로 **보고만** 한다. 결과를 바꾸지 않는다
- Qwen 사전학습 데이터의 오염은 이 도구로 알 수 없다. 우리 CPT 풀만 본다
- 13-gram 은 문자 기준이다. 짧은 수식·단위 표기는 우연 일치가 날 수 있다
