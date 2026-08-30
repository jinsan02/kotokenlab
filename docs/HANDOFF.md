# 인수인계 — 현재 상태와 다음 순서

최종 갱신 2026-08-30. 규칙은 [`RULES.md`](RULES.md), 범위와 종료 조건은
[`PLAN.md`](PLAN.md), 프롬프트는 [`PROMPTS.md`](PROMPTS.md),
도메인 라벨 신뢰 범위는 [`DOMAIN_LABELS.md`](DOMAIN_LABELS.md).

---

## 한 줄 상태

**Level 1 정식 측정까지 완료. Step 3 T2a/T2b 도 끝났다.**
코퍼스는 `phase1-tokenizer-freeze` 로 얼렸고, T2b n=30,000 이 한국어 -28.4% 로
Candidate Gate 를 통과했다. 다음은 Step 4 embedding surgery 다. 한국어 118.7만 문서(정제 4.73GB)와
영어·코드 대조군을 만들었고 `manifest_sha256` 을 [`PLAN.md`](PLAN.md) 에 고정했다.
남은 것은 그 코퍼스로 Level 1 을 다시 재고 얼리는 일이다.

```
Step 0   저장소 · 원장 · 훅 · CI · 환경 · 자원 실측              완료
Step 1   한국어 파이프라인 (규칙 v4, 호스트 상한 400)            완료 (파일럿)
Step 1b  영어 · 코드 대조군                                     완료 (파일럿)
Step 1c  도메인 라벨 블라인드 감사                               완료
Step 2   Level 1 벤치마크 + Gate 임계값 확정                     완료
Step 2b  전체 규모 코퍼스 v1 (한국어 + 영어·코드 대조군)          완료
Step 2c  Level 1 정식 측정 (v1 dev 259MB)                        완료
Step 3   T2a / T2b 수술 + N 스윕 -> N=30,000 확정               완료
         ────────────────────────────────────────────────
현재     Step 4 embedding surgery (resize, E0/E1 초기화)
금지     phase1-tokenizer-freeze 이후 코퍼스를 바꾸지 않는다
```

---

## 측정된 것

### Level 1 — 1층: 언어 간 (라벨이 출처 데이터셋이라 신뢰 가능)

| 구분 | 문자 수 | Qwen2.5 | HCX-SEED | A.X-4.0-Light |
|---|---:|---:|---:|---:|
| 한국어 | 103,657,654 | 0.6834 | −24.9% | **−39.4%** |
| 영어 | 11,549,402 | 0.2177 | +0.0% | **+7.3%** |
| 코드 | 8,350,350 | 0.2842 | **+17.6%** | **+22.5%** |

한국어 압축을 얻는 만큼 코드에서 20~27% 를 잃는다. 스펙 §16 이 경고한 trade-off 가
실측으로 확인됐다.

### Level 1 — 2층: 한국어 내부 (참고. 세분화는 보고하지 않는다)

| 구분 | 문자 수 | Qwen2.5 | A.X |
|---|---:|---:|---:|
| news | 16,136,240 | 0.7086 | −40.5% |
| 기타(한국어) | 87,521,414 | 0.6788 | −39.2% |

**차이가 1.3%p 다.** 반면 언어 간은 62%p 갈린다. 라벨을 못 믿는 축과 결과가
갈리는 축이 서로 다르므로, 세분화를 포기해도 잃는 정보가 거의 없다.

### 도메인 라벨 신뢰도

```
블라인드 감사 150건        54.7%
  news 정밀도              81%
  technical 재현율          8%
  community / ko_en_mixed   0%
```

경위는 [`DOMAIN_LABELS.md`](DOMAIN_LABELS.md).
**한국어 내부 세분화 수치를 단독으로 인용하지 마라.**

### 자원 실측 (RTX 5070 Ti 16GB)

```
학습 운영 설정      seq 2048 / micro_bs 2 / AdamW 8bit -> 13.3GB, 9,089 tok/s
50M 토큰 CPT        약 1.6시간.  코어 전체 GPU 예산 약 13시간
prefill 32,768 tok  0.5B 2,718MB / 1,310ms    1.5B 6,808MB / 2,972ms
```

전체 표: [`reports/tables/resource_probe.md`](../reports/tables/resource_probe.md)

---

## 다음에 할 일

### 1. 전체 규모 실행 — 완료 (2026-08-30)

만들어진 것 (계보와 한계는 [`PLAN.md`](PLAN.md) "고정된 계보"):

```
data_v1_seed42              한국어 1,186,892문서  정제 4,735MB  79분
data_controlen_v1_seed42    영어      51,960문서  정제  247MB   5분
data_controlcode_v1_seed42  코드      30,216문서  정제  187MB   2분
```

**dev 기준은 세분화 도메인이 아니라 보고 버킷 4개로 본다.** 한국어 news 38.2MB /
기타 201.2MB / 영어 11.6MB / 코드 8.3MB 로 전부 5MB 를 넘겼다. encyclopedia
0.4MB 등이 미달이지만 보고 대상이 아니다 ([`DOMAIN_LABELS.md`](DOMAIN_LABELS.md)).

**필터 통과율은 79.2% 로 기준(90~92%)을 밑돈다.** 사유가 `host_cap` 하나에 몰려
있고(15.79%, 파일럿 4.3%), 상한이 비율이 아니라 절대 건수라 규모가 커질수록 더
빡빡해지기 때문이다. 위키백과가 6,652건에서 400건으로 잘렸다 — 코퍼스의 약 1.2%.
v1 은 이대로 가되 백과 문체 과소대표를 전제로 해석한다.

대조군을 다시 돌릴 일이 있으면 `--max-bytes` 를 반드시 올려라. 기본값 120MB 가
`--max-docs` 보다 먼저 걸려서 조용히 절반 규모가 나온다. 처음 돌렸을 때
코드 dev 가 4.64MB 로 미달했던 원인이다.

### 2. Level 1 정식 측정 — 완료 (2026-08-30, run `tok_bench_v1`)

v1 dev 259.2MB / 63,961문서를 토크나이저 3종으로 234초에 쟀다. 수치는 위
"측정된 것" 표가 정식 값이고, 코퍼스는 `phase1-tokenizer-freeze` 로 얼렸다.
Candidate Gate 는 hcx_seed / ax_4_0_light 둘 다 하드 상한 이내라 GPU 진행이지만,
둘 다 코드 악화가 보고 기준 10% 를 넘겼다 (+17.6% / +22.5%).

### 3. Step 3 — T2a / T2b 완료 (2026-08-30)

전체 표는 [`reports/tables/t2_sweep.md`](../reports/tables/t2_sweep.md).

```
             한국어      영어     코드    vocab
T2a n=30,000  +0.0%    +0.0%   +0.1%   121,643  (embedding -26.9M)
T2b n=30,000  -28.4%   +0.0%   -0.8%   151,643  (유지)
```

**T2a 가 못한 것이 아니라 사는 물건이 다르다.** 제거 대상이 코퍼스에서 거의
안 쓰이므로(3만개 = 13.6억 토큰 중 0.0035%) 지우기만 해서는 토큰화가 바뀔
이유가 없다. T2a 가 사는 것은 파라미터 26.9M(embedding 의 19.7%) 감소이고,
T2b 가 사는 것은 압축 -28.4% 다. 압축은 **빈자리를 채워야** 나온다.

외부 참조 대비: HCX 는 한국어 -24.9% 에 코드 +17.6%, A.X 는 -39.4% 에 코드
+22.5% 다. T2b 는 HCX 보다 한국어가 좋으면서 코드 손해가 없다.

N 은 CPU 스윕으로 골랐다. 50,000 이 -30.2% 로 1.8%p 낫지만 제거 비용이 10배고
그 지점부터 T2a 코드가 +0.8% 로 악화된다.

산출물: `artifacts/tokenizers/ko{t2a,t2b}_v1_n30000`, 원장 `experiments/artifacts.tsv`.

### 4. Step 4 — embedding surgery (다음)

[`PROMPTS.md`](PROMPTS.md) 5번, 스펙 §20~21. T2a 는 vocab 이 줄어 resize 가
필요하고 T2b 는 형상이 그대로다. 치환된 30,000행만 다시 초기화한다 —
`artifacts/tokenizers/kot2b_v1_n30000/id_map.json` 에 어느 ID 가 어떤 토큰으로
바뀌었는지 들어 있다.

### 하지 않아도 되는 것

도메인 라벨 재감사. 이미 3회 했고 결론이 나왔다. 규칙을 고쳤다면
`scripts/eval_domain_rules.py` 로 **추가 라벨링 없이** 채점하면 된다.

---

## 반드시 지킬 것

1. `data/final_test/` 는 읽거나 커밋하지 않는다. 마지막 1회만 개봉한다.
2. 다른 토크나이저 비교는 PPL 이 아니라 BPB 로 한다.
3. HCX/A.X 는 external reference 다. 인과 효과는 같은 Qwen backbone 에서만.
4. attention 은 `EFFICIENT_ATTENTION + CUDNN_ATTENTION` 을 강제한다.
   강제하지 않으면 8,192 토큰에서 메모리 7.1배, 시간 10.3배가 된다.
5. byte fallback 256개와 special token 은 pruning 하지 않는다.
6. 첫 CPT 는 동일 config seed 42/123/2026 으로 노이즈 플로어부터.
7. 원장은 append-only. 실패 run 도 지우지 않는다.
8. `record` 커밋에 코드·설정을 섞지 않는다.
9. 실험 전 `tools/check_clock.py --record` 를 실행한다.
10. **해시를 손으로 적지 않는다.** 원장에서 읽어온다 — 훅이 대조해서 거부한다.
11. **도메인별 수치를 라벨 정확도 없이 인용하지 않는다.** 한국어 내부는 ~55% 다.
12. 라벨링 화면으로 정확도를 잴 때는 **반드시 `--blind`**.

전체는 [`RULES.md`](RULES.md) 17개 항목이 유일한 기준이다.

---

## 알아두면 시간 아끼는 것

- python 은 항상 `C:\llm_tokenizer\.conda\python.exe` 절대경로
- 원장 시각은 UTC 다. 로컬로 보려면 `tools/ledger_tail.py`
- 원장 테이블 10종 + manifest. 컬럼을 추가했으면 `tools/migrate_ledger.py`
- CI 는 `pytest numpy pyyaml tokenizers` 만 설치한다. 새 테스트가 서드파티를
  쓰면 `.github/workflows/ci.yml` 에 추가해야 한다
- **파이썬 문자열 치환으로 코드를 수정할 때 `\n` 이스케이프가 조용히 어긋난다.**
  실제로 네 번 당했다. 치환 후 반드시 `grep` 이나 `assert` 로 적용 여부를 확인하라
- 감사 TSV 를 브라우저에서 받으면 `&` 가 `&amp;` 로 온다. `html.unescape` 후 적용
- 라벨링 화면은 `python -m http.server` 로 띄운다 (`.claude/launch.json` 의 `labeler`).
  `file://` 로 열면 localStorage 가 막히는 브라우저가 있다

---

## 아직 안 만든 것

| 파일 | 용도 |
|---|---|
| `src/tokenizer/train.py` `prune.py` `substitute.py` | T2a / T2b 학습 |
| `src/surgery/*.py` | embedding resize, E0/E1 초기화 |
| `src/training/cpt.py` `callbacks.py` | CPT 루프 |
| `src/evaluation/bpb.py` | Level 2 BPB |
| `src/evaluation/latency.py` `memory.py` | Level 4 |

`src/` 의 스텁에는 각각 스펙 절 번호가 docstring 에 적혀 있다.

---

## 세션 시작

```bash
cd C:\llm_tokenizer
git log --oneline -20
git status
C:\llm_tokenizer\.conda\python.exe tools\ledger_tail.py
C:\llm_tokenizer\.conda\python.exe -m src.utils.env --check
C:\llm_tokenizer\.conda\python.exe tools\check_clock.py --record
C:\llm_tokenizer\.conda\python.exe -m pytest tests/ -q
C:\llm_tokenizer\.conda\python.exe tools\validate_ledger.py
```

앞선 대화를 기억한다고 가정하지 말고, 커밋과 `experiments/` 원장만 현재 상태로 믿는다.
