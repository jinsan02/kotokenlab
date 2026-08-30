# 인수인계 — 현재 상태와 다음 순서

최종 갱신 2026-08-30, Claude Code. 규칙은 [`RULES.md`](RULES.md),
범위와 종료 조건은 [`PLAN.md`](PLAN.md), 프롬프트는 [`PROMPTS.md`](PROMPTS.md).

---

## 한 줄 상태

**전체 규모 직전.** 데이터 파이프라인·Level 1·영어/코드 대조군까지 끝났고,
Candidate Gate 가 처음으로 온전히 판정됐다. 남은 것은 도메인 규칙 정확도 검증
(사람 손 필요)과 게이트 임계값 결정이다.

```
Step 0  저장소·원장·훅·CI·환경·자원 실측                완료
Step 1  한국어 파이프라인 (규칙 v3, 호스트 상한)          완료
Step 2  Level 1 벤치마크 + 영어·코드 대조군               완료
        ─────────────────────────────────────────────
현재    도메인 정확도 라벨링 · 게이트 임계값 결정
다음    전체 규모 실행 -> Step 3 토크나이저 학습
금지    임계값을 결과 보고 나서 완화하지 않는다 (§107)
```

---

## 지금까지 나온 결과

### Level 1 (dev 1,617문서 6.8MB, run_id `tok_bench_ctrl`)

| 도메인 | Qwen tok/char | HCX | A.X |
|---|---:|---:|---:|
| news | 0.7051 | −26.5% | **−40.1%** |
| blog | 0.6913 | −25.1% | −40.7% |
| web_general | 0.6914 | −25.2% | −39.8% |
| technical | 0.5388 | −20.6% | −30.8% |
| ko_en_mixed | 0.4688 | −18.4% | −26.3% |
| **english** | 0.2169 | −0.5% | **+7.3%** |
| **code** | 0.2955 | **+20.9%** | **+26.6%** |

**Candidate Gate 는 두 후보를 모두 탈락시킨다.** 스펙 §17 의 조건(영어 ≤5%,
코드 ≤10%)에 HCX 는 코드 +20.9%, A.X 는 영어 +7.3% + 코드 +26.6% 로 걸린다.

이게 이 프로젝트에서 가장 중요한 미결 판단이다 — 임계값이 현실과 안 맞는 것인지,
한국어 특화가 원래 이 비용을 치르는 것인지. **T2 를 만들기 전에** 정하고
문서에 남겨야 한다. 결과를 보고 나서 완화하면 evaluation freeze 를 어긴다.

### 도메인 분포 (규칙 v3, 호스트 상한 400)

```
web_general 67.8%  news 17.6%  blog 5.3%  ko_en_mixed 4.8%
technical 1.9%  encyclopedia 1.8%  community 0.7%  code 0.06%
```

web_general 은 여기서 더 못 내린다. 호스트가 10,015개로 흩어져 있고 1위 다음이
0.40% 이하다. "분류 실패" 가 아니라 "출처가 특정되지 않는 일반 웹" 으로 취급한다.

---

## 다음에 할 일

### 1. 도메인 규칙 정확도 — **사람 손이 필요하다**

```bash
C:\llm_tokenizer\.conda\python.exe scripts\make_label_ui.py
```

`reports/tables/domain_audit_label.html` 을 브라우저로 열고 200건에 라벨을 찍는다.
숫자키로 선택, 예측이 맞으면 Enter. 자동 저장되므로 중간에 닫아도 된다.
다 채우면 TSV 를 내려받아 `reports/tables/domain_audit.tsv` 를 덮어쓰고:

```bash
C:\llm_tokenizer\.conda\python.exe scriptsudit_domain_rules.py --mode score
```

**이 정확도 없이는 도메인별 결과를 주장할 수 없다.** 표본에 community 2건,
encyclopedia 1건뿐이라 그 두 도메인은 이 표본으로 못 잰다 — 필요하면 층화 표본을
따로 뽑아야 한다.

### 2. 게이트 임계값 결정 (사용자 판단)

선택지: (a) 스펙 §17 그대로 두고 우리 T2 도 같은 잣대로 잰다,
(b) 코드 상한을 완화하되 **지금** 정하고 근거를 남긴다,
(c) 게이트를 탈락 기준이 아니라 보고 항목으로 바꾼다.

### 3. 전체 규모 실행 — 위 둘이 끝난 뒤

```bash
C:\llm_tokenizer\.conda\python.exe scriptsun_data_pipeline.py   --max-docs 1500000 --max-bytes 6000000000 --shards 4 --tag v1
C:\llm_tokenizer\.conda\python.exe scriptsun_control_pipeline.py --lang en --max-docs 60000
C:\llm_tokenizer\.conda\python.exe scriptsun_control_pipeline.py --lang code --max-docs 60000
```

**라벨링 결과로 규칙을 고치면 manifest 가 바뀌므로 전체 규모를 다시 돌려야 한다.**
그래서 1번을 먼저 끝내는 편이 낫다. 도메인별 dev 5MB 하한도 이때 맞춘다
(지금 영어 1.6MB, 코드 2.1MB 로 미달).

완료 후 manifest_sha256 을 PLAN 에 고정하고 `data(data):` 로 커밋한다.

### 4. 그다음

Step 3 토크나이저 학습(T2a/T2b). [`PROMPTS.md`](PROMPTS.md) 4번 참조.

---

## 반드시 지킬 것

1. `data/final_test/` 는 읽거나 커밋하지 않는다. 마지막 1회만 개봉한다.
2. 다른 토크나이저 비교는 PPL 이 아니라 BPB 로 한다.
3. HCX/A.X 는 external reference 다. 인과 효과는 같은 Qwen backbone 에서만.
4. attention 은 `EFFICIENT_ATTENTION + CUDNN_ATTENTION` 을 강제한다.
5. byte fallback 256개와 special token 은 pruning 하지 않는다.
6. 첫 CPT 는 동일 config seed 42/123/2026 으로 노이즈 플로어부터.
7. 원장은 append-only. 실패 run 도 지우지 않는다.
8. `record` 커밋에 코드·설정을 섞지 않는다.
9. 실험 전 `tools/check_clock.py --record` 를 실행한다.
10. **해시를 손으로 적지 않는다.** 원장에서 읽어온다 — 훅이 대조한다.

전체는 [`RULES.md`](RULES.md) 17개 항목이 유일한 기준이다.

---

## 알아두면 시간 아끼는 것

- python 은 항상 `C:\llm_tokenizer\.conda\python.exe` 절대경로
- 원장 시각은 UTC 다. 로컬로 보려면 `tools/ledger_tail.py`
- 학습 운영 설정: seq 2048 / micro_bs 2 / AdamW 8bit -> 13.3GB, 9,089 tok/s
- CI 는 `pytest numpy pyyaml tokenizers` 만 설치한다. 새 테스트가 서드파티를
  쓰면 `.github/workflows/ci.yml` 에 추가해야 한다
- 파이썬 문자열 치환으로 코드를 수정할 때 `
` 이스케이프가 조용히 어긋난다.
  실제로 세 번 당했다 — 치환 후 반드시 `grep` 으로 적용 여부를 확인하라

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
C:\llm_tokenizer\.conda\python.exe toolsalidate_ledger.py
```
