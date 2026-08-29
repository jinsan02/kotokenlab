# 인수인계 — Claude Code가 이어서 작업하기

최종 갱신 2026-08-30, Codex. 이 문서 하나로 직전 작업과 다음 순서를 복원한다.
규칙은 [`RULES.md`](RULES.md), 범위와 종료 조건은 [`PLAN.md`](PLAN.md),
실행 진입점은 [`../CLAUDE.md`](../CLAUDE.md)다.

---

## 한 줄 상태

**아직 본격 학습 단계가 아니다.** Step 1 데이터 파이프라인과 Step 2 Level 1의
소규모 관통·재현성 검증까지 끝났고, 전체 규모 전에 도메인 규칙 정확도,
영어/코드 대조군, 도메인별 Dev 5MB 하한을 확정하는 단계다.

```
Step 0  저장소·원장·훅·CI·환경·자원 실측                    완료
Step 1  한국어 데이터 파이프라인 소규모 관통                 완료
Step 2  Qwen / HCX / A.X Level 1 파일럿·재현성 검증          완료
        ──────────────────────────────────────────────────
현재    전체 규모 전 검증                                    진행 중
다음    호스트 분포 조사 → 도메인 수동 감사 → 대조군 추가
금지    토크나이저 학습·surgery·CPT를 지금 시작하지 않는다
```

---

## Codex가 이번에 완료한 것

### 1. 파이프라인을 4개 샤드에서 다시 관통

- run: `data_verify4shardnet_seed42`
- 입력 20,000문서 → 품질 통과 19,055 → near duplicate 5건 제거 → 19,050문서
- 정제 68,439,895 bytes, 80.52초, GPU 미사용
- train/dev = 17,139 / 942. Final Test 분할은 생성됐지만 설계 판단에 사용하지 않는다.
- Dev 전체가 3.1MB라 정식 평가 표본으로는 부족
- 최초 `data_verify4shard_seed42`는 샌드박스 네트워크 거부로 `fail` 기록됨.
  데이터 처리 실패가 아니라 권한 실패이며 append-only 원장에 그대로 남겼다.

도메인 분포를 split 기준으로 다시 계산한 값:

| split | web_general | ko_en_mixed |
|---|---:|---:|
| train | 71.13% | 6.50% |
| dev | 71.76% | 6.79% |

`ko_en_mixed`는 목표 3~6%를 약간 넘고 `web_general` 쏠림은 그대로다.
도메인 규칙을 고치기 전에 여러 샤드의 상위 호스트를 먼저 봐야 한다.

검증 산출물은 git 제외 경로 `data/interim/_codex_validation_verify4shard/`에 보관했고,
기존 60,000문서 파일럿 데이터와 매니페스트는 원래 위치로 복원했다.

### 2. Level 1 재현성 확인

- run: `tok_bench_verify4shard`
- Dev 942문서, 3.1MB, 실행 28.74초

| tokenizer | 전체 tok/char |
|---|---:|
| Qwen Original | 0.6823 |
| HCX-SEED | 0.5103 |
| A.X-4.0-Light | 0.4140 |

기존 파일럿(Qwen 약 0.687 / HCX 0.516 / A.X 0.416)과 유사해 측정 경로는
재현됐다. 하지만 Dev 코드 문서가 0건이라 HCX와 A.X 모두 Candidate Gate에서
`code 도메인 판정 불가`로 탈락했다. 후보 선정 결과로 사용하면 안 된다.

### 3. 실제 실행에서 드라이버 문법 오류 수정

`scripts/run_data_pipeline.py`의 `SUMMARY.tsv` 줄바꿈 문자열이 깨져 SyntaxError가
발생했다. 수정하고 드라이버 자체를 `compile()`하는 회귀 테스트를 추가했다.

커밋: `aec9c36 fix(data): 파이프라인 요약 파일의 줄바꿈 문법 오류 수정`

### 4. 시간 검증과 산출물 레지스트리 추가

UTC 문자열만 기록하면 호스트 시계가 틀려도 알 수 없으므로 다음을 추가했다.

- `tools/check_clock.py --record`
  - 캐시되지 않은 Hugging Face HTTPS Date와 로컬 요청 구간을 비교
  - 성공 검사는 24시간 유효
  - 새 `RunContext`는 최근 성공 검사가 없으면 실행을 거부
  - run의 `clock_check_sha256`가 `experiments/clock_checks.tsv`를 참조
- `tools/register_artifact.py`
  - 프로젝트 생성 산출물의 SHA256, 크기, 경로, 생성 run을 기록
  - 같은 artifact는 중복 append하지 않음
  - `final_test` 경로는 개봉 전 등록 거부

최초 시간 검사:

```
status           ok
offset_ms        -1585.097
rtt_ms           243.123
windows_source   local_cmos_clock
clock hash       649efb43f725016159c2bf7591260a01c722700c6ad9ae7c866c77cbf019e0d7
```

Windows Time 피어는 `time.windows.com`이지만 실제 원본은 Local CMOS Clock이다.
`w32tm /resync /rediscover`는 관리자 권한 부족으로 실패했다. HTTPS 대조 시각에는
5초 이내로 맞지만, OS 자체 동기화는 관리자 PowerShell에서 따로 해야 한다.

```powershell
w32tm /resync /rediscover
```

외부 모델 원본은 계속 `experiments/models.tsv`가 담당한다. 프로젝트가 만든
토크나이저·체크포인트·adapter·매니페스트 요약·표·그림·리포트만
`experiments/artifacts.tsv`에 등록한다. 첫 artifact로 파일럿 `SUMMARY.tsv`를 등록했다.

### 5. 원장과 문서 정비

- `LEDGER.tsv` 끝에 `clock_check_sha256` 추가, 기존 12개 행은 값 변경 없이 `NA` 패딩
- `SUMMARY.tsv`를 일반 manifest로 오인하던 `migrate_ledger.py` 수정
- README / PLAN / WORKFLOW의 오래된 Step 0 표기를 검증 단계로 동기화
- 시간·모델·산출물 규칙을 RULES / LEDGER_SCHEMA / ENVIRONMENT에 반영
- 전체 테스트 89개, 원장 검사, 원장 E2E smoke 통과

주요 커밋:

```
0f78402 docs(docs): 시간 및 산출물 계보 규칙 문서화
e5aff26 feat(infra): 시간 검증과 산출물 레지스트리 추가
16f1162 record(tok): 4개 샤드 Level 1 파일럿 재현성 검증
e96c379 docs(data): 현재 위치를 전체 규모 전 검증 단계로 동기화
0f3234e record(data): 4개 샤드 소규모 파이프라인 재검증
aec9c36 fix(data): 파이프라인 요약 파일의 줄바꿈 문법 오류 수정
```

---

## Claude Code가 다음에 할 일

각 단계 결과를 사용자에게 보여주고 다음 단계로 넘어가기 전에 확인받는다.

### 1. 4개 샤드 상위 호스트 50개 조사

전체 파이프라인부터 돌리지 않는다. `stream_docs(..., shards=4)`를 재사용해 문서
20,000건의 호스트 빈도를 추출하고 다음을 보고한다.

- 상위 호스트 50개와 문서 수·비율
- 현재 규칙이 부여한 도메인
- `web_general` 상위 미분류 호스트
- 한 호스트가 표본을 과점하는지

한 샤드 앞부분만 읽으면 tripadvisor 같은 단일 호스트가 24%를 차지한 전례가 있다.
반드시 여러 샤드와 분산 row group을 사용한다.

### 2. `scripts/audit_domain_rules.py` 구현

정규화·필터를 통과한 문서에서 seed 고정 무작위 200건을 뽑아 TSV로 만든다.

권장 컬럼:

```
sample_id  url  host  text_preview  predicted_domain  gold_domain  reviewer_note
```

`gold_domain`과 `reviewer_note`는 사람이 채운다. 생성 순서와 표본 seed를 config 또는
파일 메타데이터에 남긴다. 자동 생성 파일은 기존 내용을 손으로 덮어쓰지 않게 한다.

### 3. 호스트 규칙 보강 후 소규모 재검증

상위 호스트와 수동 감사 결과를 근거로 `configs/data/domain_rules.yaml`을 수정한다.
목표는 `web_general` 40% 이하이지만, 숫자를 맞추려고 편향 표본에 과적합하지 않는다.

변경 후:

```bash
C:\llm_tokenizer\.conda\python.exe scripts\run_data_pipeline.py \
  --max-docs 20000 --max-bytes 150000000 --shards 4 --tag domainv3_smoke

C:\llm_tokenizer\.conda\python.exe -m src.evaluation.tokenizer_eval \
  --split dev --tag domainv3_smoke
```

통과율 90~96%, `ko_en_mixed` 약 3~6%, 기존 tok/char 근방인지 확인한다.
변화가 크면 원인부터 설명한다.

### 4. 영어·코드 대조군 추가

Candidate Gate의 영어 ≤5%, 코드 ≤10% regression을 판정할 데이터가 아직 없다.

- 영어: `HuggingFaceFW/fineweb-edu`, `sample-10BT`
- 코드: `codeparrot/github-code-clean`
- 도메인별 Dev 최소 5MB
- 같은 manifest 스키마와 `doc_id` 해시 분할을 사용

### 5. 그다음에만 전체 규모 실행

```bash
C:\llm_tokenizer\.conda\python.exe scripts\run_data_pipeline.py \
  --max-docs 1500000 --max-bytes 6000000000 --shards 4 --tag v1
```

예상 다운로드 수 GB, CPU 처리 1~2시간, GPU 미사용. 실행 전 예상 시간·디스크를
사용자에게 알린다. 완료 후 manifest SHA256을 PLAN에 고정하고 `data(data):` 커밋한다.

전체 규모·대조군·도메인 감사가 끝나기 전에는 Step 3 토크나이저 학습으로 넘어가지 않는다.

---

## 반드시 지킬 것

1. `data/final_test/`는 읽거나 커밋하지 않는다. Final Test는 마지막 1회만 개봉한다.
2. 다른 토크나이저의 언어 모델링 비교는 PPL이 아니라 BPB로 한다.
3. HCX/A.X는 external reference다. tokenizer 인과 효과는 같은 Qwen backbone에서만 말한다.
4. attention은 `EFFICIENT_ATTENTION + CUDNN_ATTENTION`을 강제한다.
5. byte fallback 256개와 special token은 pruning하지 않는다.
6. 첫 CPT는 동일 config seed 42/123/2026으로 노이즈 플로어부터 잰다.
7. 원장은 append-only다. 실패 run도 지우지 않는다.
8. `record` 커밋에는 코드·설정을 섞지 않는다.
9. 실험 전 `tools/check_clock.py --record`를 실행한다.

전체 하드룰은 [`RULES.md`](RULES.md)의 17개 항목이 유일한 기준이다.

---

## 세션 시작 명령

```bash
cd C:\llm_tokenizer
git log --oneline -20
git status
tail -n 15 experiments/LEDGER.tsv
C:\llm_tokenizer\.conda\python.exe -m src.utils.env --check
C:\llm_tokenizer\.conda\python.exe tools/check_clock.py --record
C:\llm_tokenizer\.conda\python.exe -m pytest tests/ -q
C:\llm_tokenizer\.conda\python.exe tools/validate_ledger.py
```

시간 검사는 네트워크 승인이 필요할 수 있다. 앞선 대화를 기억한다고 가정하지 말고,
Git 커밋과 `experiments/` 원장만 현재 상태로 믿는다.
