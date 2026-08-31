# 커밋 규칙

집행 코드는 [`tools/check_commit_msg.py`](../tools/check_commit_msg.py),
훅은 [`.githooks/commit-msg`](../.githooks/commit-msg). 규칙과 코드가 어긋나면 코드가 맞다.

```
<type>(<scope>): <제목>

<본문 — 무엇을 했는지가 아니라 왜 했는지>

<트레일러 — 기계가 읽는 계보>
```

- 제목 줄 전체 **72자 이하**, 끝에 마침표 없음
- 제목과 본문 사이 **빈 줄 필수**
- 트레일러는 메시지 **마지막 블록**에 `Key: value` 형태로

---

## type

| type | 언제 | 필수 트레일러 |
|---|---|---|
| `record` | **실험 결과 기록**. 원장 행 + 그림/표 산출물 | `Run-Id`, `Ledger`, `Config-SHA256` |
| `fix` | 잘못된 동작을 고침 | `Invalidates` |
| `upgrade` | 기존 기능 개선. 동작은 그대로, 속도·정확도·구조가 나아짐 | — |
| `feat` | 새 구성요소·스크립트 추가 | — |
| `data` | corpus·manifest·split 변경 | `Manifest-SHA256` |
| `tok` | 토크나이저 버전 산출물 등록 (`kosub_v1` 등) | `Tokenizer-SHA256` |
| `docs` | 문서 | — |
| `chore` | 환경·설정·의존성·훅 | — |
| `revert` | 되돌리기 | — |

`fix` 와 `upgrade` 의 경계: **결과가 틀렸었나?** 틀렸으면 `fix`, 맞았는데 더 나아졌으면
`upgrade`. `fix` 는 과거 실험을 무효화할 수 있으므로 `Invalidates` 를 요구한다.

## scope

`data` `tok` `surgery` `align` `cpt` `eval` `system` `infra` `docs`

스펙의 파이프라인 단계와 같다. 어디를 건드렸는지 한눈에 보이게 한다.

---

## 트레일러

| Key | 형식 | 의미 |
|---|---|---|
| `Run-Id` | `cpt_kosub_mean_50m_seed42` | **`LEDGER.tsv` 에 실재해야 한다.** 훅이 확인한다 |
| `Ledger` | 쉼표 구분 경로 | 이 커밋이 행을 추가한 원장 파일들. 존재해야 한다 |
| `Config-SHA256` | 소문자 64 hex | run 의 config 해시 |
| `Tokenizer-SHA256` | 소문자 64 hex | 토크나이저 산출물 디렉토리 해시 |
| `Manifest-SHA256` | 소문자 64 hex | dataset manifest 해시 |
| `Invalidates` | `none` 또는 run_id 목록 | 이 수정으로 **무효가 된** 과거 실험 |

---

## 두 가지 강제 규칙

### 1. `record` 커밋은 코드를 건드릴 수 없다

허용되는 경로: `experiments/`, `reports/`, `data/manifests/`, `docs/`

`src/`, `tools/`, `configs/`, `.githooks/` 가 함께 스테이지되면 훅이 거부한다.
"결과가 좋게 나오도록 코드를 조금 고치면서 같이 커밋"하는 경로를 없애기 위해서다
(스펙 §107, [RULES.md](RULES.md) 14번). 코드를 고쳐야 한다면 먼저
`fix`/`upgrade`/`feat` 로 커밋하고, 그 뒤에 다시 돌려서 `record` 한다.

### 2. `fix` 커밋은 `Invalidates` 를 반드시 적는다

`none` 이라고 적는 것도 허용한다 — 다만 **한 번은 생각해야** 적을 수 있다.
적힌 run_id 는 `LEDGER.tsv` 에 있어야 한다.

### 3. 훅과 CI 는 같은 트리를 본다

계보 트레일러(`Run-Id`, `Invalidates`)가 가리키는 run 은 **이 커밋이 만들 트리** 의
`LEDGER.tsv` 에 있어야 한다. 훅은 작업 트리가 아니라 **인덱스** 를 읽는다
(`check_commit_msg.py --staged`).

왜냐하면 한 번 이렇게 깨졌다 (커밋 `1e8b379`):

| | 읽는 곳 | 결과 |
|---|---|---|
| 로컬 훅 | 작업 트리 — run 이 있다 | 통과 |
| CI | 커밋된 트리 — run 이 없다 | **실패** |

원장 행을 아직 스테이지하지 않은 채로 그 run 을 `Invalidates` 에 적었기 때문이다.
지금은 훅이 인덱스를 보므로 같은 상황에서 로컬에서 먼저 막힌다.

따라서 순서는 둘 중 하나다:

- 원장 행을 **같은 커밋에** 함께 스테이지한다, 또는
- 원장 행을 **먼저** `record` 로 커밋하고, 그 다음 커밋에서 참조한다

원장이 비어 있으면 검사를 건너뛰던 예외도 없앴다. run 을 이름으로 가리켰으면
원장에 있어야 한다 — 원장이 아직 없다는 것은 통과 사유가 아니라 거부 사유다.

---

## 예시

```
record(cpt): C2 Ko-Substitute 50M 토큰 CPT 결과

Dev BPB 가 10M 지점 이후 거의 평탄해졌다. Equal-Raw-Data 조건이며
C0 통제군과 동일한 data order, LR schedule 을 썼다.

Run-Id: cpt_kosub_mean_50m_seed42
Ledger: experiments/LEDGER.tsv,experiments/lm_metrics.tsv,experiments/train_curve.tsv
Config-SHA256: 3f9a1c0b7e2d548a6f01b93cd7e4a25f8c6b0d13a94e7f28c5b016d3a8e7f240
```

```
fix(eval): BPB 분모에 바이트 수 대신 문자 수를 쓰던 오류 수정

한글은 UTF-8 에서 문자당 3바이트라 BPB 가 약 3배로 계산되고 있었다.
영어 corpus 에서는 차이가 작아 눈치채지 못했다.

Invalidates: cpt_original_mean_10m_seed42, cpt_kosub_mean_10m_seed42
```

```
tok(tok): kosub_v1 등록 — 저빈도 4,096개 pruning 후 한국어 토큰 치환

vocab 크기는 유지했다 (스펙 §12). 치환 ID 매핑은 artifacts 에 함께 저장.

Tokenizer-SHA256: b71e0d4c9a2f386051cd7be4a90f2c65d8134ae7026fb9c85d3a071e4f6c2b98
```

```
chore(infra): transformers 4.51 -> 4.57 업그레이드
```

```
upgrade(eval): 토크나이저 벤치마크를 멀티프로세스로 전환
```

---

## 브랜치

| 패턴 | 용도 |
|---|---|
| `main` | 항상 초록. 훅을 통과한 것만 들어온다 |
| `exp/<phase>-<slug>` | 실험. 예 `exp/tok-kosub-v1`, `exp/cpt-equal-budget` |
| `fix/<slug>` | 버그 수정 |
| `docs/<slug>` | 문서 |

## 태그 — 되돌릴 수 없는 결정 지점

| 태그 | 의미 |
|---|---|
| `eval-freeze-v1` | 평가 지표 고정 (스펙 §107). 이후 지표 변경은 `-v2` |
| `phase1-tokenizer-freeze` | 토크나이저 후보 확정 (스펙 §62) |
| `final-test-opened` | **Final Test 를 처음 연 커밋** (스펙 §10) |

`final-test-opened` 이후에 나온 설계 변경은 전부 오염된 것으로 간주한다.

---

## 훅이 막는 것

| 훅 | 검사 |
|---|---|
| `commit-msg` | 제목 형식 · type/scope 화이트리스트 · 72자 · 마침표 · 빈 줄 · 필수 트레일러 · sha256 형식 · `Run-Id`/`Invalidates` 실재 여부(**인덱스 기준**) · `Ledger` 경로 실재 · `record` 커밋의 코드 변경 |
| `pre-commit` | 5MB 초과 · 모델 바이너리 · 금지 경로 · **`final_test` 경로** · `.ipynb` 출력 셀 · 원장 무결성 |

훅 연결: `git config core.hooksPath .githooks` (클론 직후 한 번).
`--no-verify` 는 쓰지 않는다 ([RULES.md](RULES.md) 15번).
