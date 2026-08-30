# CLAUDE.md — Claude Code 진입점

> **규칙 본문은 여기에 없다.** 이 저장소의 규칙은 [`docs/RULES.md`](docs/RULES.md)
> 한 곳에만 있다. 규칙을 바꾸려면 그 파일을 고친다. 여기에 복사하지 마라 —
> 갈라진 규칙은 규칙이 아니다. 이 파일은 Claude Code 로 이 저장소를 다룰 때의
> 운영 메모다. Codex 쪽 대응 파일은 [`AGENTS.md`](AGENTS.md).

## 세션 시작할 때 이 순서로 읽는다

1. [`docs/RULES.md`](docs/RULES.md) — 하드룰 17개. **매 세션 읽는다**
2. [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — 실험 수명주기, 개발 순서, 현재 위치
3. [`docs/COMMIT_CONVENTION.md`](docs/COMMIT_CONVENTION.md) — 커밋할 일이 있으면
4. [`docs/LEDGER_SCHEMA.md`](docs/LEDGER_SCHEMA.md) — 결과를 기록할 일이 있으면
5. [`docs/HANDOFF.md`](docs/HANDOFF.md) — 지금까지 된 것과 다음 할 일
6. [`docs/DOMAIN_LABELS.md`](docs/DOMAIN_LABELS.md) — 도메인 라벨로 무엇을
   주장할 수 있는지. 도메인별 수치를 인용하기 전에 반드시 본다
6b. [`docs/DESIGN_DELTA.md`](docs/DESIGN_DELTA.md) — **스펙과 다르게 한 것과 그 이유.**
   스펙만 읽고 코드를 고치면 이미 반증된 가설을 되살리게 된다
7. [`docs/SPEC_KoTokenLab.md`](docs/SPEC_KoTokenLab.md) — 연구 설계 원본. 필요한 절만

그리고 현재 상태를 복원한다:

```bash
git log --oneline -20
git status
tail -n 15 experiments/LEDGER.tsv
.conda/python.exe -m src.utils.env --check
.conda/python.exe tools/check_clock.py --record
```

**앞 세션의 대화를 기억하고 있다고 가정하지 않는다.** 상태는 커밋과
`experiments/` 에만 있다.

## 이 환경에서의 실행

- 기본 셸은 **PowerShell**. Bash 도구를 쓴다면 Git Bash 문법(POSIX)으로 쓴다.
  두 도구는 문법이 다르다 — `$env:VAR` vs `$VAR`, `2>$null` vs `2>/dev/null`
- python 은 **항상 절대경로**로 부른다. `conda activate` 에 의존하지 않는다:

  ```
  C:\llm_tokenizer\.conda\python.exe
  ```

- 저장소 루트는 `C:\llm_tokenizer`
- GPU 는 RTX 5070 Ti 16GB 하나뿐이다. 학습을 두 개 동시에 띄우지 않는다

## 긴 작업

토크나이저 학습, CPT, 벤치마크는 수십 분~수 시간이 걸린다.

- Bash/PowerShell 도구의 `run_in_background: true` 로 띄우고, 끝나면 알림을 받는다
- 폴링하지 않는다. 완료 알림이 온다
- 시작 전에 예상 소요와 VRAM 을 사용자에게 말한다

## 절대 하지 않는 것

- `git commit --no-verify` — 훅 우회. 훅이 막으면 커밋에 문제가 있는 것이다
- 원장 TSV 의 **기존 행 편집·삭제** — append-only다. 정정도 새 행
- `data/final_test/` 를 읽거나 커밋 — 프로젝트에서 가장 되돌릴 수 없는 실수
- `C:\aimers\.conda` 패키지 업그레이드 — 다른 프로젝트의 재현성이 깨진다
- 실험 결과를 마크다운 일지로 손으로 적기 — 원장에 코드가 적는다
- **도메인별 수치를 라벨 정확도 없이 인용하기** — 한국어 내부 세분화는 ~55% 다
- **해시를 손으로 적기** — 원장에서 읽어온다. 훅이 대조해서 거부한다

## 결과를 기록할 때

`RunContext` 로 감싸면 `run_id`, `git_commit`, `config_sha256`, `ts_utc` 가 자동으로
들어간다. 손으로 TSV 를 쓰지 않는다. 사용법은
[`docs/LEDGER_SCHEMA.md`](docs/LEDGER_SCHEMA.md) 마지막 절.

기록 후 커밋은 `record(<scope>): ...` 이고, **코드를 함께 스테이지하면 훅이 거부한다**.
코드 수정이 필요하면 `fix`/`upgrade`/`feat` 로 먼저 커밋하고 다시 돌린 뒤 기록한다.

## 코드를 쓸 때

- 새 모듈은 [`docs/SPEC_KoTokenLab.md`](docs/SPEC_KoTokenLab.md) 의 어느 절을
  구현하는지 docstring 첫 줄에 밝힌다. `src/` 의 스텁들이 이미 그렇게 되어 있다
- 재사용 먼저: `src/utils/` 에 시드·해시·환경·원장·run 추적이 이미 있다
- 주석은 **왜**를 적는다. 무엇을 하는지는 코드가 말한다
- 파일 인코딩 UTF-8, 줄바꿈 LF (`.gitattributes` 가 강제한다)

## 막혔을 때

규칙과 코드가 어긋나 보이면 코드가 맞다. 훅과 검사기가 실제 집행자다:

- [`tools/check_commit_msg.py`](tools/check_commit_msg.py)
- [`tools/precheck.py`](tools/precheck.py)
- [`tools/validate_ledger.py`](tools/validate_ledger.py)
- [`src/utils/ledger.py`](src/utils/ledger.py) — 스키마 정의

규칙 자체가 틀렸다고 판단되면 사용자에게 말하고, `docs/RULES.md` 를 고쳐서
`docs(docs):` 로 커밋한다. 조용히 우회하지 않는다.
