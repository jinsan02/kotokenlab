# 작업 흐름

## 실험 1건의 수명

```
   1. config 작성            configs/<phase>/<name>.yaml
            ↓
   2. RunContext 진입        시간·환경 검증 → run_id → LEDGER status=start
            ↓
   3. 실행                   run.log(...) 로 메트릭 TSV append
            ↓
   4. RunContext 이탈        LEDGER status=ok (또는 fail)
            ↓
   5. 산출물 정리            reports/figures, reports/tables
            ↓
   6. record 커밋            git add experiments/ reports/
```

3~4번은 코드가 자동으로 한다. 사람이 하는 것은 1, 5, 6뿐이다.

### 2~4번 코드

```python
from src.utils.tracking import RunContext, make_run_id

run_id = make_run_id("cpt", "kosub", "mean", "50m", seed=42)

with RunContext(
    run_id, phase="cpt", config=cfg, seed=42,
    model="qwen2.5-0.5b", tokenizer_version="kosub_v3",
    tokenizer_sha256=tok_sha, manifest_sha256=man_sha,
    init_method="mean", target_tokens=50_000_000,
) as run:
    for step, batch in enumerate(loader):
        ...
        if is_eval_point(step):
            run.log("train_curve", step=step, tokens_seen=n_tok,
                    raw_bytes_seen=n_bytes, train_loss=loss,
                    dev_bpb=dev_bpb, lr=lr, grad_norm=gnorm)
    run.tokens_seen = n_tok
    run.raw_bytes_seen = n_bytes
```

예외가 나면 `status=fail` 행과 예외 메시지가 자동으로 남는다.
**실패한 run 도 기록한다** — 기록이 없으면 같은 실패를 반복한다.

### 6번 커밋

```
record(cpt): C2 Ko-Substitute 50M 토큰 CPT 결과

Dev BPB 가 10M 지점 이후 평탄. Equal-Raw-Data 조건.

Run-Id: cpt_kosub_mean_50m_seed42
Ledger: experiments/LEDGER.tsv,experiments/lm_metrics.tsv,experiments/train_curve.tsv
Config-SHA256: <config.json 의 sha256>
```

`record` 커밋에 코드를 섞으면 훅이 거부한다. 자세한 것은
[COMMIT_CONVENTION.md](COMMIT_CONVENTION.md).

---

## 개발 순서 (스펙 §97~§112)

현재 위치는 **검증 단계**다. Step 1 소규모 관통과 Step 2 Level 1 파일럿은
완료했지만, 전체 규모 데이터·도메인 규칙 정확도·영어/코드 대조군은 아직
고정되지 않았다. 이 셋을 확정하기 전에는 토크나이저 학습과 CPT 로 넘어가지 않는다.

| Step | 내용 | 상태 |
|---|---|---|
| 0 | 저장소 뼈대 · 원장 · 커밋 위생 · 환경 | **완료** |
| 1 | 데이터 파이프라인: 정규화 → dedup → manifest → 문서 단위 split | 파일럿 완료 · 본편 검증 중 |
| 2 | 토크나이저 벤치마크 프레임워크 (Qwen / HCX / A.X / Custom 동일 조건) | 파일럿 완료 · 정식 측정 대기 |
| 3 | 토크나이저 학습기: Extend / Substitute / New BBPE + 버전 관리 | |
| 4 | Candidate Gate — CPU 단계에서 후보 거르기 | |
| 5 | Qwen 0.5B tokenizer surgery (resize, LM head, special token, ID mapping) | |
| 6 | Embedding init ablation: Random / Mean / Weighted | |
| 7 | Embedding alignment (Transformer freeze) | |
| 8 | 0.5B Full CPT | |
| 9 | 평가 파이프라인 고정 → `eval-freeze-v1` 태그 | |
| 10 | 최종 후보 multi-seed (42 / 123 / 2026) | |
| 11 | 1.5B scale validation | |
| 12 | 시스템 벤치마크 (RTX 5070 Ti) | |

**Step 1 을 모델 학습보다 먼저 한다.** 분할이 확정되지 않은 상태에서 토크나이저를
학습하면 그 뒤의 모든 결과가 오염된다 ([RULES.md](RULES.md) 1번).

---

## GPU 예산 (스펙 §80)

단일 16GB GPU 다. 처음부터 모든 후보에 50M 토큰을 쓰지 않는다.

| Stage | 예산 | 목적 |
|---|---|---|
| 1 | 1M~5M | 버그 검증, loss 정상 감소, VRAM 확인, 명백한 후보 탈락 |
| 2 | 10M~20M | 후보 간 추세 확인 |
| 3 | 50M+ | 최종 후보만 |

**Fail-fast** (스펙 §81): 한국어 compression +3% 인데 영어 degradation +15% 인
토크나이저는 GPU 로 보내지 않는다.

---

## 두 에이전트가 교대로 일하는 법

이 저장소는 **Claude Code** 와 **Codex** 가 번갈아 작업한다.
서로의 대화 맥락은 볼 수 없다. 그래서:

> **상태 공유 채널은 커밋과 `experiments/` 뿐이다.**

### 세션을 시작할 때 (양쪽 공통)

```bash
git log --oneline -20
git status
tail -n 15 experiments/LEDGER.tsv
.conda/python.exe -m src.utils.env --check
```

이 넷이면 현재 상태가 복원된다. 앞선 대화 내용을 기억하고 있다고 가정하지 않는다.

### 세션을 끝낼 때

- 돌린 실험은 원장에 남기고 커밋한다. **커밋되지 않은 결과는 없는 결과다**
- 작업이 중간이면 그 사실을 커밋 메시지 본문에 적는다
- `git status` 를 깨끗하게 두거나, 남긴 이유를 커밋에 적는다

### 규칙을 고칠 때

[`docs/RULES.md`](RULES.md) **한 곳만** 고친다.
`CLAUDE.md` / `AGENTS.md` 에 규칙을 복사하지 않는다. 갈라진 규칙은 규칙이 아니다.

---

## 자주 쓰는 명령

```bash
# 실험 전 하루 한 번: 외부 시각과 로컬 시계 대조 + 기록
.conda/python.exe tools/check_clock.py --record

# 테스트
.conda/python.exe -m pytest tests/ -q

# 원장 검사 (pre-commit 이 자동으로 돌리지만 수동으로도)
.conda/python.exe tools/validate_ledger.py

# 환경 확인
.conda/python.exe -m src.utils.env

# 훅 연결 (클론 직후 한 번)
git config core.hooksPath .githooks
```
