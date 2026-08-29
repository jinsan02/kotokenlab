# 인수인계 — Codex 로 이어서 작업하기

작성 2026-08-29. 다른 에이전트가 이 저장소를 이어받을 때 읽는 문서다.
규칙은 [`RULES.md`](RULES.md), 계획은 [`PLAN.md`](PLAN.md), 진입점은 [`../AGENTS.md`](../AGENTS.md).

---

## 지금까지 된 것

```
Step 0  저장소·원장·훅·CI·문서·환경                          완료
        모델 5종 (models.tsv 에 revision·구조 기록)          완료
        자원 실측 + attention 백엔드 함정 발견               완료
Step 1  데이터 파이프라인 (소규모 관통)                      완료
Step 2  Level 1 벤치마크 (파일럿)                            완료
        ────────────────────────────────────────────────
        전체 규모 데이터 (raw 10~15GB -> 정제 3~5GB)         다음
```

파일럿 실적: FineWeb-2 `kor_Hang` 60,000문서 수집 → 57,078 통과 → 정제 221MB,
train/dev/final_test = 51,399 / 2,846 / 2,833. 209초.
Level 1 결과는 `experiments/tokenizer_metrics.tsv` (커밋됨, run_id `tok_bench_pilot`).

핵심 수치 (dev, tok/char, Qwen2.5 대비):

| 도메인 | Qwen | HCX | A.X |
|---|---:|---:|---:|
| news | 0.7145 | −26.9% | **−41.1%** |
| web_general | 0.6940 | −24.9% | −39.6% |
| technical | 0.6567 | −24.0% | −38.4% |
| ko_en_mixed | 0.4836 | −19.0% | −27.5% |

---

## 바로 다음에 할 일 (우선순위 순)

### 1. 전체 규모 데이터 파이프라인 — Step 1 본편

```bash
.conda/python.exe scripts/run_data_pipeline.py \
    --max-docs 1500000 --max-bytes 6000000000 --shards 4 --tag v1
```

`--shards 4` 로 서로 다른 샤드에 걸쳐 표본을 뽑는다. **한 샤드 앞부분만 읽으면
안 된다** — 실측으로 확인했듯 tripadvisor 한 호스트가 24% 를 차지한다.

예상: 다운로드 수 GB, 처리 1~2시간. 끝나면 `manifest_sha256` 을 `PLAN.md` 에 적고
`data(data):` 커밋 (트레일러 `Manifest-SHA256:` 필수).

**해결해야 할 것**: 파일럿에서 `web_general` 이 문서 수 기준 약 73% 였다.
`configs/data/domain_rules.yaml`
에 호스트 규칙을 보강하되, **편향된 표본으로 과적합하지 마라.** 전체 규모 표본의
상위 호스트를 먼저 뽑아보고 결정한다.

### 2. 도메인 규칙 정확도 검증

무작위 200건을 손으로 라벨링해 규칙의 정확도를 재고 리포트에 적는다.
**오류율을 모른 채 도메인별 결과를 주장하면 안 된다.**
`scripts/audit_domain_rules.py` 는 아직 없다 — 만들어야 한다.

### 3. 영어·코드 대조군 추가

지금 Candidate Gate 의 "통과"는 **한국어 조건만 본 것**이다. 영어와 코드
도메인이 코퍼스에 없어 regression 을 판정하지 못한다 (스펙 §17 이 요구하는
영어 ≤5%, 코드 ≤10% 를 검증할 수 없다).

- 영어: `HuggingFaceFW/fineweb-edu` `sample-10BT`
- 코드: `codeparrot/github-code-clean`

`scripts/run_data_pipeline.py` 는 지금 한국어 전용이다. 다국어 소스를 받도록
확장하거나 별도 스크립트를 만든다.

### 4. dev 크기 하한

파일럿 dev 는 도메인당 0.06~8MB 로 편차가 크다. 백과 0.06MB, 코드 1건이면
BPB 표준오차가 너무 커서 비교가 무의미하다 (검토 A5).
전체 규모에서 **도메인당 최소 5MB** 를 확보하고, 못 채우는 도메인은
결과 표에 "표본 부족"으로 표시한다.

### 5. 그 다음

`PLAN.md` 의 5~12주 일정을 따른다. 다음 마일스톤은
**Level 1 정식 측정 → Candidate Gate → `phase1-tokenizer-freeze` 태그**.

---

## 반드시 지켜야 하는 것

이 셋은 어기면 결과가 통째로 무효가 된다.

1. **attention 백엔드를 강제한다.** `attn_implementation="sdpa"` 만 주면 이 환경의
   디스패처가 MATH 로 떨어져 8,192 토큰에서 메모리 7.1배, 시간 10.3배가 된다.

   ```python
   from torch.nn.attention import SDPBackend, sdpa_kernel
   EFFICIENT_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]
   with sdpa_kernel(EFFICIENT_SDPA):
       ...
   ```

2. **byte fallback 토큰을 pruning 하지 않는다.** `src/tokenizer/protected.py` 의
   `protected_token_ids()` 를 후보에서 빼고, 직후 `assert_byte_roundtrip()` 을 부른다.

3. **노이즈 플로어를 먼저 잰다.** 첫 CPT 실험은 동일 config 3 seed 다.
   σ 를 모르면 이후 모든 비교를 해석할 수 없다.

전체 규칙: [`RULES.md`](RULES.md) 18개.

---

## 알아두면 시간 아끼는 것

- python 은 항상 `C:\llm_tokenizer\.conda\python.exe` 절대경로. `conda activate` 쓰지 않는다
- 환경을 바꿨으면 `-m src.utils.env --register "이유"` 먼저. 안 하면 `RunContext` 가 거부한다
- 매니페스트 본체는 커밋 대상이 아니다. `SUMMARY.tsv` 만 커밋한다
- `record` 커밋에 코드를 섞으면 훅이 거부한다. 코드는 먼저 `fix`/`upgrade`/`feat` 로
- 학습 운영 설정: **seq 2048 / micro_bs 2 / AdamW 8bit** → 13.3GB, 9,089 tok/s
- seq 8192 학습은 VRAM 을 넘겨 호스트 메모리로 흘러 속도가 절반이 된다. 쓰지 마라
- HF 캐시 심볼릭 링크가 실패하면 `HF_HUB_DISABLE_SYMLINKS=1` 로 재시도
- `.hf_cache/` 가 저장소 안에 있다. `git clean -xdf` 하면 5GB 가 날아간다

---

## 아직 안 만든 것

| 파일 | 용도 |
|---|---|
| `scripts/audit_domain_rules.py` | 도메인 규칙 정확도 수동 검증 |
| `src/tokenizer/train.py` | T2a(축소) / T2b(치환) 학습 |
| `src/tokenizer/prune.py` `substitute.py` | vocab pruning + 한국어 치환 |
| `src/surgery/*.py` | embedding resize, E0/E1 초기화 |
| `src/training/cpt.py` `callbacks.py` | CPT 루프 |
| `src/evaluation/bpb.py` | Level 2 BPB |
| `src/evaluation/latency.py` `memory.py` | Level 4 |

`src/` 의 스텁에는 각각 스펙 절 번호가 docstring 에 적혀 있다.

---

## 세션 시작 시 상태 복원

```bash
cd C:\llm_tokenizer
git log --oneline -20
git status
tail -n 15 experiments/LEDGER.tsv
.conda\python.exe -m src.utils.env --check
.conda\python.exe -m pytest tests/ -q
```

**앞선 대화를 알고 있다고 가정하지 않는다.** 상태는 커밋과 `experiments/` 에만 있다.
