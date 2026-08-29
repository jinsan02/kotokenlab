# 에이전트 프롬프트 모음

Codex / Claude Code 에 그대로 붙여 넣는 프롬프트다.
전제: `cd C:\llm_tokenizer` 에서 시작하고, 저장소 규칙은 [`RULES.md`](RULES.md) 를 따른다.

---

## 0. 세션 시작 (매번)

```
C:\llm_tokenizer 프로젝트를 이어서 작업한다.

먼저 이 순서로 읽고 현재 상태를 복원해라.
1. docs/RULES.md        하드룰 17개 — 매 세션 읽는다
2. docs/HANDOFF.md      지금까지 된 것과 다음 할 일
3. docs/PLAN.md         범위·일정·사전 등록 질문 6개

그 다음 아래를 실행해서 상태를 확인해라.
  git log --oneline -20
  git status
  tail -n 15 experiments/LEDGER.tsv
  .conda\python.exe -m src.utils.env --check
  .conda\python.exe -m pytest tests/ -q

python 은 항상 C:\llm_tokenizer\.conda\python.exe 절대경로로 부른다.
앞선 대화 내용을 알고 있다고 가정하지 마라. 상태는 커밋과 experiments/ 에만 있다.

확인이 끝나면 무엇을 할 차례인지 말하고, 내 확인을 받은 뒤에 시작해라.
```

---

## 1. 소규모 관통 (파이프라인 검증용)

새 필터나 도메인 규칙을 바꿨을 때, 전체 규모로 가기 전에 이걸 먼저 돌린다.
**10분 안에 끝나고 GPU 를 쓰지 않는다.**

```
데이터 파이프라인을 소규모로 한 번 관통시켜서 변경이 깨지지 않았는지 확인해라.

1) 파이프라인 실행 (약 4분, 다운로드 ~230MB)
   .conda\python.exe scripts\run_data_pipeline.py --max-docs 20000 --max-bytes 150000000 --tag smoke

2) Level 1 벤치마크 (약 3분, dev 분할 대상)
   .conda\python.exe -m src.evaluation.tokenizer_eval --split dev --tag smoke

3) 검사
   .conda\python.exe tools\validate_ledger.py
   .conda\python.exe -m pytest tests/ -q

확인할 것:
- 필터 통과율이 90~96% 범위인가 (파일럿 95.1%). 크게 벗어나면 필터가 바뀐 것이다
- 탈락 사유 분포가 파일럿과 비슷한가 (too_short ~2.6%, repeated_lines ~2.2%)
- ko_en_mixed 도메인이 3~6% 인가. 40% 가 나오면 도메인 규칙이 깨진 것이다
  (한글 비율이 아니라 latin/(hangul+latin) 을 봐야 한다 — src/data/domain.py 참조)
- web_general 비율. 파일럿은 문서 수 기준 약 73% 이고, 규칙을 보강했다면 줄어야 한다
- Level 1 의 tok/char 가 파일럿과 비슷한가
  (Qwen 0.687 / HCX 0.516 -24.9% / A.X 0.416 -39.4% 근방)

수치가 파일럿과 크게 다르면 **원인을 먼저 설명**하고, 의도한 변화인지 내게 확인받아라.
결과 기록은 record(...) 커밋으로 하되 코드는 섞지 마라 (훅이 거부한다).
파일럿 기준값은 experiments/tokenizer_metrics.tsv 의 run_id=tok_bench_pilot 행에 있다.
```

---

## 2. 전체 규모 데이터 파이프라인 (Step 1 본편)

```
Step 1 데이터 파이프라인을 전체 규모로 돌린다. docs/HANDOFF.md 의 1~4번 항목이 대상이다.

순서:
1) 먼저 표본의 호스트 분포를 조사해라. 4개 샤드에 걸쳐 2만 건만 읽어서
   상위 호스트 50개를 뽑아 보여줘라. 편향된 표본으로 규칙을 과적합하면 안 되므로
   반드시 --shards 4 로 여러 샤드에 걸쳐 뽑아라.

2) 그 결과를 보고 configs/data/domain_rules.yaml 의 호스트 규칙을 보강해라.
   파일럿의 web_general 은 문서 수 기준 약 73% 다. 목표는 40% 이하다.
   규칙을 바꾸면 manifest 가 바뀌므로 upgrade(data) 로 먼저 커밋해라.

3) scripts/audit_domain_rules.py 를 만들어라.
   무작위 200건의 (url, 본문 앞부분, 규칙이 매긴 도메인) 을 TSV 로 뽑아
   사람이 손으로 정답을 채울 수 있게 한다. 규칙 기반 분류의 오류율을 모른 채
   도메인별 결과를 주장하면 안 된다.

4) 전체 규모 실행 (1~2시간)
   .conda\python.exe scripts\run_data_pipeline.py --max-docs 1500000 --max-bytes 6000000000 --shards 4 --tag v1

5) 도메인별 dev 크기를 확인해라. 도메인당 최소 5MB 가 목표다.
   못 채우는 도메인은 리포트에 "표본 부족"으로 표시해야 한다 (docs/REVIEW.md A5).

6) manifest_sha256 을 docs/PLAN.md 에 적고 data(data) 로 커밋해라.
   트레일러에 Manifest-SHA256 이 반드시 있어야 한다.

각 단계가 끝날 때마다 결과를 보여주고 다음으로 넘어가도 되는지 물어봐라.
```

---

## 3. 영어·코드 대조군 추가

```
Candidate Gate 의 regression 조건을 판정할 수 있게 영어와 코드 도메인을 추가해라.

지금 gate 의 "통과"는 한국어 조건만 본 것이다. 스펙 §17 이 요구하는
영어 악화 ≤5%, 코드 악화 ≤10% 를 검증할 코퍼스가 없다.

- 영어: HuggingFaceFW/fineweb-edu 의 sample-10BT
- 코드: codeparrot/github-code-clean

scripts/run_data_pipeline.py 는 지금 한국어 전용이다(한글 비율 필터가 걸린다).
언어별로 필터를 다르게 적용할 수 있게 확장하거나 별도 스크립트를 만들어라.
어느 쪽이든 같은 manifest 스키마와 같은 분할 규칙(doc_id 해시)을 쓴다.

규모는 도메인당 dev 5MB 이상이면 충분하다. 학습용이 아니라 regression 측정용이다.
```

---

## 4. 토크나이저 학습 (T2a / T2b) — Step 3

```
스펙 §12 의 T2 를 두 조건으로 구현해라. 이게 이 프로젝트의 신규성 방어 지점이다.

  T2a  저빈도 토큰 pruning 만 (vocab 축소)   ← 선행연구 arXiv:2604.16235 의 설정
  T2b  pruning + 한국어 고효율 토큰 치환 (vocab 크기 유지)  ← 우리 주장

같은 파이프라인에서 직접 비교해야 "축소 vs 치환" 이 결과로 남는다.

반드시 지킬 것:
- byte fallback 256개와 special token 은 절대 pruning 하지 않는다.
  src/tokenizer/protected.py 의 protected_token_ids() 를 후보에서 빼라.
- pruning·치환 직후 assert_byte_roundtrip() 을 호출해라. 안 하면 처음 보는
  입력에서 토크나이저가 조용히 깨지고, 그 실패는 학습을 한참 돌린 뒤에 드러난다.
- Qwen 은 BPE 라서 단어만 추가해서는 확장되지 않는다. merge rule 이 필요하다.
  추가한 토큰이 실제로 사용되는지 인코딩으로 검증하는 테스트를 넣어라.
  참고: https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
        https://github.com/KaihuaTang/Qwen-Tokenizer-Pruner
- vocab 은 config.vocab_size 151,936 이고 실제 토큰은 151,665 다.
  271칸이 비어 있고 그 행들은 학습 신호를 거의 못 받은 벡터다.
  초기화 통계를 낼 때 [:len(tokenizer)] 로 잘라내라.

산출물은 artifacts/tokenizers/<version>/ 에 두고 sha256 만 원장에 기록한다.
tok(tok) 커밋에 Tokenizer-SHA256 트레일러가 필요하다.
```

---

## 5. 노이즈 플로어 측정 — Step 5 (CPT 전에 반드시)

```
동일 config 를 seed 42 / 123 / 2026 으로 3회 돌려 σ_BPB 를 측정해라.

이걸 먼저 하지 않으면 이후 모든 비교를 해석할 수 없다. "BPB 1.207 vs 1.198" 이
의미 있는 차이인지 판단할 근거가 없고, Candidate Gate 가 노이즈로 후보를
탈락시킬 수 있다 (docs/RULES.md 10번).

- 예산은 각 5M 토큰, 총 15M. 약 30분.
- run_id 는 noise_ 로 시작한다.
- 학습 설정: seq 2048 / micro_bs 2 / AdamW 8bit / bf16 / gradient checkpointing
  (실측 13.3GB, 9,089 tok/s — reports/tables/resource_probe.md)
- attention 은 반드시 sdpa_kernel([EFFICIENT_ATTENTION, CUDNN_ATTENTION]) 안에서
- LR 스케줄의 x축은 step 이 아니라 raw_bytes 다 (docs/RULES.md 12b)

측정한 σ 를 docs/PLAN.md 의 사전 등록 질문 절에 적어라.
이후 모든 비교는 Δ > 2σ 일 때만 "차이 있음" 으로 보고하고, 그 미만은
"구별 불가" 로 명시한다.
```

---

## 6. 결과 기록 (실험을 돌린 뒤 항상)

```
방금 돌린 실험 결과를 원장에 기록하고 커밋해라.

- 결과는 RunContext 를 통해서만 원장에 들어간다. 손으로 TSV 를 쓰지 마라.
- 커밋은 record(<scope>) 이고 코드·설정을 함께 스테이지하면 훅이 거부한다.
  코드를 고쳐야 하면 먼저 fix/upgrade/feat 로 커밋하고 다시 돌린 뒤 기록해라.
- 트레일러 필수: Run-Id, Ledger, Config-SHA256
- Run-Id 는 experiments/LEDGER.tsv 에 실재해야 한다 (훅이 확인한다)

커밋 메시지 본문에는 **무엇을 했는지가 아니라 무엇을 알게 됐는지**를 써라.
그리고 이 결과의 한계를 반드시 한 문단 적어라 — 표본이 작다, 대조군이 없다,
도메인 편향이 있다 같은 것. 한계를 적지 않은 기록은 나중에 과신하게 만든다.
```

---

## 프롬프트를 쓸 때

- **단계마다 확인을 받게 한다.** "각 단계가 끝나면 결과를 보여주고 물어봐라" 를 넣는다
- **기준값을 준다.** "파일럿에서 95.1% 였다" 처럼 비교 대상이 있어야 이상을 알아챈다
- **하지 말 것을 명시한다.** 규칙 문서를 읽으라고만 하면 안 읽는다
- 긴 작업은 `--tag` 를 다르게 줘서 원장에서 구분되게 한다
