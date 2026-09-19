# 에이전트 프롬프트 모음

Codex / Claude Code 에 그대로 붙여 넣는 프롬프트다.
전제: `cd C:\llm_tokenizer` 에서 시작하고, 저장소 규칙은 [`RULES.md`](RULES.md) 를 따른다.

---

## 0. 세션 시작 (매번)

```
C:\llm_tokenizer 프로젝트를 이어서 작업한다.

먼저 이 순서로 읽고 현재 상태를 복원해라.
1. docs/RULES.md         하드룰 17개 — 매 세션 읽는다
2. docs/HANDOFF.md       지금까지 된 것과 다음 할 일
3. docs/PLAN.md          범위와 사전 등록 전부 (1차 Q1~Q6 · 2차 Q7/R5/D · 3차 "P3 확장")
4. docs/DESIGN_DELTA.md  스펙과 다르게 한 것과 그 이유
   — 스펙만 읽고 코드를 고치면 이미 반증된 가설을 되살리게 된다
5. reports/FINAL_REPORT.md  1차 결과 전체 (2026-09-02 종료)
   — Q1~Q6 의 답, 반증된 가설 셋, 권고, 한계
6. docs/SPEC_P3.md · docs/SCHEDULE_P3.md   3차 설계와 일정 — 지금 여기서 일한다
   — 2차(SPEC_P2 · SCHEDULE_P2)는 2026-09-17 에 닫혔다. 결과는 SCHEDULE_P2

그 다음 아래를 실행해서 상태를 확인해라.
  git log --oneline -20
  git status
  tail -n 15 experiments/LEDGER.tsv
  .conda\python.exe -m src.utils.env --check
  .conda\python.exe tools/check_clock.py --record
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

확인할 것 (규칙 v4 + 호스트 상한 400 기준):
- 필터 통과율 90~92% (host_cap 이 4.3% 를 걷어내므로 95% 가 아니다)
- 탈락 사유: host_cap ~4.3%, too_short ~2.9%, repeated_lines ~1.8%
- 도메인 분포: web_general ~64%, news ~21%, blog ~5%, ko_en_mixed ~5%
  ko_en_mixed 가 40% 가 나오면 도메인 규칙이 깨진 것이다 (한글 비율이 아니라
  latin/(hangul+latin) 을 봐야 한다 — src/data/domain.py)
- 한 호스트가 상한을 넘겼는지. tripadvisor 가 상한 없이는 6.66% 를 차지한다
- Level 1 tok/char: 한국어 Qwen 0.683 / HCX -25.2% / A.X -39.4%
  영어 Qwen 0.217 / A.X +7.3%,  코드 Qwen 0.296 / A.X +26.6%
- **도메인별 수치를 인용하기 전에 docs/DOMAIN_LABELS.md 를 읽어라.**
  한국어 내부 세분화는 감사 정확도 ~55% 라 news / 기타 까지만 보고한다

수치가 파일럿과 크게 다르면 **원인을 먼저 설명**하고, 의도한 변화인지 내게 확인받아라.
결과 기록은 record(...) 커밋으로 하되 코드는 섞지 마라 (훅이 거부한다).
기준값은 experiments/tokenizer_metrics.tsv 의 run_id=tok_bench_ctrl 행에 있다.
tok_bench_pilot 이 아니다 — 파일럿에는 english 행이 아예 없고 code 도 한국어
코퍼스의 code 라벨이라 위 0.296 과 다른 값(0.717)이다.
```

---

## 1b. 도메인 규칙을 고쳤을 때 (라벨 재사용, 추가 라벨링 없음)

```
도메인 규칙을 고쳤으니 이미 있는 블라인드 라벨로 채점해라.
사람에게 다시 라벨링을 시키지 마라 — reports/tables/domain_audit_v5.tsv 에
블라인드로 찍은 150건이 있다.

  .conda\python.exe scripts\eval_domain_rules.py --errors      # dev 105건
  .conda\python.exe scripts\eval_domain_rules.py --holdout     # 최종 1회만

dev 로는 몇 번이든 고쳐도 된다. **holdout 은 최종 측정 전용**이고, 그 숫자를
보고 규칙을 고치면 holdout 이 dev 가 된다.

기준값: 규칙 v4 호스트만 = dev 58.1% / holdout 55.6%.
내용 신호를 켜면 dev 70.5% 로 오르지만 holdout 53.3% 로 내려간다 — 과적합이다
(docs/DOMAIN_LABELS.md 4차). dev 만 보고 채택하지 마라.

dev 와 holdout 격차가 10%p 를 넘으면 과적합을 의심하고 내게 보고해라.
```

---

## 1c. 새 감사 표본이 필요할 때 (반드시 블라인드)

```
도메인 라벨 정확도를 다시 재야 한다. 새 표본을 뽑아 라벨링 화면을 만들어라.

  .conda\python.exe scripts\audit_domain_rules.py --mode sample --shards 4 \
      --seed <기존과 다른 seed> --sample-size 150 --out reports\tables\domain_audit_v6.tsv
  .conda\python.exe scripts\make_label_ui.py --blind \
      --audit reports\tables\domain_audit_v6.tsv --out reports\tables\domain_audit_v6_label.html

**--blind 를 빼지 마라.** 예측을 보여주면 사람이 그대로 수용해서 정확도가
측정이 아니라 항등식이 된다. 실제로 두 번 그랬다 — 88%(부풀려짐)와
100%(무효). 블라인드로 재니 54.7% 였다 (docs/DOMAIN_LABELS.md).

seed 는 기존 표본(42, 123, 2026)과 달라야 하고, 겹침 건수를 확인해서 보고해라.
채워진 TSV 를 받으면 html.unescape 로 이스케이프를 되돌린 뒤 적용해라.
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

## 5. 노이즈 플로어 측정 — **새 조건이 생길 때마다**

```
동일 config 를 seed 42 / 123 / 2026 으로 3회 돌려 σ_BPB 를 측정해라.

이걸 먼저 하지 않으면 이후 모든 비교를 해석할 수 없다. "BPB 1.207 vs 1.198" 이
의미 있는 차이인지 판단할 근거가 없고, Candidate Gate 가 노이즈로 후보를
탈락시킬 수 있다 (docs/RULES.md 10번).

- 예산은 각 17.5MB **원문 바이트**. 토큰이 아니다 — 토크나이저가 다르면
  같은 토큰수가 다른 분량이 된다 (docs/RULES.md 12b번). 조건당 약 33분.
- **조건마다 따로 잰다.** sigma 는 조건별로 18.9배까지 다르다 — 손상된 조건일수록
  seed 에 따라 회복 궤적이 갈린다 (reports/tables/noise_floor.md).
  안 잰 조건의 비교는 규칙대로 "구별 불가" 로 남긴다.
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
- Run-Id 와 Invalidates 의 run_id 는 **인덱스(= 이 커밋이 만들 트리)의**
  LEDGER.tsv 에 있어야 한다. 작업 트리에 있는 것만으로는 안 된다.
  원장 행을 같은 커밋에 함께 스테이지하거나, 먼저 record 로 커밋하고 참조해라
  (docs/COMMIT_CONVENTION.md 3번 — 이 순서를 어겨 CI 가 한 번 빨개졌다)

커밋 메시지 본문에는 **무엇을 했는지가 아니라 무엇을 알게 됐는지**를 써라.
그리고 이 결과의 한계를 반드시 한 문단 적어라 — 표본이 작다, 대조군이 없다,
도메인 편향이 있다 같은 것. 한계를 적지 않은 기록은 나중에 과신하게 만든다.
```

---

## 7. Phase 4 — 등토큰 예산과 N 스윕 (**2026-09-01 완료**)

> 끝난 실험이다. 결과는 [`../reports/tables/phase4.md`](../reports/tables/phase4.md).
> 이 프롬프트는 같은 모양의 CPT 를
> 다시 짤 때의 본보기로만 남긴다.

```
scripts/run_phase4.sh 를 돌린다. 약 4.3시간.

먼저 읽어라: reports/tables/cpt_main.md 와 docs/PLAN.md "Phase 4 사전 등록".
**예측이 이미 적혀 있다.** 결과가 예측과 다르면 그게 더 중요한 발견이다 —
예측을 사후에 고치지 마라 (docs/RULES.md 14번).

시작 전에 반드시:
  - nvidia-smi 로 GPU 여유를 확인한다. 지난 CPT 에서 peak_alloc 11.4GB 에
    데스크톱 앱까지 더해져 16.3GB 중 96% 가 찼다. 여유 550MB 였다
  - 사용자에게 "게임·영상 편집을 켜면 OOM 으로 몇 시간이 날아간다" 고 알린다
  - tools/check_clock.py --record

돌리는 중에는 GPU 작업을 하나만 띄운다 (CLAUDE.md).
끝나면 6번 프롬프트로 기록하고, 조건별 최종표를 reports/tables/ 에 남겨라.
```

---

## 8. Q7 — tie 를 끊는다 (**2026-09-15 완료 — 구별 불가, S4 취소**)

> 끝난 실험이다. 결과는 [`SCHEDULE_P2.md` "Day 2"](SCHEDULE_P2.md), record `36c8b5f`.
> 게이트 단계에서 멈추는 설계의 본보기로 남긴다. 다음 작업은 **9번**.

```
Q7 을 돌린다. tie_word_embeddings 를 끊고 회복률이 달라지는지 본다.

먼저 읽어라. 순서대로다.
  docs/PLAN.md "Q7"       사전 등록 — 가설·예측·판정 수식·효과 크기 바닥·중단 기준
  docs/SPEC_P2.md §9      실행 설계 — untie 방법, 단계, 예산, VRAM, 도구 결함
  reports/tables/cpt_main.md   비교 대상인 tied 결과

**예측이 이미 적혀 있다(72~78%). 결과가 예측과 다르면 그게 더 중요한 발견이다.
예측을 사후에 고치지 마라** (docs/RULES.md 14번).

## 반드시 지킬 것 — 어기면 실험이 성립하지 않는다

- **untied-C0 대조군을 빼지 마라.** tie 를 끊으면 파라미터가 +27.6%
  (494,032,768 -> 630,167,424) 늘어난다. 대조군이 없으면 "tie 를 풀어서" 인지
  "파라미터가 늘어서" 인지 영영 못 가른다. 회복률의 기준선은 tied-C0 이 아니라
  **untied-C0 의 BPB** 다
- **from_pretrained(tie_word_embeddings=False) 를 쓰지 마라.** transformers 5.16 이
  lm_head 를 무작위 초기화한다 (검증됨, SPEC_P2 §9.1). tied 로 적재한 뒤
  embedding 사본으로 clone 해서 끊어라
- **untie 를 run_surgery.py 안에 넣지 마라.** 기존 artifacts/models/t2b_mean 에
  거는 후처리로 해야 embedding 이 1차와 비트 동일하다
- **seq_len 을 줄이지 마라.** 메모리가 모자라면 micro_bs 를 낮추고 accum 을 올려
  유효 32,768 tokens/step 을 유지한다
- **예산과 LR 스케줄은 바이트 기준(cosine_by_raw_bytes)을 유지한다.** step 이나
  token 으로 바꾸면 비교 대상이 토크나이저가 아니라 학습률이 된다
- **이 목록에 없는 새 모듈·도구·원장 테이블을 추가하지 마라.** 필요하다고
  판단되면 제안만 하고 확인받아라

## 순서 — 단계마다 결과를 보여주고 확인을 받아라

S0  scripts/probe_resources.py 에 --model 과 --out 을 먼저 붙여라.
    지금은 hub 모델을 하드코딩하고 reports/tables/resource_probe.md 를
    write_text 로 덮어쓴다 — 그대로 돌리면 1차 기록이 사라진다.
    upgrade(infra) 로 커밋한 뒤 untied 0.5B 를 재라. **reserved 기준이다.**
    (약 5분)

S1  untied-C0 / untied-T2b 의 학습 전 BPB. **정합성 게이트다.**
    2.380297 / 1.156880 이 나와야 한다 — untie 는 forward 를 안 바꾸므로
    구성상 고정값이다. 다르면 발견이 아니라 버그이고 거기서 멈춘다. (약 8분)

S2·3 17.5MB x seed 42/123/2026 x 조건 2개. seed 42 를 먼저 돌려 파이프라인을
    확인한 뒤 나머지를 이어라. **이 단계가 중단 판정의 σ 를 만든다.** (약 1h12m)

    -> 여기서 PLAN.md "Q7" 의 판정 규칙으로 "구별 불가" 가 나오면
       **S4 를 돌리지 말고 정리해라.** 그게 결과다.

S4  168.5MB. untied-T2b 3 seed + untied-C0 (SPEC_P2 §9.3 의 발동 조건대로
    1 seed 또는 3 seed). 약 5h53m ~ 9h42m.

## 시작 전에

  - nvidia-smi 로 여유를 확인한다. **12.2GB 를 예상하는데 1차 Phase 4 때
    여유가 550MB 였다** — 브라우저·Steam·Discord 를 닫아야 들어간다
  - 사용자에게 몇 시간짜리이고 그동안 GPU 를 쓰면 안 된다고 알린다
  - .conda\python.exe tools/check_clock.py --record
  - GPU 작업은 하나만 띄운다 (CLAUDE.md)

## 끝나면

6번 프롬프트로 기록하고, 조건별 최종표를 reports/tables/q7_untie.md 에 남겨라.
결과가 어느 쪽이든 docs/DESIGN_DELTA.md 에 스펙 / 실제 / 왜 / 근거 네 항목으로
적어라 — 1차 산출물이 "tie 를 풀지 않는다" 를 규칙으로 갖고 있고, Q7 은 그것을
의도적으로 깨는 별개 조건이기 때문이다.

**반증이 나오면 그것도 성과다. 긍정 결과를 찾으러 가지 마라.**
```

---

## 9. P3 W0 — 코드와 게이트 (다음 작업)

> 2차는 닫혔다. 결과는 [`SCHEDULE_P2.md`](SCHEDULE_P2.md). 3차의 어느 run 도
> 이 W0 전에는 돌릴 수 없다.

```
P3 의 W0 를 한다. GPU 는 D0 보정과 1.5B 탐침에만 수 분 쓴다.

먼저 읽어라. 순서대로다.
  docs/SPEC_P3.md §3       W0-1 ~ W0-12 — 무엇을 왜 고치는가
  docs/PLAN.md "P3 확장"    사전 등록 — 예측과 판정 경계. 여기서 고치지 않는다
  docs/SCHEDULE_P3.md "W0"  순서
  src/training/cpt.py · src/training/callbacks.py · tools/compare_runs.py

## 반드시 지킬 것

- **인자를 안 주면 P2 와 똑같이 돌아야 한다.** --warmup-bytes 와
  --pool-extend-docs 의 기본값은 "기존 동작" 이다. 이것을 테스트로 먼저 못 박고
  나서 구현하라. 과거 run 과의 비교가 여기에 달려 있다
- **--pool-extend-docs 는 기존 풀(첫 --pool-docs 개)을 기존과 같은 seed 로 섞은
  순서를 그대로 두고, 추가분을 그 뒤에 붙인다.** 추가분만 따로 섞는다.
  "기존 풀 순서가 비트 단위로 같다" 를 테스트한다
- **train_curve 에 컬럼을 붙일 때는 뒤에만 붙인다.** 과거 행은 NA 다
  (src/utils/ledger.py 의 기존 방식을 따른다)
- **compare_runs.py 에 새 필드를 분류하지 않으면 전부 "미분류 = 치명" 으로 막힌다.**
  warmup_bytes · pool_extend_docs 는 치명, eval_at · save_at 은 시간
- **W0-9 (R5 배증당 증분 비) 가 (0.5, 0.85] 밖이면 멈추고 알려라.** P3-A 는
  재등록 전에 돌리지 않는다
- **D0 보정과 1.5B 탐침의 결과는 게이트다.** 결과를 보여주고, D2 와 1.5B 본 실험을
  할지 확인받아라. 1.5B 본 실험은 별도 사전 등록 커밋이 먼저다
- **KMMLU 나머지 39과목은 받지 마라.** W3 에서 사용자 확인을 받은 뒤다
- **run 이 도는 동안 그 run 이 쓰는 모듈을 커밋하지 마라**
- **이 목록에 없는 새 모듈·도구·원장 테이블을 추가하지 마라.** SPEC_P3 §3 에 있는
  것만 만든다. 필요하다고 판단되면 제안만 하고 확인받아라

## 완료 기준

- SPEC_P3 §3 의 W0-1 ~ W0-12 가 각각 feat/fix 커밋으로 존재하고 푸시됐다
- pytest 전체 통과, tools/validate_ledger.py 통과
- tools/p3_verdicts.py 가 아직 run 이 없는 상태에서 "미실행" 표를 쓴다
- D0 · G 탐침 결과가 record 커밋으로 남았고, 게이트 판정을 사용자에게 보고했다

각 단계가 끝나면 결과를 보여주고 물어봐라.
```

---

## 10. 외부 AI 에게 전체 검토를 받을 때

> 저장소를 못 보는 상대에게 `reports/OVERVIEW.md` 하나만 주고 받는 검토다.
> 그 파일은 `tools/overview.py` 가 원장에서 만든다 — 붙여 넣기 전에 다시 돌려
> 최신인지 확인한다.

```
아래는 1인 연구 프로젝트 "KoTokenLab" 의 전체 개요다. 나는 이 프로젝트의
저자이고, 당신에게 **비판적 검토** 를 받으려 한다.

[여기에 reports/OVERVIEW.md 전문을 붙여 넣는다]

## 당신이 할 일

이 설계와 결과를 읽고 **틀린 곳 · 과잉 주장 · 빠진 대조군** 을 찾아라.
칭찬은 필요 없다. 내가 놓친 것을 찾는 것이 목적이다.

우선순위가 높은 순으로:

1. **주장이 데이터를 넘는 곳.** 이 실험 설계로는 말할 수 없는데 말하고 있는
   문장이 있으면 그 문장을 인용하고, 무엇이 더 필요한지 적어라
2. **빠진 대조군.** "이 비교에는 X 조건이 하나 더 있어야 한다" 가 있으면
   그 조건이 무엇을 가르는지까지 적어라
3. **지표와 정의.** 회복률 R 의 정의(1절)가 결론을 실제로 지탱하는가.
   분모·분자·시점 선택이 결과를 유리하게 만드는 구석은 없는가
4. **통계.** 효과 크기 바닥 5%p + 2σ 규칙, seed 1개인 run 들, bootstrap CI
   사용이 적절한가. 과소·과대 주장이 되는 지점은 어디인가
5. **선행 연구와의 거리.** 이 결과가 이미 알려진 것과 겹치는 부분이 있으면
   구체적인 논문과 함께 지적하라. "새롭지 않다" 는 판단도 환영한다
6. **투고 가능성.** 이 정도면 워크숍/단편 논문이 되는가. 안 된다면 무엇이
   더 있어야 되는가 (마감은 7주 뒤다. 장비는 GPU 한 대뿐이다)

## 규칙

- **추측으로 숫자를 만들지 마라.** 개요에 없는 수치가 필요하면 "이 값이
  없어서 판단 못 한다" 라고 적어라. 없는 값을 가정해 논증하지 마라
- **코드를 새로 써 주지 마라.** 실험 설계와 주장에 대한 검토만 필요하다
- **우선순위를 매겨라.** 지적이 10개면 "이것부터 고쳐라" 순으로 정렬해라.
  치명(결론이 흔들림) / 중간(문장 수정) / 사소 로 나눠라
- 개요는 한국어이고 답도 한국어면 좋다. 용어는 영어 그대로 써도 된다
- 이 프로젝트는 **부정 결과를 결과로 보고하는 것** 을 원칙으로 한다.
  "실패한 실험이 많다" 는 지적이라면, 그것이 왜 문제인지까지 적어라
```

### 답을 받은 뒤

**받은 지적을 저장소에 바로 반영하지 않는다.** 순서는 이렇다.

1. 지적마다 원장·코드로 사실 확인을 한다 (그쪽은 저장소를 못 봤다)
2. 맞는 지적은 `docs/REVIEW.md` 처리 현황 표에 행을 추가한다
3. 실험이 필요한 지적은 **사전 등록부터** 쓴다 (`docs/PLAN.md`)
4. 틀린 지적은 왜 틀렸는지를 한 줄로 남긴다 — 다음에 같은 지적이 또 온다

---

## 프롬프트를 쓸 때

- **단계마다 확인을 받게 한다.** "각 단계가 끝나면 결과를 보여주고 물어봐라" 를 넣는다
- **기준값을 준다.** "파일럿에서 95.1% 였다" 처럼 비교 대상이 있어야 이상을 알아챈다
- **하지 말 것을 명시한다.** 규칙 문서를 읽으라고만 하면 안 읽는다
- **목록 밖 작업을 금지한다.** "이 목록에 없는 새 모듈·도구·원장 테이블을 추가하지
  마라. 필요하다고 판단되면 제안만 하고 확인받아라." 를 넣지 않으면 에이전트가
  남는 판단력으로 안 시킨 개선 작업을 만든다
- **완료 기준을 파일명으로 못 박는다.** "scripts/xxx.py 가 존재하고 N행 TSV 를
  만들면 완료" 처럼. "조사해라" 만으로는 조사했는지 알 수 없다
- **측정 도구를 줄 때는 그 도구의 편향을 함께 적는다.** 예측을 보여주는 라벨링
  화면을 주면서 정확도를 재라고 하면 앵커링된 숫자가 돌아온다
- 긴 작업은 `--tag` 를 다르게 줘서 원장에서 구분되게 한다
