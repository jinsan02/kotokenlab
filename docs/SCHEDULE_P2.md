# P2 실행 일정 — 하루 1실험

> 설계는 [`SPEC_P2.md`](SPEC_P2.md), 사전 등록은 [`PLAN.md`](PLAN.md).
> 이 문서는 **순서와 시간** 만 담는다. 예측이나 판정 기준을 여기서 바꾸지 마라.

시간은 1차 원장 실측에서 계산했다 (168.5MB CPT 가 C0 6,237초 / T2b 4,344초).
어림한 곳은 그렇다고 적었다.

```
Day 0  준비          코드 · 스모크 · 보정      GPU 거의 안 씀
Day 1  R1  전제      40MB x2                   49분      <- 부정이면 여기서 끝
Day 2  R2  손상 종류  40MB x3                 1h 14m
Day 3  R3  손상 규모  40MB x3                 1h 14m
Day 4  Q7  S0~S3     게이트                   1h 24m      <- 구별 불가면 Day 5 취소
Day 5  Q7  S4 본편   168.5MB                  5h 53m
Day 6  R5  상수 LR   168.5MB x3               3h 37m
Day 7  1.5B + 정리   학습 없음                1h 30m
──────────────────────────────────────────────────────
                                            약 16시간
```

**중단 분기 둘은 정상 경로다.** Day 1 이 부정이면 Day 2·3 이 의미를 잃고,
Day 4 가 구별 불가면 Day 5 를 안 돈다. 둘 다 걸리면 5시간에 끝난다. 그것도 답이다.

---

## Day 0 — 준비 (2026-09-07 완료)

| 한 것 | 커밋 |
|---|---|
| 자원 프로브에 `--model` / `--out` / `--force` | `eed02d1` |
| untied 에서 `lm_head` 기울기를 잡는다 (`grad_norm_head`) | `ba4f386` |
| 상수 LR 스케줄 (`--lr-schedule constant`) | `e9e2228` |
| `scripts/untie_model.py` | `6861874` |
| `scripts/damage_rows.py` (random / mean / permute) | `bd0ad55` |
| 스모크 기록 | `9686f4f` |

만들어 둔 산출물 (`artifacts.tsv` 등록 완료):

```
artifacts/models/untied_c0_qwen     Q7 대조군
artifacts/models/untied_t2b_mean    Q7 본 조건
```

### 스모크가 확인한 것

```
untied        494,032,768 -> 630,167,424 (+27.6%)  텐서 291개  logits 비트 동일
grad 그룹      tied  emb 1 / head 0      untied  emb 1 / head 1
상수 LR        lr 1.00e-05 유지 (코사인은 같은 예산에서 1.62e-06 까지 내려감)
손상 표준편차   0.01423 -> random 0.01563 / mean 0.01057 / permute 0.01423
VRAM          tied 11,935MB  untied 12,758MB  (+823MB, 어림 +816MB 와 1% 이내)
```

### Day 1 전 보정 — 완료 (2026-09-07)

스모크가 사전 등록의 구멍 둘을 드러냈고, 학습 없이 Pre-CPT 로만 닫았다.
표는 [`k_calibration.md`](../reports/tables/k_calibration.md), 커밋 `fddcbb6`.

```
R1 의 K = 40      손상 배율 2.043배 (1차 T2b 2.058배)
R3 의 K = 10 / 40 / 160 / 640
```

**행 40개가 치환 30,000개와 같은 손상을 낸다.** 손상은 몇 행인가가 아니라
어느 행인가로 정해진다. 원래 등록했던 K=1,000 은 이미 3.114배로 과손상이었고,
1k/10k/30k/100k 라는 간격은 커버리지가 73.3% -> 83.0% 로 포화해 네 점이
사실상 한 점이었다.

부수 피해도 봐야 한다 — K>200 에서 영어가 무너지기 시작한다 (K=300 +3.9%,
K=1,000 +29%). K=640 을 보고할 때 영어·코드를 함께 적고 "1차와 같은 종류의
손상" 이라고 부르지 않는다.

---

## Day 1 — R1 (전제)

토크나이저를 안 바꾸고 임베딩 행만 망가뜨린다. **K=40**, `mean` 손상.

```
.conda/python.exe scripts/damage_rows.py --k 40 --how mean --name dmg_mean_k40
.conda/python.exe -m src.training.cpt --model artifacts/models/dmg_mean_k40 \
    --budget-bytes 40000000 --tag r1
.conda/python.exe -m src.training.cpt --model Qwen/Qwen2.5-0.5B --revision 060db64... \
    --budget-bytes 40000000 --tag r1ctrl
```

**C0 대조군은 여기서 한 번만 돌린다.** 40MB CPT 는 원장에 아직 하나도 없고,
Day 2·3 이 같은 행을 재사용한다.

| 결과 | 뜻 | 다음 |
|---|---|---|
| 회복률이 40MB 기준값(59.0~59.4%) 근처 | **법칙은 임베딩의 성질이다** | Day 2 |
| 크게 다르다 | 토크나이저 치환에 특유한 현상이다 | **멈추고 1차 보고서에 한 절** |

---

## Day 2 — R2 (손상 종류)

같은 K, `random` / `mean` / `permute`. 대조군 재사용.

`permute` 가 이 실험의 무게 중심이다. 분포도 노름도 정확히 보존되고 배정만
틀리므로, 회복률이 셋 다 같으면 **"무엇이 틀렸는가" 와 무관하게 벽이 있다** 는
뜻이 된다. 노름 보정 가설을 1차와 다른 각도에서 한 번 더 치는 셈이다.

---

## Day 3 — R3 (손상 규모)

**K = 10 / 40 / 160 / 640**, `mean` 고정. K=40 은 Day 1 것을 재사용한다.

---

## Day 4 — Q7 게이트 (S0 → S1 → S2·3)

```
S0    .conda/python.exe scripts/probe_resources.py --skip-infer --only-cpt-config \
          --model artifacts/models/untied_t2b_mean \
          --out reports/tables/resource_probe_untied.md
S1    untied 둘의 Pre-CPT BPB.  2.380297 / 1.156880 이 나와야 한다
S2.3  17.5MB x seed 42/123/2026 x 조건 2
```

판정은 [`PLAN.md` "Q7"](PLAN.md) — `ΔR >= 5%p AND Δ > 2σ`, tied 기준값 42.26%.

> **S0 전에 브라우저·Steam·Discord 를 닫는다.** untied 는 tied 보다 +823MB 이고,
> 1차 Phase 4 때 여유가 550MB 였다. 그 조건이면 안 들어간다.

---

## Day 5 — Q7 본편 (S4)

168.5MB. untied-T2b 3 seed + untied-C0 ([`SPEC_P2.md` §9.3](SPEC_P2.md) 의
발동 조건대로 1 또는 3 seed). **하루를 통째로 쓴다.**

`tools/watch_run.py` 로 감시를 건다. `fail` / `abort` 도 원장에서 지우지 않는다.

---

## Day 6 — R5 (상수 LR)

tied T2b, `--lr-schedule constant`, 168.5MB x 3 seed.

C0 기준선(1.137540)과 σ(0.000058)는 1차 것을 그대로 쓴다 — 같은 예산·같은
조건이라 재측정할 이유가 없다.

> **본 run 전에 17.5MB 로 한 번 확인한다.** 상수 LR 이 168.5MB 끝까지 발산 없이
> 도는지는 아직 모른다. 1.2MB 스모크로는 알 수 없다.

---

## Day 7 — 1.5B 확장과 정리

학습 없이 수술 + Pre-CPT BPB 만 (7.1GB). R3 의 K 스윕을 1.5B(임베딩 15.1%)에서
확인한다.

그리고 보고서. 결과가 어느 쪽이든 [`DESIGN_DELTA.md`](DESIGN_DELTA.md) 에
스펙 / 실제 / 왜 / 근거 네 항목으로 남긴다.

---

## 매일 지키는 것

```
시작   nvidia-smi 로 여유 확인 · tools/check_clock.py --record
중간   GPU 작업은 하나만
끝     tools/validate_ledger.py · pytest tests/ -q
기록   record(...) 커밋과 코드 커밋을 섞지 않는다
```

---

## 알려진 부채

- **산출물 등록이 수동이다.** `tools/register_artifact.py` 를 사람이 부른다.
  `run_surgery.py` 도 `untie_model.py` 도 `damage_rows.py` 도 자동 등록하지
  않는다. 1차 종료 검토가 찾은 "미등록 산출물 5개" 와 같은 뿌리다. 한 스크립트만
  고치면 경로가 갈리므로 셋을 같이 고쳐야 한다
- **원장의 `peak_vram_mb` 는 allocated 다.** 천장에 붙어 도는 run 인데 reserved
  컬럼이 없다. 스키마를 늘리는 대신 S0 의 프로브 리포트에 남긴다
  ([`SPEC_P2.md` §9.5](SPEC_P2.md))
- ~~`PROMPTS.md` 의 기준값 출처 오지시와 규칙 번호 오기~~ — 2026-09-07 수정
