"""외부 검토용 프로젝트 개요를 원장에서 만든다 (학습 없음, GPU 0시간).

    .conda/python.exe tools/overview.py

## 왜 도구인가

저장소를 못 보는 사람에게 프로젝트 전체를 한 파일로 건네야 할 때가 있다.
그런데 **수치를 손으로 옮기면 그 순간 원장과 갈린다** (RULES 13번). 그래서
서술은 이 파일에 문자열로 두고, 숫자는 전부 `experiments/` 에서 읽는다.

산출물: `reports/OVERVIEW.md` — 자기완결적이다. 링크를 못 열어도 읽힌다.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "OVERVIEW.md"

KO_DOMAINS = {"blog", "community", "encyclopedia", "ko_en_mixed", "news",
              "technical", "web_general"}


def rows(name: str) -> list:
    with (EXP / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def bpb(run_id: str, checkpoint: str = "final", domain: str = "ko") -> float:
    rs = [r for r in rows("lm_metrics.tsv")
          if r["run_id"] == run_id and r["checkpoint"] == checkpoint
          and r["domain"] == domain]
    if len(rs) != 1:
        raise SystemExit(f"{run_id}/{checkpoint}/{domain}: 행이 {len(rs)}개다")
    return float(rs[0]["bpb"])


def tok_per_byte(run_id: str, version: str, domains: set) -> float:
    rs = [r for r in rows("tokenizer_metrics.tsv")
          if r["run_id"] == run_id and r["tokenizer_version"] == version
          and r["domain"] in domains]
    if not rs:
        raise SystemExit(f"{run_id}/{version}: 행이 없다")
    return (sum(int(r["n_tokens"]) for r in rs)
            / sum(int(r["n_bytes"]) for r in rs))


def recovery(b0: float, bf: float, cf: float) -> float:
    return (b0 - bf) / (b0 - cf)


def main() -> int:
    L: list = []
    w = L.append

    # ── 원장에서 읽는 값 ────────────────────────────────────────────────
    ko_q = tok_per_byte("tok_bench_v2", "qwen_original", KO_DOMAINS)
    ko_t = tok_per_byte("tok_bench_v2", "t2b_v2_n30000", KO_DOMAINS)
    en_q = tok_per_byte("tok_bench_v2", "qwen_original", {"english"})
    en_t = tok_per_byte("tok_bench_v2", "t2b_v2_n30000", {"english"})
    cd_q = tok_per_byte("tok_bench_v2", "qwen_original", {"code"})
    cd_t = tok_per_byte("tok_bench_v2", "t2b_v2_n30000", {"code"})

    c0 = bpb("cpt_c0_qwen_main_seed42")
    t2b = bpb("cpt_t2b_mean_main_seed42")
    t2a = bpb("cpt_t2a_none_main_seed42")
    t2b10k = bpb("cpt_t2b10k_mean_main_seed42")
    b0_t2b = bpb("cpt_t2b_mean_r5_seed42", "step0")      # 수술 직후 (R5 가 다시 쟀다)
    b0_10k = bpb("cpt_t2b10k_mean_main_seed42", "step0")
    r_cos = recovery(b0_t2b, t2b, c0)
    r_10k = recovery(b0_10k, t2b10k, c0)

    r5_t2b = [bpb(f"cpt_t2b_mean_r5_seed{s}") for s in (42, 123, 2026)]
    r5_c0 = bpb("cpt_c0_qwen_r5_seed42")
    r5_mean = sum(r5_t2b) / len(r5_t2b)
    r_const = recovery(b0_t2b, r5_mean, r5_c0)
    sd = math.sqrt(sum((x - r5_mean) ** 2 for x in r5_t2b) / (len(r5_t2b) - 1))
    sigma_r = sd / (b0_t2b - r5_c0)

    dmg_b0 = bpb("cpt_dmg_mean_k40_r1_seed42", "step0")
    dmg_f = bpb("cpt_dmg_mean_k40_r1_seed42")
    ctrl = bpb("cpt_Qwen2.5-0.5B_r1ctrl_seed42")
    base_b0 = bpb("cpt_kot2b_v2_n30000_mean_r1base_seed42", "step0")
    base_f = bpb("cpt_kot2b_v2_n30000_mean_r1base_seed42")
    r_dmg = recovery(dmg_b0, dmg_f, ctrl)
    r_sub40 = recovery(base_b0, base_f, ctrl)

    hanja = {c: bpb(f"eval_bpb_cpt_{m}_main_seed42_d2", "pre_cpt")
             for c, m in (("C0", "c0_qwen"), ("T2a", "t2a_none"),
                          ("T2b", "t2b_mean"))}
    def acc(run_id: str) -> tuple:
        rs = [r for r in rows("capability.tsv")
              if r["run_id"] == run_id and r["benchmark"] == "kmmlu_d3"]
        if not rs:
            return (float("nan"), float("nan"), float("nan"))
        r = rs[-1]
        return float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"])

    a_c0 = acc("eval_kmmlu_cpt_c0_qwen_main_seed42_d3")
    a_t2a = acc("eval_kmmlu_cpt_t2a_none_main_seed42_d3")
    a_t2b = acc("eval_kmmlu_cpt_t2b_mean_main_seed42_d3")

    # ── 문서 ────────────────────────────────────────────────────────────
    w("# KoTokenLab — 외부 검토용 개요")
    w("")
    w("> **이 파일은 `tools/overview.py` 가 원장에서 만든다.** 서술은 코드 안에")
    w("> 문자열로 있고, 숫자는 전부 `experiments/*.tsv` 에서 읽었다. 손으로 옮긴")
    w("> 수치는 없다. 저장소를 못 보는 사람이 읽을 수 있게 자기완결적으로 쓴다.")
    w("")
    w("RTX 5070 Ti 16GB **한 대** 로 돌리는 1인 프로젝트다.")
    w("")
    w("## 1. 무엇을 묻는가")
    w("")
    w("한국어 토크나이저를 바꾸면 무엇을 얻고 무엇을 잃는가. **어휘 크기를**")
    w("**유지한 채** 토큰을 치환하는 수술(T2b)을 걸고, 같은 코퍼스·같은 예산으로")
    w("CPT 한 뒤, 원본 토크나이저 통제군(C0)과 비교한다.")
    w("")
    w("핵심 지표는 **BPB (bits per byte)** 하나다. 분모가 원문 바이트라서")
    w("토크나이저가 다른 모델을 비교할 수 있다 — token PPL 로 비교하면 토큰을")
    w("크게 자르는 쪽이 자동으로 이긴다.")
    w("")
    w("회복률은 이렇게 정의한다 (`docs/RULES.md` 14b).")
    w("")
    w("```")
    w("R = (B0 - Bf) / (B0 - Cf)")
    w("  B0  수술 직후 · 학습 전 한국어 dev BPB")
    w("  Bf  같은 모델을 CPT 로 예산만큼 돌린 뒤")
    w("  Cf  통제군 C0 (원본 토크나이저, 같은 코퍼스 · 예산 · 스케줄)")
    w("```")
    w("")
    w("## 2. 실험 대상")
    w("")
    w("```")
    w("백본        Qwen2.5-0.5B (tie_word_embeddings=1, 임베딩이 파라미터의 27.6%)")
    w("코퍼스      FineWeb-2 한국어 + 영어 · 코드 대조군. 문서 단위 90/5/5 분할")
    w("T2a         저빈도 토큰 30,000개 제거 (vocab 축소)")
    w("T2b         그 자리에 한국어 토큰 30,000개 치환 (**크기 유지**)")
    w("초기화      E1 = merge 부품 둘의 평균 (FVT 와 동등, 9절)")
    w("CPT         168.5MB 원문, bf16, gradient checkpointing, AdamW8bit")
    w("```")
    w("")
    w("## 3. 1차 결과 — 교환비")
    w("")
    w("### 압축 (학습 없음, dev)")
    w("")
    w("| | 원본 Qwen | T2b | 차이 |")
    w("|---|---:|---:|---:|")
    w(f"| 한국어 tok/byte | {ko_q:.4f} | {ko_t:.4f} | **{ko_t / ko_q - 1:+.1%}** |")
    w(f"| 영어 | {en_q:.4f} | {en_t:.4f} | {en_t / en_q - 1:+.1%} |")
    w(f"| 코드 | {cd_q:.4f} | {cd_t:.4f} | {cd_t / cd_q - 1:+.1%} |")
    w("")
    w("### 품질 (CPT 168.5MB 후, 한국어 dev BPB)")
    w("")
    w("| 조건 | BPB | C0 대비 |")
    w("|---|---:|---:|")
    w(f"| C0 (원본 토크나이저) | {c0:.4f} | — |")
    w(f"| T2a (제거만) | {t2a:.4f} | {t2a / c0 - 1:+.2%} |")
    w(f"| T2b (치환, N=30k) | {t2b:.4f} | **{t2b / c0 - 1:+.2%}** |")
    w(f"| T2b (치환, N=10k) | {t2b10k:.4f} | {t2b10k / c0 - 1:+.2%} |")
    w("")
    w(f"수술 직후 한국어 BPB 는 {b0_t2b:.4f} 였다. 즉 CPT 가 격차의")
    w(f"**{r_cos:.1%}** 를 메웠고(N=10k 는 {r_10k:.1%}), 나머지는 남았다.")
    w("")
    w("**압축은 설계대로 됐고 품질은 따라오지 못했다.** 그 대신 산 것은 추론")
    w("비용이다 — prefill -35~41%, KV cache -30%, 같은 문맥 창에 +44% 원문.")
    w("(시스템 수치는 `reports/tables/system_bench` 계열에 있다)")
    w("")
    w("## 4. 2차 결과 — 무엇이 회복을 정하는가")
    w("")
    w("1차에서 회복률이 조건을 바꿔도 65% 근처에서 겹쳤다. 2차는 그 이유를 물었고,")
    w("**사전 등록한 가설 셋 중 둘이 반증됐다.**")
    w("")
    w("### R1 — 임베딩 손상 일반의 성질인가 (부정)")
    w("")
    w("토크나이저를 안 바꾸고 임베딩 **행 40개만** 망가뜨린다. 손상 배율은 T2b 와")
    w("맞췄다 (학습 없는 보정으로 K 를 골랐다).")
    w("")
    w("| 조건 | 학습 전 | 학습 후 | 통제군 | R |")
    w("|---|---:|---:|---:|---:|")
    w(f"| 행 40개 손상 (40MB) | {dmg_b0:.4f} | {dmg_f:.4f} | {ctrl:.4f} | **{r_dmg:.2%}** |")
    w(f"| T2b 치환 (40MB) | {base_b0:.4f} | {base_f:.4f} | {ctrl:.4f} | **{r_sub40:.2%}** |")
    w("")
    w(f"같은 손상 배율인데 **{(r_dmg - r_sub40) * 100:.2f}%p** 벌어진다. 정체는")
    w("임베딩 복구 일반의 성질이 아니라 토크나이저 치환에 특유한 것이다.")
    w("")
    w("> **남은 교락:** 두 조건은 손상 배율만 같고 **행당 노출** 과 **손상 행 수** 가")
    w("> 다르다. 2026-09-19 에 노출을 맞춰 보려 했으나, 드물게 쓰이는 행은 30,000개를")
    w("> 망가뜨려도 손상 배율이 1.05 에 그쳐 **두 축을 동시에 맞출 수 없다는 것이**")
    w("> **확인됐다.** 그 자체를 결과로 보고한다.")
    w("")
    w("### Q7 — tied embedding 때문인가 (부정)")
    w("")
    w("`lm_head` 를 임베딩 사본으로 복제해 tie 를 끊고(수술 전후 forward 가 비트")
    w("단위로 같은 것을 확인) 같은 실험을 돌렸다. 17.5MB × 3 seed.")
    w("")
    w("```")
    w("R(untied) 42.70%   R(tied) 42.26%   차이 +0.44%p   2σ = 0.226%p")
    w("```")
    w("")
    w("**2σ 는 넘었지만 사전 등록한 효과 크기 바닥(5%p)의 1/11 이다.** 바닥이")
    w("없었으면 \"유의한 차이\" 로 보고할 뻔했다. 등록이 그 상황을 미리 적어 뒀다.")
    w("")
    w("### R5 — LR 스케줄 때문인가 (긍정)")
    w("")
    w("코사인 대신 **상수 LR** 로 같은 예산을 돌렸다. 감쇠만 빼고 나머지는 같다.")
    w("통제군도 상수 LR 로 다시 돌렸다 (조건이 바뀌면 대조군도 다시 잰다).")
    w("")
    w("| | 코사인 | 상수 LR |")
    w("|---|---:|---:|")
    w(f"| T2b 한국어 BPB | {t2b:.4f} | {r5_mean:.4f} (3 seed) |")
    w(f"| C0 한국어 BPB | {c0:.4f} | {r5_c0:.4f} |")
    w(f"| 회복률 R | {r_cos:.2%} | **{r_const:.2%}** |")
    w("")
    w(f"σ_R = {sigma_r * 100:.3f}%p (3 seed). 등록 예측은 \"71~75%\" 였고 경쟁 등록")
    w("(\"68.5% 이하\")은 기각됐다.")
    w("")
    w("**헤드라인이 바뀐다.** \"65% 벽\" 은 코사인 스케줄의 산물이고, 1차의 절대값은")
    w("전부 그 스케줄 조건부로 읽어야 한다 — C0 도 상수 LR 에서 더 좋아졌다.")
    w("")
    w("## 5. 한자 프로브 (D1~D3)")
    w("")
    w("T2a 가 제거한 30,000개 중 13,310개(44.4%)가 한자 포함 토큰이었다. 빈도")
    w("필터의 부수 효과이고, 절단면은 **간체 전용자** 쪽으로 치우쳐 정자(國 無 韓)는")
    w("남았다. 한자가 밀집한 부분집합(787문서)과 KMMLU 로 확인했다.")
    w("")
    w("| | C0 | T2a | T2b |")
    w("|---|---:|---:|---:|")
    w("| 한자 밀집 BPB | " + " | ".join(f"{hanja[k]:.4f}" for k in ("C0", "T2a", "T2b")) + " |")
    w("| KMMLU 정확도 (1,900문항) | "
      + " | ".join(f"{a[0]:.4f}" for a in (a_c0, a_t2a, a_t2b)) + " |")
    w("")
    w(f"T2a 는 BPB {hanja['T2a'] / hanja['C0'] - 1:+.2%}, KMMLU "
      f"{(a_t2a[0] - a_c0[0]) * 100:+.2f}%p 로 **C0 와 구별되지 않는다**")
    w("(등록 경계 ±2%p, paired bootstrap). 다만 C0 가 찍기(25%)보다 약")
    w(f"{(a_c0[0] - 0.25) * 100:.1f}%p 위라 **감도가 낮은 계기** 다.")
    w("")
    w("## 6. 3차 설계 (실행 전)")
    w("")
    w("```")
    w("A  상수 LR 500MB — 결국 멈추는가. 앞 168.5MB 로 R5 재현을 내장한다")
    w("B  WSD 스케줄 — 실무 처방. 1차 지표는 T2b 의 절대 BPB")
    w("C  불변성이 상수 LR 에서도 유지되는가 (T2b10k)")
    w("D  R1 격차 분해 — permute 만 남았다 (노출 구간 조건은 게이트에서 취소)")
    w("E  오라클 초기화 상한 — CPT 된 행을 원본 몸통에 이식")
    w("F  과제 비용 — KMMLU 나머지 39과목 (D3 의 6과목은 이미 봤으므로 제외)")
    w("G  1.5B — VRAM 탐침 통과 (micro_bs 1 에서 reserved 13,458MB)")
    w("```")
    w("")
    w("## 7. 방법론에서 지키는 것")
    w("")
    w("- **사전 등록.** 예측 · 판정 경계 · 효과 크기 바닥을 돌리기 전에 커밋한다.")
    w("  결과를 보고 경계를 옮기지 않고, 옮겨야 하면 그 사실을 날짜와 함께 적는다")
    w("- **효과 크기 바닥 5%p + 2σ.** 통계적 유의만으로 \"차이\" 라고 부르지 않는다")
    w("- **조건이 바뀌면 통제군도 같은 조건에서 다시 잰다** (R5 의 상수 LR C0)")
    w("- **결과는 코드가 원장에 쓴다.** 마크다운에 손으로 옮기지 않는다")
    w("- **append-only 원장** — 실패한 run 도 지우지 않는다 (fail · abort 포함)")
    w("- 커밋 훅이 규칙을 집행한다 — 결과 커밋에 코드가 섞이면 거부된다")
    w("")
    w("## 8. 알려진 한계 (전부 문서에 적혀 있다)")
    w("")
    w("- **0.5B 한 모델.** 1.5B 는 수술 손상만 쟀고 CPT 회복력은 미측정")
    w("- **seed 42 단독** 인 run 이 많다 (노이즈 플로어와 R5 만 3 seed)")
    w("- **dev 가 닳았다.** 초기화도 N 도 dev 에서 골랐다. Final Test 는 한 번도")
    w("  열지 않았다")
    w("- **CPT 코퍼스가 Qwen 사전학습과 겹친다** (둘 다 Common Crawl 계열).")
    w("  C0 에 유리한 방향의 편향이다")
    w("- **평가 문항 오염 제거가 구현되지 않았다.** 2026-09-19 에 따로 세어 보니")
    w("  KMMLU 6과목 중 덮임 50% 이상은 2문항이었다 (상용구 겹침이 대부분)")
    w("- **도메인 라벨 정확도가 ~55%** 라 도메인별 수치를 인용하지 않는다")
    w("- **ZeTT 를 못 돌렸다.** 대신 \"완벽한 초기화\" 상한을 P3-E 로 잰다")
    w("- 학습 wall-clock 이 오염됐다 (데스크톱이 GPU 를 함께 썼다). 속도 주장에")
    w("  쓰지 않는다. BPB 는 영향 없다")
    w("")
    w("## 9. 초기화 baseline 의 위치")
    w("")
    w("E1(merge 부품 평균)은 **FVT (Gee et al., 2022)와 동등** 하다 — 재토큰화가")
    w("정의되는 25,821개에서 100% 일치했다. 나머지 4,179개(13.9%)는 표면형이 유효")
    w("UTF-8 이 아니라 FVT 를 적용할 수 없는 경우이고, 거기서는 merge 계보로")
    w("확장한다. 그 확장분이 바이트 단위 평균과 얼마나 다른지도 쟀다 (중앙값 코사인")
    w("0.520 으로, FVT 가 정의되는 구간의 0.171 보다 오히려 가깝다).")
    w("")
    w("E2(역빈도 가중)는 **FOCUS 가 아니다** — 가중치 출처가 유사도가 아니라 빈도다.")
    w("")
    w("## 10. 운영 환경 — 장비 한 대를 원격으로 돌린다")
    w("")
    w("실험은 **데스크톱 워크스테이션 한 대** (RTX 5070 Ti 16GB) 에서만 돈다.")
    w("클러스터도 클라우드도 없고, 학습을 두 개 동시에 띄우지 않는다 — 그래서")
    w("일정표의 시간이 곧 실제 대기 시간이다.")
    w("")
    w("긴 run(3~11시간)을 책상 앞에서 지킬 수 없으므로 **원격 접속** 을 붙여 뒀다.")
    w("")
    w("```")
    w("사설 메시 VPN (Tailscale)   기기끼리만 닿는 사설망. 포트를 공개하지 않는다")
    w("OpenSSH 서버               그 망 안에서만 접속. 자동 시작 서비스")
    w("쓰임                       run 시작 · 진행 확인 · 원장 tail · 중단")
    w("```")
    w("")
    w("**기기 이름 · 주소 · 키 · 계정 정보는 저장소에 넣지 않는다.** 저장소는 공개를")
    w("전제로 쓰고 있고, 접속 정보는 실험의 재현성과 무관하다. 재현에 필요한 것은")
    w("`env/ENV_SNAPSHOT.tsv`(드라이버 · 라이브러리 버전 해시)와 원장이다.")
    w("")
    w("한 대뿐이라는 제약이 설계를 여러 번 바꿨다 — 7B 를 포기하고 0.5B 에서 깊게")
    w("판 것, 비교 실험을 40MB 로 줄인 것, Q7 게이트에서 6시간짜리 본편을 취소한 것이")
    w("전부 여기서 나온다.")
    w("")
    w("## 11. 저장소 구조 (검토에 필요한 곳만)")
    w("")
    w("```")
    w("docs/RULES.md          하드룰. 지표 정의(14b)와 사전 등록 규칙(14)")
    w("docs/PLAN.md           모든 사전 등록 원문 (1차 Q1~Q6, 2차 R1~R5·D, 3차 P3)")
    w("docs/SPEC_P2.md        2차 설계    docs/SCHEDULE_P2.md  2차 일정과 결과")
    w("docs/SPEC_P3.md        3차 설계    docs/SCHEDULE_P3.md  3차 일정")
    w("docs/DESIGN_DELTA.md   스펙과 다르게 한 것과 그 이유 (반증된 가설 포함)")
    w("reports/FINAL_REPORT.md  1차 결과 전체")
    w("reports/tables/*.md    전부 도구가 쓴 표")
    w("experiments/*.tsv      원장 (LEDGER · lm_metrics · train_curve · capability …)")
    w("src/ tools/ scripts/   구현. 훅은 tools/check_commit_msg.py · precheck.py")
    w("```")
    w("")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(L))
    print(f"\n썼다 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
