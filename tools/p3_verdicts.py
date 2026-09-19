"""P3 판정을 원장에서 계산해 표로 쓴다 (GPU 0시간).

    .conda/python.exe tools/p3_verdicts.py

등록은 [`docs/PLAN.md` "P3 확장"](../docs/PLAN.md) (2026-09-17, 어떤 run 보다 먼저).
경계는 아래 상수로 옮겨 적었고 **결과를 보고 고치지 않는다.**

지금 들어 있는 것
    사전 점검   R5 곡선의 **예산 배증당 회복률 증분 비** (W0-9)
                P3-A 의 예측 "감속이 이어진다" 가 이 비의 구간으로 정의된다.
                최초 등록 (0.5, 0.85] 에서 R5 의 40->80MB 비가 벗어나(0.869),
                등록 절차대로 **A 를 돌리기 전에** (0.5, 0.9] 로 재등록했다
                (2026-09-19, PLAN.md "P3-A 예측 재등록")
    A~F         아직 run 이 없다. 입력이 생기면 여기에 붙인다

아직 없는 입력은 "미실행" 으로 적고 멈추지 않는다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "tables" / "p3_gates.md"

# 2026-09-17 PLAN.md "P3 확장" 등록값
# 2026-09-19 재등록. 최초 등록은 (0.5, 0.85] 였는데, W0-9 점검에서 R5 의
# 40->80MB 비가 0.869 로 그 위였다. 등록 절차대로 **P3-A 를 돌리기 전에** 고친다
# (PLAN.md "P3-A 예측 재등록"). 0.9 위는 "감속이 없다"(멱법칙) 쪽으로 읽는다.
SAT_LOW, SAT_HIGH = 0.5, 0.9
MIN_INCREMENT = 0.02                   # 증분이 2%p 미만이면 비가 불안정 — 판정 안 함

# D0 (P3-D 게이트). 노출 구간에서 1차 T2b 의 손상 배율에 닿는가.
D0_RUN = "eval_kcal_mean_d0band"
TARGET_RATIO = 2.058                   # 1차 T2b 의 ko 손상 배율 (PLAN 등록)
RATIO_TOL = 0.05                       # 등록된 허용 폭
D0_MAX_K = 30_000                      # 이 안에서 닿아야 D2 를 돌린다

# R5 (상수 LR, 168.5MB). 앵커는 같은 스케줄의 C0 다.
R5_T2B = "cpt_t2b_mean_r5_seed42"
R5_C0 = "cpt_c0_qwen_r5_seed42"
DOUBLINGS = (20, 40, 80, 160)          # MB. 배증 세 번


def in_band(ratio: float) -> bool:
    """등록된 "감속이 이어진다" 구간."""
    return SAT_LOW < ratio <= SAT_HIGH


def _rows(name: str) -> list:
    with (EXP / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def ok_runs() -> set:
    return {r["run_id"] for r in _rows("LEDGER.tsv") if r["status"] == "ok"}


def curve(run_id: str) -> list:
    """[(raw_bytes, ko dev BPB)] — train_curve 의 평가 지점."""
    return sorted((int(r["raw_bytes_seen"]), float(r["dev_bpb"]))
                  for r in _rows("train_curve.tsv")
                  if r["run_id"] == run_id and r["dev_bpb"] not in ("", "NA"))


def pre_cpt_bpb(run_id: str) -> float:
    """그 run 의 학습 전 한국어 BPB (step0 행). 손으로 적지 않는다."""
    rs = [r for r in _rows("lm_metrics.tsv")
          if r["run_id"] == run_id and r["checkpoint"] == "step0"
          and r["domain"] == "ko"]
    if len(rs) != 1:
        raise SystemExit(f"{run_id}: step0 ko 행이 {len(rs)}개다")
    return float(rs[0]["bpb"])


def at(points: list, target_mb: int, tol_mb: float = 1.0):
    """target 에 가장 가까운 평가 지점. 없으면 None.

    두 run 의 평가 지점은 정확히 같은 바이트가 아니다 — 조각 경계에서 멈추므로
    20MB 지점이 20.04MB 와 20.13MB 로 갈린다. 1MB 안이면 같은 지점으로 본다.
    """
    t = target_mb * 1_000_000
    best = min(points, key=lambda p: abs(p[0] - t), default=None)
    if best is None or abs(best[0] - t) > tol_mb * 1_000_000:
        return None
    return best


def recovery_trajectory() -> list:
    """[(MB, R)] — R5 의 회복률. R = (B0 - Bf) / (B0 - Bf_C0), 앵커는 상수 LR C0."""
    b0 = pre_cpt_bpb(R5_T2B)
    t, c = curve(R5_T2B), curve(R5_C0)
    out = []
    for mb in DOUBLINGS:
        pt, pc = at(t, mb), at(c, mb)
        if pt is None or pc is None:
            continue
        denom = b0 - pc[1]
        if denom <= 0:
            continue
        out.append((mb, (b0 - pt[1]) / denom))
    return out


def d0_ratios() -> list:
    """[(K, ko 배율, 영어 변화율)] — D0 보정 run 에서."""
    rows = [r for r in _rows("lm_metrics.tsv") if r["run_id"] == D0_RUN]
    by_k: dict = {}
    for r in rows:
        k = int(r["checkpoint"][1:])          # "k30000" -> 30000
        by_k.setdefault(k, {})[r["domain"]] = float(r["bpb"])
    if 0 not in by_k:
        return []
    base = by_k[0]
    out = []
    for k in sorted(by_k):
        v = by_k[k]
        out.append((k, v["ko"] / base["ko"], v["en"] / base["en"] - 1))
    return out


def main() -> int:
    runs = ok_runs()
    L = ["# P3 게이트와 판정", "",
         "`tools/p3_verdicts.py` 가 원장에서 쓴다. 손으로 고치지 않는다.",
         "등록: [`docs/PLAN.md` \"P3 확장\"](../../docs/PLAN.md) (2026-09-17).", "",
         "## 사전 점검 — R5 의 예산 배증당 회복률 증분 (W0-9)", ""]

    if R5_T2B not in runs or R5_C0 not in runs:
        L += ["미실행 — R5 run 이 원장에 없다", ""]
    else:
        traj = recovery_trajectory()
        L += ["회복률은 상수 LR C0 를 앵커로 그 run 의 평가 지점에서 다시 계산했다.", "",
              "| 예산 | 회복률 | 직전 배증 대비 증분 |", "|---:|---:|---:|"]
        incs = []
        for i, (mb, r) in enumerate(traj):
            inc = "" if i == 0 else f"{(r - traj[i - 1][1]) * 100:+.2f}%p"
            if i:
                incs.append(r - traj[i - 1][1])
            L.append(f"| {mb}MB | {r * 100:.2f}% | {inc} |")
        L += [""]
        if len(incs) < 2:
            L += ["증분이 둘 미만이라 비를 낼 수 없다.", ""]
        else:
            L += ["| 구간 | 증분 비 | 등록 구간 안인가 |", "|---|---:|---|"]
            inside = []
            for i in range(1, len(incs)):
                prev, cur = incs[i - 1], incs[i]
                if prev < MIN_INCREMENT:
                    L.append(f"| {DOUBLINGS[i]}→{DOUBLINGS[i + 1]}MB | — | "
                             f"직전 증분 {prev * 100:.2f}%p 가 "
                             f"{MIN_INCREMENT:.0%}p 미만이라 판정 안 함 |")
                    continue
                ratio = cur / prev
                ok = in_band(ratio)
                inside.append(ok)
                L.append(f"| {DOUBLINGS[i]}→{DOUBLINGS[i + 1]}MB | {ratio:.3f} | "
                         f"{'예' if ok else '**아니오**'} |")
            L += ["", f"등록 구간은 ({SAT_LOW}, {SAT_HIGH}] 다 — "
                  "P3-A 의 예측 \"감속이 이어진다\" 가 이 구간이다.", ""]
            if inside and all(inside):
                L += ["**점검 통과.** R5 에서 관측된 비가 등록 구간 안이다. "
                      "P3-A 를 등록된 예측 그대로 돌린다.", ""]
            else:
                L += ["**점검 실패.** 관측된 비가 등록 구간 밖이다. "
                      "P3-A 를 돌리기 전에 예측을 다시 등록하고, "
                      "재등록했다는 사실을 PLAN 에 남긴다.", ""]
        L += ["이 값들은 R5 의 **사후** 관측이다. P3-A 의 판정은 A 자신의 run 에서 "
              "같은 방식으로 계산한 증분 비로 내린다.", ""]

    # ── D0 (P3-D 게이트) ───────────────────────────────────────────────
    L += ["## 게이트 — D0 노출 구간 보정 (P3-D)", ""]
    if D0_RUN not in runs:
        L += ["미실행", ""]
    else:
        d0 = d0_ratios()
        L += [f"행당 노출 `count_ko` 구간 안에서만 고른다 (T2b 새 토큰 노출 중앙값의 "
              "10배 안). 학습 없음, 언어별 2MB.", "",
              "| K | ko 손상 배율 | 영어 변화 |", "|---:|---:|---:|"]
        for k, ratio, en in d0:
            L.append(f"| {k:,} | {ratio:.3f} | {en:+.1%} |")
        best = max(d0, key=lambda x: x[1]) if d0 else None
        hit = best is not None and abs(best[1] - TARGET_RATIO) <= RATIO_TOL
        L += ["", f"목표는 1차 T2b 의 배율 {TARGET_RATIO} ± {RATIO_TOL} 이고, "
              f"K 상한은 {D0_MAX_K:,} 다 (등록).", ""]
        if hit:
            L += [f"**통과.** K={best[0]:,} 에서 배율 {best[1]:.3f}. D2 를 돌린다.", ""]
        else:
            L += [f"**불통과.** K={D0_MAX_K:,} 에서도 배율이 {best[1]:.3f} 에 그친다 "
                  f"(목표 {TARGET_RATIO}).", "",
                  "**D2 를 돌리지 않는다.** 등록에 적어 둔 그대로, 이것 자체가 결과다 —",
                  "기존 어휘를 망가뜨리는 방법으로는 **행당 노출과 손상 크기를 동시에**",
                  "T2b 에 맞출 수 없다. 드물게 쓰이는 행은 많이 망가뜨려도 한국어 BPB 를",
                  "거의 못 움직인다.", "",
                  f"부수 피해도 다르다 — 같은 K 에서 영어가 {best[2]:+.1%} 다. "
                  "1차 T2b 는 영어가 거의 그대로였다 (+0.1%). 한국어 배율만 맞춰도",
                  "\"같은 손상\" 이라고 부를 수 없다는 뜻이다.", ""]

    L += ["## A~F 판정", "", "미실행 — P3 run 이 아직 없다.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("\n".join(L))
    print(f"\n썼다 {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
