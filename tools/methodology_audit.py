"""외부 비판 검토용 수치 감사를 원장과 산출물에서 생성한다 (GPU 0).

이 표는 새 실험 결과가 아니다. 기존 원장의 학습량, BPB, 신규 토큰 노출을
한곳에 모아 해석상의 통제축을 확인한다. 수치를 마크다운에 손으로 복사하지
않기 위해 코드로 생성한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "tables" / "methodology_audit.md"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def ok_runs() -> dict[str, dict[str, str]]:
    return {r["run_id"]: r for r in read_tsv(EXP / "LEDGER.tsv")
            if r["status"] == "ok"}


def final_bpb() -> dict[tuple[str, str, str], float]:
    return {(r["run_id"], r["checkpoint"], r["domain"]): float(r["bpb"])
            for r in read_tsv(EXP / "lm_metrics.tsv")}


def steps(row: dict[str, str]) -> int:
    match = re.search(r"(?:^|\s)steps=(\d+)(?:\s|$)", row["note"])
    if not match:
        raise SystemExit(f"{row['run_id']}: note 에 steps= 가 없다")
    return int(match.group(1))


def config(run_id: str) -> dict:
    path = EXP / "runs" / run_id / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def recovery(b0: float, bf: float, cf: float) -> float:
    return (b0 - bf) / (b0 - cf)


def exposure(path: Path) -> tuple[int, int, int, int]:
    rows = read_tsv(path)
    vals = sorted(int(r["fires_scaled"]) for r in rows)
    zero_sample = sum(int(r["fires_sample"]) == 0 for r in rows)
    below_100 = sum(v < 100 for v in vals)
    return len(vals), zero_sample, below_100, vals[len(vals) // 2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="방법론 감사 표 생성")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    ledger = ok_runs()
    bpb = final_bpb()
    lines: list[str] = []
    w = lines.append

    w("# 방법론 감사 — 원장 재계산")
    w("")
    w("> **이 표는 코드가 적는다.** `C:\\llm_tokenizer\\.conda\\python.exe "
      "tools/methodology_audit.py`")
    w("> 입력은 `experiments/LEDGER.tsv`, `experiments/lm_metrics.tsv`, 각 run의 "
      "`config.json`, 등록된 `token_exposure.tsv`다. 새 GPU 실험은 없다.")
    w("")

    w("## 1. 168.5MB 학습 회계")
    w("")
    w("| run | 기록 원문 B | 기록 토큰 | optimizer update | update에 반영된 토큰 | "
      "마지막 미반영 토큰 | seq / micro / accum |")
    w("|---|---:|---:|---:|---:|---:|---|")
    run_ids = (
        "cpt_c0_qwen_main_seed42",
        "cpt_t2b_mean_main_seed42",
        "cpt_t2b10k_mean_main_seed42",
        "cpt_t2b_mean_eqtok_seed42",
    )
    for rid in run_ids:
        row = ledger[rid]
        cfg = config(rid)
        n_step = steps(row)
        per_update = int(cfg["seq_len"]) * int(cfg["micro_bs"]) * int(cfg["accum"])
        applied = n_step * per_update
        recorded = int(row["tokens_seen"])
        w(f"| `{rid}` | {int(row['raw_bytes_seen']):,} | {recorded:,} | {n_step:,} | "
          f"{applied:,} | {recorded - applied:,} | {cfg['seq_len']} / "
          f"{cfg['micro_bs']} / {cfg['accum']} |")
    w("")
    w("`tokens_seen`과 `raw_bytes_seen`은 마지막 완전한 optimizer update 뒤에 쌓인 "
      "microbatch까지 포함할 수 있다. 현재 원장은 그 꼬리의 원문 바이트를 따로 "
      "기록하지 않으므로 **update에 실제 반영된 raw bytes는 사후 복원할 수 없다.** "
      "기존 결과의 최대 차이는 한 update 미만이지만, 다음 run 전 중단·회계 로직을 "
      "고쳐야 한다.")
    w("")

    u0 = bpb[("cpt_c0_qwen_r5_seed42", "step0", "ko")]
    c_cos = bpb[("cpt_c0_qwen_main_seed42", "final", "ko")]
    c_const = bpb[("cpt_c0_qwen_r5_seed42", "final", "ko")]
    b0_30 = bpb[("cpt_t2b_mean_r5_seed42", "step0", "ko")]
    b0_10 = bpb[("cpt_t2b10k_mean_main_seed42", "step0", "ko")]
    bf_const_vals = [bpb[(f"cpt_t2b_mean_r5_seed{s}", "final", "ko")]
                     for s in (42, 123, 2026)]
    bf_const = statistics.mean(bf_const_vals)
    recovery_rows = (
        ("N=30k cosine, seed42", b0_30,
         bpb[("cpt_t2b_mean_main_seed42", "final", "ko")], c_cos),
        ("N=10k cosine, seed42", b0_10,
         bpb[("cpt_t2b10k_mean_main_seed42", "final", "ko")], c_cos),
        ("N=30k constant, T2b 3-seed 평균 / C0 seed42", b0_30,
         bf_const, c_const),
    )
    w("## 2. 절대 BPB와 회복률 해석")
    w("")
    w(f"수술 전 원본 기준 `U0`는 {u0:.6f}이다. `D0=B0-U0`, "
      "`Df=Bf-Cf`는 **2026-09-19 사후 기술 분석**이며 인과적 손상 분해가 아니다.")
    w("")
    w("| 조건 | B0 | Bf | Cf | Bf-Cf (우선) | R (보조) | D0 | Df |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, b0, bf, cf in recovery_rows:
        w(f"| {label} | {b0:.6f} | {bf:.6f} | {cf:.6f} | {bf-cf:+.6f} | "
          f"{recovery(b0, bf, cf):.2%} | {b0-u0:+.6f} | {bf-cf:+.6f} |")
    cosine_gap = recovery_rows[0][2] - c_cos
    constant_gap = bf_const - c_const
    w("")
    w(f"N=30k의 종점 격차는 cosine {cosine_gap:.6f}에서 constant "
      f"{constant_gap:.6f}로 **{cosine_gap-constant_gap:.6f} BPB 감소**했다. "
      "이는 유한 예산 종점 비교이지 점근선 추정이 아니다.")
    w("")

    seeds = (42, 123, 2026)
    paired: list[float] = []
    q7_rows: list[tuple[int, float, float, float]] = []
    for seed in seeds:
        tied = recovery(
            b0_30,
            bpb[(f"cpt_t2b_mean_noise_seed{seed}", "final", "ko")],
            bpb[(f"cpt_c0_qwen_noise2_seed{seed}", "final", "ko")],
        )
        untied = recovery(
            b0_30,
            bpb[(f"cpt_untied_t2b_mean_q7g_seed{seed}", "final", "ko")],
            bpb[(f"cpt_untied_c0_qwen_q7g_seed{seed}", "final", "ko")],
        )
        diff = untied - tied
        paired.append(diff)
        q7_rows.append((seed, tied, untied, diff))
    mean_diff = statistics.mean(paired)
    sd_diff = statistics.stdev(paired)
    se_diff = sd_diff / math.sqrt(len(paired))
    # t(0.975, df=2). n=3이고 정규성 검증이 불가능하므로 탐색적 구간이다.
    t_df2 = 4.302652729911275
    ci = (mean_diff - t_df2 * se_diff, mean_diff + t_df2 * se_diff)
    w("## 3. Q7 paired seed 재계산")
    w("")
    w("| seed | R tied | R untied | paired 차이 |")
    w("|---:|---:|---:|---:|")
    for seed, tied, untied, diff in q7_rows:
        w(f"| {seed} | {tied:.3%} | {untied:.3%} | {diff*100:+.3f}%p |")
    w("")
    w(f"평균 차이는 **{mean_diff*100:+.3f}%p**, paired SD는 "
      f"{sd_diff*100:.3f}%p, 평균의 탐색적 t 95% CI는 "
      f"[{ci[0]*100:+.3f}, {ci[1]*100:+.3f}]%p다. 이 구간은 0과 다르지만 "
      "사전 등록 실용 경계 ±5%p 안에 있다. n=3 정규성 가정의 사후 구간이므로 "
      "일반적 동등성 증명이나 장기 예산 결론으로 쓰지 않는다.")
    w("")

    w("## 4. 신규 토큰 노출")
    w("")
    w("| 조건 | 표에 있는 신규 토큰 | 20MB 표본 0회 | 168.5MB 환산 <100회 | "
      "환산 중앙값 |")
    w("|---|---:|---:|---:|---:|")
    for label, rel in (
        ("N=30k", "artifacts/vocab_stats/kot2b_v2_n30000/token_exposure.tsv"),
        ("N=10k", "artifacts/vocab_stats/kot2b_v2_n10000/token_exposure.tsv"),
    ):
        n, zero, low, median = exposure(ROOT / rel)
        w(f"| {label} | {n:,} | {zero:,} ({zero/n:.1%}) | "
          f"{low:,} ({low/n:.1%}) | {median:,} |")
    w("")
    w("0회는 20MB 표본에서의 값이라 전체 168.5MB 미출현율의 **상한**이다. "
      "따라서 토큰 종류 수와 실제 학습 노출량을 같은 것으로 취급하지 않는다.")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {dest.relative_to(ROOT)} ({len(lines):,}줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
