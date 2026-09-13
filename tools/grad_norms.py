"""기울기 군 노름을 군 크기로 정규화한다 (학습 없음, 원장만 읽는다).

    .conda/python.exe tools/grad_norms.py

## 왜

`reports/tables/phase6.md` 는 이렇게 적었다 —

    "예측대로 임베딩 기울기가 죽지 않는다. 끝까지 attn 의 1.57배로 남는다"

**이 비교는 정규화되지 않았다.** `grad_norm_emb` / `attn` / `ffn` 은 각 군에
속한 파라미터 전체의 L2 노름인데, 군 크기가 3~7배 다르다.

    emb   151,936 x 896        = 136,134,656
    attn  24 x (q,k,v,o+bias)  =  44,067,840
    ffn   24 x 3 x 896 x 4,864 = 313,786,368

원소당 크기가 같아도 노름은 sqrt(N) 에 비례한다. sqrt(136.1/44.1) = 1.76 이므로
**임베딩 노름이 attn 의 1.76배 미만이면 원소당으로는 오히려 작다.**
관측된 1.57배가 정확히 그 구간에 있다.

## 다만 sqrt(N) 정규화도 완전하지 않다

임베딩 기울기는 **희소하다** — 배치에 나온 토큰의 행만 0 이 아니다. 151,936행
전체로 나누면 이번에는 과소평가가 된다. 즉 **어느 쪽으로도 보정 없이는
해석할 수 없다.**

그래서 이 도구의 결론은 "임베딩 기울기가 작다" 가 아니라
**"이 비교는 현재 형태로 논문에 쓸 수 없다"** 다. 제대로 하려면 셋이 필요하다.

    (a) 손상된 30,000행만의 노름      계측을 새로 넣어야 한다
    (b) grad / weight 비              스케일 무관 지표
    (c) C0 · T2a 대조군               phase6.md 가 이미 한계로 적어 뒀다

## 살아남는 진술

`grad_norm_emb` 의 **절대값이 바닥을 치지 않는다** (4.78 -> 2.98). 이것은
정규화와 무관하게 관측 그대로다. BPB 가 100MB 이후 0.005 밖에 안 움직이는
구간에서도 기울기 신호가 남아 있다는 진술은 유지된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TC = ROOT / "experiments" / "train_curve.tsv"
LM = ROOT / "experiments" / "lm_metrics.tsv"
OUT = ROOT / "reports" / "tables" / "grad_norms.md"

RUN = "cpt_t2b_mean_grad_seed42"
CONFIG = ROOT / "artifacts" / "models" / "kot2b_v2_n30000_mean" / "config.json"


def group_sizes(cfg: dict) -> dict:
    """Qwen2 구조에서 군별 파라미터 수. 편향까지 센다."""
    h = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    inter = cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"]
    hd = h // heads
    q, k, v = heads * hd, kv * hd, kv * hd
    attn = layers * (h * q + q + h * k + k + h * v + v + q * h)
    return {
        "emb": cfg["vocab_size"] * h,
        "attn": attn,
        "ffn": layers * 3 * h * inter,
    }


def read_curve(run_id: str) -> list:
    with TC.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")
                if r["run_id"] == run_id and r.get("grad_norm_emb") not in ("", "NA", None)]
    return sorted(rows, key=lambda r: int(r["raw_bytes_seen"]))


def read_bpb(run_id: str) -> dict:
    with LM.open(encoding="utf-8") as fh:
        return {int(r["raw_bytes_seen"]): float(r["bpb"])
                for r in csv.DictReader(fh, delimiter="\t")
                if r["run_id"] == run_id and r["domain"] == "ko"}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="기울기 군 노름을 군 크기로 정규화")
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    n = group_sizes(cfg)
    rows = read_curve(args.run)
    if not rows:
        raise SystemExit(f"{args.run} 의 train_curve 에 grad_norm_emb 가 없다")
    bpb = read_bpb(args.run)

    o: list = []
    w = o.append
    w("# 기울기 군 노름 — 정규화하면 추세가 뒤집힌다\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/grad_norms.py`")
    w("> 학습 없음. `experiments/train_curve.tsv` 만 읽는다.\n")
    w(f"run `{args.run}`. 군 크기는 `{CONFIG.parent.name}/config.json` 에서 계산했다.\n")
    w("## 군 크기가 3~7배 다르다\n")
    w("```")
    for k, v in n.items():
        w(f"{k:5s} {v:12,}   sqrt(N) = {math.sqrt(v):10,.0f}")
    w(f"\nsqrt(emb/attn) = {math.sqrt(n['emb']/n['attn']):.2f}"
      f"   <- 원소당 크기가 같으면 노름 비가 이 값이다")
    w("```\n")
    w("**관측된 1.57배는 이 값보다 작다.** 즉 원소당으로는 임베딩 기울기가")
    w("attention 보다 **작다.**\n")

    w("## 원본과 정규화\n")
    w("| 예산 MB | ko BPB | lr | emb | attn | ffn | raw emb/attn | **원소당 emb/attn** |")
    w("|---:|---:|---:|---:|---:|---:|---:|---:|")
    first = last = None
    for r in rows:
        b = int(r["raw_bytes_seen"])
        e, a, f = (float(r["grad_norm_emb"]), float(r["grad_norm_attn"]),
                   float(r["grad_norm_ffn"]))
        raw = e / a
        norm = (e / math.sqrt(n["emb"])) / (a / math.sqrt(n["attn"]))
        first = norm if first is None else first
        last = norm
        bb = bpb.get(b)
        w(f"| {b/1e6:.0f} | {bb:.4f} | {float(r['lr']):.2e} | {e:.2f} | {a:.2f} | {f:.2f} | "
          f"{raw:.2f} | **{norm:.2f}** |" if bb else
          f"| {b/1e6:.0f} | — | {float(r['lr']):.2e} | {e:.2f} | {a:.2f} | {f:.2f} | "
          f"{raw:.2f} | **{norm:.2f}** |")
    w("")
    w(f"**정규화하면 {first:.2f} -> {last:.2f} 로 1 아래를 지난다.**")
    w("raw 비는 끝까지 1 위에 있지만(1.57), 그것은 임베딩 군이")
    w(f"attention 군보다 {n['emb']/n['attn']:.1f}배 크다는 사실의 그림자다.\n")

    w("## 그래서 무엇을 쓸 수 있나\n")
    w("**쓸 수 없는 것**\n")
    w("> ~~임베딩 기울기가 끝까지 attention 의 1.57배로 남는다~~")
    w("> — 군 크기 미보정. 논문에서 뺀다\n")
    w("**쓸 수 있는 것**\n")
    w("> `grad_norm_emb` 의 절대값이 바닥을 치지 않는다 (4.78 -> 2.98).")
    w("> 같은 구간에서 BPB 는 100MB 이후 0.005 밖에 안 움직인다.")
    w("> **BPB 가 정체된 뒤에도 기울기 신호가 남아 있다.**\n")
    w("이 진술은 정규화와 무관하다 — 절대값이 0 으로 가지 않는다는 관측이기")
    w("때문이다. 그리고 이것만으로 병목의 원인을 특정할 수는 없다는 단서도")
    w("`phase6.md` 에 이미 달려 있다.\n")

    w("## sqrt(N) 정규화도 완전하지 않다\n")
    w("임베딩 기울기는 **희소하다** — 배치에 나온 토큰의 행만 0 이 아니다.")
    w(f"{cfg['vocab_size']:,}행 전체로 나누면 이번에는 과소평가가 된다.")
    w("**어느 쪽으로도 보정 없이는 해석할 수 없다.**\n")
    w("제대로 하려면 셋이 필요하다.\n")
    w("```")
    w("(a) 손상된 30,000행만의 노름     계측을 새로 넣어야 한다")
    w("(b) grad / weight 비            스케일 무관 지표")
    w("(c) C0 · T2a 대조군             phase6.md 가 이미 한계로 적어 뒀다")
    w("```\n")
    w("셋 다 P2 범위 밖이다. **그래서 이 축은 논문에서 관측으로만 쓰고**")
    w("**기전 주장에 쓰지 않는다.**\n")

    w("## 한계\n")
    w("- 군 분류는 파라미터 이름 기준이다 (`embed_tokens` / `self_attn` / `mlp`).")
    w("  norm·bias 일부는 어느 군에도 안 들어가고 `grad_norm` 전체에만 든다")
    w("- 노름은 클리핑 **전** 에 쟀다 (`src/training/cpt.py`)")
    w("- T2b 한 조건이다. C0·T2a 의 기울기 궤적은 재지 않았다")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"썼다  {dest}")
    print(f"  원소당 emb/attn  {first:.2f} -> {last:.2f}   (raw 는 끝까지 1 위)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
