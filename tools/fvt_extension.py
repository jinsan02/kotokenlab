"""E1 의 "FVT 확장분" 이 FVT 평균과 얼마나 다른 값인가 (학습 없음, GPU 0시간).

    .conda/python.exe tools/fvt_extension.py

## 왜

[`fvt_check.py`](fvt_check.py) 가 "E1 은 FVT 와 동등하며, **FVT 가 정의되지 않는**
**13.9% 까지 merge 계보로 확장한다**" 를 뒷받침했다. 그런데 **확장분이 어떤 값인지**
는 아직 안 쟀다. 정의되지 않는 구간에서 merge 계보가 내놓는 벡터가, FVT 정신을
가장 곧이곧대로 따른 대안(토큰의 **바이트를 쪼개 평균**)과 크게 다르면,
"확장" 이 아니라 "다른 방법" 이라고 불러야 한다.

## 무엇을 비교하나

새 토큰 하나마다 둘을 만든다. 둘 다 **원본 Qwen 임베딩** 위에서다.

```
E1        merge 부품 둘의 평균          (emb[left] + emb[right]) / 2
바이트    토큰이 나타내는 바이트들의     mean(emb[byte token] for b in bytes)
          byte-level 토큰 평균          ByteLevel BPE 라 항상 정의된다
```

그리고 코사인 유사도 분포를 **두 무리로 나눠** 본다.

```
정의됨     표면형이 유효 UTF-8 이라 FVT 가 정의되는 토큰 (fvt_check 의 25,821개)
           여기서는 E1 = FVT 이므로, 이 무리의 분포가 **기준선** 이다
확장분     표면형이 유효 UTF-8 이 아니어서 FVT 가 정의되지 않는 토큰 (4,179개)
```

확장분의 분포가 기준선과 비슷하면 "확장이 안전하다" 의 근거가 된다. 크게 낮으면
그 13.9% 가 E1 성능에 미치는 영향을 따로 논해야 한다.

**바이트 평균은 "정답" 이 아니다.** FVT 를 그 구간까지 억지로 밀었을 때 나오는
가장 자연스러운 값일 뿐이다. 이 도구는 둘의 **거리** 를 재는 것이지 어느 쪽이
옳은지 정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(ROOT / "tools"))
from fvt_check import byte_decoder, surface  # noqa: E402

OUT = ROOT / "reports" / "tables" / "fvt_extension.md"
DEFAULT_MAP = ROOT / "artifacts" / "tokenizers" / "kot2b_v2_n30000" / "id_map.json"

QUANTILES = (5, 25, 50, 75, 95)


def cosine(a, b) -> float:
    import numpy as np

    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def describe(xs: list) -> dict:
    xs = sorted(xs)
    n = len(xs)
    out = {"n": n, "mean": statistics.fmean(xs) if n else float("nan")}
    for q in QUANTILES:
        out[f"p{q}"] = xs[min(n - 1, int(q / 100 * n))] if n else float("nan")
    return out


def main(argv: list | None = None) -> int:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description="E1 확장분 vs 바이트 평균")
    ap.add_argument("--id-map", default=str(DEFAULT_MAP))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    spec = json.loads(Path(args.id_map).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(spec["base"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["base"], revision=spec["revision"], dtype=torch.float32)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    vocab = tok.get_vocab()

    dec = byte_decoder()
    enc = {b: ch for ch, b in dec.items()}          # 바이트 -> ByteLevel 문자
    byte_id = {b: vocab[enc[b]] for b in range(256) if enc[b] in vocab}
    missing_byte = 256 - len(byte_id)

    groups: dict = {"정의됨": [], "확장분": []}
    norms: dict = {"정의됨": [], "확장분": []}
    skipped = 0

    for info in spec["map"].values():
        left, right = info["left"], info["right"]
        li, ri = vocab.get(left), vocab.get(right)
        if li is None or ri is None:
            skipped += 1
            continue
        token = info["token"]
        try:
            raw = bytes(dec[ch] for ch in token)
        except KeyError:
            skipped += 1
            continue
        ids = [byte_id[b] for b in raw if b in byte_id]
        if len(ids) != len(raw):
            skipped += 1
            continue
        e1 = (emb[li] + emb[ri]) / 2.0
        bmean = emb[ids].mean(axis=0)
        key = "정의됨" if surface(token, dec) is not None else "확장분"
        groups[key].append(cosine(e1, bmean))
        nb = float(np.linalg.norm(bmean))
        norms[key].append(float(np.linalg.norm(e1)) / nb if nb else float("nan"))

    stats = {k: describe(v) for k, v in groups.items()}
    nstats = {k: describe(v) for k, v in norms.items()}

    o: list = []
    w = o.append
    w("# E1 의 FVT 확장분은 바이트 평균과 얼마나 다른가\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/fvt_extension.py`")
    w("> 학습 없음, GPU 0시간. 원본 Qwen 임베딩 위에서 계산한다.\n")
    w("[`fvt_check.md`](fvt_check.md) 가 \"E1 은 FVT 와 동등하며 FVT 가 정의되지")
    w("않는 구간까지 merge 계보로 확장한다\" 를 뒷받침했다. 이 표는 **그 확장분이**")
    w("**어떤 값인지** 를 잰다 — FVT 를 그 구간까지 억지로 밀었을 때 나오는 값")
    w("(토큰의 바이트를 쪼개 평균)과 코사인으로 비교한다.\n")
    w(f"대상: `{Path(args.id_map).parent.name}` 의 새 토큰 "
      f"{sum(len(v) for v in groups.values()):,}개"
      + (f" (제외 {skipped:,}개)" if skipped else ""))
    if missing_byte:
        w(f"주의: vocab 에 없는 byte-level 토큰 {missing_byte}개")
    w("")
    w("## 코사인 유사도 — E1 vs 바이트 평균\n")
    w("| 무리 | 개수 | 평균 | p5 | p25 | p50 | p75 | p95 |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in ("정의됨", "확장분"):
        s = stats[k]
        w(f"| {k} | {s['n']:,} | {s['mean']:.4f} | " +
          " | ".join(f"{s[f'p{q}']:.4f}" for q in QUANTILES) + " |")
    w("")
    near = {k: sum(1 for x in v if x > 0.999) for k, v in groups.items()}
    for k in ("정의됨", "확장분"):
        if groups[k]:
            w(f"- {k}: 코사인 > 0.999 인 토큰 {near[k]:,}개 "
              f"({near[k] / len(groups[k]):.1%}) — merge 부품이 곧 바이트인 경우다")
    w("")
    w("**\"정의됨\" 무리가 기준선이다** — 그 구간에서는 E1 이 곧 FVT 이므로,")
    w("거기서 나온 코사인이 \"FVT 와 바이트 평균 사이의 원래 거리\" 다.")
    w("확장분이 그와 비슷하면 확장이 기준선 밖으로 벗어나지 않았다는 뜻이다.\n")
    w("## 노름 비 — ||E1|| / ||바이트 평균||\n")
    w("| 무리 | 평균 | p5 | p50 | p95 |")
    w("|---|---:|---:|---:|---:|")
    for k in ("정의됨", "확장분"):
        s = nstats[k]
        w(f"| {k} | {s['mean']:.3f} | {s['p5']:.3f} | {s['p50']:.3f} | {s['p95']:.3f} |")
    w("")
    w("노름은 방향과 따로 본다. 1차에서 **노름을 키우는 보정이 오히려 해로웠다**")
    w("([`DESIGN_DELTA.md`](../../docs/DESIGN_DELTA.md) 3-6) — tied 임베딩이라")
    w("출력 로짓 방향을 겸하기 때문이다. 그래서 여기서도 노름 차이를 \"틀림\" 으로")
    w("읽지 않는다.\n")
    w("## 한계\n")
    w("- **바이트 평균은 정답이 아니다.** FVT 를 정의되지 않는 구간까지 밀었을 때")
    w("  나오는 자연스러운 대안일 뿐이고, 원 논문이 그렇게 하라고 한 적은 없다")
    w("- 두 벡터의 코사인은 **초기값** 의 거리다. CPT 를 돌리면 달라진다 —")
    w("  이 표는 초기화 선택의 차이를 재는 것이지 성능 차이를 재는 것이 아니다")
    w("- 성능 영향을 보려면 바이트 평균으로 초기화한 조건을 따로 CPT 해야 한다.")
    w("  **이 표는 그 실험을 대신하지 않는다**")

    dest = Path(args.out)
    dest.write_text("\n".join(o) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(o))
    print(f"\n썼다  {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
