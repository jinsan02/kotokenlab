"""README 그림 둘을 원장에서 그린다 (GPU 0시간). 손으로 숫자를 옮기지 않는다.

    .conda/python.exe tools/readme_figures.py

    docs/img/gap_summary.svg   처방별 한국어 잔차 d 와 문서 95% 구간 — Final Test 강조
    docs/img/wsd_curve.svg     감쇠 구간의 한국어 dev BPB — 상수 LR 대 WSD (C0 · T2b)

색은 dataviz 기본 팔레트의 1 · 2번 슬롯(파랑 · 주황)과 회색 하나다. GitHub 의 밝은 바탕
(#ffffff)과 어두운 바탕(#0d1117) 모두에서 validate_palette 를 통과했다 (2026-10-19).
README 의 그림은 이미지로 렌더되므로 hover 는 없다 — 같은 수치는 README 표와
reports/tables 에 있다 (표 보기 대체).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_GAP = ROOT / "docs" / "img" / "gap_summary.svg"
OUT_WSD = ROOT / "docs" / "img" / "wsd_curve.svg"
FONT = '"Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", system-ui, -apple-system, sans-serif'

STYLE = f"""<style>
  text {{ font-family: {FONT}; fill: #1f2328; }}
  .sub {{ fill: #59636e; }} .muted {{ fill: #818b98; }}
  .grid {{ stroke: #eaeef2; stroke-width: 1; }} .base {{ stroke: #afb8c1; stroke-width: 1; }}
  .c1 {{ fill: #2a78d6; }} .l1 {{ stroke: #2a78d6; stroke-width: 2; fill: none; }}
  .c2 {{ fill: #eb6834; }} .l2 {{ stroke: #eb6834; stroke-width: 2; fill: none; }}
  .cg {{ fill: #8c959f; }} .lg {{ stroke: #8c959f; stroke-width: 2; fill: none; }}
  .ring {{ stroke: #ffffff; stroke-width: 2; }}
  .band {{ fill: #f6f8fa; }}
  @media (prefers-color-scheme: dark) {{
    text {{ fill: #e6edf3; }} .sub {{ fill: #9198a1; }} .muted {{ fill: #8b949e; }}
    .grid {{ stroke: #262c36; }} .base {{ stroke: #59636e; }}
    .c1 {{ fill: #3987e5; }} .l1 {{ stroke: #3987e5; }}
    .c2 {{ fill: #d95926; }} .l2 {{ stroke: #d95926; }}
    .cg {{ fill: #6e7681; }} .lg {{ stroke: #6e7681; }}
    .ring {{ stroke: #0d1117; }} .band {{ fill: #161b22; }}
  }}
</style>"""


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def gap_rows() -> list:
    """(이름, 설명, d, lo, hi, 강조) — 전부 문서 단위 paired bootstrap (같은 함수 · 같은 설정)."""
    from tools import doc_bootstrap as D, heldout_spec as H, w5_spec as W5, w6_spec as W6

    def pair(a_docs, b_docs, b_seed=W5.RNG_SEED):
        r = D.paired_bootstrap(a_docs["ko"], b_docs["ko"], W5.B, b_seed)
        return r["diff"], r["lo"], r["hi"]

    dev = {n: D.load_docs(n) for n in ("t2b_cos", "c0_cos", "t2b_const", "c0_const",
                                        "t2b_upd", "t2b_warm")}
    wsd = {n: D.load_docs_run(W6.doc_run_id(n)) for n in ("t2b_wsd", "c0_wsd")}
    held = {n: D.load_docs_run(H.doc_run_id(n)) for n in H.CHECKPOINTS}
    rows = [
        ("1차 코사인 LR", "dev · seed 42", *pair(dev["t2b_cos"], dev["c0_cos"]), False),
        ("상수 LR", "dev · seed 42 재학습", *pair(dev["t2b_const"], dev["c0_const"]), False),
        ("같은 update 1,523", "dev · seed 42 · C0 는 상수 LR", *pair(dev["t2b_upd"], dev["c0_const"]), False),
        ("신규 행 warm-start", "dev · seed 42 · C0 는 상수 LR", *pair(dev["t2b_warm"], dev["c0_const"]), False),
        ("WSD 마지막 20% 감쇠", "dev · seed 42 · C0 도 WSD", *pair(wsd["t2b_wsd"], wsd["c0_wsd"]), False),
        ("Final Test", f"처음 보는 {len(held['c0']['ko']):,}문서 · 상수 LR 대표 쌍",
         *pair(held["t2b"], held["c0"]), True),
    ]
    return rows


def draw_gap(rows: list) -> str:
    W, left, right, top, rowh = 1000, 300, 150, 108, 46
    xmax = 0.45
    pw = W - left - right
    x = lambda v: left + v / xmax * pw  # noqa: E731
    H = top + rowh * len(rows) + 64
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         'aria-label="처방별 한국어 잔차 d = BPB(T2b) - BPB(C0) 와 문서 95% 구간. '
         + _esc(" · ".join(f"{r[0]} {r[2]:+.4f}" for r in rows)) + '">',
         "<title>어느 처방도 격차를 0 쪽으로 옮기지 못했다 — 처음 보는 문서에서도 그대로다</title>", STYLE,
         '<text x="0" y="26" font-size="18" font-weight="600">어느 처방도 격차를 0 쪽으로 옮기지 못했다</text>',
         '<text x="0" y="50" font-size="14" class="sub">한국어 잔차 d = BPB(T2b) − BPB(C0), 같은 원문 168.5MB CPT 뒤. '
         '0 이면 격차 없음.</text>',
         '<text x="0" y="70" font-size="14" class="sub">점 = 값, 가로선 = 문서 단위 paired bootstrap 95% 구간 '
         '(대부분 점보다 짧다).</text>']
    y0, y1 = top - 10, top + rowh * len(rows) - 6
    for t in (0.0, 0.1, 0.2, 0.3, 0.4):
        cls = "base" if t == 0 else "grid"
        o.append(f'<line class="{cls}" x1="{x(t):.1f}" y1="{y0}" x2="{x(t):.1f}" y2="{y1}"/>')
        o.append(f'<text x="{x(t):.1f}" y="{y1 + 20}" font-size="12" text-anchor="middle" class="muted">{t:.1f}</text>')
    o.append(f'<text x="{x(0) + 6:.1f}" y="{y0 + 4}" font-size="12" class="muted">격차 없음</text>')
    for i, (name, desc, d, lo, hi, em) in enumerate(rows):
        cy = top + rowh * i + rowh / 2 - 6
        dot, line = ("c1", "l1") if em else ("cg", "lg")
        weight = ' font-weight="600"' if em else ""
        o.append(f'<text x="0" y="{cy - 2:.1f}" font-size="14"{weight}>{_esc(name)}</text>')
        o.append(f'<text x="0" y="{cy + 15:.1f}" font-size="12" class="muted">{_esc(desc)}</text>')
        o.append(f'<line class="{line}" x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}"/>')
        o.append(f'<circle class="{dot} ring" cx="{x(d):.1f}" cy="{cy:.1f}" r="{6 if em else 5}"/>')
        sub = "" if em else ' class="sub"'
        o.append(f'<text x="{x(xmax) + 14:.1f}" y="{cy + 5:.1f}" font-size="13"{weight}{sub}>{d:+.4f}</text>')
    o.append(f'<text x="0" y="{H - 10}" font-size="12" class="muted">구간은 문서 표본의 불확실성만 담는다 — '
             '학습 seed 불확실성(상수 LR seed 3쌍 SD 0.0002)은 따로다. 축은 BPB.</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def wsd_series() -> dict:
    """{조건: {스케줄: [(MB, ko BPB)]}} — 100MB 이후 평가 지점 + 종점."""
    from tools import w6_spec as S
    from tools.wsd_pairs import lm_rows
    lm = lm_rows()
    out: dict = {}
    for cond in ("c0", "t2b"):
        out[cond] = {}
        for sched, run in (("상수 LR", S.BASE[cond]), ("WSD", S.WSD[cond])):
            pts = sorted((int(r["raw_bytes_seen"]) / 1e6, float(r["bpb"])) for r in lm
                         if r["run_id"] == run and r["domain"] == "ko"
                         and int(r["raw_bytes_seen"]) >= 100_000_000)
            out[cond][sched] = pts
    return out


def draw_wsd(series: dict) -> str:
    from tools import w6_spec as S
    W, H = 1000, 420
    panels = (("c0", "C0 (원본 Qwen)", 0), ("t2b", "T2b (어휘 치환)", 1))
    pw, ph, top, gap = 400, 240, 110, 100
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         'aria-label="감쇠 구간의 한국어 dev BPB, 상수 LR 과 WSD. '
         + _esc(" · ".join(f"{c} {s} 종점 {p[-1][1]:.4f}" for c, d in series.items() for s, p in d.items()))
         + '">', "<title>마지막 감쇠는 끝에서 학습을 멈추게 했다 — 손해는 T2b 가 더 크다</title>", STYLE,
         '<text x="0" y="26" font-size="18" font-weight="600">마지막 감쇠는 끝에서 학습을 멈추게 했다</text>',
         '<text x="0" y="50" font-size="14" class="sub">한국어 dev BPB (낮을수록 좋다), 원문 100MB 이후 · '
         f'회색 띠 = WSD 감쇠 구간 ({S.DECAY_START / 1e6:.1f}MB~, 끝 학습률 1e-6) · seed 42</text>']
    # 범례 (2 계열 — 범례 + 끝점 직접 라벨)
    o.append('<line class="l1" x1="0" y1="78" x2="22" y2="78"/><circle class="c1" cx="11" cy="78" r="4"/>'
             '<text x="30" y="83" font-size="13">상수 LR</text>'
             '<line class="l2" x1="110" y1="78" x2="132" y2="78"/><circle class="c2" cx="121" cy="78" r="4"/>'
             '<text x="140" y="83" font-size="13">WSD</text>')
    xmin, xmax = 100.0, 170.0
    for cond, title, k in panels:
        ox = 40 + k * (pw + gap)
        ys = [v for pts in series[cond].values() for _, v in pts]
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.12
        lo, hi = lo - pad, hi + pad
        X = lambda mb: ox + (mb - xmin) / (xmax - xmin) * pw  # noqa: E731
        Y = lambda v: top + (hi - v) / (hi - lo) * ph  # noqa: E731
        o.append(f'<text x="{ox}" y="{top - 10}" font-size="14" font-weight="600">{_esc(title)}</text>')
        o.append(f'<rect class="band" x="{X(S.DECAY_START / 1e6):.1f}" y="{top}" '
                 f'width="{X(S.BUDGET / 1e6) - X(S.DECAY_START / 1e6):.1f}" height="{ph}"/>')
        for i in range(5):
            v = lo + (hi - lo) * i / 4
            o.append(f'<line class="grid" x1="{ox}" y1="{Y(v):.1f}" x2="{ox + pw}" y2="{Y(v):.1f}"/>')
            o.append(f'<text x="{ox - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end" class="muted">{v:.3f}</text>')
        o.append(f'<line class="base" x1="{ox}" y1="{top + ph}" x2="{ox + pw}" y2="{top + ph}"/>')
        for mb in (100, 120, 140, 160):
            o.append(f'<text x="{X(mb):.1f}" y="{top + ph + 18}" font-size="11" text-anchor="middle" class="muted">{mb}MB</text>')
        ends = []
        for sched, (cl, cc) in (("상수 LR", ("l1", "c1")), ("WSD", ("l2", "c2"))):
            pts = series[cond][sched]
            o.append(f'<polyline class="{cl}" points="' + " ".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in pts) + '"/>')
            for m, v in pts:
                o.append(f'<circle class="{cc} ring" cx="{X(m):.1f}" cy="{Y(v):.1f}" r="4"/>')
            ends.append((Y(pts[-1][1]), pts[-1][1], X(pts[-1][0])))
        # 끝점 라벨 — 겹치지 않게 위아래로 벌린다
        ends.sort()
        if len(ends) == 2 and ends[1][0] - ends[0][0] < 16:
            ends[0] = (ends[1][0] - 16, *ends[0][1:])
        for yy, v, xx in ends:
            o.append(f'<text x="{xx + 8:.1f}" y="{yy + 4:.1f}" font-size="12" class="sub">{v:.4f}</text>')
        d = series[cond]["WSD"][-1][1] - series[cond]["상수 LR"][-1][1]
        o.append(f'<text x="{ox}" y="{top + ph + 44}" font-size="13">종점 WSD − 상수 LR = <tspan font-weight="600">{d:+.4f}</tspan></text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main() -> int:
    OUT_GAP.write_text(draw_gap(gap_rows()), encoding="utf-8", newline="\n")
    OUT_WSD.write_text(draw_wsd(wsd_series()), encoding="utf-8", newline="\n")
    print(f"썼다 {OUT_GAP.relative_to(ROOT)} · {OUT_WSD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
