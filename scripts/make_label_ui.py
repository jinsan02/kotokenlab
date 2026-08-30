"""도메인 감사 표본을 손으로 라벨링하는 화면을 만든다.

TSV 200줄을 편집기로 채우는 건 고역이고, 엑셀은 UTF-8 TSV 인코딩을 자주 깨뜨린다.
그래서 자기완결적인 HTML 한 장을 만든다 — 브라우저로 열고 숫자키로 라벨을 찍은 뒤
채워진 TSV 를 내려받는다.

    .conda/python.exe scripts/make_label_ui.py
    (생성된 reports/tables/domain_audit_label.html 을 브라우저로 연다)

작업은 localStorage 에 자동 저장되므로 중간에 닫아도 이어서 할 수 있다.
다 채우면 TSV 를 내려받아 reports/tables/domain_audit.tsv 를 덮어쓰고:

    .conda/python.exe scripts/audit_domain_rules.py --mode score
"""

from __future__ import annotations

import argparse
import html
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.domain import DOMAINS  # noqa: E402

LABELS = [d for d in DOMAINS if d not in ("conversational", "noisy")]

PAGE = """<title>도메인 감사 — {n}건</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#ddd;--acc:#0b57d0;--ok:#0a7c3f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e6e6e6;--mut:#9aa0a6;--line:#333;--acc:#8ab4f8;--ok:#7ee2a8}}}}
*{{box-sizing:border-box}}
body{{margin:0;font:15px/1.6 system-ui,'Malgun Gothic',sans-serif;background:var(--bg);color:var(--fg)}}
header{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:9}}
.bar{{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:8px}}
.bar>div{{height:100%;background:var(--ok);width:0%;transition:width .2s}}
main{{max-width:900px;margin:0 auto;padding:16px}}
.card{{border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}}
.meta{{color:var(--mut);font-size:13px;word-break:break-all}}
.prev{{margin:10px 0;padding:10px;background:rgba(128,128,128,.10);border-radius:8px;white-space:pre-wrap}}
.pred{{display:inline-block;padding:2px 8px;border-radius:6px;background:rgba(128,128,128,.18);font-size:13px}}
.btns{{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}}
button{{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:8px;
background:transparent;color:var(--fg);cursor:pointer}}
button:hover{{border-color:var(--acc)}}
button.sel{{background:var(--ok);border-color:var(--ok);color:#fff}}
button.same{{border-color:var(--acc);color:var(--acc)}}
kbd{{font:12px monospace;color:var(--mut)}}
.done{{opacity:.45}}
.top{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.top b{{font-size:16px}}
#dl{{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}}
</style>
<header>
  <div class="top">
    <b>도메인 감사</b>
    <span id="cnt" class="meta"></span>
    <button id="dl">채워진 TSV 내려받기</button>
    <button id="jump">다음 미완료로</button>
    <button id="clr">초기화</button>
  </div>
  <div class="bar"><div id="pg"></div></div>
  <div class="meta" style="margin-top:6px">
    숫자키로 라벨 선택 · <kbd>Enter</kbd> 예측이 맞으면 그대로 확정 · 자동 저장됨
  </div>
</header>
<main id="list"></main>
<script>
const ROWS = {rows};
const LABELS = {labels};
const KEY = "domain_audit_v1";
let gold = {{}};
try {{ gold = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ gold = {{}}; }}

function save() {{
  try {{ localStorage.setItem(KEY, JSON.stringify(gold)); }} catch (e) {{}}
  render();
}}
function render() {{
  const done = Object.keys(gold).length;
  document.getElementById("cnt").textContent = done + " / " + ROWS.length + " 완료";
  document.getElementById("pg").style.width = (done / ROWS.length * 100) + "%";
  ROWS.forEach((r, i) => {{
    const card = document.getElementById("c" + i);
    if (!card) return;
    card.classList.toggle("done", !!gold[r.sample_id]);
    card.querySelectorAll("button[data-l]").forEach(b => {{
      b.classList.toggle("sel", gold[r.sample_id] === b.dataset.l);
    }});
  }});
}}
const list = document.getElementById("list");
ROWS.forEach((r, i) => {{
  const d = document.createElement("div");
  d.className = "card"; d.id = "c" + i;
  const btns = LABELS.map((l, k) =>
    `<button data-l="${{l}}" data-i="${{i}}" class="${{l === r.predicted_domain ? 'same' : ''}}">`
    + `${{k + 1}}. ${{l}}</button>`).join("");
  d.innerHTML =
    `<div class="meta">${{i + 1}}/${{ROWS.length}} · ${{r.sample_id}} · <b>${{r.host}}</b></div>`
    + `<div class="meta">${{r.url}}</div>`
    + `<div class="prev">${{r.text_preview}}</div>`
    + `<div>규칙 예측: <span class="pred">${{r.predicted_domain}}</span></div>`
    + `<div class="btns">${{btns}}</div>`;
  list.appendChild(d);
}});
list.addEventListener("click", e => {{
  const b = e.target.closest("button[data-l]");
  if (!b) return;
  const r = ROWS[+b.dataset.i];
  if (gold[r.sample_id] === b.dataset.l) delete gold[r.sample_id];
  else gold[r.sample_id] = b.dataset.l;
  save();
}});
document.addEventListener("keydown", e => {{
  const card = document.elementFromPoint(window.innerWidth / 2, 200);
  const host = card && card.closest(".card");
  if (!host) return;
  const i = +host.id.slice(1), r = ROWS[i];
  if (e.key === "Enter") {{ gold[r.sample_id] = r.predicted_domain; save(); scrollNext(i); }}
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= LABELS.length) {{
    gold[r.sample_id] = LABELS[n - 1]; save(); scrollNext(i);
  }}
}});
function scrollNext(from) {{
  for (let j = from + 1; j < ROWS.length; j++)
    if (!gold[ROWS[j].sample_id]) {{
      document.getElementById("c" + j).scrollIntoView({{block: "start", behavior: "smooth"}});
      return;
    }}
}}
document.getElementById("jump").onclick = () => scrollNext(-1);
document.getElementById("clr").onclick = () => {{
  if (confirm("라벨을 전부 지웁니다. 계속?")) {{ gold = {{}}; save(); }}
}};
document.getElementById("dl").onclick = () => {{
  const head = {header};
  const lines = [head.join("\\t")];
  ROWS.forEach(r => lines.push(head.map(c =>
    c === "gold_domain" ? (gold[r.sample_id] || "") : (r[c] || "")).join("\\t")));
  const blob = new Blob([lines.join("\\n") + "\\n"], {{type: "text/tab-separated-values"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "domain_audit.tsv";
  a.click();
}};
render();
</script>
"""


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="도메인 감사 라벨링 화면 생성")
    ap.add_argument("--audit", default="reports/tables/domain_audit.tsv")
    ap.add_argument("--out", default="reports/tables/domain_audit_label.html")
    args = ap.parse_args(argv)

    src = ROOT / args.audit
    if not src.exists():
        print(f"{src} 가 없다. --mode sample 로 먼저 뽑아라.", file=sys.stderr)
        return 1

    with io.open(src, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, ln.rstrip("\n").split("\t")))
                for ln in fh if ln.strip()]
    for r in rows:
        for k in r:
            r[k] = html.escape(r[k])

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        PAGE.format(n=len(rows),
                    rows=json.dumps(rows, ensure_ascii=False),
                    labels=json.dumps(LABELS, ensure_ascii=False),
                    header=json.dumps(header, ensure_ascii=False)),
        encoding="utf-8", newline="\n")

    print(f"{out}\n  표본 {len(rows)}건, 라벨 {len(LABELS)}종\n")
    print("브라우저로 열어서 숫자키로 라벨을 찍는다. 예측이 맞으면 Enter.")
    print("작업은 자동 저장되니 중간에 닫아도 된다.\n")
    print("다 채우면 TSV 를 내려받아 아래 경로를 덮어쓰고:")
    print(f"  {src}")
    print("  .conda/python.exe scripts/audit_domain_rules.py --mode score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
