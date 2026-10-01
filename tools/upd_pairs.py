"""3주차 같은-update 대조를 원장에서 집계하고 PLAN 의 경계로 판정한다 (학습 없음).

    .conda/python.exe tools/upd_pairs.py

동결 설정은 `tools/upd_spec.py` (PLAN.md "3주차 run 설정 동결 — 2026-10-02").
**결과를 보기 전에 만들었다** — 판정 경계를 새로 만들지 않고 등록된 것만 계산한다.

## 무엇을 계산하나

seed 마다 같은 seed 의 C0 상수 LR 과 짝짓는다. 지표는 한국어 dev BPB.

```
Cf      C0 최종                                   (cpt_c0_qwen_r5_seed<s>)
Bf_upd  새 T2b 종점 — C0 와 같은 update 수        (cpt_t2b_mean_upd_seed<s> final)
Bf_168  새 T2b 가 168.5MB 를 지난 첫 update 경계   (같은 run 의 --eval-at 지점)
d_upd   Bf_upd - Cf
d_168   Bf_168 - Cf
rho     (d_168 - d_upd) / d_168      update 부족이 설명하는 잔차의 몫
```

`d_168 <= 0` 이면 rho 를 계산하지 않는다 (분모가 몫의 기준이 아니게 된다).

## 판정하기 전에 run 을 검증한다

하나라도 어긋나면 그 seed 는 판정에서 빠지고 이유를 표에 적는다.

- 원장 argv 가 동결 명령과 문자 그대로 같다
- update 수가 PLAN 표와 같고, 관찰 토큰 = 반영 토큰 = update x 32,768 (꼬리 없음)
- git_dirty = 0, 짝 C0(새 코드) 이후 측정 경로 커밋이 동결이 다룬 것뿐
- config 의 eval_at · pool_extend_docs · budget_tokens 가 동결값
- **사전 점검 예측과 같다** — 종료 원문 바이트, d_168 지점의 update 번호·원문
  바이트 (`reports/tables/upd_preflight.json`). 어긋나면 데이터 순서가 계획과
  다르게 돈 것이다
- 학습 전 B0 가 기존 T2b r5 의 B0 와 같다 (같은 체크포인트인가)

## 판정

PLAN 대로 **세 seed 평균 rho** 로 한다. 세 seed 가 모두 검증을 통과하기 전에는
seed 별 값을 기술만 하고 판정하지 않는다. 기존 T2b r5 와의 비교는 워밍업(21 vs
30 update)과 코드 계보가 달라 **교차 확인** 으로만 싣는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import upd_spec as S  # noqa: E402
from tools.seed_pairs import describe  # noqa: E402

LM = ROOT / "experiments" / "lm_metrics.tsv"
LEDGER = ROOT / "experiments" / "LEDGER.tsv"
RUNS = ROOT / "experiments" / "runs"
PRED = ROOT / "reports" / "tables" / "upd_preflight.json"
OUT = ROOT / "reports" / "tables" / "upd_pairs.md"

DOMAINS = (("ko", "한국어"), ("en", "영어"), ("code", "코드"))
B0_TOL = 1e-6


def _tsv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def ok_rows() -> dict:
    return {r["run_id"]: r for r in _tsv(LEDGER) if r["status"] == "ok"}


def lm_rows() -> list:
    return [r for r in _tsv(LM) if r["split"] == "dev"]


def bpb_at(rows: list, run_id: str, checkpoint: str, domain: str):
    hits = [float(r["bpb"]) for r in rows
            if r["run_id"] == run_id and r["checkpoint"] == checkpoint and r["domain"] == domain]
    return hits[-1] if hits else None


def eval_at_row(rows: list, run_id: str, domain: str = "ko"):
    """--eval-at 지점의 행. EVAL_AT 이상 EVAL_AT+SLACK 미만의 중간 체크포인트.

    --eval-at 을 빠뜨린 run 이면 그 구간에 행이 없다 — 다음 격자(180MB)를 대신
    잡지 않고 None 을 돌려준다. 둘 이상이면 무엇이 d_168 인지 모르므로 None.
    """
    hits = [r for r in rows
            if r["run_id"] == run_id and r["domain"] == domain
            and r["checkpoint"] not in ("final", "step0")
            and S.EVAL_AT <= int(r["raw_bytes_seen"]) < S.EVAL_AT + S.EVAL_AT_SLACK]
    return hits[0] if len(hits) == 1 else None


def rho(d_168, d_upd):
    if d_168 is None or d_upd is None or d_168 <= 0:
        return None
    return (d_168 - d_upd) / d_168


def git_code_commits(a: str, b: str) -> list:
    out = subprocess.run(["git", "log", "--format=%h", f"{a}..{b}", "--", "src", "configs"],
                         cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    return [x for x in out.stdout.split() if x] if out.returncode == 0 else ["<git 실패>"]


def validate(seed: int, ok: dict, lm: list, pred: dict) -> list:
    """이 seed 의 run 이 동결대로 돌았는지. 문제 목록 (비면 통과)."""
    rid = S.run_id(seed)
    r = ok.get(rid)
    if r is None:
        return ["아직 없음"]
    p: list = []
    if r.get("argv", "") != S.argv_str(seed):
        p.append("argv 가 동결 명령과 다르다")
    want_u = S.UPDATES[seed]
    want_t = S.budget_tokens(seed)
    if r.get("updates") != str(want_u):
        p.append(f"update {r.get('updates')} != {want_u}")
    if r.get("tokens_applied") != str(want_t) or r.get("tokens_seen") != str(want_t):
        p.append(f"토큰 관찰 {r.get('tokens_seen')} · 반영 {r.get('tokens_applied')} != {want_t}")
    if r.get("git_dirty") != "0":
        p.append(f"git_dirty={r.get('git_dirty')}")
    ref = ok.get(f"{S.CTRL}2026")
    if ref is not None:
        extra = [h for h in git_code_commits(ref["git_commit"], r["git_commit"])
                 if not any(h.startswith(a) or a.startswith(h)
                            for a in S.ALLOWED_CODE_COMMITS_SINCE_CTRL)]
        if extra:
            p.append("동결 밖 측정 경로 커밋: " + ", ".join(extra))
    cfg_p = RUNS / rid / "config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        if cfg.get("eval_at") != [S.EVAL_AT]:
            p.append(f"config eval_at={cfg.get('eval_at')}")
        if cfg.get("pool_extend_docs") != S.POOL_EXTEND_DOCS:
            p.append(f"config pool_extend_docs={cfg.get('pool_extend_docs')}")
        if cfg.get("budget_tokens") != want_t:
            p.append(f"config budget_tokens={cfg.get('budget_tokens')}")
    else:
        p.append("config.json 없음")
    pr = pred.get(str(seed))
    if pr is None:
        p.append("사전 점검 예측 없음 (upd_preflight.json)")
    else:
        if r.get("raw_bytes_seen") != str(pr["raw_bytes_final"]):
            p.append(f"종료 원문 {r.get('raw_bytes_seen')} != 예측 {pr['raw_bytes_final']}")
        row = eval_at_row(lm, rid)
        if row is None:
            p.append("d_168 행이 없거나 하나가 아니다 (--eval-at 확인)")
        else:
            if row["checkpoint"] != f"step{pr['d168_update']}":
                p.append(f"d_168 지점 {row['checkpoint']} != 예측 step{pr['d168_update']}")
            if row["raw_bytes_seen"] != str(pr["d168_raw_bytes"]):
                p.append(f"d_168 원문 {row['raw_bytes_seen']} != 예측 {pr['d168_raw_bytes']}")
    b0 = bpb_at(lm, rid, "step0", "ko")
    b0_old = bpb_at(lm, f"{S.OLD_TREAT}{seed}", "step0", "ko")
    if b0 is None:
        p.append("step0 행 없음")
    elif b0_old is not None and abs(b0 - b0_old) > B0_TOL:
        p.append(f"B0 {b0} != 기존 r5 {b0_old} (다른 체크포인트?)")
    return p


def per_seed(seed: int, lm: list) -> dict:
    rid = S.run_id(seed)
    out: dict = {"seed": seed}
    for dom, _ in DOMAINS:
        cf = bpb_at(lm, f"{S.CTRL}{seed}", "final", dom)
        bf = bpb_at(lm, rid, "final", dom)
        row = eval_at_row(lm, rid, dom)
        b168 = float(row["bpb"]) if row else None
        old = bpb_at(lm, f"{S.OLD_TREAT}{seed}", "final", dom)
        out[dom] = {
            "cf": cf, "bf_upd": bf, "bf_168": b168, "old_final": old,
            "d_upd": None if bf is None or cf is None else bf - cf,
            "d_168": None if b168 is None or cf is None else b168 - cf,
        }
    k = out["ko"]
    k["rho"] = rho(k["d_168"], k["d_upd"])
    return out


def f6(x) -> str:
    return "NA" if x is None else f"{x:.6f}"


def sf6(x) -> str:
    return "NA" if x is None else f"{x:+.6f}"


def pct(x) -> str:
    return "NA" if x is None else f"{x:.1%}"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="3주차 같은-update 대조 집계")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    ok = ok_rows()
    lm = lm_rows()
    pred = json.loads(PRED.read_text(encoding="utf-8")) if PRED.exists() else {}
    checks = {s: validate(s, ok, lm, pred) for s in S.SEEDS}
    done = [s for s in S.SEEDS if checks[s] != ["아직 없음"]]
    valid = [s for s in done if not checks[s]]
    data = {s: per_seed(s, lm) for s in done}

    L: list = []
    w = L.append
    w("# 3주차 — 같은 update 대조 (T2b 상수 LR)\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/upd_pairs.py`")
    w("> 동결 설정 `tools/upd_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md) \"3주차 run")
    w("> 설정 동결\". 사전 점검 [`upd_preflight.md`](upd_preflight.md). 손으로 고치지 마라.\n")
    w("168.5MB 에서 T2b 는 update 를 1,063 번 받았고 C0 는 1,523~1,524 번 받았다.")
    w("T2b 를 C0 와 **같은 update 수** 까지 돌려, 원문량 잔차 중 update 부족의 몫을 잰다.\n")

    w("## run 검증\n")
    w("| seed | run | 상태 |")
    w("|---:|---|---|")
    for s in S.SEEDS:
        st = ("통과" if s in valid else
              "아직 없음" if checks[s] == ["아직 없음"] else
              "**제외** — " + " · ".join(checks[s]))
        w(f"| {s} | `{S.run_id(s)}` | {st} |")
    w("")

    if not done:
        w("**아직 돈 run 이 없다.** 판정 없음.\n")
    else:
        w("## 한국어 잔차 (seed 짝)\n")
        w("| seed | update | Cf (C0) | Bf_168 | Bf_upd | d_168 | d_upd | rho |")
        w("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in done:
            k = data[s]["ko"]
            w(f"| {s} | {S.UPDATES[s]:,} | {f6(k['cf'])} | {f6(k['bf_168'])} | {f6(k['bf_upd'])} | "
              f"{sf6(k['d_168'])} | {sf6(k['d_upd'])} | {pct(k['rho'])} |")
        w("")
        rhos = [data[s]["ko"]["rho"] for s in valid if data[s]["ko"]["rho"] is not None]
        d_upds = [data[s]["ko"]["d_upd"] for s in valid if data[s]["ko"]["d_upd"] is not None]
        for label, vals, fmt in (("rho", rhos, pct), ("d_upd", d_upds, sf6)):
            d = describe(vals)
            if d["n"]:
                ci = (f" · 탐색적 t(2) 95% CI [{fmt(d['lo'])}, {fmt(d['hi'])}]"
                      if d["lo"] is not None else "")
                # SD 는 부호가 없는 크기다. rho 는 백분율, d 는 BPB 로 적는다.
                sd = "" if d["sd"] is None else (
                    f" · 표본 SD {d['sd']:.1%}" if label == "rho" else f" · 표본 SD {d['sd']:.6f}")
                w(f"- {label} (검증 통과 {d['n']}개): 평균 {fmt(d['mean'])}{sd}{ci}")
        w("")

        w("## 판정\n")
        if len(valid) == len(S.SEEDS) and len(rhos) == len(S.SEEDS):
            m = describe(rhos)["mean"]
            v = S.classify(m)
            w(f"**{v}** — 세 seed 평균 rho {pct(m)} "
              f"(경계: <= {S.RHO_PREDICT:.0%} 예측 적중 · <= {S.RHO_HEADLINE:.0%} 부분 설명 · "
              f"그 위 헤드라인 변경).")
            cross = [s for s in valid
                     if S.classify(data[s]["ko"]["rho"]) != v]
            if cross:
                w(f"seed {', '.join(map(str, cross))} 는 다른 구간에 있다 — 함께 적는다.")
        else:
            w(f"**판정하지 않는다** — 검증을 통과한 seed {len(valid)}/{len(S.SEEDS)}. PLAN 은 세 seed")
            w("평균으로 판정한다. 위 값은 기술일 뿐이고, 효과 방향을 보고 seed 수를 줄이지 않는다.")
        w("")

        w("## 교차 확인 — 기존 T2b r5 (판정에 쓰지 않는다)\n")
        w("기존 r5 는 `7683b83` 코드, 워밍업 약 21 update, 회계 수정 이전 종료다. 새 run 의")
        w("168.5MB 지점은 같은 데이터를 같은 순서로 지나 온 같은 update 수(1,063)지만 워밍업이")
        w("약 30 update 다. 둘이 얼마나 가까운지만 본다.\n")
        w("| seed | 기존 r5 최종 | 새 run 168.5MB | 차이 |")
        w("|---:|---:|---:|---:|")
        for s in done:
            k = data[s]["ko"]
            diff = (None if k["old_final"] is None or k["bf_168"] is None
                    else k["bf_168"] - k["old_final"])
            w(f"| {s} | {f6(k['old_final'])} | {f6(k['bf_168'])} | {sf6(diff)} |")
        w("")

        w("## 영어 · 코드 (기술만)\n")
        w("rho 는 한국어에만 정의한다 (RULES 14b — 손상 축이 아니다).\n")
        w("| seed | 축 | Cf | Bf_168 | Bf_upd | d_168 | d_upd |")
        w("|---:|---|---:|---:|---:|---:|---:|")
        for s in done:
            for dom, name in DOMAINS[1:]:
                x = data[s][dom]
                w(f"| {s} | {name} | {f6(x['cf'])} | {f6(x['bf_168'])} | {f6(x['bf_upd'])} | "
                  f"{sf6(x['d_168'])} | {sf6(x['d_upd'])} |")
        w("")

    w("## 한계\n")
    w("- `d_168` 은 같은 run 의 중간 지점이다. 상수 LR 이 이어지는 중이라, 168.5MB 에서")
    w("  멈춘 run 의 종점과 학습 상태가 같지 않다")
    w("- 같은 update 는 같은 FLOPs 가 아니다 (vocab softmax 비용 등). compute-matched proxy 다")
    w("- T2b 는 같은 update 를 채우려고 원문을 약 241MB 본다 — 같은 정보량 비교가 아니다")
    w("- n=3 이다. t 구간은 정규성을 가정한 탐색적 값이다 (RULES 14c)")
    w("- seed 42 의 C0 는 회계 수정 이전 코드다 (PLAN 이 09-20 커밋별 확인으로 허용)")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"썼다 {Path(args.out).relative_to(ROOT)}  (검증 통과 {len(valid)}/{len(S.SEEDS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
