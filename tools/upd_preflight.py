"""3주차 같은-update 대조를 돌리기 **전에** 빈틈을 찾는다 (GPU 0시간).

    .conda/python.exe tools/upd_preflight.py           # 전부 (토큰화 포함, 수 분)
    .conda/python.exe tools/upd_preflight.py --quick   # 데이터 흐름 재현 생략

동결 설정은 `tools/upd_spec.py` 하나에 있다 (PLAN.md "3주차 run 설정 동결").

## 왜

5.3시간짜리 run 셋을 돌리고 나서 "풀이 모자랐다", "--eval-at 을 빠뜨렸다",
"그 사이 학습 코드가 바뀌었다" 를 알면 GPU 시간이 통째로 버려진다. 이 프로젝트는
그런 일을 실제로 겪었다 — 예산보다 작은 문서 풀(103.8MB 로 168.5MB 를 채우려
했다), 기본값이 달라 거짓 실패한 게이트, 기록이 끝난 뒤에야 거부된 run_id.

## 무엇을 보나

정적 점검 — 원장·git·환경·디스크·GPU

    측정 코드가 깨끗한가 · C0 이후 측정 경로 커밋이 동결이 다룬 것뿐인가 ·
    run_id 가 비어 있는가 · 짝 C0 의 update 수가 PLAN 표와 같은가 ·
    짝 C0 와 공유 플래그가 같은가 · 모델 해시가 등록값과 같은가 ·
    환경 등록 · 시각 검증 유효시간 · 디스크 · GPU 여유

데이터 흐름 재현 — **학습 코드의 함수를 그대로 쓴다**

    `src.training.cpt` 의 `load_pool` · `order_pool` 과 같은 패킹 규칙으로
    토큰 스트림의 누적 원문 바이트를 CPU 에서 계산한다. 먼저 **이미 돈 run
    6개의 원장 기록(관찰 토큰 -> 원문 바이트)을 바이트 단위까지 재현하는지**
    검증하고, 재현될 때만 새 run 의 예측을 믿는다.

    예측: 풀이 예산을 채우는가(여유) · 종료 시 원문 바이트 · d_168 을 재는
    update 번호와 그 원문 바이트. 예측은 `reports/tables/upd_preflight.json` 에
    남기고 `tools/upd_pairs.py` 가 실제 run 과 대조한다 — 어긋나면 데이터 순서가
    계획과 다르게 돈 것이라 판정하지 않는다.

## 판정하지 않는다

이 도구는 "돌려도 되는가" 만 본다. 결과 해석은 `tools/upd_pairs.py` 가 한다.
상태는 셋이다 — 통과 / 실패 / 경고(돌릴 수는 있지만 사람이 봐야 한다).
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 점검이 네트워크를 타지 않는다

from tools import upd_spec as S  # noqa: E402

LEDGER = ROOT / "experiments" / "LEDGER.tsv"
ARTIFACTS = ROOT / "experiments" / "artifacts.tsv"
RUNS = ROOT / "experiments" / "runs"
POOL = ROOT / "data" / "interim" / "docs" / "train.jsonl"
OUT_MD = ROOT / "reports" / "tables" / "upd_preflight.md"
OUT_JSON = ROOT / "reports" / "tables" / "upd_preflight.json"

PASS, FAIL, WARN = "통과", "실패", "경고"
MIN_FREE_DISK = 10 * 1024 ** 3
MIN_FREE_VRAM_MB = 12_000          # 기존 T2b run 의 peak 11,382MB + 여유


# ── 원장 ─────────────────────────────────────────────────────────────────

def ledger_rows() -> list:
    with LEDGER.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE))


def ok_row(rows: list, run_id: str):
    hits = [r for r in rows if r["run_id"] == run_id and r["status"] == "ok"]
    return hits[-1] if hits else None


def parse_argv(s: str) -> dict:
    """`--flag value` 와 값 없는 `--flag` 를 dict 로. 원장 argv 컬럼용."""
    toks = s.split()
    out: dict = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[t] = toks[i + 1]
                i += 2
                continue
            out[t] = True
        i += 1
    return out


def ctrl_updates(row: dict) -> tuple:
    """짝 C0 의 update 수와 그 출처.

    회계 수정(2da591e) 이후 run 은 `updates` 컬럼이 있다. 그 전 run 은 NA 라
    관찰 토큰을 update 크기로 **내림** 한다 — 마지막 미완성 update 는 step 되지
    않았기 때문이다. 출처를 함께 돌려 표에 적는다.
    """
    u = row.get("updates", "NA")
    if u not in ("", "NA"):
        return int(u), "원장 updates"
    return int(row["tokens_seen"]) // S.TOKENS_PER_UPDATE, "관찰 토큰에서 내림 (회계 수정 이전)"


# ── git ──────────────────────────────────────────────────────────────────

def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                         text=True, encoding="utf-8", timeout=60)
    return out.stdout.strip() if out.returncode == 0 else ""


def code_commits_since(sha: str) -> list:
    """sha 이후 HEAD 까지 측정 경로(src·configs)를 건드린 커밋 (짧은 해시)."""
    raw = git("log", "--format=%h", f"{sha}..HEAD", "--", "src", "configs")
    return [x for x in raw.splitlines() if x]


def running_training_pids():
    """이 저장소의 `src.training.cpt` 를 돌리는 python 프로세스 PID. 못 읽으면 None."""
    ps = shutil.which("powershell") or shutil.which("powershell.exe")
    if not ps:
        return None
    q = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*src.training.cpt*' } | "
         "ForEach-Object { $_.ProcessId }")
    try:
        out = subprocess.run([ps, "-NoProfile", "-Command", q], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


# ── 데이터 흐름 재현 ─────────────────────────────────────────────────────

def byte_table(tokenizer):
    """vocab id -> 원문 바이트 수. cpt.py 의 byte_len_fn 과 같은 규칙이다."""
    import numpy as np
    from tokenizers.pre_tokenizers import ByteLevel

    from src.evaluation.bpb import token_byte_length
    alphabet = set(ByteLevel.alphabet())
    n = max(tokenizer.get_vocab().values()) + 1
    table = np.zeros(n, dtype=np.int64)
    for tok, i in tokenizer.get_vocab().items():
        table[i] = token_byte_length(tok, alphabet)
    eos = tokenizer.eos_token_id or 0
    table[eos] = 0          # cpt.py: EOS 는 코퍼스 원문이 아니다
    return table, eos


def doc_arrays(tokenizer, docs: list, table, batch: int = 512) -> list:
    """문서마다 토큰별 바이트 배열. 토큰이 없는 문서는 None (pack 이 건너뛴다)."""
    import numpy as np
    out: list = []
    for k in range(0, len(docs), batch):
        enc = tokenizer(docs[k:k + batch], add_special_tokens=False)["input_ids"]
        for ids in enc:
            out.append(table[np.asarray(ids, dtype=np.int64)] if ids else None)
    return out


class Stream:
    """`pack()` 이 내놓는 토큰 스트림의 누적 원문 바이트.

    pack 은 문서를 순서대로 토큰화해 `ids + [eos]` 를 이어 붙이고 seq_len 조각만
    내보낸다 (남은 꼬리는 버린다). EOS 는 0 바이트. 그러면 앞 N 토큰의 원문
    바이트는 문서 단위 누적합과 경계 문서 안의 부분합으로 정확히 나온다.
    """

    def __init__(self, arrays: list, order: list, seq_len: int) -> None:
        import numpy as np
        self.docs = [arrays[i] for i in order if arrays[i] is not None]
        lens = np.array([len(a) + 1 for a in self.docs], dtype=np.int64)
        sums = np.array([int(a.sum()) for a in self.docs], dtype=np.int64)
        self.cum_tok = np.cumsum(lens)
        self.cum_bytes = np.cumsum(sums)
        self.seq_len = seq_len

    @property
    def full_tokens(self) -> int:
        """pack 이 실제로 내보내는 토큰 수 (꽉 찬 조각만)."""
        total = int(self.cum_tok[-1]) if len(self.cum_tok) else 0
        return total // self.seq_len * self.seq_len

    def bytes_at(self, n: int) -> int:
        import numpy as np
        if n <= 0:
            return 0
        if n > self.full_tokens:
            raise ValueError(f"풀이 {n:,} 토큰을 채우지 못한다 (최대 {self.full_tokens:,})")
        j = int(np.searchsorted(self.cum_tok, n, side="left"))
        prev_tok = int(self.cum_tok[j - 1]) if j else 0
        prev_bytes = int(self.cum_bytes[j - 1]) if j else 0
        k = n - prev_tok
        arr = self.docs[j]
        return prev_bytes + int(arr[:min(k, len(arr))].sum())

    def first_update_at(self, target_bytes: int, tpu: int, max_updates: int):
        """원문 바이트가 target 이상이 되는 첫 update 번호. 없으면 None."""
        lo, hi = 1, max_updates
        if self.bytes_at(hi * tpu) < target_bytes:
            return None
        while lo < hi:
            mid = (lo + hi) // 2
            if self.bytes_at(mid * tpu) >= target_bytes:
                hi = mid
            else:
                lo = mid + 1
        return lo


def order_indices(pool_docs: int, extend: int, seed: int) -> list:
    """학습 코드의 order_pool 을 그대로 쓴다. 섞기는 내용과 무관하게 길이와
    seed 만으로 정해지므로 인덱스를 섞으면 문서를 섞은 것과 같은 순서다."""
    from src.training.cpt import order_pool
    return order_pool(list(range(pool_docs)), seed,
                      list(range(pool_docs, pool_docs + extend)))


def load_tokenizer(model: str, revision):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, revision=revision)


# ── 점검 ─────────────────────────────────────────────────────────────────

def static_checks(rows: list) -> list:
    res: list = []
    add = lambda name, st, detail: res.append((name, st, detail))  # noqa: E731

    dirty = git("status", "--porcelain", "--", "src", "configs")
    add("측정 코드가 깨끗한가", PASS if not dirty else FAIL,
        "src·configs 에 커밋 안 된 변경 없음" if not dirty else dirty.replace("\n", " · "))

    for s in S.SEEDS:
        c = ok_row(rows, f"{S.CTRL}{s}")
        if c is None:
            add(f"짝 C0 seed{s}", FAIL, "원장에 ok 행이 없다")
            continue
        u, src = ctrl_updates(c)
        want = S.UPDATES[s]
        add(f"짝 C0 seed{s} update 수", PASS if u == want else FAIL,
            f"{u:,} ({src}) vs PLAN {want:,}")

        if c.get("updates", "NA") not in ("", "NA"):     # 새 코드 C0 만 계보로 묶는다
            extra = [h for h in code_commits_since(c["git_commit"])
                     if not any(h.startswith(a) or a.startswith(h)
                                for a in S.ALLOWED_CODE_COMMITS_SINCE_CTRL)]
            add(f"C0 seed{s} 이후 측정 경로 커밋", PASS if not extra else FAIL,
                "동결이 다룬 것뿐 (" + ", ".join(S.ALLOWED_CODE_COMMITS_SINCE_CTRL) + ")"
                if not extra else "동결 밖: " + ", ".join(extra))
        else:
            n = len(code_commits_since(c["git_commit"]))
            add(f"C0 seed{s} 이후 측정 경로 커밋", PASS,
                f"{n}개 — 회계 수정 이전 run. PLAN 이 09-20 커밋별 확인으로 허용")

        a = parse_argv(c.get("argv", ""))
        want_flags = {"--pool-docs": str(S.POOL_DOCS), "--eval-bytes": str(S.EVAL_BYTES),
                      "--eval-budget": str(S.EVAL_BUDGET), "--lr-schedule": "constant"}
        bad = {k: (a.get(k), v) for k, v in want_flags.items() if a.get(k) != v}
        add(f"짝 C0 seed{s} 공유 플래그", PASS if not bad else FAIL,
            "pool-docs · eval-bytes · eval-budget · lr-schedule 같음" if not bad
            else "; ".join(f"{k} C0={x} 계획={y}" for k, (x, y) in bad.items()))

        cfg_p = RUNS / f"{S.CTRL}{s}" / "config.json"
        if cfg_p.exists():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            got = (cfg.get("seq_len"), cfg.get("micro_bs"), cfg.get("accum"))
            add(f"짝 C0 seed{s} update 크기", PASS if got == (S.SEQ_LEN, S.MICRO_BS, S.ACCUM) else FAIL,
                f"seq_len·micro_bs·accum = {got} -> {S.TOKENS_PER_UPDATE:,} 토큰/update")
            # 옛 config 에 없는 치명 필드는 compare_runs 가 "비교 불가" 로 막는다.
            # 2026-10-02 seed 42 짝의 warmup_bytes 를 실행 뒤에야 알았다 — 미리 본다.
            # 기준은 현재 코드의 C0 config 가 가진 키다 (max_bytes·split 같은 평가 전용
            # 필드는 학습 config 에 원래 없다).
            from tools.compare_runs import CRITICAL
            ref_p = RUNS / f"{S.CTRL}{S.SEEDS[-1]}" / "config.json"
            ref = json.loads(ref_p.read_text(encoding="utf-8")) if ref_p.exists() else cfg
            missing = sorted(k for k in CRITICAL & set(ref) if k not in cfg)
            add(f"짝 C0 seed{s} config 기록 없는 치명 필드", PASS if not missing else WARN,
                "없음" if not missing else
                ", ".join(missing) + " — '모른다' 다. compare_runs 에서 허용하려면 근거를 PLAN 에 적는다")

    # 끝난 seed 는 건너뛰고, 남은 seed 의 run_id 만 비어 있어야 한다. 시작 행만 있고
    # ok 가 없으면(중단) 같은 id 로 다시 돌릴 수 없으니 사람이 봐야 한다.
    for s in S.SEEDS:
        rid = S.run_id(s)
        sts = [r["status"] for r in rows if r["run_id"] == rid]
        if "ok" in sts:
            add(f"run_id seed{s}", PASS, f"{rid} — 이미 돌았다 (ok)")
        elif sts:
            add(f"run_id seed{s}", FAIL, f"{rid} — 끝나지 않은 행 {sts} — 원인을 먼저 본다")
        else:
            add(f"run_id seed{s}", PASS, f"{rid} — 비어 있음")

    for s in S.SEEDS:
        o = ok_row(rows, f"{S.OLD_TREAT}{s}")
        if o is not None:
            m = parse_argv(o.get("argv", "")).get("--model")
            add(f"기존 T2b seed{s} 와 같은 모델", PASS if m == S.MODEL else FAIL, f"{m}")

    reg = None
    with ARTIFACTS.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["kind"] == "checkpoint" and r["name"] == S.MODEL_ARTIFACT_NAME:
                reg = r["artifact_sha256"]
    model_dir = ROOT / S.MODEL
    if not model_dir.exists():
        add("모델 체크포인트", FAIL, f"{S.MODEL} 가 없다")
    elif reg is None:
        add("모델 체크포인트", FAIL, "artifacts.tsv 에 등록이 없다")
    else:
        from src.utils.hashing import sha256_dir, short
        got = sha256_dir(model_dir)
        add("모델 체크포인트 해시", PASS if got == reg else FAIL,
            f"{short(got)} vs 등록 {short(reg)}")

    from src.utils import env as env_mod
    try:
        env_ok = env_mod.is_registered(ROOT)
    except Exception as e:  # noqa: BLE001 — 점검 도구는 멈추지 않고 보고한다
        env_ok, detail = None, f"검사 실패: {e}"
    else:
        detail = "env_sha256 등록됨" if env_ok else "미등록 — RunContext 가 시작을 막는다"
    add("환경 등록", PASS if env_ok else FAIL, detail)

    from src.utils.clock import latest_valid_check
    chk = latest_valid_check(ROOT)
    if chk is None:
        add("시각 검증", FAIL, "24시간 안의 성공 기록 없음 — tools/check_clock.py --record")
    else:
        t = calendar.timegm(time.strptime(chk["ts_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        left = 24 - (time.time() - t) / 3600
        add("시각 검증", PASS if left >= S.PLANNED_HOURS else WARN,
            f"남은 유효시간 {left:.1f}h / 계획 {S.PLANNED_HOURS:.1f}h"
            + ("" if left >= S.PLANNED_HOURS else " — 세 번째 run 전에 다시 기록해야 할 수 있다"))

    free = shutil.disk_usage(ROOT).free
    add("디스크 여유", PASS if free >= MIN_FREE_DISK else FAIL,
        f"{free / 1024 ** 3:.0f}GB (체크포인트 3 x 약 1GB)")

    # 2026-10-02: seed 2026 이 같은 run_id 로 두 번 떴다. 중복은 원장만 어지럽힌 게
    # 아니라 남은 run 의 메모리 일부를 시스템 RAM 으로 밀어내 약 2배 느리게 만든
    # 것으로 보인다. GPU 여유 메모리만 보면 시작 직후의 중복은 못 잡는다 — 프로세스를 본다.
    running = running_training_pids()
    if running is None:
        add("돌고 있는 학습 프로세스", WARN, "프로세스 목록을 못 읽었다 — 사람이 확인")
    else:
        add("돌고 있는 학습 프로세스", PASS if not running else FAIL,
            "없음" if not running else
            "이미 돌고 있다: PID " + ", ".join(map(str, running)) + " — 끝나기 전에 새로 띄우지 않는다")

    smi = shutil.which("nvidia-smi")
    if not smi:
        add("GPU 여유", WARN, "nvidia-smi 없음 — 사람이 확인")
    else:
        q = subprocess.run([smi, "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
        try:
            used, total = (int(x) for x in q.splitlines()[0].split(","))
            free_mb = total - used
            add("GPU 여유", PASS if free_mb >= MIN_FREE_VRAM_MB else WARN,
                f"{free_mb:,}MB 비어 있음 (필요 약 {MIN_FREE_VRAM_MB:,}MB). "
                "다른 학습이 돌고 있으면 시작하지 않는다")
        except (ValueError, IndexError):
            add("GPU 여유", WARN, f"nvidia-smi 출력 해석 실패: {q!r}")
    return res


def replay_and_predict(rows: list) -> tuple:
    """기존 run 재현으로 시뮬레이터를 검증하고, 새 run 을 예측한다."""
    from src.training.cpt import load_pool

    res: list = []
    pred: dict = {}
    pool = load_pool(POOL, S.POOL_DOCS + S.POOL_EXTEND_DOCS, 0)
    if len(pool) < S.POOL_DOCS + S.POOL_EXTEND_DOCS:
        res.append(("문서 풀 크기", FAIL, f"{len(pool):,}개뿐"))
        return res, pred

    # 재현 대상: 같은 풀(앞 50,000)을 쓴 기존 run 들
    replays = [(f"{S.OLD_TREAT}{s}", s) for s in S.SEEDS] + [(f"{S.CTRL}{s}", s) for s in S.SEEDS]
    tok_cache: dict = {}

    def arrays_for(model: str, revision, n_docs: int):
        key = (model, revision)
        if model == S.MODEL:     # 예측에 70,000문서가 필요하니 한 번에 토큰화한다
            n_docs = max(n_docs, S.POOL_DOCS + S.POOL_EXTEND_DOCS)
        if key not in tok_cache or len(tok_cache[key][0]) < n_docs:
            tk = load_tokenizer(model, revision)
            table, _ = byte_table(tk)
            print(f"  토큰화  {model}  {n_docs:,}문서 ...", flush=True)
            tok_cache[key] = (doc_arrays(tk, pool[:n_docs], table), tk)
        return tok_cache[key][0]

    all_ok = True
    for rid, s in replays:
        r = ok_row(rows, rid)
        if r is None:
            res.append((f"재현 {rid}", FAIL, "원장 ok 행 없음"))
            all_ok = False
            continue
        a = parse_argv(r["argv"])
        pd = int(a.get("--pool-docs", 0))
        ext = int(a.get("--pool-extend-docs", 0) or 0)
        arrs = arrays_for(a["--model"], a.get("--revision"), pd + ext)
        st = Stream(arrs, order_indices(pd, ext, s), S.SEQ_LEN)
        n = int(r["tokens_seen"])
        got = st.bytes_at(n)
        want = int(r["raw_bytes_seen"])
        ok = got == want
        all_ok &= ok
        res.append((f"재현 {rid}", PASS if ok else FAIL,
                    f"토큰 {n:,} -> 바이트 {got:,} vs 원장 {want:,}"))

    if not all_ok:
        res.append(("새 run 예측", FAIL, "재현이 어긋나 예측을 믿을 수 없다 — 원인부터 찾는다"))
        return res, pred

    arrs = arrays_for(S.MODEL, None, S.POOL_DOCS + S.POOL_EXTEND_DOCS)
    for s in S.SEEDS:
        st = Stream(arrs, order_indices(S.POOL_DOCS, S.POOL_EXTEND_DOCS, s), S.SEQ_LEN)
        n = S.budget_tokens(s)
        if st.full_tokens < n:
            res.append((f"풀 여유 seed{s}", FAIL,
                        f"{st.full_tokens:,} 토큰 < 예산 {n:,} — run 이 RuntimeError 로 멈춘다"))
            continue
        final_bytes = st.bytes_at(n)
        k = st.first_update_at(S.EVAL_AT, S.TOKENS_PER_UPDATE, S.UPDATES[s])
        k_bytes = st.bytes_at(k * S.TOKENS_PER_UPDATE) if k else None
        margin = st.full_tokens / n - 1
        res.append((f"풀 여유 seed{s}", PASS,
                    f"최대 {st.full_tokens:,} 토큰 / 예산 {n:,} (여유 {margin:.1%})"))
        res.append((f"d_168 지점 seed{s}", PASS if k else FAIL,
                    f"update {k:,} · 원문 {k_bytes:,} 바이트" if k else "168.5MB 에 닿지 않는다"))
        old = ok_row(rows, f"{S.OLD_TREAT}{s}")
        pred[str(s)] = {
            "run_id": S.run_id(s), "updates": S.UPDATES[s], "budget_tokens": n,
            "raw_bytes_final": final_bytes, "d168_update": k, "d168_raw_bytes": k_bytes,
            "pool_full_tokens": st.full_tokens,
            "old_r5_tokens": int(old["tokens_seen"]) if old else None,
            "old_r5_raw_bytes": int(old["raw_bytes_seen"]) if old else None,
        }
    return res, pred


def write(res: list, pred: dict, quick: bool) -> None:
    from src.utils.hashing import sha256_file, short
    L: list = []
    w = L.append
    w("# 3주차 사전 점검 — 같은 update 대조\n")
    w("> **이 표는 코드가 적는다.** `.conda/python.exe tools/upd_preflight.py`")
    w("> 동결 설정은 `tools/upd_spec.py` · [`docs/PLAN.md`](../../docs/PLAN.md)")
    w("> \"3주차 run 설정 동결\". GPU 0시간. 판정하지 않는다 — 돌려도 되는지만 본다.\n")
    w(f"HEAD `{git('rev-parse', '--short', 'HEAD')}` · 풀 `data/interim/docs/train.jsonl` "
      f"sha256 `{short(sha256_file(POOL))}`" if POOL.exists() else "")
    w("")
    w("| 점검 | 상태 | 내용 |")
    w("|---|---|---|")
    for name, st, detail in res:
        w(f"| {name} | **{st}** | {detail} |")
    w("")
    if quick:
        w("`--quick` — 데이터 흐름 재현과 예측을 건너뛰었다.\n")
    if pred:
        w("## 예측 (`upd_preflight.json`)\n")
        w("| seed | update | 예산 토큰 | 종료 원문 바이트 | d_168 update | d_168 원문 바이트 | 기존 r5 종료 |")
        w("|---:|---:|---:|---:|---:|---:|---|")
        for s, p in pred.items():
            old = (f"{p['old_r5_tokens'] // S.TOKENS_PER_UPDATE:,} update · "
                   f"{p['old_r5_raw_bytes']:,}" if p["old_r5_tokens"] else "NA")
            w(f"| {s} | {p['updates']:,} | {p['budget_tokens']:,} | {p['raw_bytes_final']:,} | "
              f"{p['d168_update']:,} | {p['d168_raw_bytes']:,} | {old} |")
        w("")
        w("반복 바이트는 0 이다 — 풀(50,000 + 20,000문서)이 예산을 반복 없이 채우고,")
        w("`cpt.py` 에는 풀을 되감는 경로가 없다 (모자라면 멈춘다).\n")
        w("## 실행 명령 (seed 42 부터, 한 번에 하나)\n")
        w("```")
        for s in S.SEEDS:
            w(S.command(s))
        w("```\n")
    w("## 한계\n")
    w("- 재현은 원문 바이트 회계와 데이터 순서를 검증한다. GPU 연산·난수는 보지 않는다")
    w("- 기존 r5 T2b 는 회계 수정 이전 코드라 종료가 update 경계가 아닐 수 있다 —")
    w("  표의 \"기존 r5 종료\" update 수는 관찰 토큰을 내린 값이다")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    if pred:
        OUT_JSON.write_text(json.dumps(pred, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="3주차 사전 점검")
    ap.add_argument("--quick", action="store_true", help="데이터 흐름 재현 생략")
    args = ap.parse_args(argv)

    rows = ledger_rows()
    res = static_checks(rows)
    pred: dict = {}
    if not args.quick:
        more, pred = replay_and_predict(rows)
        res += more
    for name, st, detail in res:
        print(f"  [{st}] {name}  — {detail}")
    write(res, pred, args.quick)
    print(f"썼다 {OUT_MD.relative_to(ROOT)}")
    fails = [r for r in res if r[1] == FAIL]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
