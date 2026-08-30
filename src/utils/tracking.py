"""run 하나의 수명을 관리한다 (스펙 §58 Experiment Tracking).

RunContext 로 감싸면 아래가 자동으로 일어난다.

    진입   최근 시간 검증 -> 환경 등록 확인 -> config_sha256 계산 -> run_id 생성
           -> experiments/runs/<run_id>/ 생성 (config.json, env.json)
           -> LEDGER.tsv 에 status=start 행
    본문   run.log(...) 로 메트릭 TSV 에 append (run_id/config_sha256 자동 주입)
    이탈   LEDGER.tsv 에 status=ok 또는 status=fail 행 (append-only, 수정 아님)

status 행을 고치지 않고 새로 붙이는 이유는, 죽은 run 의 흔적을 지우지 않기
위해서다. 실패한 실험도 실험이다.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from . import clock as clock_mod
from . import env as env_mod
from . import ledger
from .hashing import sha256_obj
from .seed import set_seed

PHASES: tuple[str, ...] = ("data", "tok", "surgery", "align", "cpt", "eval", "sys")


class _Tee:
    """화면과 log.txt 에 동시에 쓴다.

    백그라운드로 돌린 파이프라인의 stdout 이 통째로 사라진 적이 있다. 원장에는
    종료 행만 남아서, 필터 탈락 사유 분포처럼 화면에만 찍히는 진단을 잃었다.
    run 디렉터리에 남겨두면 세션이 끝나도 읽을 수 있다.
    """

    def __init__(self, stream: Any, fh: Any) -> None:
        self._stream = stream
        self._fh = fh

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._fh.write(s)
        self._fh.flush()      # 죽어도 남아야 하므로 매번 flush 한다
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def make_run_id(phase: str, *parts: Any, seed: int | None = None) -> str:
    """스펙 §61 의 명명 규칙. 이름만 보고 어떤 실험인지 알 수 있어야 한다.

    >>> make_run_id("cpt", "kosub", "mean", "50m", seed=42)
    'cpt_kosub_mean_50m_seed42'
    >>> make_run_id("tok", "qwen", "original", "v1")
    'tok_qwen_original_v1'
    """
    if phase not in PHASES:
        raise ValueError(f"알 수 없는 phase: {phase!r} (가능: {PHASES})")
    chunks = [phase] + [str(p) for p in parts if p not in (None, "")]
    if seed is not None:
        chunks.append(f"seed{seed}")
    return "_".join(chunks)


def _peak_vram_mb() -> Any:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return None


def _reset_vram_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


class RunContext:
    """실험 1건. `with` 로 쓴다."""

    def __init__(
        self,
        run_id: str,
        phase: str,
        config: Mapping[str, Any] | None = None,
        *,
        seed: int | None = None,
        root: Path | str | None = None,
        skip_env_check: bool = False,
        set_seeds: bool = True,
        **ledger_fields: Any,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"알 수 없는 phase: {phase!r} (가능: {PHASES})")
        self.run_id = run_id
        self.phase = phase
        self.config = dict(config or {})
        self.seed = seed
        self.root = Path(root) if root else None
        self.skip_env_check = skip_env_check
        self.set_seeds = set_seeds
        self.config_sha256 = sha256_obj(self.config) if self.config else ledger.NA
        self.env_sha256 = ledger.NA
        self.clock_check_sha256 = ledger.NA
        self.extra = dict(ledger_fields)

        # 본문에서 갱신하면 종료 행에 반영된다.
        self.tokens_seen: Any = None
        self.raw_bytes_seen: Any = None
        self.note: str = ""

        self._t0 = 0.0
        self._dir: Path | None = None
        self._log_fh: Any = None
        self._saved_streams: Any = None

    # ── 경로 ──────────────────────────────────────────────────────────
    @property
    def dir(self) -> Path:
        """experiments/runs/<run_id>/ — 이 run 의 산출물이 모이는 곳."""
        if self._dir is None:
            base = Path(self.root or ledger.repo_root())
            self._dir = base / "experiments" / "runs" / self.run_id
        return self._dir

    # ── 수명 ──────────────────────────────────────────────────────────
    def __enter__(self) -> "RunContext":
        self.clock_check_sha256 = clock_mod.require_recent_check(self.root)
        if self.skip_env_check:
            self.env_sha256 = env_mod.env_sha256()
        else:
            self.env_sha256 = env_mod.require_registered(self.root)

        if self.set_seeds and self.seed is not None:
            set_seed(self.seed)

        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(
            json.dumps(self.config, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8", newline="\n",
        )
        (self.dir / "env.json").write_text(
            json.dumps(env_mod.collect(), sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8", newline="\n",
        )

        self._log_fh = (self.dir / "log.txt").open("a", encoding="utf-8", newline=chr(10))
        self._saved_streams = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._log_fh)
        sys.stderr = _Tee(sys.stderr, self._log_fh)

        _reset_vram_stats()
        self._t0 = time.time()
        self._ledger_row("start")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = "ok" if exc_type is None else "fail"
        if exc_type is not None and not self.note:
            self.note = f"{exc_type.__name__}: {exc}"
        # 트레이스백은 __exit__ 이후에 인터프리터가 찍으므로 tee 로는 못 잡는다.
        if exc_type is not None:
            print("".join(traceback.format_exception(exc_type, exc, tb)), file=sys.stderr)
        self._ledger_row(status)
        self._restore_streams()
        return False  # 예외를 삼키지 않는다

    def _restore_streams(self) -> None:
        saved = getattr(self, "_saved_streams", None)
        if saved is not None:
            sys.stdout, sys.stderr = saved
            self._saved_streams = None
        fh = getattr(self, "_log_fh", None)
        if fh is not None:
            fh.close()
            self._log_fh = None

    def _ledger_row(self, status: str) -> None:
        row: dict = {
            "run_id": self.run_id,
            "phase": self.phase,
            "status": status,
            "seed": self.seed,
            "config_sha256": self.config_sha256,
            "env_sha256": self.env_sha256,
            "clock_check_sha256": self.clock_check_sha256,
            "argv": " ".join(sys.argv[1:]),
        }
        row.update(self.extra)
        if status != "start":
            row["wall_sec"] = round(time.time() - self._t0, 2)
            row["tokens_seen"] = self.tokens_seen
            row["raw_bytes_seen"] = self.raw_bytes_seen
            row["peak_vram_mb"] = _peak_vram_mb()
            row["note"] = self.note or None
        ledger.append_row("ledger", {k: v for k, v in row.items() if v is not None}, self.root)

    # ── 기록 ──────────────────────────────────────────────────────────
    def log(self, table: str, **row: Any) -> Path:
        """메트릭 TSV 에 한 행 추가. run_id 와 config_sha256 은 자동으로 붙는다."""
        if table not in ledger.METRIC_TABLES:
            raise ledger.LedgerError(
                f"{table!r} 은 메트릭 테이블이 아니다 (가능: {ledger.METRIC_TABLES})"
            )
        payload = dict(row)
        payload.setdefault("run_id", self.run_id)
        if "config_sha256" in ledger.columns(table):
            payload.setdefault("config_sha256", self.config_sha256)
        return ledger.append_row(table, payload, self.root)

    def log_rows(self, table: str, rows) -> Path:
        path = None
        for row in rows:
            path = self.log(table, **row)
        return path if path is not None else ledger.table_path(table, self.root)
