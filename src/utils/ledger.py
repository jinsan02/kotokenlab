"""실험 원장 (TSV) — 스키마 정의와 append-only 기록기.

왜 TSV 인가
    쉼표·따옴표 이스케이프 규칙이 없어서 cut, awk, sort, pandas 어디서나
    같은 결과가 나온다. 원장은 사람이 grep 으로 읽을 수 있어야 한다.

왜 append-only 인가
    이미 기록된 행은 고치지 않는다. 정정도 새 행으로 남긴다.
    "결과를 보고 기록을 손보는" 경로 자체를 없애기 위해서다 (스펙 §107).

규약
    - 헤더 1행, UTF-8(BOM 없음), LF, 탭 구분
    - 결측은 빈칸이 아니라 'NA'
    - 필드 안의 탭/개행은 리터럴 두 글자로 이스케이프
    - 모든 행에 git_commit 이 들어간다 (스펙 §59 Data Lineage)
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gitinfo import git_commit, git_dirty, repo_root

NA = "NA"

# ── 스키마 ────────────────────────────────────────────────────────────────
# 컬럼 순서가 곧 파일 포맷이다. 컬럼은 뒤에만 추가한다 (중간 삽입 금지).

LEDGER_COLUMNS: tuple[str, ...] = (
    "ts_utc", "run_id", "phase", "status",
    "tokenizer_version", "vocab_size", "tokenizer_sha256",
    "model", "init_method", "seed",
    "target_tokens", "tokens_seen", "raw_bytes_seen",
    "wall_sec", "peak_vram_mb",
    "git_commit", "git_dirty", "config_sha256", "manifest_sha256", "env_sha256",
    "argv", "note",
    # ── 아래는 검토(docs/REVIEW.md) 이후 추가된 컬럼. 뒤에만 붙인다 ──────────
    "model_revision",    # HF snapshot 해시. 저장소가 갱신돼도 계보가 끊기지 않게 (§58)
    "embedding_share",   # 임베딩이 전체 파라미터에서 차지하는 비율. 결론의 유효 범위 (REVIEW A2)
    "clock_check_sha256",  # 실행 직전 외부 시각 검증. clock_checks.tsv 로 연결된다 (§59)
)

# 외부 모델·토크나이저 레지스트리 (REVIEW D1).
# 어떤 revision 을 무슨 구성으로 받았는지 못 박는다. 검토에서 쓴 숫자들의 출처다.
MODELS_COLUMNS: tuple[str, ...] = (
    "ts_utc", "name", "repo_id", "revision", "role", "scope",
    "vocab_size", "tokenizer_len", "hidden_size", "n_layers",
    "n_heads", "n_kv_heads", "head_dim",
    "embedding_params", "total_params", "embedding_share",
    "tie_word_embeddings", "kv_bytes_per_token", "files_mb", "note",
)

# 외부 HTTPS 시각과 호스트 시계를 대조한 증거. OS 시계 자체를 바꾸지 않는다.
CLOCK_CHECKS_COLUMNS: tuple[str, ...] = (
    "ts_utc", "clock_check_sha256", "status", "server", "server_utc",
    "local_midpoint_utc", "offset_ms", "rtt_ms", "windows_source",
    "method", "git_commit", "note",
)

# 저장소가 만든 산출물 레지스트리. 외부 모델 원본은 models.tsv 가 담당한다.
ARTIFACTS_COLUMNS: tuple[str, ...] = (
    "ts_utc", "artifact_id", "run_id", "kind", "name", "path",
    "artifact_sha256", "size_bytes", "model_revision", "tokenizer_version",
    "manifest_sha256", "git_commit", "note",
)

TOKENIZER_METRICS_COLUMNS: tuple[str, ...] = (
    "ts_utc", "run_id", "tokenizer_version", "tokenizer_sha256",
    "split", "domain",
    "n_docs", "n_chars", "n_bytes", "n_eojeol", "n_tokens",
    "tok_per_char", "tok_per_byte", "bytes_per_tok", "tok_per_eojeol",
    "fertility_mean",
    "p50_len", "p90_len", "p95_len", "p99_len", "max_len",
    "git_commit", "config_sha256",
)

LM_METRICS_COLUMNS: tuple[str, ...] = (
    "ts_utc", "run_id", "checkpoint", "tokens_seen", "raw_bytes_seen",
    "split", "domain", "n_bytes",
    "total_nll", "bpb", "bpc", "token_ppl",
    "git_commit", "config_sha256",
)

CAPABILITY_COLUMNS: tuple[str, ...] = (
    "ts_utc", "run_id", "checkpoint", "benchmark", "lang",
    "n_items", "n_shot", "metric", "value", "ci_lo", "ci_hi",
    "git_commit", "config_sha256",
)

SYSTEM_BENCH_COLUMNS: tuple[str, ...] = (
    "ts_utc", "run_id", "model", "tokenizer_version", "mode",
    "raw_chars", "raw_bytes", "input_tokens", "gen_tokens",
    "n_warmup", "n_runs",
    "tokenize_ms_mean",
    "prefill_ms_mean", "prefill_ms_p95",
    "ttft_ms_mean", "ttft_ms_p95",
    "decode_tok_s_mean", "total_ms_mean", "total_ms_std",
    "kv_cache_mb_est", "peak_alloc_mb", "peak_reserved_mb",
    "git_commit", "config_sha256",
)

TRAIN_CURVE_COLUMNS: tuple[str, ...] = (
    "ts_utc", "run_id", "step", "tokens_seen", "raw_bytes_seen",
    "train_loss", "dev_loss", "dev_bpb", "lr",
    "grad_norm", "grad_norm_emb", "grad_norm_attn", "grad_norm_ffn",
    "peak_vram_mb", "tok_per_s", "raw_bytes_per_s", "elapsed_sec",
    "git_commit", "config_sha256",
)

ENV_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "ts_utc", "env_sha256",
    "python", "torch", "cuda", "cudnn",
    "transformers", "tokenizers", "datasets", "accelerate",
    "bitsandbytes", "peft", "numpy",
    "driver", "gpu_name", "vram_mb",
    "change_note",
    # 검토 A7 이후 추가. kiwipiepy 버전이 바뀌면 fertility 정의가 바뀐다.
    "kiwipiepy", "psutil", "attn_backend",
)

MANIFEST_COLUMNS: tuple[str, ...] = (
    "doc_id", "source", "domain", "date", "language",
    "sha256", "split", "char_count", "byte_count",
    # 검토(docs/DOMAIN_LABELS.md) 이후 추가. ko_en_mixed 라벨은 블라인드 감사에서
    # 정밀도·재현율 모두 0% 였다. 라벨 대신 연속값을 남겨서 사후에 임의의 구간으로
    # 문서군을 나눌 수 있게 한다 — 도메인 라벨이 아니라 문서 속성이므로 규칙
    # 정확도 문제를 우회한다.
    "latin_share", "hangul_ratio",
)

# 테이블명 -> (저장소 상대경로, 컬럼)
TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "ledger":            ("experiments/LEDGER.tsv",            LEDGER_COLUMNS),
    "tokenizer_metrics": ("experiments/tokenizer_metrics.tsv", TOKENIZER_METRICS_COLUMNS),
    "lm_metrics":        ("experiments/lm_metrics.tsv",        LM_METRICS_COLUMNS),
    "capability":        ("experiments/capability.tsv",        CAPABILITY_COLUMNS),
    "system_bench":      ("experiments/system_bench.tsv",      SYSTEM_BENCH_COLUMNS),
    "train_curve":       ("experiments/train_curve.tsv",       TRAIN_CURVE_COLUMNS),
    "env_snapshot":      ("env/ENV_SNAPSHOT.tsv",              ENV_SNAPSHOT_COLUMNS),
    "models":            ("experiments/models.tsv",            MODELS_COLUMNS),
    "clock_checks":      ("experiments/clock_checks.tsv",      CLOCK_CHECKS_COLUMNS),
    "artifacts":         ("experiments/artifacts.tsv",         ARTIFACTS_COLUMNS),
}

# 테이블별로 반드시 호출자가 채워야 하는 컬럼.
REQUIRED: dict[str, frozenset] = {
    "ledger":            frozenset({"run_id", "phase", "status"}),
    "tokenizer_metrics": frozenset({"run_id", "tokenizer_version", "split", "domain"}),
    "lm_metrics":        frozenset({"run_id", "split", "domain", "bpb"}),
    "capability":        frozenset({"run_id", "benchmark", "metric", "value"}),
    "system_bench":      frozenset({"run_id", "model", "mode"}),
    "train_curve":       frozenset({"run_id", "tokens_seen"}),
    "env_snapshot":      frozenset({"env_sha256"}),
    "models":            frozenset({"name", "repo_id", "revision"}),
    "clock_checks":      frozenset({"clock_check_sha256", "status", "server"}),
    "artifacts":         frozenset({"artifact_id", "kind", "name", "path",
                                     "artifact_sha256"}),
}

# 메트릭 테이블의 run_id 는 LEDGER.tsv 에 존재해야 한다 (참조 무결성).
METRIC_TABLES: tuple[str, ...] = (
    "tokenizer_metrics", "lm_metrics", "capability", "system_bench", "train_curve",
)


# run 의 상태. start 이후 반드시 종료 상태 행이 하나 더 붙어야 한다 (REVIEW D2).
RUN_STATUSES: tuple[str, ...] = ("start", "ok", "fail", "abort")
TERMINAL_STATUSES: tuple[str, ...] = ("ok", "fail", "abort")

PHASES: tuple[str, ...] = ("data", "tok", "surgery", "align", "cpt", "eval", "sys")


class LedgerError(ValueError):
    """스키마를 어긴 기록 시도."""


# ── 경로 ──────────────────────────────────────────────────────────────────
def columns(table: str) -> tuple[str, ...]:
    if table not in TABLES:
        raise LedgerError(f"알 수 없는 테이블: {table!r} (가능: {sorted(TABLES)})")
    return TABLES[table][1]


def table_path(table: str, root: Path | str | None = None) -> Path:
    if table not in TABLES:
        raise LedgerError(f"알 수 없는 테이블: {table!r} (가능: {sorted(TABLES)})")
    return Path(root or repo_root()) / TABLES[table][0]


def manifest_path(split: str, root: Path | str | None = None) -> Path:
    return Path(root or repo_root()) / "data" / "manifests" / f"{split}.tsv"


# ── 포맷 ──────────────────────────────────────────────────────────────────
def utcnow() -> str:
    """ISO-8601 UTC, 초 단위. 로컬 타임존은 쓰지 않는다 (기기 간 비교용)."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_value(value: Any) -> str:
    """TSV 한 필드로 안전하게 직렬화한다."""
    if value is None:
        return NA
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if value != value:  # NaN
            return NA
        text = format(value, ".10g")
    else:
        text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("\t", "\\t")
    text = text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return text if text else NA


def _row_to_line(table: str, row: Mapping[str, Any]) -> str:
    cols = columns(table)
    unknown = set(row) - set(cols)
    if unknown:
        raise LedgerError(
            f"{table}: 스키마에 없는 컬럼 {sorted(unknown)}. "
            "컬럼을 추가하려면 src/utils/ledger.py 의 스키마 끝에 붙이고 "
            "docs/LEDGER_SCHEMA.md 를 함께 고쳐라."
        )
    missing = REQUIRED.get(table, frozenset()) - set(row)
    if missing:
        raise LedgerError(f"{table}: 필수 컬럼 누락 {sorted(missing)}")

    # 사람이 적는 단계를 없앤다 — 계보 컬럼은 생략하면 자동으로 채워진다.
    filled = dict(row)
    if "ts_utc" in cols and not filled.get("ts_utc"):
        filled["ts_utc"] = utcnow()
    if "git_commit" in cols and not filled.get("git_commit"):
        filled["git_commit"] = git_commit()
    if "git_dirty" in cols and not filled.get("git_dirty"):
        filled["git_dirty"] = git_dirty()

    return "\t".join(format_value(filled.get(c, NA)) for c in cols)


# ── 기록 ──────────────────────────────────────────────────────────────────
def _append_line(path: Path, header: Sequence[str], line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 'a' 는 O_APPEND. 여러 프로세스가 한 줄씩 붙여도 줄이 섞이지 않는다.
    with io.open(path, "a", encoding="utf-8", newline="\n") as fh:
        if fh.tell() == 0:
            fh.write("\t".join(header) + "\n")
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def append_row(
    table: str,
    row: Mapping[str, Any],
    root: Path | str | None = None,
) -> Path:
    """원장에 한 행을 덧붙인다. 헤더가 없으면 먼저 만든다."""
    line = _row_to_line(table, row)
    path = table_path(table, root)
    _append_line(path, columns(table), line)
    return path


def append_rows(
    table: str,
    rows: Iterable[Mapping[str, Any]],
    root: Path | str | None = None,
) -> Path:
    lines = [_row_to_line(table, r) for r in rows]  # 전부 검증한 뒤에 쓴다
    path = table_path(table, root)
    for line in lines:
        _append_line(path, columns(table), line)
    return path


def append_manifest_rows(
    split: str,
    rows: Iterable[Mapping[str, Any]],
    root: Path | str | None = None,
) -> Path:
    """문서 단위 manifest (스펙 §7). split 당 파일 하나."""
    path = manifest_path(split, root)
    lines = []
    for row in rows:
        unknown = set(row) - set(MANIFEST_COLUMNS)
        if unknown:
            raise LedgerError(f"manifest: 스키마에 없는 컬럼 {sorted(unknown)}")
        lines.append("\t".join(format_value(row.get(c, NA)) for c in MANIFEST_COLUMNS))
    for line in lines:
        _append_line(path, MANIFEST_COLUMNS, line)
    return path


# ── 읽기 ──────────────────────────────────────────────────────────────────
def read_rows(table: str, root: Path | str | None = None) -> list:
    """원장을 dict 리스트로 읽는다. 파일이 없으면 빈 리스트."""
    path = table_path(table, root)
    if not path.exists():
        return []
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read().replace("\r\n", "\n")
    lines = [ln for ln in raw.split("\n") if ln]
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def known_run_ids(root: Path | str | None = None) -> set:
    """LEDGER.tsv 에 등록된 run_id 집합. 참조 무결성 검사에 쓴다."""
    return {r.get("run_id", "") for r in read_rows("ledger", root)} - {"", NA}
