"""원장이 잘못된 기록을 실제로 거부하는지 확인한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402
from src.utils.hashing import is_sha256, sha256_obj, sha256_text  # noqa: E402
from src.utils.tracking import make_run_id  # noqa: E402
from tools.validate_ledger import validate  # noqa: E402


# ── 스키마 ────────────────────────────────────────────────────────────────
def test_unknown_column_rejected(tmp_path):
    with pytest.raises(ledger.LedgerError, match="스키마에 없는 컬럼"):
        ledger.append_row(
            "ledger",
            {"run_id": "cpt_x_seed42", "phase": "cpt", "status": "ok", "오타컬럼": 1},
            root=tmp_path,
        )


def test_missing_required_column_rejected(tmp_path):
    with pytest.raises(ledger.LedgerError, match="필수 컬럼 누락"):
        ledger.append_row("ledger", {"run_id": "cpt_x_seed42"}, root=tmp_path)


def test_unknown_table_rejected(tmp_path):
    with pytest.raises(ledger.LedgerError, match="알 수 없는 테이블"):
        ledger.append_row("없는테이블", {"run_id": "x"}, root=tmp_path)


def test_log_rejects_non_metric_table(tmp_path):
    from src.utils.tracking import RunContext

    run = RunContext("cpt_x_seed42", phase="cpt", root=tmp_path)
    with pytest.raises(ledger.LedgerError, match="메트릭 테이블이 아니다"):
        run.log("ledger", status="ok")


# ── 포맷 ──────────────────────────────────────────────────────────────────
def test_header_written_once_and_rows_appended(tmp_path):
    for i in range(3):
        ledger.append_row(
            "ledger",
            {"run_id": f"cpt_x_seed{i}", "phase": "cpt", "status": "ok"},
            root=tmp_path,
        )
    path = ledger.table_path("ledger", tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.count("\n") == 4  # 헤더 1 + 행 3
    assert text.split("\n")[0].split("\t") == list(ledger.LEDGER_COLUMNS)
    assert "\r" not in text  # LF only


def test_tab_and_newline_are_escaped(tmp_path):
    ledger.append_row(
        "ledger",
        {"run_id": "cpt_x_seed42", "phase": "cpt", "status": "ok",
         "note": "탭\t개행\n둘 다"},
        root=tmp_path,
    )
    rows = ledger.read_rows("ledger", tmp_path)
    assert len(rows) == 1
    assert rows[0]["note"] == "탭\\t개행\\n둘 다"


def test_missing_values_become_na(tmp_path):
    ledger.append_row(
        "ledger",
        {"run_id": "cpt_x_seed42", "phase": "cpt", "status": "start"},
        root=tmp_path,
    )
    row = ledger.read_rows("ledger", tmp_path)[0]
    assert row["seed"] == "NA"
    assert row["note"] == "NA"
    assert "" not in row.values()


def test_ts_utc_is_autofilled(tmp_path):
    ledger.append_row(
        "ledger", {"run_id": "cpt_x_seed42", "phase": "cpt", "status": "ok"},
        root=tmp_path,
    )
    ts = ledger.read_rows("ledger", tmp_path)[0]["ts_utc"]
    assert ts.endswith("Z") and len(ts) == 20


def test_float_formatting_and_nan(tmp_path):
    assert ledger.format_value(1.10298) == "1.10298"
    assert ledger.format_value(float("nan")) == "NA"
    assert ledger.format_value(None) == "NA"
    assert ledger.format_value(True) == "1"
    assert ledger.format_value("") == "NA"


def test_manifest_schema(tmp_path):
    ledger.append_manifest_rows(
        "train",
        [{"doc_id": "news_00001234", "source": "news", "domain": "news",
          "language": "ko", "sha256": "a" * 64, "split": "train",
          "char_count": 2381, "byte_count": 6137}],
        root=tmp_path,
    )
    path = ledger.manifest_path("train", tmp_path)
    header = path.read_text(encoding="utf-8").split("\n")[0].split("\t")
    assert header == list(ledger.MANIFEST_COLUMNS)


# ── 검사기 ────────────────────────────────────────────────────────────────
def _seed_ledger(tmp_path, run_id="cpt_kosub_mean_50m_seed42"):
    ledger.append_row(
        "ledger",
        {"run_id": run_id, "phase": "cpt", "status": "ok", "git_commit": "NA"},
        root=tmp_path,
    )
    return run_id


def test_validate_passes_on_clean_ledger(tmp_path):
    run_id = _seed_ledger(tmp_path)
    ledger.append_row(
        "lm_metrics",
        {"run_id": run_id, "split": "dev", "domain": "news", "bpb": 1.207,
         "git_commit": "NA"},
        root=tmp_path,
    )
    assert validate(tmp_path) == []


def test_validate_catches_orphan_run_id(tmp_path):
    _seed_ledger(tmp_path)
    ledger.append_row(
        "lm_metrics",
        {"run_id": "존재하지_않는_run", "split": "dev", "domain": "news",
         "bpb": 1.2, "git_commit": "NA"},
        root=tmp_path,
    )
    errors = validate(tmp_path)
    assert any("LEDGER.tsv 에 없다" in e for e in errors)


def test_validate_catches_crlf(tmp_path):
    _seed_ledger(tmp_path)
    path = ledger.table_path("ledger", tmp_path)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert any("CRLF" in e for e in validate(tmp_path))


def test_validate_catches_column_count_mismatch(tmp_path):
    _seed_ledger(tmp_path)
    path = ledger.table_path("ledger", tmp_path)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write("행이\t너무\t짧다\n")
    assert any("컬럼 수" in e for e in validate(tmp_path))


def test_validate_catches_bad_status(tmp_path):
    ledger.append_row(
        "ledger",
        {"run_id": "cpt_x_seed42", "phase": "cpt", "status": "성공", "git_commit": "NA"},
        root=tmp_path,
    )
    assert any("status" in e for e in validate(tmp_path))


def test_validate_catches_bad_sha(tmp_path):
    ledger.append_row(
        "ledger",
        {"run_id": "cpt_x_seed42", "phase": "cpt", "status": "ok",
         "config_sha256": "짧은해시", "git_commit": "NA"},
        root=tmp_path,
    )
    assert any("sha256" in e for e in validate(tmp_path))


# ── 해시 · 명명 ───────────────────────────────────────────────────────────
def test_sha256_obj_is_order_independent():
    assert sha256_obj({"a": 1, "b": 2}) == sha256_obj({"b": 2, "a": 1})
    assert sha256_obj({"a": 1}) != sha256_obj({"a": 2})
    assert is_sha256(sha256_obj({"a": 1}))


def test_sha256_text_normalizes_line_endings():
    assert sha256_text("a\r\nb") == sha256_text("a\nb")


def test_make_run_id():
    assert make_run_id("cpt", "kosub", "mean", "50m", seed=42) == "cpt_kosub_mean_50m_seed42"
    assert make_run_id("tok", "qwen", "original", "v1") == "tok_qwen_original_v1"
    with pytest.raises(ValueError, match="알 수 없는 phase"):
        make_run_id("없는단계", "x")
