"""시간 검증과 산출물 레지스트리의 계보를 검사한다."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.utils import ledger
from src.utils.artifacts import describe_artifact, register_artifact
from src.utils.clock import evaluate_probe, latest_valid_check, require_recent_check
from src.utils.tracking import RunContext
from tools.validate_ledger import validate


def test_clock_probe_uses_request_midpoint():
    row = evaluate_probe(100.5, 100.0, 101.0, max_offset_ms=100)
    assert row["status"] == "ok"
    assert row["offset_ms"] == 0.0
    assert row["rtt_ms"] == 1000.0


def test_clock_probe_rejects_large_offset():
    row = evaluate_probe(110.0, 100.0, 101.0, max_offset_ms=100)
    assert row["status"] == "fail"


def test_recent_clock_check_is_required(tmp_path):
    digest = "a" * 64
    ledger.append_row(
        "clock_checks",
        {"clock_check_sha256": digest, "status": "ok", "server": "https://time"},
        root=tmp_path,
    )
    assert latest_valid_check(tmp_path)["clock_check_sha256"] == digest
    assert require_recent_check(tmp_path) == digest


def test_old_clock_check_is_not_accepted(tmp_path):
    ledger.append_row(
        "clock_checks",
        {"ts_utc": "2020-01-01T00:00:00Z", "clock_check_sha256": "a" * 64,
         "status": "ok", "server": "https://time"},
        root=tmp_path,
    )
    assert latest_valid_check(tmp_path, now_epoch=time.time()) is None


def _seed_run(root: Path, run_id: str = "data_probe_seed42") -> str:
    ledger.append_row(
        "ledger",
        {"run_id": run_id, "phase": "data", "status": "ok", "git_commit": "NA"},
        root=root,
    )
    return run_id


def test_artifact_registration_is_idempotent(tmp_path):
    run_id = _seed_run(tmp_path)
    artifact = tmp_path / "reports" / "table.tsv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("a\tb\n1\t2\n", encoding="utf-8", newline="\n")

    first, created_first = register_artifact(
        artifact, kind="table", run_id=run_id, root=tmp_path,
    )
    second, created_second = register_artifact(
        artifact, kind="table", run_id=run_id, root=tmp_path,
    )

    assert created_first is True and created_second is False
    assert first["artifact_id"] == second["artifact_id"]
    assert len(ledger.read_rows("artifacts", tmp_path)) == 1
    assert validate(tmp_path) == []


def test_artifact_rejects_unknown_run(tmp_path):
    artifact = tmp_path / "reports" / "x.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("x", encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match="LEDGER.tsv 에 없다"):
        describe_artifact(
            artifact, kind="report", run_id="ghost_run", root=tmp_path,
        )


def test_artifact_rejects_final_test_path(tmp_path):
    artifact = tmp_path / "data" / "final_test" / "result.txt"
    with pytest.raises(ValueError, match="final_test"):
        describe_artifact(artifact, kind="other", root=tmp_path)


def test_validator_checks_clock_reference(tmp_path):
    ledger.append_row(
        "ledger",
        {"run_id": "data_x_seed42", "phase": "data", "status": "ok",
         "clock_check_sha256": "b" * 64, "git_commit": "NA"},
        root=tmp_path,
    )
    assert any("clock_checks.tsv 에 없다" in error for error in validate(tmp_path))


def test_run_context_links_recent_clock_check(tmp_path, monkeypatch):
    digest = "c" * 64
    ledger.append_row(
        "clock_checks",
        {"clock_check_sha256": digest, "status": "ok", "server": "https://time",
         "git_commit": "NA"},
        root=tmp_path,
    )
    monkeypatch.setattr("src.utils.tracking.env_mod.env_sha256", lambda: "d" * 64)
    monkeypatch.setattr("src.utils.tracking.env_mod.collect", lambda: {})
    monkeypatch.setattr("src.utils.tracking._reset_vram_stats", lambda: None)
    monkeypatch.setattr("src.utils.tracking._peak_vram_mb", lambda: 0)

    with RunContext(
        "data_clock_link_seed42", phase="data", root=tmp_path,
        skip_env_check=True, set_seeds=False,
    ):
        pass

    rows = ledger.read_rows("ledger", tmp_path)
    assert {row["clock_check_sha256"] for row in rows} == {digest}
