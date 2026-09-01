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


# ── 스키마 진화 (컬럼 추가 후 마이그레이션) ───────────────────────────────
def test_migration_pads_old_rows(tmp_path):
    """컬럼을 뒤에 추가했을 때 기존 파일이 자동으로 맞춰지는지."""
    from tools.migrate_ledger import migrate_file

    path = tmp_path / "old.tsv"
    path.write_text("a\tb\nx\ty\n", encoding="utf-8", newline="\n")

    assert "변경 예정" in migrate_file(path, ("a", "b", "c"), dry_run=True)
    assert path.read_text(encoding="utf-8") == "a\tb\nx\ty\n"  # dry-run 은 안 건드린다

    assert "완료" in migrate_file(path, ("a", "b", "c"), dry_run=False)
    assert path.read_text(encoding="utf-8") == "a\tb\tc\nx\ty\tNA\n"
    assert (tmp_path / "old.tsv.bak").exists()


def test_migration_refuses_reordered_header(tmp_path):
    """컬럼 순서가 바뀐 경우는 자동으로 손대지 않는다 — 사람이 봐야 한다."""
    from tools.migrate_ledger import migrate_file

    path = tmp_path / "reordered.tsv"
    path.write_text("b\ta\nx\ty\n", encoding="utf-8", newline="\n")
    assert "수동 확인 필요" in migrate_file(path, ("a", "b", "c"), dry_run=False)


def test_migration_is_noop_when_current(tmp_path):
    from tools.migrate_ledger import migrate_file

    path = tmp_path / "cur.tsv"
    path.write_text("a\tb\nx\ty\n", encoding="utf-8", newline="\n")
    assert migrate_file(path, ("a", "b"), dry_run=False) == "이미 최신"


def test_migration_skips_summary_manifest(tmp_path):
    from tools.migrate_ledger import manifest_paths

    manifest_dir = tmp_path / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "train.tsv").write_text("x\n", encoding="utf-8")
    (manifest_dir / "SUMMARY.tsv").write_text("summary\n", encoding="utf-8")

    assert [path.name for path in manifest_paths(tmp_path)] == ["train.tsv"]


# ── 훅과 CI 는 같은 트리를 본다 (validate 편) ─────────────────────────────
#
# check_commit_msg 와 같은 이유다. 훅이 작업 트리를 보면, 깨진 것을 스테이지해
# 놓고 작업 트리에서만 고쳐도 로컬은 통과하고 CI 는 거부한다.

def _git(root, *args):
    import subprocess
    subprocess.run(["git", *args], cwd=str(root), check=True, capture_output=True)


def test_validate_index_mode_sees_the_staged_tree(tmp_path):
    import subprocess

    import pytest as _p
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError):
        _p.skip("git 이 없는 환경")

    _git(tmp_path, "init", "-q")
    ledger.append_row(
        "ledger",
        {"run_id": "cpt_ok_seed42", "phase": "cpt", "status": "ok",
         "git_commit": "NA"},
        root=tmp_path,
    )
    _git(tmp_path, "add", "experiments/LEDGER.tsv")     # 멀쩡한 것을 스테이지

    # 작업 트리만 망가뜨린다 — 컬럼 수가 안 맞는 행
    path = ledger.table_path("ledger", tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "깨진행\n",
                    encoding="utf-8", newline="\n")

    assert validate(tmp_path, source="worktree"), "작업 트리는 깨져 있다"
    assert validate(tmp_path, source="index") == [], \
        "커밋될 트리는 멀쩡하므로 통과해야 한다"


def test_validate_index_mode_catches_a_staged_break(tmp_path):
    import subprocess

    import pytest as _p
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError):
        _p.skip("git 이 없는 환경")

    _git(tmp_path, "init", "-q")
    ledger.append_row(
        "ledger",
        {"run_id": "cpt_ok_seed42", "phase": "cpt", "status": "ok",
         "git_commit": "NA"},
        root=tmp_path,
    )
    path = ledger.table_path("ledger", tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "깨진행\n",
                    encoding="utf-8", newline="\n")
    _git(tmp_path, "add", "experiments/LEDGER.tsv")     # 깨진 것을 스테이지

    # 작업 트리는 되돌려 놔도 인덱스가 깨졌으면 거부해야 한다
    lines = path.read_text(encoding="utf-8").split("\n")
    path.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8", newline="\n")

    assert validate(tmp_path, source="worktree") == [], "작업 트리는 멀쩡하다"
    assert validate(tmp_path, source="index"), "커밋될 트리가 깨졌으므로 거부"


def test_validate_falls_back_outside_a_repo(tmp_path):
    """git 밖에서는 인덱스가 없다. 작업 트리로 폴백한다."""
    ledger.append_row(
        "ledger",
        {"run_id": "cpt_ok_seed42", "phase": "cpt", "status": "ok",
         "git_commit": "NA"},
        root=tmp_path,
    )
    assert validate(tmp_path, source="index") == []


# ── 돌고 있는 run 을 죽은 것으로 보지 않는다 ──────────────────────────────
#
# 학습 중인 run 은 정상적으로 start 만 있다. 그때 다른 실험 결과를 기록하려
# 하면 예전 검사는 살아 있는 run 을 죽은 것으로 보고 커밋을 막았다.
# 추측이 아니라 증거를 본다 — 최근에 지표 행을 썼는가.

def _ts(minutes_ago: int) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc)
            - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _running_repo(tmp_path, curve_age_min: int):
    """끝난 run 하나 + start 만 있는 run 하나. 후자가 지표를 쓴 지 curve_age_min 분."""
    ledger.append_row("ledger", {"run_id": "cpt_done_seed42", "phase": "cpt",
                                 "status": "start", "git_commit": "NA"},
                      root=tmp_path)
    ledger.append_row("ledger", {"run_id": "cpt_done_seed42", "phase": "cpt",
                                 "status": "ok", "git_commit": "NA"},
                      root=tmp_path)
    ledger.append_row("ledger", {"run_id": "cpt_live_seed42", "phase": "cpt",
                                 "status": "start", "git_commit": "NA"},
                      root=tmp_path)
    ledger.append_row("train_curve", {"run_id": "cpt_live_seed42", "step": 10,
                                      "tokens_seen": 1000,
                                      "ts_utc": _ts(curve_age_min),
                                      "git_commit": "NA"},
                      root=tmp_path)
    return tmp_path


def test_lifecycle_lets_a_live_run_through(tmp_path):
    """방금 지표를 쓴 run 은 돌고 있는 것이다. 기록 커밋을 막으면 안 된다."""
    root = _running_repo(tmp_path, curve_age_min=5)
    assert validate(root) == []


def test_lifecycle_still_catches_a_dead_run(tmp_path):
    """지표가 멎은 지 오래면 죽은 것이다. 그대로 잡아야 한다."""
    root = _running_repo(tmp_path, curve_age_min=600)
    errors = validate(root)
    assert any("cpt_live_seed42" in e for e in errors)
    assert not any("cpt_done_seed42" in e for e in errors)


def test_lifecycle_catches_a_run_with_no_metrics_at_all(tmp_path):
    """지표 행이 아예 없으면 살아 있다는 증거가 없다."""
    ledger.append_row("ledger", {"run_id": "cpt_ghost_seed42", "phase": "cpt",
                                 "status": "start", "git_commit": "NA"},
                      root=tmp_path)
    assert any("cpt_ghost_seed42" in e for e in validate(tmp_path))
