"""src/utils/gpulock.py · RunContext 의 GPU 잠금 — 2026-10-02 중복 실행의 재발 방지."""

import os
import subprocess
import sys

import pytest

from src.utils import gpulock, ledger


def _dead_pid() -> int:
    """방금 끝난 프로세스의 PID."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_pid_alive_distinguishes_live_and_dead():
    assert gpulock.pid_alive(os.getpid())
    assert not gpulock.pid_alive(_dead_pid())
    assert not gpulock.pid_alive(0)


def test_second_acquire_is_refused_while_holder_lives(tmp_path):
    gpulock.acquire(tmp_path, "cpt_a_seed1", "cpt")
    with pytest.raises(gpulock.GpuBusy, match="cpt_a_seed1"):
        gpulock.acquire(tmp_path, "cpt_b_seed1", "cpt")
    gpulock.release(tmp_path)
    assert not gpulock.lock_path(tmp_path).exists()


def test_dead_holder_lock_is_reclaimed(tmp_path):
    p = gpulock.lock_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text('{"pid": %d, "run_id": "cpt_dead_seed1"}' % _dead_pid(), encoding="utf-8")
    info = gpulock.acquire(tmp_path, "cpt_new_seed1", "cpt")
    assert info["pid"] == os.getpid()
    assert gpulock.read_lock(tmp_path)["run_id"] == "cpt_new_seed1"
    gpulock.release(tmp_path)


def test_release_never_removes_someone_elses_lock(tmp_path):
    p = gpulock.lock_path(tmp_path)
    p.parent.mkdir(parents=True)
    other = os.getppid()          # 살아 있는 다른 프로세스
    p.write_text('{"pid": %d, "run_id": "cpt_other_seed1"}' % other, encoding="utf-8")
    gpulock.release(tmp_path)
    assert p.exists()


def _quiet_runcontext(monkeypatch):
    import src.utils.tracking as tr
    monkeypatch.setattr(tr, "git_commit", lambda root=None: "a" * 40)
    monkeypatch.setattr(tr, "git_dirty", lambda root=None: "0")
    monkeypatch.setattr(tr.clock_mod, "require_recent_check", lambda root=None: "c" * 64)
    monkeypatch.setattr(tr.env_mod, "env_sha256", lambda: "d" * 64)
    monkeypatch.setattr(tr.env_mod, "collect", lambda: {})
    monkeypatch.setattr(ledger, "git_commit", lambda: "a" * 40)
    return tr


def test_duplicate_runcontext_is_refused_before_any_ledger_row(tmp_path, monkeypatch):
    """2026-10-02 의 재현: 같은 run 을 두 번 띄우면 두 번째는 원장에 아무것도 안 쓴다."""
    tr = _quiet_runcontext(monkeypatch)
    with tr.RunContext("cpt_dup_seed1", phase="cpt", config={"k": 1}, root=tmp_path,
                       skip_env_check=True, set_seeds=False):
        with pytest.raises(gpulock.GpuBusy):
            with tr.RunContext("cpt_dup_seed1", phase="cpt", config={"k": 1}, root=tmp_path,
                               skip_env_check=True, set_seeds=False):
                pass
    rows = ledger.read_rows("ledger", tmp_path)
    assert [r["status"] for r in rows] == ["start", "ok"]
    assert not gpulock.lock_path(tmp_path).exists()


def test_cpu_phase_takes_no_lock(tmp_path, monkeypatch):
    tr = _quiet_runcontext(monkeypatch)
    gpulock.acquire(tmp_path, "cpt_busy_seed1", "cpt")
    try:
        with tr.RunContext("tok_x_v1", phase="tok", config={"k": 1}, root=tmp_path,
                           skip_env_check=True, set_seeds=False):
            pass
    finally:
        gpulock.release(tmp_path)


def test_failed_entry_releases_the_lock(tmp_path, monkeypatch):
    tr = _quiet_runcontext(monkeypatch)

    def no_clock(root=None):
        raise RuntimeError("최근 시간 검증 기록이 없다")

    monkeypatch.setattr(tr.clock_mod, "require_recent_check", no_clock)
    with pytest.raises(RuntimeError, match="시간 검증"):
        with tr.RunContext("cpt_noclock_seed1", phase="cpt", root=tmp_path,
                           skip_env_check=True, set_seeds=False):
            pass
    assert not gpulock.lock_path(tmp_path).exists()
