"""GPU 작업 잠금 — "학습을 두 개 동시에 띄우지 않는다" (CLAUDE.md · RULES 운영 규칙) 의 집행.

## 왜 실행 시점에, 파일로

2026-10-02 3주차 seed 2026 이 같은 run_id 로 9초 차이로 두 번 떴다. 연쇄 스크립트
둘이 각자 사전 점검을 돌렸는데, **두 점검 모두 어느 학습도 뜨기 전에** "돌고 있는
학습 없음" 을 확인했다 (13:28:35 · 13:28:44 점검, 13:29:16 · 13:29:25 실행). 점검과
실행 사이에 40초가 비면, 점검을 아무리 정확히 해도 막을 수 없다 — 확인 후 실행
(check-then-act) 경쟁이다.

그래서 확인과 점유를 **한 동작** 으로 만든다. `O_CREAT | O_EXCL` 로 잠금 파일을
만들면 운영체제가 둘 중 하나만 성공시킨다. RunContext 는 원장 start 행을 쓰기
**전에** 잠금을 잡으므로, 거부된 쪽은 원장에 흔적을 남기지 않는다.

## 죽은 잠금

학습 프로세스가 강제 종료되면 `__exit__` 이 돌지 않아 파일이 남는다. 파일에 PID 를
적어 두고, 그 PID 가 살아 있지 않으면 회수한다. PID 가 재사용돼 엉뚱한 프로세스가
살아 있으면 회수하지 않는다 — 안전한 쪽(거부)으로 틀린다. 그때는 사람이 파일을
지운다 (메시지가 경로를 알려 준다).

Windows 에서는 `os.kill(pid, 0)` 을 쓰지 않는다. POSIX 에서는 존재 확인이지만
Windows 에서는 `TerminateProcess` 를 불러 **그 프로세스를 죽인다.**
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

LOCK_NAME = ".gpu.lock"


class GpuBusy(RuntimeError):
    """다른 GPU 작업이 잠금을 쥐고 있다."""


def lock_path(root: Path | str) -> Path:
    # artifacts/ 는 git 에서 제외된다. 잠금은 이 기계의 상태이지 기록이 아니다.
    return Path(root) / "artifacts" / LOCK_NAME


def pid_alive(pid: int) -> bool:
    """프로세스가 살아 있는가. 확인할 수 없으면 True (안전한 쪽 — 회수하지 않는다)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            # ERROR_INVALID_PARAMETER(87) 는 그런 PID 가 없다는 뜻이다. 권한 문제 등
            # 다른 이유로 못 열면 살아 있다고 본다.
            return ctypes.get_last_error() != 87
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(root: Path | str) -> dict | None:
    p = lock_path(root)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"pid": -1, "run_id": "<읽을 수 없는 잠금>"}


def acquire(root: Path | str, run_id: str, phase: str) -> dict:
    """잠금을 잡는다. 살아 있는 다른 작업이 쥐고 있으면 GpuBusy."""
    p = lock_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "pid": os.getpid(), "run_id": run_id, "phase": phase,
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for attempt in (1, 2):
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = read_lock(root) or {}
            pid = int(held.get("pid", -1) or -1)
            if attempt == 1 and pid > 0 and not pid_alive(pid):
                # 죽은 잠금. 회수하고 한 번만 다시 시도한다 — 동시에 회수한 다른
                # 프로세스가 먼저 잡았으면 두 번째 O_EXCL 이 실패해 거부된다.
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
                print(f"  [잠금] 죽은 잠금을 회수했다: PID {pid} ({held.get('run_id')})")
                continue
            raise GpuBusy(
                f"GPU 를 다른 작업이 쓰고 있다: PID {pid} · {held.get('run_id')} · "
                f"{held.get('phase')} · {held.get('ts_utc')}\n"
                "학습을 두 개 동시에 띄우지 않는다 (CLAUDE.md). 그 작업이 끝난 뒤 다시 시작하라.\n"
                f"그 PID 가 이 작업이 아닌데 살아 있다면(PID 재사용) {p} 를 지운다."
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False)
        return info
    raise GpuBusy(f"잠금을 회수한 직후 다른 프로세스가 먼저 잡았다: {p}")


def release(root: Path | str) -> None:
    """내가 쥔 잠금만 푼다. 다른 프로세스의 잠금은 건드리지 않는다."""
    held = read_lock(root)
    if held and int(held.get("pid", -1) or -1) == os.getpid():
        try:
            lock_path(root).unlink()
        except FileNotFoundError:
            pass
