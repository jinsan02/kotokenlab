"""git 인덱스(스테이지 영역) 읽기.

왜 이 모듈이 있는가
    훅은 작업 트리를 보고 CI 는 커밋된 트리를 본다. 두 곳이 다른 것을 보면
    "로컬 통과 → CI 실패"가 난다. 실제로 그랬다 (커밋 1e8b379):
    Invalidates 가 가리킨 run 이 작업 트리 원장에만 있고 커밋된 원장에는
    없었다. 훅은 통과시켰고 CI 는 거부했다.

    인덱스는 **이 커밋이 만들어 낼 트리** 다. 훅이 인덱스를 읽으면 CI 가
    나중에 보게 될 것과 같은 것을 지금 본다. 사람이 커밋 순서를 기억하는
    대신 도구가 막는다.

없음의 세 가지 뜻을 구분한다
    None  git 저장소가 아니다 — 호출자가 작업 트리로 폴백해야 한다
    ""    저장소는 맞지만 그 경로가 인덱스에 없다 — 커밋될 트리에도 없다
    text  인덱스에 있는 내용
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .gitinfo import repo_root


def _run(args: list, root: Path) -> tuple:
    """(returncode, stdout_bytes). git 이 없으면 (None, b'')."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None, b""
    return out.returncode, out.stdout


def in_repo(root: Path | str | None = None) -> bool:
    code, _ = _run(["rev-parse", "--git-dir"], Path(root or repo_root()))
    return code == 0


def read_text(rel: str, root: Path | str | None = None) -> str | None:
    """인덱스에 스테이지된 rel 의 내용.

    저장소가 아니면 None (호출자가 작업 트리로 폴백한다).
    저장소인데 인덱스에 없으면 "" — 이 커밋의 트리에 그 파일은 없다는 뜻이다.
    """
    root = Path(root or repo_root())
    if not in_repo(root):
        return None
    # git 은 항상 posix 구분자를 쓴다. Windows 경로가 섞여 들어와도 맞춰 준다.
    spec = ":" + str(rel).replace("\\", "/")
    code, data = _run(["show", spec], root)
    if code != 0:
        return ""
    # text=True 를 쓰지 않는 이유: Windows 에서 locale(cp949) 로 디코딩돼
    # 원장의 한국어 note 가 깨진다. 원장은 언제나 UTF-8 이다.
    return data.decode("utf-8", errors="replace")


def is_staged(rel: str, root: Path | str | None = None) -> bool | None:
    """이 커밋의 트리에 rel 이 존재하는가. 저장소가 아니면 None."""
    text = read_text(rel, root)
    if text is None:
        return None
    root = Path(root or repo_root())
    spec = str(rel).replace("\\", "/")
    code, _ = _run(["ls-files", "--error-unmatch", "--", spec], root)
    return code == 0
