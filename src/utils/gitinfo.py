"""저장소 위치와 git 상태 조회.

원장의 모든 행은 `git_commit` 을 갖는다 (스펙 §59 Data Lineage).
사람이 적는 단계를 없애기 위해, 커밋 해시는 여기서 자동으로 읽어온다.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

_MARKERS = (".git", "pyproject.toml")


@lru_cache(maxsize=8)
def repo_root(start: str | None = None) -> Path:
    """프로젝트 루트를 찾는다.

    우선순위: 인자 > 환경변수 KOTOKENLAB_ROOT > 이 파일에서 위로 탐색.
    """
    env = os.environ.get("KOTOKENLAB_ROOT")
    here = Path(start or env or __file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if any((candidate / m).exists() for m in _MARKERS):
            return candidate
    return here


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd or repo_root()),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_commit(cwd: Path | None = None) -> str:
    """현재 HEAD 의 전체 해시. 저장소가 아니거나 커밋이 없으면 'NA'."""
    return _git("rev-parse", "HEAD", cwd=cwd) or "NA"


def git_dirty(cwd: Path | None = None) -> str:
    """워킹트리에 커밋되지 않은 변경이 있는가. '1' / '0' / 'NA'.

    dirty 상태에서 나온 결과는 커밋 해시만으로 재현할 수 없다.
    그래서 값을 숨기지 않고 원장에 그대로 남긴다.
    """
    status = _git("status", "--porcelain", cwd=cwd)
    if status is None:
        return "NA"
    return "1" if status else "0"
