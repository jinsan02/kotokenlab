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


# 결과 기록은 run 이 도는 동안 늘 커밋 전 상태다. 이것까지 dirty 로 치면
# 모든 행이 1 이 되어 플래그가 아무것도 말하지 않는다 (2026-09-17 까지 실제로
# 그랬다). dirty 는 **실행 결과를 바꿀 수 있는 파일** 만 본다.
RECORD_PREFIXES: tuple[str, ...] = (
    "experiments/", "reports/", "data/manifests/", "env/ENV_SNAPSHOT.tsv",
)


def dirty_paths(porcelain_z: str) -> list:
    """`git status --porcelain -z` 출력에서 기록 경로를 뺀 경로 목록."""
    out = []
    entries = porcelain_z.split("\0")
    i = 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if len(e) < 4:
            continue
        path = e[3:]
        if e[0] in "RC":          # 이름 바꾸기는 원래 경로가 다음 항목에 온다
            i += 1
        if not path.startswith(RECORD_PREFIXES):
            out.append(path)
    return out


def git_dirty(cwd: Path | None = None) -> str:
    """실행 결과를 바꿀 수 있는 커밋 안 된 변경이 있는가. '1' / '0' / 'NA'.

    dirty 상태에서 나온 결과는 커밋 해시만으로 재현할 수 없다.
    그래서 값을 숨기지 않고 원장에 그대로 남긴다.

    **2026-09-17 부터** 결과 기록 경로(`RECORD_PREFIXES`)는 세지 않는다.
    그 전 행의 `1` 은 대부분 커밋 안 된 원장 행 때문이라 정보가 없다.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            cwd=str(cwd or repo_root()), capture_output=True, text=True,
            encoding="utf-8", timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "NA"
    if out.returncode != 0:
        return "NA"
    return "1" if dirty_paths(out.stdout) else "0"


# `git_dirty` 가 기록 경로를 빼고 세기 시작한 커밋 (2026-09-17).
# 이 커밋을 조상으로 갖지 않는 run 의 dirty=1 은 **코드가 아니라 원장·리포트**
# 때문일 수 있다. 둘을 구분할 수 없으므로 그렇게 말해야 한다.
DIRTY_SCOPE_COMMIT = "a4784269ed3d5586ba415a4b5ca8a27f56adbf99"


def dirty_is_code_scoped(sha: str, cwd: Path | None = None):
    """그 run 의 `git_dirty` 가 **코드 기준**이었나. True/False/None(모름).

    RULES 는 결과를 커밋 해시로 고정하라고 한다. 그런데 옛 정의의 dirty=1 을
    "코드가 커밋 안 된 채 돌았다" 로 읽으면 없는 결함을 만든다 — 대부분은
    그 run 이 직접 append 한 원장 행이었다. 반대로 "괜찮다" 로 읽으면 진짜
    코드 오염을 놓친다. **모르는 것은 모른다고 답한다.**
    """
    if not sha:
        return None
    try:
        out = subprocess.run(
            ["git", "merge-base", "--is-ancestor", DIRTY_SCOPE_COMMIT, sha],
            cwd=str(cwd or repo_root()), capture_output=True, text=True,
            encoding="utf-8", timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode == 0:
        return True
    # 1 은 "조상 아님", 그 외(128 등)는 "커밋을 모른다".
    return False if out.returncode == 1 else None
