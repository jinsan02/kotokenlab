"""pre-commit 검사 본체 — 저장소에 들어가면 안 되는 것을 막는다.

이름은 C:/aimers 의 tools/precheck.py 에서 따왔다. 같은 역할이다:
사람이 기억해서 지키는 규칙을 기계가 대신 지킨다.

막는 것
    1. 5MB 초과 파일
    2. 체크포인트·모델 바이너리 확장자
    3. .conda/, data/raw/, data/interim/, artifacts/ 아래 전부
    4. 경로에 final_test 가 들어간 모든 파일  ← 스펙 §10, §122-2 lockbox
    5. 출력 셀이 남아 있는 .ipynb
    6. 원장 TSV 무결성 위반 (tools/validate_ledger.py)

4번은 이 프로젝트에서 가장 되돌릴 수 없는 실수를 막는 장치다.
Final Test 를 한 번 보면 그 다음의 모든 선택이 오염된다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402
from tools.validate_ledger import validate  # noqa: E402

MAX_BYTES = 5 * 1024 * 1024

BLOCKED_SUFFIXES = (
    ".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".npz", ".model", ".arrow",
)

BLOCKED_PREFIXES = (
    ".conda/", "data/raw/", "data/interim/", "artifacts/",
)

LOCKBOX_TOKEN = "final_test"


def _git(*args: str, root: Path) -> str:
    out = subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=30
    )
    return out.stdout if out.returncode == 0 else ""


def staged_files(root: Path) -> list:
    raw = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", root=root)
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def staged_size(path: str, root: Path) -> int:
    raw = _git("cat-file", "-s", f":{path}", root=root).strip()
    if raw.isdigit():
        return int(raw)
    full = root / path
    return full.stat().st_size if full.exists() else 0


def staged_blob(path: str, root: Path) -> str:
    return _git("show", f":{path}", root=root)


def check(root: Path | None = None) -> list:
    root = Path(root or ledger.repo_root())
    errors: list = []
    files = staged_files(root)

    for path in files:
        lower = path.lower()

        if LOCKBOX_TOKEN in lower:
            errors.append(
                f"{path}: Final Test 는 커밋할 수 없다 (스펙 §10 / docs/RULES.md 규칙 2). "
                "최종 평가를 실제로 개봉할 때는 final-test-opened 태그로 기록하라"
            )
            continue

        if any(lower.startswith(p) for p in BLOCKED_PREFIXES):
            errors.append(f"{path}: 커밋 금지 경로다 (.gitignore 확인)")
            continue

        if lower.endswith(BLOCKED_SUFFIXES):
            errors.append(
                f"{path}: 모델·체크포인트 바이너리는 커밋하지 않는다. "
                "artifacts/ 에 두고 sha256 만 원장에 기록하라"
            )
            continue

        size = staged_size(path, root)
        if size > MAX_BYTES:
            errors.append(
                f"{path}: {size / 1024 / 1024:.1f}MB 로 상한 "
                f"{MAX_BYTES // 1024 // 1024}MB 를 넘는다"
            )

        if lower.endswith(".ipynb"):
            try:
                nb = json.loads(staged_blob(path, root) or "{}")
            except json.JSONDecodeError:
                nb = {}
            for cell in nb.get("cells", []):
                if cell.get("outputs") or cell.get("execution_count"):
                    errors.append(f"{path}: 출력 셀을 비우고 커밋하라")
                    break

    errors.extend(validate(root))
    return errors


def main() -> int:
    errors = check()
    if not errors:
        return 0
    print("", file=sys.stderr)
    print("  커밋 거부 — pre-commit 검사 실패", file=sys.stderr)
    print("", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  규칙: docs/RULES.md   |   --no-verify 로 우회하지 마라", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
