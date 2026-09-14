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


CODE_SUFFIXES = (".py", ".ps1", ".sh")
CODE_PREFIXES = ("src/", "scripts/", "tools/", "configs/", "tests/")
TAB = chr(9)


def is_code(path: str) -> bool:
    return (path.lower().endswith(CODE_SUFFIXES)
            or any(path.startswith(pre) for pre in CODE_PREFIXES))


def result_rows_added(staged_text: str, head_text: str) -> list:
    """스테이지된 LEDGER 에 새로 붙은 **실험 결과** 행의 run_id 목록.

    결과 행 = `status` 가 ok 또는 fail. `start` 행만 있는 것은 학습이 도는
    중이라는 뜻이므로 결과가 아니다.

    스키마 이행(컬럼 추가로 기존 행이 전부 다시 쓰이는 경우)에서는 status 가
    ok 인 옛 행들도 "새 줄" 로 보인다. 그래서 호출부가 헤더 변경 여부를 먼저
    본다 — 헤더가 바뀌었으면 이행으로 보고 검사하지 않는다.
    """
    before = set(head_text.splitlines())
    out = []
    for line in staged_text.splitlines():
        if not line or line in before:
            continue
        cells = line.split(TAB)
        if len(cells) > 3 and cells[3] in ("ok", "fail"):
            out.append(cells[1])
    return out


def mixed_results_and_code(run_ids: list, code_files: list, rel: str) -> list:
    """실험 결과와 코드가 한 커밋에 있으면 거부 사유를 돌려준다.

    훅은 원래 한 방향만 막았다 — `record:` 커밋에 코드가 섞이는 것. 그런데
    2026-09-14 에 반대 방향이 터졌다. `fix(infra)` 커밋이 스테이지에 남아 있던
    R1 의 원장 행·메트릭·run 디렉터리를 통째로 삼켰고 훅은 통과시켰다.
    어느 커밋이 그 결과를 기록했는지 이력만 보고는 알 수 없게 된다.
    """
    if not run_ids or not code_files:
        return []
    uniq = sorted(set(run_ids))
    shown = ", ".join(uniq[:3]) + (" 외" if len(uniq) > 3 else "")
    return [
        f"{rel}: 실험 결과 행 {len(run_ids)}개(run {shown})와 "
        f"코드 {len(code_files)}개가 같은 커밋에 있다. 나눠라 — 코드를 먼저 "
        f"fix/upgrade/feat 로, 결과는 record(...) 로 "
        f"(CLAUDE.md \"record 커밋과 코드 커밋을 섞지 않는다\")"
    ]


def _check_results_not_mixed_with_code(files: list, root: Path) -> list:
    code = [f for f in files if is_code(f)]
    rel = ledger.TABLES["ledger"][0]
    if not code or rel not in files:
        return []
    try:
        staged = staged_blob(rel, root) or ""
        head = _git("show", f"HEAD:{rel}", root=root)
    except Exception:
        return []          # 첫 커밋 등 — 비교 대상이 없으면 검사하지 않는다
    if not staged or not head:
        return []
    if staged.splitlines()[0] != head.splitlines()[0]:
        return []          # 헤더가 바뀌었다 = 스키마 이행. 코드와 같이 가는 게 정상
    return mixed_results_and_code(result_rows_added(staged, head), code, rel)


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

    errors += _check_results_not_mixed_with_code(files, root)

    # 미종료 run 검사는 원장을 건드리는 커밋에서만 한다. 학습이 도는 동안에는
    # 그 run 이 정상적으로 start 만 있는 상태라, 문서만 고치는 커밋까지 막는 것은
    # 오탐이다. 실제로 6시간짜리 CPT 중에 문서 커밋이 막혔다.
    touches_ledger = any(
        f.startswith("experiments/") or f.startswith("data/manifests/")
        for f in files)
    # 인덱스를 본다 — CI 는 커밋된 트리를 보므로, 작업 트리에서 고쳐 놓고 깨진
    # 것을 스테이지하면 로컬만 통과하는 경로가 생긴다 (check_commit_msg 와 같은
    # 이유. docs/COMMIT_CONVENTION.md "훅과 CI 는 같은 트리를 본다").
    errors.extend(validate(root, check_lifecycle=touches_ledger, source="index"))
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
