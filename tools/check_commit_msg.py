"""커밋 메시지 검사 — .githooks/commit-msg 의 본체.

규칙 전문은 docs/COMMIT_CONVENTION.md 에 있다. 여기는 그 집행 코드다.

    <type>(<scope>): <제목>

    <본문 — 왜 바꿨는지>

    <트레일러 — 기계가 읽는 계보>

핵심 두 가지
    1. record 커밋은 코드를 건드릴 수 없다.
       결과를 보고 코드를 슬쩍 고치면서 기록하는 경로를 없앤다 (스펙 §107).
    2. fix 커밋은 Invalidates 를 반드시 적는다.
       그 버그가 어떤 기존 실험 결과를 무효화하는지 남긴다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402
from src.utils.hashing import is_sha256  # noqa: E402

TYPES: dict = {
    "record":  "실험 기록 추가 (원장 행 + 산출물). 코드 변경 금지",
    "fix":     "잘못된 동작 수정",
    "upgrade": "기존 기능 개선 (동작은 유지)",
    "feat":    "새 구성요소·스크립트 추가",
    "data":    "corpus·manifest·split 변경",
    "tok":     "tokenizer 버전 산출물 등록",
    "docs":    "문서",
    "chore":   "환경·설정·의존성·훅",
    "revert":  "되돌리기",
}

SCOPES: tuple = (
    "data", "tok", "surgery", "align", "cpt", "eval", "system", "infra", "docs",
)

REQUIRED_TRAILERS: dict = {
    "record": ("Run-Id", "Ledger", "Config-SHA256"),
    "fix": ("Invalidates",),
    "data": ("Manifest-SHA256",),
    "tok": ("Tokenizer-SHA256",),
}

SUBJECT_MAX = 72

# record 커밋에서 스테이지가 허용되는 경로.
RECORD_ALLOWED_PREFIXES = ("experiments/", "reports/", "data/manifests/", "docs/")

HEADER_RE = re.compile(r"^(?P<type>[a-z]+)\((?P<scope>[a-z]+)\): (?P<subject>.+)$")
TRAILER_RE = re.compile(r"^(?P<key>[A-Z][A-Za-z0-9-]*): (?P<value>.+)$")
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def staged_files(root: Path) -> list:
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=str(root), capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def strip_comments(text: str) -> str:
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n")
             if not ln.startswith("#")]
    # verbose diff 구간 제거
    for i, ln in enumerate(lines):
        if ln.startswith("diff --git "):
            lines = lines[:i]
            break
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def parse_trailers(lines: list) -> dict:
    """마지막 연속 블록에서 'Key: value' 트레일러를 읽는다."""
    block: list = []
    for line in reversed(lines):
        if not line.strip():
            break
        block.append(line)
    trailers: dict = {}
    for line in reversed(block):
        m = TRAILER_RE.match(line.strip())
        if m:
            trailers[m.group("key")] = m.group("value").strip()
    return trailers


def check(message: str, root: Path | None = None) -> list:
    """규칙 위반 목록을 돌려준다. 빈 리스트면 통과."""
    root = Path(root or ledger.repo_root())
    errors: list = []

    text = strip_comments(message)
    lines = text.split("\n")
    if not lines or not lines[0].strip():
        return ["커밋 메시지가 비어 있다"]

    header = lines[0].rstrip()

    # git 이 자동 생성하는 메시지는 통과시킨다.
    if header.startswith(("Merge ", "Revert ", "fixup!", "squash!")):
        return []

    m = HEADER_RE.match(header)
    if not m:
        return [
            f"제목 형식이 틀렸다: {header!r}",
            "  올바른 형식: <type>(<scope>): <제목>",
            f"  type  : {', '.join(sorted(TYPES))}",
            f"  scope : {', '.join(SCOPES)}",
            "  예: record(cpt): C2 Ko-Substitute 50M 토큰 CPT 결과",
        ]

    ctype, scope, subject = m.group("type"), m.group("scope"), m.group("subject")

    if ctype not in TYPES:
        errors.append(f"알 수 없는 type: {ctype!r} (가능: {', '.join(sorted(TYPES))})")
    if scope not in SCOPES:
        errors.append(f"알 수 없는 scope: {scope!r} (가능: {', '.join(SCOPES)})")
    if len(header) > SUBJECT_MAX:
        errors.append(f"제목이 {len(header)}자다. {SUBJECT_MAX}자 이하로 줄여라")
    if subject.endswith("."):
        errors.append("제목 끝에 마침표를 쓰지 않는다")
    if len(lines) > 1 and lines[1].strip():
        errors.append("제목과 본문 사이에 빈 줄이 있어야 한다")

    trailers = parse_trailers(lines[1:]) if len(lines) > 1 else {}
    for key in REQUIRED_TRAILERS.get(ctype, ()):  # type: ignore[arg-type]
        if key not in trailers:
            errors.append(
                f"{ctype} 커밋에는 '{key}:' 트레일러가 필요하다 "
                f"(docs/COMMIT_CONVENTION.md 참조)"
            )

    known_runs = ledger.known_run_ids(root)

    for key in ("Config-SHA256", "Tokenizer-SHA256", "Manifest-SHA256"):
        value = trailers.get(key)
        if value and not is_sha256(value):
            errors.append(f"{key} 는 소문자 64자리 sha256 이어야 한다: {value!r}")

    run_id = trailers.get("Run-Id")
    if run_id:
        if not RUN_ID_RE.match(run_id):
            errors.append(f"Run-Id 형식이 틀렸다: {run_id!r} (소문자·숫자·_.- 만)")
        elif known_runs and run_id not in known_runs:
            errors.append(
                f"Run-Id {run_id!r} 가 experiments/LEDGER.tsv 에 없다. "
                "원장에 없는 run 은 기록할 수 없다"
            )

    invalidates = trailers.get("Invalidates")
    if invalidates and invalidates.lower() not in ("none", "없음"):
        for rid in [r.strip() for r in invalidates.split(",") if r.strip()]:
            if known_runs and rid not in known_runs:
                errors.append(f"Invalidates 의 run_id {rid!r} 가 LEDGER.tsv 에 없다")

    ledger_field = trailers.get("Ledger")
    if ledger_field:
        for rel in [p.strip() for p in ledger_field.split(",") if p.strip()]:
            if not (root / rel).exists():
                errors.append(f"Ledger 에 적힌 경로가 없다: {rel}")

    if ctype == "record":
        offenders = [
            f for f in staged_files(root)
            if not f.startswith(RECORD_ALLOWED_PREFIXES)
        ]
        if offenders:
            errors.append(
                "record 커밋은 코드·설정을 건드릴 수 없다. 다음을 분리해서 "
                "먼저 fix/upgrade/feat 로 커밋하라:"
            )
            errors.extend(f"    {f}" for f in offenders[:20])

    return errors


def main(argv: list | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: check_commit_msg.py <COMMIT_MSG_FILE>", file=sys.stderr)
        return 2
    msg_path = Path(args[0])
    message = msg_path.read_text(encoding="utf-8", errors="replace")

    errors = check(message)
    if not errors:
        return 0

    print("", file=sys.stderr)
    print("  커밋 거부 — 커밋 메시지 규칙 위반", file=sys.stderr)
    print("", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  규칙: docs/COMMIT_CONVENTION.md", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
