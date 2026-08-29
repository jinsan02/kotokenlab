"""원장 TSV 를 현재 스키마로 맞춘다 (컬럼 추가 후 1회 실행).

규칙상 컬럼은 **뒤에만** 추가한다 (docs/LEDGER_SCHEMA.md). 그런데 이미 존재하는
파일의 헤더는 옛날 그대로라서, 컬럼을 늘리면 헤더와 행이 어긋난다.
이 도구가 그 간극을 메운다:

    - 헤더를 현재 스키마로 교체
    - 기존 행의 부족한 뒤쪽 컬럼을 'NA' 로 채움
    - 이미 새 컬럼을 갖고 붙은 행은 그대로 둠

**값은 절대 바꾸지 않는다.** append-only 원칙을 지키기 위해 뒤에 NA 를 덧대기만 한다.
헤더가 뒤섞이거나 행이 스키마보다 길면 손대지 않고 보고만 한다 — 사람이 봐야 한다.

    .conda/python.exe tools/migrate_ledger.py --dry-run
    .conda/python.exe tools/migrate_ledger.py
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402


def _load(path: Path) -> list:
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read().replace("\r\n", "\n")
    return [ln for ln in raw.split("\n") if ln != ""]


def migrate_file(path: Path, expected: tuple, dry_run: bool) -> str:
    if not path.exists():
        return "없음"
    lines = _load(path)
    if not lines:
        return "빈 파일"

    header = tuple(lines[0].split("\t"))
    if header == expected:
        return "이미 최신"

    # 기존 헤더가 새 스키마의 접두사가 아니면 자동으로 손대지 않는다.
    if header != expected[: len(header)]:
        return (f"수동 확인 필요 — 기존 헤더가 현재 스키마의 접두사가 아니다\n"
                f"      기존: {list(header)}\n      현재: {list(expected)}")

    added = len(expected) - len(header)
    if added < 0:
        return "수동 확인 필요 — 현재 스키마가 파일보다 컬럼이 적다"

    out = ["\t".join(expected)]
    padded = 0
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < len(expected):
            fields += [ledger.NA] * (len(expected) - len(fields))
            padded += 1
        elif len(fields) > len(expected):
            return f"수동 확인 필요 — {len(fields)}개 필드 행이 있다"
        out.append("\t".join(fields))

    if dry_run:
        return f"변경 예정: 컬럼 +{added}, 행 {padded}개 패딩"

    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")
    return f"완료: 컬럼 +{added}, 행 {padded}개 패딩 (.bak 보관)"


def manifest_paths(root: Path) -> list[Path]:
    """문서 매니페스트만 반환한다. SUMMARY.tsv 는 집계 스키마라 제외한다."""
    manifest_dir = root / "data" / "manifests"
    if not manifest_dir.is_dir():
        return []
    return [
        path for path in sorted(manifest_dir.glob("*.tsv"))
        if path.name != "SUMMARY.tsv"
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="원장 스키마 마이그레이션")
    ap.add_argument("--dry-run", action="store_true", help="바꾸지 않고 보고만 한다")
    args = ap.parse_args()

    root = ledger.repo_root()
    problems = 0
    for table, (rel, cols) in ledger.TABLES.items():
        result = migrate_file(root / rel, cols, args.dry_run)
        print(f"  {Path(rel).name:<24} {result}")
        if "수동" in result:
            problems += 1

    for path in manifest_paths(root):
        result = migrate_file(path, ledger.MANIFEST_COLUMNS, args.dry_run)
        print(f"  {path.name:<24} {result}")
        if "수동" in result:
            problems += 1

    if problems:
        print(f"\n{problems}개 파일은 수동 확인이 필요하다", file=sys.stderr)
        return 1
    if not args.dry_run:
        print("\n검사: .conda/python.exe tools/validate_ledger.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
