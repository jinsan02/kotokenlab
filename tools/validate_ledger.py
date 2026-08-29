"""원장 TSV 무결성 검사 — pre-commit 이 호출하고, 사람이 직접 돌려도 된다.

검사 항목
    - CRLF 없음 (.gitattributes 가 LF 를 강제하지만 실수로 섞이는 경우가 있다)
    - 헤더가 src/utils/ledger.py 의 스키마와 정확히 일치
    - 모든 행의 컬럼 수 == 헤더 컬럼 수
    - 빈 필드 없음 (결측은 'NA')
    - sha256 컬럼은 64자리 hex 또는 'NA'
    - git_commit 은 40자리 hex 또는 'NA'
    - 참조 무결성: 메트릭 TSV 의 run_id 는 LEDGER.tsv 에 존재해야 한다

마지막 항목이 핵심이다. 원장에 없는 run 의 메트릭은 계보가 끊긴 숫자다.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402

PHASES = ("data", "tok", "surgery", "align", "cpt", "eval", "sys")

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

SHA_COLUMNS = (
    "config_sha256", "tokenizer_sha256", "manifest_sha256", "env_sha256", "sha256",
)


def _read(path: Path) -> tuple:
    with io.open(path, "r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    return raw, raw.count("\r\n")


def _check_table(path: Path, expected: tuple, label: str) -> tuple:
    """(errors, rows) 를 돌려준다. rows 는 dict 리스트."""
    errors: list = []
    raw, crlf = _read(path)
    if crlf:
        errors.append(f"{label}: CRLF 줄바꿈 {crlf}개. LF 로 통일하라")
    lines = [ln for ln in raw.replace("\r\n", "\n").split("\n") if ln != ""]
    if not lines:
        return [f"{label}: 파일이 비어 있다 (헤더가 없다)"], []

    header = tuple(lines[0].split("\t"))
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        errors.append(f"{label}: 헤더가 스키마와 다르다")
        if missing:
            errors.append(f"    누락: {missing}")
        if extra:
            errors.append(f"    잉여: {extra}")
        if not missing and not extra:
            errors.append("    컬럼 순서가 다르다")
        return errors, []

    rows: list = []
    for n, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(header):
            errors.append(
                f"{label}:{n}: 컬럼 수 {len(fields)} != 헤더 {len(header)}"
            )
            continue
        row = dict(zip(header, fields))
        rows.append(row)

        for col, value in row.items():
            if value == "":
                errors.append(f"{label}:{n}: '{col}' 이 빈칸이다. 결측은 'NA' 로 적어라")
            elif col in SHA_COLUMNS and value != "NA" and not HEX64.match(value):
                errors.append(f"{label}:{n}: '{col}' 이 sha256 형식이 아니다: {value!r}")
            elif col == "git_commit" and value != "NA" and not HEX40.match(value):
                errors.append(f"{label}:{n}: git_commit 형식이 아니다: {value!r}")
            elif col == "ts_utc" and value != "NA" and not TS_RE.match(value):
                errors.append(
                    f"{label}:{n}: ts_utc 는 ISO-8601 UTC(...Z) 여야 한다: {value!r}"
                )
    return errors, rows


def validate(root: Path | str | None = None) -> list:
    root = Path(root or ledger.repo_root())
    errors: list = []
    run_ids: set = set()

    # 1) LEDGER 먼저 — 다른 테이블의 참조 대상이다.
    ledger_path = ledger.table_path("ledger", root)
    if ledger_path.exists():
        errs, rows = _check_table(ledger_path, ledger.LEDGER_COLUMNS, "LEDGER.tsv")
        errors += errs
        run_ids = {r["run_id"] for r in rows if r.get("run_id") not in (None, "", "NA")}
        for n, row in enumerate(rows, start=2):
            if row.get("phase") not in PHASES:
                errors.append(f"LEDGER.tsv:{n}: 알 수 없는 phase {row.get('phase')!r}")
            if row.get("status") not in ("start", "ok", "fail", "abort"):
                errors.append(f"LEDGER.tsv:{n}: 알 수 없는 status {row.get('status')!r}")

    # 2) 나머지 테이블
    for table, (rel, cols) in ledger.TABLES.items():
        if table == "ledger":
            continue
        path = Path(root) / rel
        if not path.exists():
            continue
        errs, rows = _check_table(path, cols, Path(rel).name)
        errors += errs
        if table in ledger.METRIC_TABLES and run_ids:
            for n, row in enumerate(rows, start=2):
                rid = row.get("run_id")
                if rid and rid != "NA" and rid not in run_ids:
                    errors.append(
                        f"{Path(rel).name}:{n}: run_id {rid!r} 가 LEDGER.tsv 에 없다"
                    )

    # 3) manifest
    manifest_dir = Path(root) / "data" / "manifests"
    if manifest_dir.is_dir():
        for path in sorted(manifest_dir.glob("*.tsv")):
            errs, _ = _check_table(path, ledger.MANIFEST_COLUMNS, path.name)
            errors += errs

    return errors


def main() -> int:
    errors = validate()
    if not errors:
        print("원장 검사 통과")
        return 0
    print("", file=sys.stderr)
    print("  원장 무결성 위반", file=sys.stderr)
    print("", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  스키마: docs/LEDGER_SCHEMA.md / src/utils/ledger.py", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
