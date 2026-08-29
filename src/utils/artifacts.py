"""스펙 §59 — 로컬 산출물의 내용 해시·크기·생성 run 계보 등록.

외부 모델 revision과 구조는 models.tsv가 담당한다. 이 모듈은 프로젝트가 만든
토크나이저, 체크포인트, 어댑터, 매니페스트 요약, 표·그림·리포트를 담당한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ledger
from .gitinfo import repo_root
from .hashing import sha256_dir, sha256_file, sha256_obj

KINDS: tuple[str, ...] = (
    "tokenizer", "checkpoint", "adapter", "manifest_summary",
    "table", "figure", "report", "other",
)


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def describe_artifact(
    path: Path | str,
    *,
    kind: str,
    name: str | None = None,
    run_id: str = ledger.NA,
    model_revision: str = ledger.NA,
    tokenizer_version: str = ledger.NA,
    manifest_sha256: str = ledger.NA,
    note: str = ledger.NA,
    root: Path | str | None = None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"알 수 없는 artifact kind: {kind!r} (가능: {KINDS})")
    base = Path(root or repo_root()).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = base / target
    target = target.resolve()
    try:
        rel = target.relative_to(base)
    except ValueError as exc:
        raise ValueError("산출물은 저장소 안에 있어야 한다") from exc
    if "final_test" in {part.lower() for part in rel.parts}:
        raise ValueError("final_test 산출물은 개봉 태그 전에는 등록할 수 없다")
    if not target.exists():
        raise FileNotFoundError(target)
    if run_id != ledger.NA and run_id not in ledger.known_run_ids(base):
        raise ledger.LedgerError(f"run_id {run_id!r} 가 LEDGER.tsv 에 없다")

    digest = sha256_file(target) if target.is_file() else sha256_dir(target)
    rel_text = rel.as_posix()
    artifact_name = name or target.name
    identity = {
        "kind": kind, "name": artifact_name, "path": rel_text,
        "artifact_sha256": digest, "run_id": run_id,
    }
    return {
        "artifact_id": sha256_obj(identity),
        "run_id": run_id,
        "kind": kind,
        "name": artifact_name,
        "path": rel_text,
        "artifact_sha256": digest,
        "size_bytes": _size_bytes(target),
        "model_revision": model_revision,
        "tokenizer_version": tokenizer_version,
        "manifest_sha256": manifest_sha256,
        "note": note,
    }


def register_artifact(
    path: Path | str,
    *,
    root: Path | str | None = None,
    **metadata: Any,
) -> tuple[dict[str, Any], bool]:
    """산출물을 등록한다. 같은 artifact_id가 있으면 중복 기록하지 않는다."""
    row = describe_artifact(path, root=root, **metadata)
    known = {r.get("artifact_id") for r in ledger.read_rows("artifacts", root)}
    if row["artifact_id"] in known:
        return row, False
    ledger.append_row("artifacts", row, root)
    return row, True
