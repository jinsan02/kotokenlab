"""로컬 산출물을 experiments/artifacts.tsv 에 등록한다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402
from src.utils.artifacts import KINDS, register_artifact  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="산출물 해시·크기·run 계보 등록")
    ap.add_argument("path")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--name")
    ap.add_argument("--run-id", default=ledger.NA)
    ap.add_argument("--model-revision", default=ledger.NA)
    ap.add_argument("--tokenizer-version", default=ledger.NA)
    ap.add_argument("--manifest-sha256", default=ledger.NA)
    ap.add_argument("--note", default=ledger.NA)
    args = ap.parse_args()

    row, created = register_artifact(
        args.path,
        kind=args.kind,
        name=args.name,
        run_id=args.run_id,
        model_revision=args.model_revision,
        tokenizer_version=args.tokenizer_version,
        manifest_sha256=args.manifest_sha256,
        note=args.note,
    )
    print("등록" if created else "이미 등록됨")
    print(f"artifact_id      {row['artifact_id']}")
    print(f"artifact_sha256  {row['artifact_sha256']}")
    print(f"size_bytes       {row['size_bytes']}")
    print(f"path             {row['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
