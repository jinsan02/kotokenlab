"""D3 용 KMMLU 파일을 받는다 (docs/PLAN.md "D3 실행 조건", 2026-09-17 등록).

    .conda/python.exe scripts/download_kmmlu.py

등록한 6과목의 dev/test CSV 12개만 받는다. train 과 나머지 과목은 받지 않는다 —
받아 두면 "받은 김에 과목을 더하자" 가 가능해지고, 그것이 사후 선택이다.

판본은 revision 해시로 고정한다. 파일 sha256 은 artifacts.tsv 에 코드가 적는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402

from src.evaluation.capability import KMMLU_REPO, KMMLU_REVISION, SUBJECTS  # noqa: E402
from src.utils.artifacts import register_artifact  # noqa: E402

DEST = ROOT / "data" / "raw" / "kmmlu"


def main() -> int:
    if "final_test" in str(DEST):
        raise SystemExit("final_test 에는 쓰지 않는다")
    for subject in SUBJECTS:
        for split in ("dev", "test"):
            fn = f"data/{subject}-{split}.csv"
            p = Path(hf_hub_download(KMMLU_REPO, fn, repo_type="dataset",
                                     revision=KMMLU_REVISION, local_dir=DEST))
            row, created = register_artifact(
                p, kind="other", name=f"kmmlu_{subject}_{split}",
                model_revision=KMMLU_REVISION,
                note=f"{KMMLU_REPO} {fn} (D3, CC-BY-ND-4.0)")
            print(f"{'등록' if created else '이미'}  {fn:<52} "
                  f"{row['size_bytes']:>8}B  {row['artifact_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
