"""HF 파케이 데이터셋을 **고르게 퍼진** 표본으로 읽는다.

앞에서부터 순차로 읽으면 안 된다. FineWeb-2 kor_Hang 에서 실제로 해보니 앞
2,868건 중 24% 가 tripadvisor 한 호스트였다 — 파케이가 출처별로 뭉쳐 있기 때문이다.
그 표본으로 도메인 분포나 압축률을 논하면 전부 틀린다.

row group 을 일정 간격으로 골라 읽으면 파일 전체를 가로지르는 표본이 된다.
HfFileSystem 이 range 요청을 하므로 수 GB 짜리 샤드를 통째로 받지 않는다.

`datasets` 라이브러리를 쓰지 않는다. 5.x 가 스크립트 데이터셋을 못 읽어서
저장소마다 되고 안 되고가 갈리는데, 파케이를 직접 읽으면 그 차이가 없다.
"""

from __future__ import annotations

from typing import Iterator, Sequence


def list_parquet(repo: str, prefix: str) -> list:
    """저장소에서 prefix 로 시작하는 파케이 경로를 정렬해 돌려준다."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files
                  if f.startswith(prefix) and f.endswith(".parquet"))


def stream_parquet(
    repo: str,
    paths: Sequence[str],
    columns: Sequence[str],
    max_docs: int,
    max_bytes: int,
    text_column: str = "text",
) -> Iterator[dict]:
    """여러 파케이 파일에 걸쳐 row group 을 일정 간격으로 골라 읽는다.

    `paths` 가 여러 개면 각 파일에서 고르게 나눠 읽는다. 한 파일만 주면
    그 파일 전체를 가로지른다.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    if not paths:
        raise ValueError(f"{repo}: 읽을 파케이 파일이 없다")

    fs = HfFileSystem()
    rows_per_group = 1000
    want_groups = max(1, -(-max_docs // rows_per_group))
    per_file = max(1, -(-want_groups // len(paths)))

    seen_bytes = 0
    emitted = 0
    for path in paths:
        with fs.open(f"datasets/{repo}/{path}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            total = pf.metadata.num_row_groups
            step = max(1, total // per_file)
            available = [c for c in columns if c in pf.schema_arrow.names]
            for rg in list(range(0, total, step))[:per_file]:
                table = pf.read_row_group(rg, columns=available)
                for i, row in enumerate(table.to_pylist()):
                    if emitted >= max_docs or seen_bytes >= max_bytes:
                        return
                    text = row.get(text_column) or ""
                    seen_bytes += len(text.encode("utf-8"))
                    emitted += 1
                    row["_source_path"] = path
                    row["_fallback_id"] = f"{path}#{rg}:{i}"
                    yield row
