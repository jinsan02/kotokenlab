"""해시 유틸 — 결과에서 원본까지 추적 가능하게 만드는 도구 (스펙 §59).

원장에 들어가는 해시는 전부 sha256 소문자 64자다.
config / tokenizer / manifest / env 네 가지 해시가 하나의 run 을 고정한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

_CHUNK = 1 << 20  # 1MB


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """텍스트 해시. 개행을 LF 로 정규화한 뒤 UTF-8 로 인코딩한다.

    정규화를 하지 않으면 같은 파일이 Windows 와 Linux 에서 다른 해시를 갖는다.
    """
    return sha256_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """파이썬 객체(dict/list/scalar)의 해시.

    키 정렬 + 공백 고정으로 정규화하므로, 같은 내용이면 항상 같은 해시가 나온다.
    config dict 와 환경 버전 dict 에 쓴다.
    """
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(canonical)


def sha256_dir(
    root: Path | str,
    patterns: Iterable[str] = ("**/*",),
    exclude_names: Iterable[str] = ("__pycache__",),
) -> str:
    """디렉토리 전체의 해시. 토크나이저 산출물 폴더에 쓴다.

    파일 경로(POSIX, 상대)와 내용 해시를 정렬해 이어붙인 뒤 다시 해시한다.
    파일명이 바뀌어도 해시가 바뀐다.
    """
    root = Path(root)
    exclude = set(exclude_names)
    entries: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            if not p.is_file() or p in seen:
                continue
            if any(part in exclude for part in p.relative_to(root).parts):
                continue
            seen.add(p)
            entries.append((p.relative_to(root).as_posix(), sha256_file(p)))
    entries.sort()
    return sha256_text("\n".join(f"{name}  {digest}" for name, digest in entries))


def is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def short(digest: str, n: int = 12) -> str:
    return digest[:n] if is_sha256(digest) else digest
