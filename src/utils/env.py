"""가상환경을 실험 변수로 다룬다 (스펙 §60).

환경은 "설치하고 잊는 것"이 아니다. torch 나 transformers 를 올리면 결과가
달라질 수 있고, 그 사실이 원장에 남아 있지 않으면 나중에 두 실험을 비교할 수
없다. 그래서 다음을 강제한다.

    현재 환경의 env_sha256 이 env/ENV_SNAPSHOT.tsv 에 없으면 실험을 시작할 수 없다.

패키지를 하나라도 바꿨다면 먼저

    .conda/python.exe -m src.utils.env --register "왜 바꿨는지"

로 스냅샷 행을 추가하고 chore(infra) 커밋을 한 뒤에 실험을 돌린다.

.conda/ 자체는 git 에 들어가지 않는다. 커밋되는 것은
env/environment.yml, env/requirements-lock.txt, env/ENV_SNAPSHOT.tsv 셋이다.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from . import ledger
from .hashing import sha256_obj

# env_sha256 계산에 들어가는 키. 여기에 없는 값(타임스탬프·메모)은 해시에 영향이 없다.
_HASH_KEYS: tuple[str, ...] = (
    "python", "torch", "cuda", "cudnn",
    "transformers", "tokenizers", "datasets", "accelerate",
    "bitsandbytes", "peft", "numpy",
    "driver", "gpu_name", "vram_mb",
)

_PACKAGES: tuple[str, ...] = (
    "transformers", "tokenizers", "datasets", "accelerate",
    "bitsandbytes", "peft", "numpy",
)


class EnvironmentNotRegistered(RuntimeError):
    """현재 환경이 ENV_SNAPSHOT.tsv 에 등록되어 있지 않다."""


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return ledger.NA
    try:
        return version(name)
    except PackageNotFoundError:
        return ledger.NA
    except Exception:
        return ledger.NA


def _nvidia_driver() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ledger.NA
    if out.returncode != 0 or not out.stdout.strip():
        return ledger.NA
    return out.stdout.strip().splitlines()[0].strip()


def collect() -> dict:
    """지금 이 인터프리터가 보고 있는 환경을 수집한다."""
    snapshot = {
        "python": platform.python_version(),
        "torch": ledger.NA,
        "cuda": ledger.NA,
        "cudnn": ledger.NA,
        "gpu_name": ledger.NA,
        "vram_mb": ledger.NA,
        "driver": _nvidia_driver(),
    }
    for pkg in _PACKAGES:
        snapshot[pkg] = _pkg_version(pkg)

    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda"] = torch.version.cuda or ledger.NA
        cudnn = getattr(torch.backends.cudnn, "version", lambda: None)()
        snapshot["cudnn"] = str(cudnn) if cudnn else ledger.NA
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            snapshot["gpu_name"] = props.name
            snapshot["vram_mb"] = str(props.total_memory // (1024 * 1024))
    except ImportError:
        pass

    return snapshot


def env_sha256(snapshot: dict | None = None) -> str:
    """환경 스냅샷의 해시. 버전 하나만 달라져도 값이 바뀐다."""
    snap = snapshot if snapshot is not None else collect()
    return sha256_obj({k: snap.get(k, ledger.NA) for k in _HASH_KEYS})


def registered_hashes(root: Path | str | None = None) -> set:
    return {r.get("env_sha256", "") for r in ledger.read_rows("env_snapshot", root)}


def is_registered(root: Path | str | None = None, sha: str | None = None) -> bool:
    return (sha or env_sha256()) in registered_hashes(root)


def register(change_note: str, root: Path | str | None = None) -> str:
    """현재 환경을 ENV_SNAPSHOT.tsv 에 한 행으로 등록한다. 이미 있으면 그대로 둔다."""
    snapshot = collect()
    sha = env_sha256(snapshot)
    if sha in registered_hashes(root):
        return sha
    row = dict(snapshot)
    row["env_sha256"] = sha
    row["change_note"] = change_note or "NA"
    ledger.append_row("env_snapshot", row, root)
    return sha


def require_registered(root: Path | str | None = None) -> str:
    """등록되지 않은 환경이면 실행을 막는다. RunContext 가 진입 시 호출한다."""
    snapshot = collect()
    sha = env_sha256(snapshot)
    if sha in registered_hashes(root):
        return sha
    versions = "\n".join(f"    {k:<14} {snapshot.get(k, ledger.NA)}" for k in _HASH_KEYS)
    raise EnvironmentNotRegistered(
        "현재 환경이 env/ENV_SNAPSHOT.tsv 에 등록되어 있지 않다.\n"
        f"  env_sha256 = {sha}\n{versions}\n\n"
        "환경이 바뀌었다면 아래를 실행하고 chore(infra) 로 커밋한 뒤 다시 시도하라.\n"
        '    .conda/python.exe -m src.utils.env --register "무엇을 왜 바꿨는지"'
    )


def _main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="환경 스냅샷 조회/등록")
    parser.add_argument("--register", metavar="NOTE",
                        help="현재 환경을 ENV_SNAPSHOT.tsv 에 등록한다")
    parser.add_argument("--check", action="store_true",
                        help="등록 여부만 확인한다 (미등록이면 종료코드 1)")
    args = parser.parse_args(argv)

    snapshot = collect()
    sha = env_sha256(snapshot)
    print(f"env_sha256  {sha}")
    for key in _HASH_KEYS:
        print(f"  {key:<14} {snapshot.get(key, ledger.NA)}")

    if args.register is not None:
        register(args.register)
        print(f"\n등록 완료 -> {ledger.table_path('env_snapshot')}")
        return 0
    if args.check:
        if is_registered(sha=sha):
            print("\n등록됨")
            return 0
        print("\n미등록 — --register 로 먼저 기록하라")
        return 1
    print(f"\n등록됨: {is_registered(sha=sha)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
