"""스펙 §59 — 실험 타임스탬프의 외부 시각 검증과 계보 연결.

UTC 문자열만 남기면 호스트 시계가 틀렸을 때도 그럴듯한 기록이 만들어진다.
캐시되지 않은 HTTPS 응답의 Date 헤더와 로컬 왕복 구간의 중간값을 비교해
오차를 기록한다. 이 도구는 운영체제 시계를 변경하지 않는다.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from . import ledger
from .hashing import sha256_obj

DEFAULT_SERVER = "https://huggingface.co/api/models?limit=1"
DEFAULT_MAX_OFFSET_MS = 5_000.0
DEFAULT_MAX_AGE_HOURS = 24.0


def _iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _with_nonce(url: str, nonce: str) -> str:
    parts = urlsplit(url)
    query = parts.query + ("&" if parts.query else "") + urlencode({"clock_probe": nonce})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _windows_time_source() -> str:
    if os.name != "nt":
        return "non_windows"
    try:
        proc = subprocess.run(
            ["w32tm", "/query", "/status"], capture_output=True, timeout=10,
            check=False,
        )
        raw = (proc.stdout + proc.stderr).decode(errors="replace")
        if "Local CMOS Clock" in raw:
            return "local_cmos_clock"
        if proc.returncode == 0:
            return "w32time_synchronized_source"
        return f"w32tm_unavailable_{proc.returncode}"
    except (OSError, subprocess.SubprocessError):
        return "w32tm_unavailable"


def evaluate_probe(
    server_epoch: float,
    local_before: float,
    local_after: float,
    *,
    max_offset_ms: float = DEFAULT_MAX_OFFSET_MS,
) -> dict[str, Any]:
    """서버 시각과 로컬 요청 구간으로 오차·RTT·상태를 계산한다."""
    midpoint = (local_before + local_after) / 2.0
    offset_ms = (server_epoch - midpoint) * 1_000.0
    rtt_ms = (local_after - local_before) * 1_000.0
    return {
        "status": "ok" if abs(offset_ms) <= max_offset_ms else "fail",
        "server_utc": _iso_utc(server_epoch),
        "local_midpoint_utc": _iso_utc(midpoint),
        "offset_ms": round(offset_ms, 3),
        "rtt_ms": round(rtt_ms, 3),
    }


def probe_https_time(
    server: str = DEFAULT_SERVER,
    *,
    max_offset_ms: float = DEFAULT_MAX_OFFSET_MS,
    timeout: float = 15.0,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """캐시 우회를 넣은 HTTPS 요청으로 외부 시각을 한 번 검증한다."""
    url = _with_nonce(server, str(time.time_ns()))
    request = Request(
        url,
        headers={"Cache-Control": "no-cache", "User-Agent": "KoTokenLab-clock-check/1"},
        method="GET",
    )
    before = time.time()
    try:
        with opener(request, timeout=timeout) as response:
            date_header = response.headers.get("Date")
            age_header = response.headers.get("Age")
        after = time.time()
        if not date_header:
            raise RuntimeError("HTTPS 응답에 Date 헤더가 없다")
        if age_header not in (None, "0"):
            raise RuntimeError(f"캐시된 응답이다 (Age={age_header})")
        server_dt = parsedate_to_datetime(date_header)
        server_epoch = server_dt.astimezone(timezone.utc).timestamp()
        row = evaluate_probe(
            server_epoch, before, after, max_offset_ms=max_offset_ms,
        )
        row["note"] = "HTTPS Date 헤더는 초 단위이므로 5초 이내 여부만 판정한다"
    except Exception as exc:
        after = time.time()
        row = {
            "status": "fail", "server_utc": ledger.NA,
            "local_midpoint_utc": _iso_utc((before + after) / 2.0),
            "offset_ms": ledger.NA, "rtt_ms": round((after - before) * 1_000.0, 3),
            "note": f"{type(exc).__name__}: {exc}",
        }
    row.update({
        "server": server,
        "windows_source": _windows_time_source(),
        "method": "https_date",
    })
    identity = {k: row[k] for k in (
        "status", "server", "server_utc", "local_midpoint_utc", "offset_ms",
        "rtt_ms", "windows_source", "method", "note",
    )}
    row["clock_check_sha256"] = sha256_obj(identity)
    return row


def record_check(row: dict[str, Any], root: Path | str | None = None) -> Path:
    return ledger.append_row("clock_checks", row, root)


def latest_valid_check(
    root: Path | str | None = None,
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    now_epoch: float | None = None,
) -> dict[str, str] | None:
    """최근 max_age_hours 안의 성공한 검사를 돌려준다."""
    now = time.time() if now_epoch is None else now_epoch
    for row in reversed(ledger.read_rows("clock_checks", root)):
        if row.get("status") != "ok":
            continue
        try:
            checked = datetime.strptime(row["ts_utc"], "%Y-%m-%dT%H:%M:%SZ")
            checked = checked.replace(tzinfo=timezone.utc).timestamp()
        except (KeyError, TypeError, ValueError):
            continue
        age_hours = (now - checked) / 3600.0
        if 0 <= age_hours <= max_age_hours:
            return row
    return None


def require_recent_check(
    root: Path | str | None = None,
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> str:
    row = latest_valid_check(root, max_age_hours=max_age_hours)
    if row is None:
        raise RuntimeError(
            "최근 시간 검증 기록이 없다. 실험 전에 "
            "C:\\llm_tokenizer\\.conda\\python.exe tools/check_clock.py --record 를 실행하라."
        )
    return row["clock_check_sha256"]
