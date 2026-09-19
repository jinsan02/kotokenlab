"""pre-commit 훅이 실제로 거부하는지 확인한다.

2026-09-14 에 훅이 **통과시킨** 커밋이 있었다 — `fix(infra)` 하나가 스테이지에
남아 있던 R1 의 원장 행·메트릭·run 디렉터리를 통째로 삼켰다. 훅은 `record:`
커밋에 코드가 섞이는 것만 막고 반대 방향을 안 봤기 때문이다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.precheck import (  # noqa: E402
    is_code,
    mixed_results_and_code,
    result_rows_added,
)

TAB = chr(9)
NL = chr(10)
HEADER = TAB.join(("ts_utc", "run_id", "phase", "status"))


def _tsv(*rows: tuple) -> str:
    return NL.join([HEADER] + [TAB.join(r) for r in rows])


def test_result_rows_added_sees_only_new_ok_and_fail():
    head = _tsv(("t0", "old_run", "cpt", "ok"))
    staged = _tsv(
        ("t0", "old_run", "cpt", "ok"),      # 이미 있던 행
        ("t1", "new_run", "cpt", "ok"),      # 새 결과
        ("t2", "dead_run", "cpt", "fail"),   # 실패도 결과다
        ("t3", "live_run", "cpt", "start"),  # 아직 도는 중 — 결과 아님
    )
    assert result_rows_added(staged, head) == ["new_run", "dead_run"]


def test_start_only_is_not_a_result():
    """학습이 도는 동안 문서를 커밋하는 것은 막지 않는다 (실제로 겪은 오탐)."""
    head = _tsv()
    staged = _tsv(("t1", "live_run", "cpt", "start"))
    assert result_rows_added(staged, head) == []
    assert mixed_results_and_code([], ["src/a.py"], "L.tsv") == []


def test_results_with_code_is_rejected():
    errs = mixed_results_and_code(["r1", "r2"], ["src/utils/tracking.py"], "L.tsv")
    assert len(errs) == 1
    assert "나눠라" in errs[0]


def test_results_alone_is_fine():
    assert mixed_results_and_code(["r1"], [], "L.tsv") == []


def test_code_alone_is_fine():
    assert mixed_results_and_code([], ["src/a.py"], "L.tsv") == []


def test_is_code_classification():
    for p in ("src/utils/env.py", "tools/x.py", "scripts/run.ps1",
              "tests/test_a.py", "configs/cpt/base.yaml"):
        assert is_code(p), p
    for p in ("docs/PLAN.md", "reports/tables/cpt_main.md",
              "experiments/LEDGER.tsv", "README.md"):
        assert not is_code(p), p


def test_precheck_and_hook_and_generator_share_run_id_rule():
    """규칙이 세 곳에 있다. 갈리면 run 을 다 돌린 뒤 커밋에서 거부당한다.

    2026-09-14 R1 대조군이 정확히 그렇게 됐다 — 40분을 쓰고 나서 막혔고,
    원장은 append-only 라 되돌릴 수도 없었다.
    """
    from src.utils.tracking import RUN_ID_RE as GEN
    from tools.validate_ledger import RUN_ID_RE as VAL

    root = Path(__file__).resolve().parents[1]
    src = (root / "tools" / "check_commit_msg.py").read_text(encoding="utf-8")
    m = re.search(r'RUN_ID_RE\s*=\s*re\.compile\(r"([^"]+)"\)', src)
    assert m, "훅에서 RUN_ID_RE 를 못 찾았다"
    assert m.group(1) == GEN.pattern == VAL.pattern, (
        f"훅 {m.group(1)!r} / 생성기 {GEN.pattern!r} / 검증기 {VAL.pattern!r}")


def test_legacy_run_id_exemption_is_frozen():
    """면제 목록은 늘어나면 안 된다. 생성기가 막으므로 새로 생길 수 없다."""
    from tools.validate_ledger import LEGACY_RUN_IDS

    assert LEGACY_RUN_IDS == frozenset({"cpt_Qwen2.5-0.5B_r1ctrl_seed42"})


def test_compare_runs_classifies_p3_fields():
    """새 필드를 분류하지 않으면 "미분류 = 치명" 으로 모든 비교가 막힌다.

    warmup_bytes · pool_extend_docs 는 학습 경로를 바꾸므로 치명,
    eval_at · save_at 은 벽시계만 바꾼다 (docs/SPEC_P3.md §3 W0-10).
    """
    from tools.compare_runs import CRITICAL, IDENTITY, TIMING

    assert {"warmup_bytes", "pool_extend_docs"} <= CRITICAL
    assert {"eval_at", "save_at"} <= TIMING
    assert not (CRITICAL & TIMING) and not (CRITICAL & IDENTITY)


def test_cpt_config_fields_are_all_classified():
    """cpt.py 가 config 에 남기는 필드는 전부 분류돼 있어야 한다."""
    import re

    from tools.compare_runs import CRITICAL, IDENTITY, TIMING

    src = (Path(__file__).resolve().parents[1] / "src" / "training" / "cpt.py").read_text(
        encoding="utf-8")
    body = src[src.index("    config = {"):src.index("    run_id = make_run_id(\"cpt\"")]
    keys = set(re.findall(r'"([a-z_0-9]+)":', body))
    known = CRITICAL | TIMING | IDENTITY
    assert keys <= known, f"분류되지 않은 config 필드: {sorted(keys - known)}"
