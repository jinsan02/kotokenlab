"""Final Test 도구 — heldout_spec · doc_nll --split heldout · heldout_pairs · heldout_preflight.

이 테스트는 그 평가 파일을 열지 않는다. 모든 입력은 가짜다.
"""

import subprocess
from types import SimpleNamespace

import pytest

from tools import heldout_spec as S


def test_run_ids_follow_make_run_id_and_avoid_lockbox_word():
    from src.utils.tracking import make_run_id
    for n, (path, _) in S.CHECKPOINTS.items():
        rid = S.doc_run_id(n)
        assert rid == make_run_id("eval", "docnll", path.rsplit("/", 1)[-1], S.TAG)
        assert "final_test" not in rid


def test_spec_matches_plan_d():
    from tools.heldout_preflight import spec_problems
    assert spec_problems() == []


def test_plan_files_dev_is_unchanged_and_heldout_is_fixed():
    from src.evaluation.doc_nll import DEV, plan_files
    files, label = plan_files("dev", 1234)
    assert label == "dev" and files == [(lang, f, 1234) for lang, f in DEV]
    files, label = plan_files("heldout", 1234)          # --max-bytes 는 무시된다
    assert label == "final_test" and files == list(S.FILES)


def _fake_git(tag: str):
    def run(cmd, **kw):
        return SimpleNamespace(stdout=tag, returncode=0)
    return run


def test_guard_refuses_without_opening_tag(monkeypatch):
    from src.evaluation import doc_nll
    monkeypatch.setattr(subprocess, "run", _fake_git(""))
    args = SimpleNamespace(tag=S.TAG, seq_len=S.SEQ_LEN, model=S.CHECKPOINTS["c0"][0])
    with pytest.raises(SystemExit, match="태그가 없다"):
        doc_nll.heldout_guard(args)


def test_guard_refuses_other_settings_even_after_opening(monkeypatch):
    from src.evaluation import doc_nll
    monkeypatch.setattr(subprocess, "run", _fake_git("final-test-opened"))
    ok = SimpleNamespace(tag=S.TAG, seq_len=S.SEQ_LEN, model=S.CHECKPOINTS["t2b"][0])
    doc_nll.heldout_guard(ok)                               # 통과
    for bad in (dict(tag="x"), dict(seq_len=1024), dict(model="artifacts/models/other")):
        with pytest.raises(SystemExit):
            doc_nll.heldout_guard(SimpleNamespace(**{**vars(ok), **bad}))


def test_verdict_is_three_valued():
    from tools.heldout_pairs import verdict
    assert verdict(0.01) == "확증"
    assert verdict(0.0) == "미확증" and verdict(-0.2) == "미확증"
    assert verdict(None) is None


def _docs(n, raw=100):
    return [(i, 1.0, raw - 1, raw) for i in range(n)]


def test_validate_reports_missing_runs_budget_and_mismatch(monkeypatch):
    import tools.heldout_pairs as H
    monkeypatch.setattr(H, "checkpoint_hash_ok", lambda n: True)
    full = {lang: _docs(b // 100 + 1) for lang, _, b in S.FILES}
    ok = {S.doc_run_id(n): {"note": "doc_nll ledger_check=none(heldout)"} for n in S.CHECKPOINTS}
    assert H.validate({"c0": full, "t2b": full}, ok) == []
    assert any("원장 ok 행 없음" in p for p in H.validate({"c0": full, "t2b": full}, {}))
    short = {**full, "ko": _docs(3)}
    assert any("예산" in p for p in H.validate({"c0": full, "t2b": short}, ok))
    other = {**full, "en": [(i, 1.0, 99, 101) for i in range(len(full["en"]))]}
    assert any("짝 불일치" in p for p in H.validate({"c0": full, "t2b": other}, ok))
    monkeypatch.setattr(H, "checkpoint_hash_ok", lambda n: False)
    assert any("해시" in p for p in H.validate({"c0": full, "t2b": full}, ok))
