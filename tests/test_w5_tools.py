"""5주차 도구 — doc_bootstrap · exposure_strata · w5_spec · w5_preflight."""

import math

import pytest

from tools import w5_spec as S
from tools.doc_bootstrap import pair_problem, reproduced
from tools.exposure_strata import bins_for
from tools.w5_preflight import argv_diff


def _docs(nll, nbytes):
    return [(i, n, b, b + 1) for i, (n, b) in enumerate(zip(nll, nbytes))]


def test_bootstrap_point_is_ratio_of_sums_and_ci_brackets_it():
    pytest.importorskip("numpy")
    from tools.doc_bootstrap import paired_bootstrap
    a = _docs([10.0, 20.0, 15.0, 30.0] * 10, [100, 150, 120, 200] * 10)
    b = _docs([9.0, 18.0, 14.0, 26.0] * 10, [100, 150, 120, 200] * 10)
    r = paired_bootstrap(a, b, 2000, 0)
    want = (sum(x[1] for x in a) - sum(x[1] for x in b)) / (math.log(2) * sum(x[2] for x in a))
    assert abs(r["diff"] - want) < 1e-12
    assert r["lo"] <= r["diff"] <= r["hi"] and r["n"] == 40


def test_pairing_refuses_different_documents():
    a = _docs([1.0, 2.0], [10, 20])
    b = [(0, 1.0, 10, 11), (1, 2.0, 25, 21)]  # 채점 바이트는 토크나이저마다 달라도 된다
    assert pair_problem(a, b) is None
    c = [(0, 1.0, 10, 11), (1, 2.0, 20, 99)]  # 원문 바이트가 다르면 다른 문서다
    assert "원문 바이트" in pair_problem(a, c)
    assert "문서 수" in pair_problem(a, a[:1])


def test_reproduced_is_three_valued():
    assert reproduced("doc_nll 3행 ko_vs_ledger=+0.000000 en_vs_ledger=-0.000001", "ko") is True
    assert reproduced("ko_vs_ledger=+0.000000 en_vs_ledger=-0.000001", "en") is False
    assert reproduced("doc_nll 3행", "ko") is None


def test_reproduction_is_checked_against_ledger_directly(monkeypatch):
    """note 가 비어 있어도(2026-10-07 CRLF 사고) 문서별 합을 원장과 직접 대조한다."""
    import tools.doc_bootstrap as D
    rows = [(0, 1.0, 10, 11), (1, 3.0, 30, 31)]
    good = round(4.0 / (math.log(2) * 40), 6)
    monkeypatch.setattr(D, "ledger_final", lambda rid: {"ko": good, "en": good + 0.001})
    docs = {"ko": rows, "en": rows}
    assert D.reproduction("c0_cos", docs) == {"ko": True, "en": False, "code": None}
    assert D.status("c0_cos", docs).startswith("원장 재현 실패: en")


def test_bins_use_median_of_nonzero_counts():
    names, m = bins_for([0, 0, 1, 2, 3, 10, 0])
    assert m == 2.5
    assert names == ["0", "0", "low", "low", "high", "high", "0"]
    assert bins_for([0, 0])[0] == ["0", "0"]


def test_ft_argv_differs_from_r5_only_by_tag_and_save():
    r5 = ("--model artifacts/models/kot2b_v2_n30000_mean --name t2b_mean --budget-bytes 168500000 "
          "--pool-docs 50000 --eval-bytes 20000000 --eval-budget 2000000 --lr-schedule constant "
          "--seed 42 --tag r5")
    assert argv_diff(S.FT_RUNS["cpt_t2b_mean_ft_seed42"][1], r5) == ["--save", "--tag"]


def test_doc_run_id_matches_make_run_id():
    from src.utils.tracking import make_run_id
    for n, (path, _) in S.CHECKPOINTS.items():
        assert S.doc_run_id(n) == make_run_id("eval", "docnll", path.rsplit("/", 1)[-1], S.EVAL_TAG)


def test_every_pair_and_strata_name_is_a_known_checkpoint():
    for _, a, b, _ in S.PAIRS:
        assert a in S.CHECKPOINTS and b in S.CHECKPOINTS
    assert all(n in S.CHECKPOINTS for n in S.STRATA_PAIR)
