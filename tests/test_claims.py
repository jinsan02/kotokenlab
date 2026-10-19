"""핵심 주장 표 — 상태 어휘 · 미확인 처리 · 다른 도구와 같은 숫자."""

import statistics

STATUSES = ("Final Test 확증", "Final Test 미확증", "dev 재현", "dev 판정", "dev 측정", "미확인")


def test_rows_have_known_status_and_unique_ids():
    from tools.claims import WITHDRAWN, rows
    rs = rows(compression=False)
    ids = [r["id"] for r in rs] + [w[0] for w in WITHDRAWN]
    assert len(ids) == len(set(ids))
    for r in rs:
        assert r["status"].startswith(STATUSES), r
        assert all(r[k] for k in ("claim", "value", "axis", "unc", "src"))


def test_heldout_claim_is_unknown_without_evaluations(monkeypatch):
    import tools.doc_bootstrap as D
    from tools.claims import claim_heldout
    monkeypatch.setattr(D, "load_docs_run", lambda rid: {})
    r = claim_heldout()
    assert r["status"].startswith("미확인") and r["value"] == "—"


def test_gap_claim_matches_seed_pair_numbers():
    from tools.claims import SEEDS, claim_gap_seeds
    from tools.wsd_pairs import bpb_at, lm_rows, sigmas
    lm = lm_rows()
    d = [bpb_at(lm, f"cpt_t2b_mean_r5_seed{s}", "final", "ko")
         - bpb_at(lm, f"cpt_c0_qwen_r5_seed{s}", "final", "ko") for s in SEEDS]
    v = claim_gap_seeds(lm)["value"]
    assert f"{statistics.mean(d):+.6f}" in v and f"{sigmas(lm)['sigma_d']:.6f}" in v
