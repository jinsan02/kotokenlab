"""6주차 도구 — w6_spec · wsd_pairs · w6_preflight (WSD, P3-B)."""

import inspect

from tools import w5_spec as W5
from tools import w6_spec as S
from tools.w5_preflight import argv_diff
from tools.wsd_pairs import bound, predecay_repro, smoke_check, verdict


def test_wsd_argv_differs_from_retrain_only_by_schedule_and_tag():
    for rid, (base, argv) in S.WSD_RUNS.items():
        assert argv_diff(argv, W5.FT_RUNS[base][1]) == ["--lr-schedule", "--tag"]
        assert "--lr-schedule wsd" in argv and "--save" in argv


def test_run_ids_match_make_run_id():
    import shlex
    from src.utils.tracking import make_run_id
    for rid, (_, argv) in list(S.WSD_RUNS.items()) + [(S.SMOKE_RUN, (None, S.SMOKE_ARGV))]:
        t = shlex.split(argv)
        name, tag, seed = (t[t.index(k) + 1] for k in ("--name", "--tag", "--seed"))
        assert make_run_id("cpt", name, tag, seed=int(seed)) == rid


def test_schedule_defaults_are_frozen():
    """config 에는 스케줄 이름만 남는다 — 기본값이 바뀌면 같은 이름으로 다른 스케줄이 돈다."""
    from src.training.callbacks import wsd_lr_by_bytes
    p = inspect.signature(wsd_lr_by_bytes).parameters
    assert p["decay_frac"].default == S.DECAY_FRAC == 0.2
    assert p["min_frac"].default == S.MIN_FRAC == 0.1
    assert S.DECAY_START == 134_800_000
    assert wsd_lr_by_bytes(S.BUDGET, S.BUDGET, S.PEAK_LR, S.WARMUP_FRAC) == \
        S.PEAK_LR * S.MIN_FRAC


def test_verdict_is_three_valued_with_floor():
    assert bound(0.0001) == S.FLOOR and bound(0.01) == 0.02 and bound(None) == S.FLOOR
    assert verdict(-0.006, 0.005) == "개선"
    assert verdict(0.004, 0.005) == "구별 불가" and verdict(-0.005, 0.005) == "구별 불가"
    assert verdict(0.006, 0.005) == "악화"
    assert verdict(None, 0.005) is None


def _lm(run, pts):
    return [{"run_id": run, "checkpoint": ck, "raw_bytes_seen": str(raw), "domain": "ko",
             "bpb": str(v), "split": "dev"} for ck, raw, v in pts]


def test_predecay_repro_uses_only_points_before_decay():
    base = _lm("b", [("step0", 0, 2.38), ("step100", 20_000_000, 1.74),
                     ("step900", 140_000_000, 1.48)])
    good = _lm("w", [("step0", 0, 2.38), ("step100", 20_000_000, 1.7405),
                     ("step900", 140_000_000, 1.40)])   # 감쇠 뒤 지점은 달라도 된다
    ok, rows = predecay_repro(base + good, "w", "b")
    assert ok is True and len(rows) == 2
    bad = _lm("w", [("step0", 0, 2.38), ("step100", 20_000_000, 1.75)])
    assert predecay_repro(base + bad, "w", "b")[0] is False
    assert predecay_repro(base, "w", "b")[0] is None
    orphan = _lm("w", [("step55", 10_000_000, 1.8)])          # 기준에 없는 지점
    assert predecay_repro(base + orphan, "w", "b")[0] is None


def test_smoke_check_compares_lr_with_schedule(monkeypatch):
    from src.training.callbacks import wsd_lr_by_bytes
    pts = [1_000_000, 2_000_000, 2_700_000, 3_000_000]

    def rows(f=lambda x: x):
        return [{"run_id": S.SMOKE_RUN, "raw_bytes_seen": str(r),
                 "lr": repr(f(wsd_lr_by_bytes(r, S.SMOKE_BUDGET, S.PEAK_LR, S.WARMUP_FRAC)))}
                for r in pts]
    assert smoke_check(rows())[0] is True
    assert smoke_check(rows(lambda x: x * 1.01))[0] is False
    early = [r for r in rows() if int(r["raw_bytes_seen"]) <= 2_000_000]
    assert smoke_check(early)[0] is False          # 감쇠 경로를 안 지났다
    assert smoke_check([])[0] is None


def test_doc_pairs_name_known_checkpoints():
    for _, a, b in S.DOC_PAIRS:
        assert a in S.CHECKPOINTS and b in S.CHECKPOINTS
    assert S.doc_run_id("t2b_const") == W5.doc_run_id("t2b_const")
