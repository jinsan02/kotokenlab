"""tools/warm_spec · warm_preflight · warm_pairs — 4주차 동결 설정과 그 집행."""

from tools import warm_spec as W
from tools.upd_preflight import parse_argv
from tools.warm_pairs import delta, note_fields, switch_row
from tools.warm_preflight import config_diff


def test_argv_has_both_warm_flags_and_same_budget_as_direct():
    a = parse_argv(W.argv_str(42))
    assert a["--warm-rows"] == W.WARM_ROWS and a["--warm-bytes"] == "33700000"
    assert a["--budget-bytes"] == "168500000" and a["--eval-at"] == "33700000"
    assert a["--tag"] == "warm" and a["--save"] is True
    assert W.run_id(42) == "cpt_t2b_mean_warm_seed42"


def test_gate_boundary_is_inclusive():
    assert W.gate(0.010) == "통과"
    assert W.gate(0.0099) == "미달"
    assert W.gate(-0.02) == "미달"
    assert W.gate(None) == "판정 불가"


def test_delta_sign_positive_means_warm_better():
    assert abs(delta(0.346, 0.330) - 0.016) < 1e-12
    assert delta(0.346, 0.350) < 0
    assert delta(None, 0.3) is None


def test_note_fields_parse_what_cpt_writes():
    note = ("steps=1063 ko 2.3803->1.4500 lr=1e-05 seed=42 warm_end_step=213 "
            "warm_end_bytes=33796179 warm_end_tokens=6979584 warm_old_rows_max_diff=0.0e+00")
    f = note_fields(note)
    assert f == {"warm_end_step": 213, "warm_end_bytes": 33796179,
                 "warm_end_tokens": 6979584, "warm_old_rows_max_diff": 0.0}
    assert note_fields("")["warm_old_rows_max_diff"] is None


def test_switch_row_needs_exactly_one_row_at_the_boundary():
    def row(ck, raw):
        return {"run_id": "r", "checkpoint": ck, "raw_bytes_seen": str(raw),
                "domain": "ko", "bpb": "1.6"}
    rows = [row("step126", 20_000_000), row("step253", 40_000_000)]
    assert switch_row(rows, "r") is None                 # --eval-at 을 빠뜨린 run
    rows.append(row("step213", 33_796_179))
    assert switch_row(rows, "r")["checkpoint"] == "step213"


def test_config_diff_reports_missing_and_changed_critical_fields_only():
    planned = {"lr": 1e-5, "warm_bytes": 33_700_000, "eval_at": [1], "seed": 42}
    direct = {"lr": 1e-5, "eval_at": [], "seed": 42}
    diff = config_diff(planned, direct)
    assert diff == [("warm_bytes", 33_700_000, "<기록 없음>")]   # eval_at 은 시간 필드


def test_every_allowed_difference_has_a_reason():
    assert all(W.ALLOW_FIELDS.values()) and all(W.REVIEWED_COMMITS.values())
