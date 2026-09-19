import csv

from tools.methodology_audit import exposure, recovery, steps


def test_recovery_uses_control_endpoint():
    assert recovery(2.0, 1.5, 1.0) == 0.5


def test_steps_parses_ledger_note():
    assert steps({"run_id": "x", "note": "steps=1523 ko 2.0->1.5"}) == 1523


def test_exposure_keeps_zero_sample_separate_from_scaled_bucket(tmp_path):
    path = tmp_path / "exposure.tsv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            delimiter="\t",
            fieldnames=("token_id", "token", "fires_sample", "fires_scaled"),
        )
        writer.writeheader()
        writer.writerows([
            {"token_id": 1, "token": "a", "fires_sample": 0, "fires_scaled": 0},
            {"token_id": 2, "token": "b", "fires_sample": 1, "fires_scaled": 8},
            {"token_id": 3, "token": "c", "fires_sample": 20, "fires_scaled": 160},
        ])
    assert exposure(path) == (3, 1, 2, 8)
