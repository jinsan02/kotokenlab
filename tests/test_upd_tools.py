"""tools/upd_spec · upd_preflight · upd_pairs — 3주차 동결 설정과 그 집행."""

import numpy as np

from tools import upd_spec as S
from tools.upd_pairs import eval_at_row, rho
from tools.upd_preflight import Stream, ctrl_updates, parse_argv


def test_spec_budget_is_whole_updates():
    assert S.TOKENS_PER_UPDATE == 32_768
    assert S.budget_tokens(42) == 1523 * 32_768 == 49_905_664
    assert S.budget_tokens(123) == S.budget_tokens(2026) == 1524 * 32_768


def test_spec_argv_matches_ledger_format():
    s = S.argv_str(42)
    a = parse_argv(s)
    assert a["--budget-tokens"] == "49905664" and a["--tag"] == "upd"
    assert a["--pool-extend-docs"] == "20000" and a["--eval-at"] == "168500000"
    assert a["--save"] is True and "--budget-bytes" not in a
    assert S.run_id(42) == "cpt_t2b_mean_upd_seed42"


def test_classify_boundaries_are_inclusive_as_registered():
    assert S.classify(0.25) == "예측 적중"
    assert S.classify(0.2501) == "부분 설명"
    assert S.classify(0.50) == "부분 설명"
    assert S.classify(0.5001) == "헤드라인 변경"
    assert S.classify(None) == "판정 불가"


def test_rho_refuses_nonpositive_denominator():
    assert abs(rho(0.4, 0.3) - 0.25) < 1e-12
    assert rho(0.0, 0.1) is None and rho(-0.1, 0.1) is None and rho(None, 0.1) is None


def test_ctrl_updates_floors_old_runs():
    assert ctrl_updates({"updates": "1524", "tokens_seen": "49938432"}) == (1524, "원장 updates")
    u, src = ctrl_updates({"updates": "NA", "tokens_seen": "49922048"})
    assert u == 1523 and "내림" in src


def _row(ck, raw, dom="ko"):
    return {"run_id": "r", "checkpoint": ck, "raw_bytes_seen": str(raw), "domain": dom, "bpb": "1.5"}


def test_eval_at_row_does_not_fall_back_to_next_grid_point():
    rows = [_row("step1010", 160_000_000), _row("step1136", 180_000_000), _row("final", 241_000_000)]
    assert eval_at_row(rows, "r") is None          # --eval-at 을 빠뜨린 run
    rows.append(_row("step1063", 168_502_276))
    assert eval_at_row(rows, "r")["checkpoint"] == "step1063"
    rows.append(_row("step1064", 168_660_000))     # 둘이면 어느 것인지 모른다
    assert eval_at_row(rows, "r") is None


class _FakeTok:
    """pack() 이 부르는 형태: tokenizer(text, add_special_tokens=False)["input_ids"]."""

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self(t)["input_ids"] for t in text]}
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


def test_stream_matches_real_pack_byte_for_byte():
    from src.training.cpt import pack
    rng = np.random.default_rng(0)
    docs = ["".join(chr(97 + int(x)) for x in rng.integers(0, 26, int(n)))
            for n in rng.integers(0, 40, 200)]
    docs[3] = ""                                   # 토큰 없는 문서는 pack 이 건너뛴다
    table = np.arange(60, dtype=np.int64) % 4 + 1  # 토큰마다 1~4 바이트
    eos = 0
    table[eos] = 0
    tok = _FakeTok()
    seq = 16
    stream = pack(tok, docs, seq, eos, lambda ids: int(table[ids].sum()))
    real_tokens, real_bytes, pairs = 0, 0, []
    for chunk, take in stream:
        real_tokens += len(chunk)
        real_bytes += take
        pairs.append((real_tokens, real_bytes))

    arrays = [table[np.asarray(tok(d)["input_ids"], dtype=np.int64)] if d else None for d in docs]
    st = Stream(arrays, list(range(len(docs))), seq)
    assert st.full_tokens == real_tokens
    for n, b in pairs:
        assert st.bytes_at(n) == b


def test_stream_first_update_and_exhaustion():
    arrays = [np.full(9, 2, dtype=np.int64) for _ in range(10)]   # 문서당 9토큰+EOS = 10
    st = Stream(arrays, list(range(10)), seq_len=10)
    assert st.full_tokens == 100
    assert st.bytes_at(20) == 36                    # 두 문서 x 9토큰 x 2바이트
    # update 1 = 20토큰 = 36바이트 < 40, update 2 = 40토큰 = 72바이트 >= 40
    assert st.first_update_at(40, tpu=20, max_updates=5) == 2
    assert st.first_update_at(10_000, tpu=20, max_updates=5) is None
