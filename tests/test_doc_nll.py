"""bpb.evaluate 의 docs_out — 문서별 값을 모아도 합계가 바뀌지 않는다."""

import json

import pytest


def _setup(tmp_path):
    torch = pytest.importorskip("torch")

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 20 + 1 for c in text]}

        def decode(self, ids):
            return "x" * len(ids)

    class Out:
        def __init__(self, logits):
            self.logits = logits

    class Model:
        def __init__(self):
            g = torch.Generator().manual_seed(0)
            self.table = torch.randn(32, 32, generator=g)

        def __call__(self, x):
            return Out(self.table[x])

    p = tmp_path / "dev.jsonl"
    docs = ["hello world " * 5, "", "abc", "korean text here " * 9, "z"]
    p.write_text("\n".join(json.dumps({"text": t}) for t in docs if t is not None) + "\n",
                 encoding="utf-8")
    return Tok(), Model(), p


def test_docs_out_does_not_change_totals_and_sums_match(tmp_path):
    from src.evaluation.bpb import evaluate
    tok, model, p = _setup(tmp_path)
    blen = lambda ids: sum(1 + (i % 3) for i in ids)  # noqa: E731
    a = evaluate(model, tok, p, 10_000, 8, "cpu", blen)
    rows: list = []
    b = evaluate(model, tok, p, 10_000, 8, "cpu", blen, docs_out=rows)
    assert a == b                                            # 반환값이 비트 단위로 같다
    assert len(rows) == a["n_docs"]
    assert abs(sum(r["nll"] for r in rows) - a["total_nll"]) < 1e-6
    assert sum(r["bytes"] for r in rows) == a["n_bytes"]
    assert sum(r["tokens"] for r in rows) == a["n_tokens"]
    assert [r["doc"] for r in rows] == list(range(len(rows)))


def test_budget_stops_at_the_same_document(tmp_path):
    from src.evaluation.bpb import evaluate
    tok, model, p = _setup(tmp_path)
    blen = lambda ids: len(ids)  # noqa: E731
    rows: list = []
    m = evaluate(model, tok, p, 30, 8, "cpu", blen, docs_out=rows)
    assert m["n_docs"] == len(rows) == 1                     # 첫 문서가 60바이트라 거기서 멈춘다
