"""src/training/cpt.py — 신규 행 warm-start (amendment 4주차) 와 config 불변성."""

import csv
import shlex

import pytest

from src.training.cpt import WarmStart, build_config, build_parser
from src.utils.hashing import sha256_obj
from src.utils.ledger import repo_root


# torch 가 없는 CI 에서는 학습 테스트만 건너뛴다. config 테스트는 torch 없이 돈다.
def _torch():
    return pytest.importorskip("torch")


def TinyTied():
    """임베딩이 입력이자 출력(lm_head)인 작은 모델 — Qwen2.5-0.5B 의 tie 와 같은 구조."""
    torch = _torch()
    nn = torch.nn

    class _M(nn.Module):
        def __init__(self, vocab=12, dim=4):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.body = nn.Linear(dim, dim)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, x):
            h = torch.tanh(self.body(self.emb(x)))
            return h @ self.emb.weight.T

    return _M()


def _steps(model, opt, n=5):
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(0)
    for _ in range(n):
        x = torch.randint(0, 12, (8, 6))
        loss = nn.functional.cross_entropy(model(x).reshape(-1, 12), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_stage1_moves_only_new_rows_and_keeps_body_bitwise():
    torch = _torch()
    torch.manual_seed(1)
    m = TinyTied()
    new = [3, 7, 9]
    old = [i for i in range(12) if i not in new]
    emb0 = m.emb.weight.detach().clone()
    body0 = {k: v.detach().clone() for k, v in m.body.state_dict().items()}

    warm = WarmStart(m, m.emb.weight, new)
    opt = torch.optim.AdamW(warm.param_groups(0.1), lr=1e-2)
    _steps(m, opt)

    assert torch.equal(m.emb.weight.detach()[old], emb0[old])         # 옛 행: 비트 단위로 그대로
    assert not torch.equal(m.emb.weight.detach()[new], emb0[new])     # 새 행: 움직였다
    for k, v in m.body.state_dict().items():
        assert torch.equal(v, body0[k])                                # 몸통: 그대로

    info = warm.finish(opt)
    assert info == {"old_rows": 9, "new_rows": 3, "old_rows_max_diff": 0.0}
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert all(p.requires_grad for p in m.parameters())

    _steps(m, opt, 2)                                                  # 2단계: 전부 움직인다
    assert not torch.equal(m.emb.weight.detach()[old], emb0[old])
    assert not torch.equal(m.body.weight.detach(), body0["weight"])


def test_weight_decay_alone_would_move_old_rows():
    torch = _torch()
    """마스크만 하고 decay 를 안 끄면 옛 행이 줄어든다 — decay 끄기가 왜 필요한지."""
    torch.manual_seed(1)
    m = TinyTied()
    new = [3, 7, 9]
    old = [i for i in range(12) if i not in new]
    emb0 = m.emb.weight.detach().clone()
    warm = WarmStart(m, m.emb.weight, new)
    groups = warm.param_groups(0.1)
    groups[0]["weight_decay"] = 0.1                                   # 일부러 decay 를 켠다
    opt = torch.optim.AdamW(groups, lr=1e-2)
    _steps(m, opt)
    assert not torch.equal(m.emb.weight.detach()[old], emb0[old])
    with pytest.raises(RuntimeError, match="옛 행이 움직였다"):
        warm.finish(opt)


def test_rows_outside_vocab_are_refused():
    torch = _torch()
    m = TinyTied()
    with pytest.raises(SystemExit, match="임베딩 밖"):
        WarmStart(m, m.emb.weight, [3, 99])


def _ledger_argv(run_id: str):
    with (repo_root() / "experiments" / "LEDGER.tsv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["run_id"] == run_id and r["status"] == "ok":
                return r["argv"], r["config_sha256"]
    pytest.skip(f"원장에 {run_id} 가 없다")


@pytest.mark.parametrize("run_id", ["cpt_t2b_mean_upd_seed42", "cpt_c0_qwen_r5_seed123"])
def test_direct_cpt_config_is_unchanged_by_refactor(run_id):
    """build_config 로 떼어 낸 뒤에도 직접 CPT 의 config 해시가 원장과 같다."""
    argv, sha = _ledger_argv(run_id)
    args = build_parser().parse_args(shlex.split(argv))
    assert sha256_obj(build_config(args)) == sha


def test_warm_fields_appear_only_in_warm_mode():
    base = ["--model", "m", "--budget-bytes", "168500000"]
    assert "warm_bytes" not in build_config(build_parser().parse_args(base))
    cfg = build_config(build_parser().parse_args(
        base + ["--warm-rows", "ids.json", "--warm-bytes", "33700000"]))
    assert cfg["warm_rows"] == "ids.json" and cfg["warm_bytes"] == 33_700_000
