"""커밋 메시지 규칙이 실제로 집행되는지 확인한다."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ledger  # noqa: E402
from tools import check_commit_msg as cm  # noqa: E402

SHA = "3f9a1c0b7e2d548a6f01b93cd7e4a25f8c6b0d13a94e7f28c5b016d3a8e7f240"
RUN = "cpt_kosub_mean_50m_seed42"


@pytest.fixture
def repo(tmp_path):
    """LEDGER.tsv 에 run 하나가 등록된 가짜 저장소."""
    ledger.append_row(
        "ledger",
        {"run_id": RUN, "phase": "cpt", "status": "ok", "git_commit": "NA"},
        root=tmp_path,
    )
    return tmp_path


def record_msg(**over) -> str:
    parts = {
        "header": "record(cpt): C2 Ko-Substitute 50M 토큰 CPT 결과",
        "body": "Dev BPB 가 10M 지점 이후 평탄하다.",
        "trailers": f"Run-Id: {RUN}\nLedger: experiments/LEDGER.tsv\nConfig-SHA256: {SHA}",
    }
    parts.update(over)
    return f"{parts['header']}\n\n{parts['body']}\n\n{parts['trailers']}\n"


# ── 통과해야 하는 것 ──────────────────────────────────────────────────────
def test_valid_record_passes(repo):
    assert cm.check(record_msg(), repo) == []


def test_valid_fix_passes(repo):
    msg = ("fix(eval): BPB 분모에 문자 수를 쓰던 오류 수정\n\n"
           "한글은 UTF-8 에서 문자당 3바이트다.\n\n"
           f"Invalidates: {RUN}\n")
    assert cm.check(msg, repo) == []


def test_fix_with_invalidates_none_passes(repo):
    assert cm.check("fix(infra): 훅 경로 오타 수정\n\nInvalidates: none\n", repo) == []


def test_simple_upgrade_passes(repo):
    assert cm.check("upgrade(eval): 토크나이저 벤치마크를 멀티프로세스로 전환\n", repo) == []


def test_merge_commit_passes(repo):
    assert cm.check("Merge branch 'exp/tok-kosub-v1'\n", repo) == []


def test_comment_lines_are_ignored(repo):
    msg = ("chore(infra): 훅 연결\n"
           "# Please enter the commit message for your changes.\n"
           "# On branch main\n")
    assert cm.check(msg, repo) == []


# ── 거부해야 하는 것 ──────────────────────────────────────────────────────
def test_freeform_message_rejected(repo):
    errors = cm.check("fixed stuff\n", repo)
    assert errors and "제목 형식이 틀렸다" in errors[0]


def test_missing_scope_rejected(repo):
    assert cm.check("fix: 뭔가 고침\n", repo)


def test_unknown_type_rejected(repo):
    assert cm.check("hotfix(cpt): 급한 수정\n", repo)


def test_unknown_scope_rejected(repo):
    errors = cm.check("fix(없는스코프): 수정\n\nInvalidates: none\n", repo)
    assert errors and "제목 형식이 틀렸다" in errors[0]


def test_trailing_period_rejected(repo):
    errors = cm.check("upgrade(eval): 벤치마크 개선.\n", repo)
    assert any("마침표" in e for e in errors)


def test_too_long_subject_rejected(repo):
    errors = cm.check("upgrade(eval): " + "가" * 80 + "\n", repo)
    assert any("자 이하로 줄여라" in e for e in errors)


def test_missing_blank_line_rejected(repo):
    errors = cm.check("upgrade(eval): 개선\n본문이 바로 붙었다\n", repo)
    assert any("빈 줄" in e for e in errors)


def test_fix_without_invalidates_rejected(repo):
    errors = cm.check("fix(eval): BPB 계산 오류 수정\n", repo)
    assert any("Invalidates" in e for e in errors)


def test_record_without_trailers_rejected(repo):
    errors = cm.check("record(cpt): 결과 기록\n", repo)
    assert any("Run-Id" in e for e in errors)
    assert any("Ledger" in e for e in errors)
    assert any("Config-SHA256" in e for e in errors)


def test_record_with_unknown_run_id_rejected(repo):
    errors = cm.check(record_msg(trailers=(
        "Run-Id: cpt_ghost_mean_50m_seed99\nLedger: experiments/LEDGER.tsv\n"
        f"Config-SHA256: {SHA}"
    )), repo)
    assert any("LEDGER.tsv 에 없다" in e for e in errors)


def test_malformed_run_id_rejected(repo):
    errors = cm.check(record_msg(trailers=(
        "Run-Id: CPT 없는런!\nLedger: experiments/LEDGER.tsv\n"
        f"Config-SHA256: {SHA}"
    )), repo)
    assert any("Run-Id 형식이 틀렸다" in e for e in errors)


def test_bad_sha_rejected(repo):
    errors = cm.check(record_msg(trailers=(
        f"Run-Id: {RUN}\nLedger: experiments/LEDGER.tsv\nConfig-SHA256: deadbeef"
    )), repo)
    assert any("sha256" in e for e in errors)


def test_ledger_path_must_exist(repo):
    errors = cm.check(record_msg(trailers=(
        f"Run-Id: {RUN}\nLedger: experiments/없는파일.tsv\nConfig-SHA256: {SHA}"
    )), repo)
    assert any("경로가 없다" in e for e in errors)


def test_tok_requires_tokenizer_sha(repo):
    errors = cm.check("tok(tok): kosub_v1 등록\n", repo)
    assert any("Tokenizer-SHA256" in e for e in errors)


def test_data_requires_manifest_sha(repo):
    errors = cm.check("data(data): 뉴스 corpus 추가\n", repo)
    assert any("Manifest-SHA256" in e for e in errors)


# ── record 커밋은 코드를 건드릴 수 없다 ───────────────────────────────────
def test_record_with_staged_code_rejected(repo, monkeypatch):
    monkeypatch.setattr(
        cm, "staged_files",
        lambda root: ["experiments/LEDGER.tsv", "src/training/cpt.py"],
    )
    errors = cm.check(record_msg(), repo)
    assert any("코드·설정을 건드릴 수 없다" in e for e in errors)
    assert any("src/training/cpt.py" in e for e in errors)


def test_record_with_only_results_passes(repo, monkeypatch):
    monkeypatch.setattr(
        cm, "staged_files",
        lambda root: ["experiments/LEDGER.tsv", "experiments/lm_metrics.tsv",
                      "reports/figures/bpb_curve.png"],
    )
    assert cm.check(record_msg(), repo) == []


def test_non_record_commit_may_touch_code(repo, monkeypatch):
    monkeypatch.setattr(cm, "staged_files", lambda root: ["src/training/cpt.py"])
    assert cm.check("upgrade(cpt): gradient checkpointing 활성화\n", repo) == []


# ── Config-SHA256 은 원장의 실제 값과 대조한다 ────────────────────────────
def test_fabricated_config_sha_rejected(repo):
    """형식만 맞는 지어낸 해시를 통과시키면 계보 시스템이 무의미해진다."""
    real = "a" * 64
    ledger.append_row(
        "ledger",
        {"run_id": "data_real_seed42", "phase": "data", "status": "ok",
         "config_sha256": real, "git_commit": "NA"},
        root=repo,
    )
    fabricated = "a" * 16 + "b" * 48          # 앞자리만 맞는 가짜
    errors = cm.check(
        "record(data): 결과 기록\n\n본문\n\n"
        f"Run-Id: data_real_seed42\nLedger: experiments/LEDGER.tsv\n"
        f"Config-SHA256: {fabricated}\n", repo)
    assert any("실제 값과 다르다" in e for e in errors)


def test_matching_config_sha_passes(repo):
    real = "c" * 64
    ledger.append_row(
        "ledger",
        {"run_id": "data_ok_seed42", "phase": "data", "status": "ok",
         "config_sha256": real, "git_commit": "NA"},
        root=repo,
    )
    assert cm.check(
        "record(data): 결과 기록\n\n본문\n\n"
        f"Run-Id: data_ok_seed42\nLedger: experiments/LEDGER.tsv\n"
        f"Config-SHA256: {real}\n", repo) == []
