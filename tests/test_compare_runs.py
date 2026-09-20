"""코드 계보 검사가 실제 저장소 이력에서 동작하는지 본다.

커밋 해시를 박아 둔다. 이 저장소의 이력은 append-only 라 바뀌지 않고,
**실제로 일어난 사건** 을 고정하는 것이 이 테스트의 목적이다.
"""

import pytest

from tools.compare_runs import known_commit, measure_commits

# 2026-09-20 2주차. 이 둘 사이에는 tools/tests/docs 만 바뀌었다.
NO_MEASURE_A = "409930aecf2d89f1ffe868bf9754c1af0d341ffe"
NO_MEASURE_B = "0fde665d36b37181bc80bca51ff3f83da8b52562"
# cpt_c0_qwen_r5_seed42 가 돌던 커밋. 그 뒤로 src/ 가 크게 바뀌었다.
OLD_RUN = "23237ccad09f6714af92a8d13fc47c932438ddae"


def _skip_if_missing(*shas):
    for s in shas:
        if not known_commit(s):
            pytest.skip(f"{s[:12]} 가 이 클론에 없다 (얕은 클론)")


def test_없는_커밋은_모른다고_답한다():
    assert known_commit("0" * 40) is False


def test_tools_만_바뀐_구간은_측정_코드_변경으로_치지_않는다():
    _skip_if_missing(NO_MEASURE_A, NO_MEASURE_B)
    assert measure_commits(NO_MEASURE_A, NO_MEASURE_B) == []


def test_src_가_바뀐_구간은_커밋을_찾아낸다():
    _skip_if_missing(OLD_RUN, NO_MEASURE_A)
    hits = measure_commits(OLD_RUN, NO_MEASURE_A)
    assert hits, "seed42 이후 src/ 가 바뀌었는데 못 찾았다"
    # 예산 꼬리 회계 수정은 실제로 학습 동작을 바꿨다. 반드시 걸려야 한다.
    assert any(h.startswith("2da591e") for h in hits)


def test_방향이_바뀌어도_같은_커밋을_찾는다():
    _skip_if_missing(OLD_RUN, NO_MEASURE_A)
    assert (sorted(measure_commits(OLD_RUN, NO_MEASURE_A))
            == sorted(measure_commits(NO_MEASURE_A, OLD_RUN)))
