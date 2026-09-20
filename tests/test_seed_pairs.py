import math

from tools.seed_pairs import describe, recovery


def test_손상_축에서는_보통의_회복률이_나온다():
    # B0 2.38 -> Bf 1.47, 통제군 Cf 1.13. 분모가 양수인 정상 구간.
    assert recovery(2.380297, 1.473036, 1.127397) == (
        (2.380297 - 1.473036) / (2.380297 - 1.127397))


def test_통제군이_더_나쁜_축에서는_R_을_내지_않는다():
    # 실제 영어 값. 분모가 -0.003049 라 음수/음수가 369% 를 만든다.
    assert recovery(0.812640, 0.823919, 0.815689) is None


def test_분모가_정확히_0_이어도_내지_않는다():
    assert recovery(1.0, 1.2, 1.0) is None


def test_쌍이_하나면_표본_SD_를_내지_않는다():
    st = describe([0.3456])
    assert st["n"] == 1
    assert st["sd"] is None
    assert st["lo"] is None


def test_세_쌍이면_표본_SD_와_t_구간을_낸다():
    st = describe([0.34, 0.35, 0.36])
    assert st["n"] == 3
    assert math.isclose(st["sd"], 0.01, rel_tol=1e-9)
    # 구간은 평균을 감싼다
    assert st["lo"] < st["mean"] < st["hi"]


def test_R_이_정의되지_않는_축이면_빈_목록이_와도_지어내지_않는다():
    st = describe([])
    assert st["n"] == 0
    assert st["mean"] is None
    assert st["sd"] is None
