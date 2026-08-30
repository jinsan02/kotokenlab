"""내용 기반 도메인 신호 (검토: 블라인드 감사 54.7% 대응).

**왜 필요한가**: 규칙이 호스트만 보는데 사람은 본문을 보고 판단한다. 블라인드
감사에서 technical 재현율 8%, community 와 ko_en_mixed 는 0% 였다. 호스트에
`docs.` 나 `cafe.` 가 없으면 규칙이 아무것도 못 잡기 때문이다.

여기 있는 신호는 **150건 블라인드 라벨 중 dev 105건을 실제로 읽고** 뽑은 것이다.
추측이 아니다. holdout 45건은 건드리지 않았고, 최종 측정에만 쓴다.

관측된 것:

    technical   "지원하지 않는 브라우저", "기술 자료: 831733", "참조하십시오",
                "다음 계정으로 실행" — 절차와 안내의 문체
    community   "작성일 : 10-08-13", "글쓴이 : 관리자", "조회 : 14,985",
                "작성자 jessica 날짜 2018.11.06" — 게시판 UI 잔해
    ko_en_mixed "모형: HS15025D", "ISO 9001 : 2000", "Clinical Cancer Research"
                — 영문 모델명·학술지명이 한국어 사이에 박힌다
    news        "밝혔다", "전했다", "기자" — 인용 보도의 종결어미

호스트 규칙이 먼저다. 내용 신호는 호스트가 아무것도 못 잡았을 때만 본다
(호스트 규칙의 정밀도가 news 81% 로 더 높기 때문).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 게시판 UI 잔해. 본문이 아니라 페이지 구조가 새어 나온 것이라 신호가 강하다.
_BOARD = re.compile(
    r"작성일|글쓴이|작성자|조회\s*[:：]|조회수|댓글\s*\d|추천\s*\d|"
    r"등록일|스크랩|신고하기|답변\s*\d|게시글|말머리"
)
# 구어체·감정 표현. 커뮤니티와 블로그를 일반 웹에서 가른다.
_COLLOQUIAL = re.compile(
    r"[ㅋㅎㅠㅜ]{2,}|\^\^|해요[.!?]|했어요|하네요|거든요|인데요|드려요|"
    r"싶어요|같아요|주세요[.!?]|나요\?|까요\?|답니다"
)
# 절차·지원 문서의 어미와 상투어.
_PROCEDURE = re.compile(
    r"하십시오|하시기 바랍니다|참조하|참고하십시오|다음과 같이|다음 단계|"
    r"클릭하|설정하|설치하|실행하|입력하|선택하십시오|지원하지 않는|"
    r"기술 자료|사용 설명서|매뉴얼|문제 해결|오류가 발생|버전을"
)
# 인용 보도의 종결어미. 한국 기사에서 매우 안정적이다.
_NEWS = re.compile(
    r"밝혔다|전했다|말했다|덧붙였다|설명했다|강조했다|지적했다|알려졌다|"
    r"기자\b|연합뉴스|뉴시스|보도자료|취재|인터뷰에서"
)
# 1인칭 체험. 블로그.
_FIRST_PERSON = re.compile(r"제가|저는|저희|내가|필자|오늘은|리뷰하려고|다녀왔")
# 제품 스펙·학술 서지. 영문 고유명사가 한국어 사이에 박힌 형태.
_SPEC = re.compile(
    r"모형\s*[:：]|모델명|규격\s*[:：]|사양\s*[:：]|ISO\s*\d{4}|"
    r"[A-Z]{2,}[-\s]?\d{3,}|저널명|논문제목|저자\s*[:：]|발행일\s*[:：]|"
    r"편의 시설|가격대\s*\(|체크인|객실"
)


# 판정에 쓰는 창 길이. **감사 표본의 text_preview 와 같아야 한다.**
# 이걸 맞추지 않으면 180자로 튜닝한 임계값이 production 에서는 수천 자에 적용되어
# 훨씬 자주 발화한다 — 측정한 정확도가 실제 동작에 적용되지 않는다.
# 문서 앞부분은 대개 대표적이다 (뉴스 리드, 게시판 메타, 기술문서 서두).
WINDOW = 180


@dataclass(frozen=True)
class ContentConfig:
    """임계값은 dev 105건의 180자 미리보기에서 맞췄다.

    관측된 평균 (dev):
        technical    procedure 0.70   나머지 도메인 0.00
        community    board     1.25   나머지 도메인 0.02~0.03
        ko_en_mixed  spec      1.00   나머지 도메인 0.00~0.02
        news         news      0.49   나머지 도메인 0.00

    180자 창에서는 신호가 희박하지만 분별력은 크다 — web_general 은 전부 0 이다.
    그래서 임계값을 1 로 둔다. 2 이상이면 아예 발화하지 않는다.
    """

    board_min: int = 1
    procedure_min: int = 1
    news_min: int = 1
    colloquial_min: int = 2
    first_person_min: int = 2
    spec_min: int = 1


def signals(text: str) -> dict:
    """각 신호가 몇 번 나왔는지. 디버깅과 임계값 조정에 쓴다."""
    return {
        "board": len(_BOARD.findall(text)),
        "colloquial": len(_COLLOQUIAL.findall(text)),
        "procedure": len(_PROCEDURE.findall(text)),
        "news": len(_NEWS.findall(text)),
        "first_person": len(_FIRST_PERSON.findall(text)),
        "spec": len(_SPEC.findall(text)),
    }


def classify_content(text: str, cfg: ContentConfig | None = None) -> str | None:
    """본문만 보고 도메인을 정한다. 확신이 없으면 None.

    순서가 곧 우선순위다. 게시판 잔해가 가장 확실한 신호라 먼저 본다.
    """
    cfg = cfg or ContentConfig()
    s = signals(text[:WINDOW])

    if s["board"] >= cfg.board_min:
        return "community"
    if s["news"] >= cfg.news_min:
        return "news"
    if s["procedure"] >= cfg.procedure_min:
        return "technical"
    if s["spec"] >= cfg.spec_min:
        return "ko_en_mixed"
    if s["colloquial"] >= cfg.colloquial_min:
        return "community" if s["board"] else "blog"
    if s["first_person"] >= cfg.first_person_min and s["colloquial"]:
        return "blog"
    return None
