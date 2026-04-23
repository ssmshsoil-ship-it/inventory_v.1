"""
텍스트 전처리 모듈 (농업 뉴스 도메인 특화)
- HTML 태그 제거, 특수문자 정제, 종결어미 통일
- 형태소 분석 (kiwipiepy 우선, KoNLPy Okt 대체)
- 불용어 처리 (외부 사전 파일 지원)
- 지역명 추출 (17개 시도 + 228개 시군구)
- 날짜 추출 (모내기/이앙 근처 날짜 우선 태깅)
- 배치 처리: news_corpus.db → news_tokenized.parquet
"""

import re
import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd

# ─────────────────────────────────────────
# 프로젝트 루트 및 주요 경로 설정
# ─────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\ai_workspace\sh-ai-model")
DB_PATH      = PROJECT_ROOT / "data" / "master_db" / "news_corpus.db"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed"
STOPWORDS_FILE = OUTPUT_DIR / "stopwords.txt"
OUTPUT_PARQUET = OUTPUT_DIR / "news_tokenized.parquet"

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 기본 불용어 목록 (한국어 일반 + 농업 도메인)
# ─────────────────────────────────────────
DEFAULT_STOPWORDS: list[str] = [
    # 한국어 일반 불용어
    "것", "수", "있다", "하다", "되다", "등", "및", "또", "의",
    "에서", "에", "이", "가", "을", "를", "은", "는", "이다",
    "있다", "없다", "하다", "이런", "저런",
    # 농업 뉴스 도메인 불용어 (언론사·보도 관련)
    "기자", "뉴스", "보도", "취재", "기사", "뉴스1",
    "연합뉴스", "뉴시스", "한국농어민신문", "농민신문",
]

# ─────────────────────────────────────────
# 종결어미 통일 패턴 (합쇼체 → 해라체)
# ─────────────────────────────────────────
FORMAL_TO_PLAIN: list[tuple[str, str]] = [
    (r"했습니다", "했다"),
    (r"합니다", "한다"),
    (r"입니다", "이다"),
    (r"됩니다", "된다"),
    (r"있습니다", "있다"),
    (r"없습니다", "없다"),
    (r"였습니다", "였다"),
    (r"겠습니다", "겠다"),
    (r"습니다", "다"),
    (r"ㅂ니다", "다"),
    (r"하였다", "했다"),
    (r"하였습니다", "했다"),
]

# ─────────────────────────────────────────
# 날짜 추출 정규식 패턴
# ─────────────────────────────────────────
DATE_PATTERNS: list[tuple[str, str]] = [
    # YYYY년 MM월 DD일
    (r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일", "full_kor"),
    # YYYY년 MM월
    (r"\d{4}년\s*\d{1,2}월", "ym_kor"),
    # YYYY.MM.DD
    (r"\d{4}\.\d{1,2}\.\d{1,2}", "full_dot"),
    # YYYY/MM/DD
    (r"\d{4}/\d{1,2}/\d{1,2}", "full_slash"),
    # YYYY-MM-DD
    (r"\d{4}-\d{1,2}-\d{1,2}", "full_dash"),
    # MM월 DD일
    (r"\d{1,2}월\s*\d{1,2}일", "md_kor"),
]

# 모내기/이앙 관련 키워드 (날짜 우선 태깅 기준)
TRANSPLANTING_KEYWORDS = ["모내기", "이앙", "이식", "모심기", "벼 심기"]


# ─────────────────────────────────────────
# 전국 17개 시도 목록
# ─────────────────────────────────────────
SIDO_LIST: list[str] = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시",
    "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
    "경기도", "강원도", "충청북도", "충청남도",
    "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도",
    # 약칭
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

# ─────────────────────────────────────────
# 228개 시군구 전체 목록 (행정안전부 기준)
# ─────────────────────────────────────────
SIGUNGU_LIST: list[str] = [
    # 서울특별시 (25개 자치구)
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구",
    "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구",
    "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
    "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
    # 부산광역시 (16개 구·군)
    "중구", "서구", "동구", "영도구", "부산진구", "동래구",
    "남구", "북구", "해운대구", "사하구", "금정구", "강서구",
    "연제구", "수영구", "사상구", "기장군",
    # 대구광역시 (7개 구, 1개 군)
    "중구", "동구", "서구", "남구", "북구", "수성구", "달서구", "달성군",
    # 인천광역시 (8개 구, 2개 군)
    "중구", "동구", "미추홀구", "연수구", "남동구",
    "부평구", "계양구", "서구", "강화군", "옹진군",
    # 광주광역시 (5개 구)
    "동구", "서구", "남구", "북구", "광산구",
    # 대전광역시 (5개 구)
    "동구", "중구", "서구", "유성구", "대덕구",
    # 울산광역시 (4개 구, 1개 군)
    "중구", "남구", "동구", "북구", "울주군",
    # 세종특별자치시
    "세종시",
    # 경기도 (28개 시, 3개 군)
    "수원시", "성남시", "의정부시", "안양시", "부천시", "광명시",
    "평택시", "동두천시", "안산시", "고양시", "과천시", "구리시",
    "남양주시", "오산시", "시흥시", "군포시", "의왕시", "하남시",
    "용인시", "파주시", "이천시", "안성시", "김포시", "화성시",
    "광주시", "양주시", "포천시", "여주시",
    "연천군", "가평군", "양평군",
    # 강원도 (7개 시, 11개 군)
    "춘천시", "원주시", "강릉시", "동해시", "태백시",
    "속초시", "삼척시",
    "홍천군", "횡성군", "영월군", "평창군", "정선군",
    "철원군", "화천군", "양구군", "인제군", "고성군", "양양군",
    # 충청북도 (3개 시, 8개 군)
    "청주시", "충주시", "제천시",
    "보은군", "옥천군", "영동군", "증평군",
    "진천군", "괴산군", "음성군", "단양군",
    # 충청남도 (8개 시, 7개 군)
    "천안시", "공주시", "보령시", "아산시", "서산시",
    "논산시", "계룡시", "당진시",
    "금산군", "부여군", "서천군", "청양군",
    "홍성군", "예산군", "태안군",
    # 전라북도 (6개 시, 8개 군)
    "전주시", "군산시", "익산시", "정읍시", "남원시", "김제시",
    "완주군", "진안군", "무주군", "장수군",
    "임실군", "순창군", "고창군", "부안군",
    # 전라남도 (5개 시, 17개 군)
    "목포시", "여수시", "순천시", "나주시", "광양시",
    "담양군", "곡성군", "구례군", "고흥군", "보성군",
    "화순군", "장흥군", "강진군", "해남군", "영암군",
    "무안군", "함평군", "영광군", "장성군", "완도군",
    "진도군", "신안군",
    # 경상북도 (10개 시, 13개 군)
    "포항시", "경주시", "김천시", "안동시", "구미시",
    "영주시", "영천시", "상주시", "문경시", "경산시",
    "군위군", "의성군", "청송군", "영양군", "영덕군",
    "청도군", "고령군", "성주군", "칠곡군", "예천군",
    "봉화군", "울진군", "울릉군",
    # 경상남도 (8개 시, 10개 군)
    "창원시", "진주시", "통영시", "사천시", "김해시",
    "밀양시", "거제시", "양산시",
    "의령군", "함안군", "창녕군", "고성군", "남해군",
    "하동군", "산청군", "함양군", "거창군", "합천군",
    # 제주특별자치도 (2개 시)
    "제주시", "서귀포시",
]

# 전체 지역명 통합 (시도 + 시군구), 긴 이름 우선 매칭을 위해 길이 내림차순 정렬
ALL_REGIONS: list[str] = sorted(
    list(dict.fromkeys(SIDO_LIST + SIGUNGU_LIST)),  # 중복 제거 후
    key=len,
    reverse=True,
)


class TextPreprocessor:
    """
    농업 뉴스 텍스트 전처리기
    - 정규화, 형태소 분석, 불용어 처리, 지역명/날짜 추출 기능 제공
    - 배치 처리: SQLite DB → Parquet
    """

    def __init__(self, stopwords_file: Path = STOPWORDS_FILE):
        """
        초기화: 불용어 사전 로드, 형태소 분석기 초기화
        """
        self.stopwords_file = stopwords_file
        self.stopwords: set[str] = set()

        # 불용어 사전 파일 생성 또는 로드
        self._init_stopwords()

        # 형태소 분석기 초기화 (kiwipiepy 우선)
        self.kiwi = None
        self.okt  = None
        self._init_morpheme_analyzer()

    # ─────────────────────────────────────────
    # [기능3] 불용어 처리: 사전 파일 초기화
    # ─────────────────────────────────────────
    def _init_stopwords(self) -> None:
        """불용어 사전 파일을 생성하거나 기존 파일을 로드한다."""
        self.stopwords_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.stopwords_file.exists():
            # 기본 불용어로 사전 파일 최초 생성
            with open(self.stopwords_file, "w", encoding="utf-8") as f:
                f.write("# 한국어 일반 불용어\n")
                f.write("\n".join(DEFAULT_STOPWORDS[:17]) + "\n")
                f.write("\n# 농업 뉴스 도메인 불용어 (언론사·보도 관련)\n")
                f.write("\n".join(DEFAULT_STOPWORDS[17:]) + "\n")
                f.write("\n# 사용자 추가 불용어 (아래에 한 줄씩 추가)\n")
            logger.info(f"불용어 사전 파일 생성: {self.stopwords_file}")
        else:
            logger.info(f"기존 불용어 사전 파일 로드: {self.stopwords_file}")

        # 파일에서 불용어 로드 (주석 및 빈 줄 제외)
        with open(self.stopwords_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith("#"):
                    self.stopwords.add(word)
        logger.info(f"불용어 {len(self.stopwords)}개 로드 완료")

    # ─────────────────────────────────────────
    # [기능2] 형태소 분석기 초기화 (이중 엔진)
    # ─────────────────────────────────────────
    def _init_morpheme_analyzer(self) -> None:
        """
        kiwipiepy를 1순위로 초기화하고,
        실패 시 KoNLPy Okt로 자동 전환한다.
        """
        # 1순위: kiwipiepy
        try:
            from kiwipiepy import Kiwi  # type: ignore
            self.kiwi = Kiwi()
            logger.info("형태소 분석기: kiwipiepy 로드 성공")
        except Exception as e:
            logger.warning(f"kiwipiepy 로드 실패: {e}")

        # 2순위: KoNLPy Okt (kiwipiepy가 없을 때만 초기화)
        if self.kiwi is None:
            try:
                from konlpy.tag import Okt  # type: ignore
                self.okt = Okt()
                logger.info("형태소 분석기: KoNLPy Okt 로드 성공 (대체 엔진)")
            except Exception as e:
                logger.error(f"KoNLPy Okt 로드 실패: {e}")
                logger.warning("형태소 분석기 없음 - 공백 분리 토크나이저로 동작합니다.")

    # ─────────────────────────────────────────
    # [기능1] 텍스트 정규화
    # ─────────────────────────────────────────
    def normalize(self, text: str) -> str:
        """
        1. HTML 태그 제거
        2. 특수문자 제거 (한글·영문·숫자·공백 유지)
        3. 연속 공백 단일화
        4. 종결어미 통일 (합쇼체 → 해라체)
        """
        if not text or not isinstance(text, str):
            return ""

        # 1) HTML 태그 제거: bs4 우선, 없으면 re 폴백
        text = self._remove_html(text)

        # 2) 종결어미 통일 (정제 전에 먼저 처리해야 어미 패턴이 깨지지 않음)
        text = self._unify_endings(text)

        # 3) 특수문자 제거 (한글·영문·숫자·공백만 유지)
        text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)

        # 4) 연속 공백 단일화 및 좌우 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _remove_html(self, text: str) -> str:
        """HTML 태그를 제거한다. bs4 사용 가능 시 우선 적용."""
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text(separator=" ")
        except ImportError:
            # bs4 미설치 시 정규식으로 대체
            return re.sub(r"<[^>]+>", " ", text)

    def _unify_endings(self, text: str) -> str:
        """합쇼체 종결어미를 해라체로 통일한다."""
        for pattern, replacement in FORMAL_TO_PLAIN:
            text = re.sub(pattern, replacement, text)
        return text

    # ─────────────────────────────────────────
    # [기능2] 형태소 분석 (명사·동사어근·형용사어근)
    # ─────────────────────────────────────────
    def tokenize(self, text: str) -> list[str]:
        """
        텍스트를 형태소 분석하여 명사/동사어근/형용사어근 토큰 리스트를 반환한다.
        불용어는 자동 제거한다.
        """
        if not text:
            return []

        tokens: list[str] = []

        if self.kiwi is not None:
            tokens = self._tokenize_kiwi(text)
        elif self.okt is not None:
            tokens = self._tokenize_okt(text)
        else:
            # 형태소 분석기 없음 - 공백 분리 후 한글만 필터
            tokens = [t for t in text.split() if re.search(r"[가-힣]", t)]

        # 불용어 제거 및 1글자 이하 토큰 제거
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]

        return tokens

    def _tokenize_kiwi(self, text: str) -> list[str]:
        """
        kiwipiepy를 이용한 형태소 분석.
        추출 품사: NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사)
        동사·형용사는 기본형 어근 추출.
        """
        # 추출 대상 품사 태그
        TARGET_POS = {"NNG", "NNP", "VV", "VA"}

        tokens: list[str] = []
        try:
            result = self.kiwi.analyze(text)
            # kiwi.analyze() 반환값: list of KiwiResult
            # result[0].tokens: list of Token(form, tag, ...)
            best = result[0].tokens if result else []
            for token in best:
                tag = str(token.tag)
                # kiwipiepy 태그는 "Tag.NNG" 형태일 수 있으므로 접미 추출
                short_tag = tag.split(".")[-1]
                if short_tag in TARGET_POS:
                    form = token.form
                    # 동사·형용사 어간 정규화 (종결형 제거)
                    if short_tag in {"VV", "VA"}:
                        form = re.sub(r"(다|고|며|서|면|아|어|지|니|어서|아서)$", "", form)
                    tokens.append(form)
        except Exception as e:
            logger.debug(f"kiwipiepy 분석 오류: {e}")

        return tokens

    def _tokenize_okt(self, text: str) -> list[str]:
        """
        KoNLPy Okt를 이용한 형태소 분석.
        추출 품사: Noun, Verb, Adjective
        동사·형용사는 어근(stem) 형태로 추출.
        """
        TARGET_POS = {"Noun", "Verb", "Adjective"}

        tokens: list[str] = []
        try:
            # norm=True: 정규화, stem=True: 어간 추출
            pos_result = self.okt.pos(text, norm=True, stem=True)
            for form, tag in pos_result:
                if tag in TARGET_POS:
                    tokens.append(form)
        except Exception as e:
            logger.debug(f"KoNLPy Okt 분석 오류: {e}")

        return tokens

    # ─────────────────────────────────────────
    # [기능4] 지역명 추출
    # ─────────────────────────────────────────
    def extract_regions(
        self,
        text: str,
        doc_id: Optional[int] = None,
    ) -> list[tuple[str, int, Optional[int]]]:
        """
        텍스트에서 지역명을 추출하여 (지역명, 등장횟수, 문서ID) 리스트를 반환한다.
        긴 이름 우선 매칭 전략으로 중복 카운트를 방지한다.
        """
        if not text:
            return []

        region_counts: dict[str, int] = {}

        for region in ALL_REGIONS:
            count = len(re.findall(region, text))
            if count > 0:
                region_counts[region] = count

        # (지역명, 등장횟수, 문서ID) 형태로 반환, 등장횟수 내림차순 정렬
        result = [
            (region, cnt, doc_id)
            for region, cnt in sorted(
                region_counts.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return result

    # ─────────────────────────────────────────
    # [기능5] 날짜 추출
    # ─────────────────────────────────────────
    def extract_dates(self, text: str) -> dict[str, list[str]]:
        """
        텍스트에서 날짜 표현을 추출한다.
        모내기/이앙 관련 날짜는 'transplanting_dates' 키로 우선 태깅하여 반환한다.

        반환값 예시:
        {
            "all_dates": ["2025년 5월", "5월 10일", ...],
            "transplanting_dates": ["5월 10일"],  # 모내기 근처 날짜
        }
        """
        if not text:
            return {"all_dates": [], "transplanting_dates": []}

        all_dates: list[str] = []

        # 모든 날짜 패턴 추출
        for pattern, _ in DATE_PATTERNS:
            matches = re.findall(pattern, text)
            all_dates.extend(matches)

        # 중복 제거 (순서 유지)
        seen: set[str] = set()
        unique_dates: list[str] = []
        for d in all_dates:
            if d not in seen:
                seen.add(d)
                unique_dates.append(d)

        # 모내기/이앙 키워드 근처(±50자) 날짜 우선 태깅
        transplanting_dates: list[str] = []

        # 모내기 관련 키워드 위치 수집
        keyword_positions: list[int] = []
        for kw in TRANSPLANTING_KEYWORDS:
            for m in re.finditer(kw, text):
                keyword_positions.append(m.start())

        if keyword_positions:
            for pattern, _ in DATE_PATTERNS:
                for m in re.finditer(pattern, text):
                    date_start = m.start()
                    date_str   = m.group()
                    # 키워드와의 최소 거리 확인 (±50자 이내)
                    min_dist = min(abs(date_start - pos) for pos in keyword_positions)
                    if min_dist <= 50 and date_str not in transplanting_dates:
                        transplanting_dates.append(date_str)

        return {
            "all_dates":          unique_dates,
            "transplanting_dates": transplanting_dates,
        }

    # ─────────────────────────────────────────
    # 통합 처리: 단건 문서 전처리
    # ─────────────────────────────────────────
    def process_document(
        self,
        doc_id:   Optional[int],
        title:    Optional[str],
        content:  Optional[str],
        date_str: Optional[str] = None,
    ) -> dict:
        """
        단건 뉴스 문서를 전처리하여 결과 딕셔너리를 반환한다.
        제목과 본문을 합산하여 처리한다.
        """
        # 제목 + 본문 합산 (None 안전 처리)
        combined = " ".join(filter(None, [title, content]))

        # 1) 정규화
        normalized = self.normalize(combined)

        # 2) 형태소 분석 및 불용어 제거
        tokens = self.tokenize(normalized)

        # 3) 지역명 추출 (원문 기준, 정규화 전)
        regions = self.extract_regions(combined, doc_id=doc_id)

        # 4) 날짜 추출 (원문 기준)
        dates = self.extract_dates(combined)

        return {
            "doc_id":              doc_id,
            "date":                date_str,
            "title":               title or "",
            "normalized_text":     normalized,
            "tokens":              tokens,
            "token_count":         len(tokens),
            "regions":             regions,
            "all_dates":           dates["all_dates"],
            "transplanting_dates": dates["transplanting_dates"],
        }


# ─────────────────────────────────────────
# 배치 처리: DB 전체 → Parquet
# ─────────────────────────────────────────
def load_from_db(db_path: Path) -> pd.DataFrame:
    """news_corpus.db에서 전체 기사를 DataFrame으로 로드한다."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT id, keyword, date, title, press, content_summary FROM news_articles",
            conn,
        )
        logger.info(f"DB 로드 완료: {len(df)}건 ({db_path})")
    finally:
        conn.close()
    return df


def get_sample_texts() -> pd.DataFrame:
    """
    DB가 없거나 비어 있을 때 동작 확인용 샘플 텍스트를 반환한다.
    """
    samples = [
        {
            "id": 1,
            "keyword": "모내기",
            "date": "2025-05-10",
            "title": "전남 나주시 농가, 5월 15일 모내기 본격 시작",
            "press": "농민신문",
            "content_summary": (
                "전라남도 나주시 일대 벼농사 농가에서 2025년 5월 15일부터 모내기가 본격 시작됩니다. "
                "기상청에 따르면 경상남도 창원시와 경기도 화성시 지역도 이번 주 이앙 작업이 예정되어 있습니다. "
                "농가 관계자는 모내기 시기가 작황에 크게 영향을 미친다고 했습니다. "
                "연합뉴스 기자 취재 보도입니다."
            ),
        },
        {
            "id": 2,
            "keyword": "벼 수확",
            "date": "2025-10-05",
            "title": "충청남도 논산시, 2025년 10월 벼 수확 풍년 전망",
            "press": "뉴시스",
            "content_summary": (
                "충청남도 논산시 농업기술센터는 2025.10.05 기준으로 지역 벼 수확량이 "
                "전년 대비 7% 증가할 것으로 전망했습니다. 금산군과 부여군도 비슷한 추세입니다. "
                "이는 5월에 이앙 시기를 앞당긴 덕분이라고 합니다. 뉴스1 뉴스 보도."
            ),
        },
        {
            "id": 3,
            "keyword": "농업 기상",
            "date": "2024-06-20",
            "title": "강원도 홍천군 6월 가뭄 피해 대책 마련",
            "press": "한국농어민신문",
            "content_summary": (
                "강원도 홍천군은 2024년 6월 20일 가뭄 피해 대책 회의를 열었습니다. "
                "인제군 및 양구군도 상황을 점검하고 있습니다. "
                "농림부는 빠른 시일 내에 피해 보상 절차를 안내할 것이라고 했습니다."
            ),
        },
    ]
    df = pd.DataFrame(samples)
    logger.info("샘플 텍스트 3건으로 동작 확인 모드 시작")
    return df


def run_batch(
    db_path:        Path = DB_PATH,
    output_parquet: Path = OUTPUT_PARQUET,
    batch_size:     int  = 500,
) -> Path:
    """
    전체 뉴스 기사를 배치로 전처리하여 Parquet 파일로 저장한다.
    - DB가 없거나 비어 있으면 샘플 텍스트로 동작 확인
    - tqdm 진행률 표시
    """
    # tqdm 로드 (없으면 대체 이터레이터 사용)
    try:
        from tqdm import tqdm  # type: ignore
        _use_tqdm = True
    except ImportError:
        logger.warning("tqdm 미설치 - 진행률 표시 생략")
        _use_tqdm = False

    # 데이터 로드
    if db_path.exists():
        df = load_from_db(db_path)
        if df.empty:
            logger.warning("DB에 데이터 없음 - 샘플 텍스트로 대체")
            df = get_sample_texts()
    else:
        logger.warning(f"DB 파일 없음: {db_path} - 샘플 텍스트로 대체")
        df = get_sample_texts()

    # 전처리기 초기화
    preprocessor = TextPreprocessor()

    # 배치 전처리
    records: list[dict] = []
    total = len(df)
    iterator = df.itertuples(index=False)

    if _use_tqdm:
        iterator = tqdm(iterator, total=total, desc="전처리 진행", unit="건")

    for row in iterator:
        doc_id   = getattr(row, "id", None)
        title    = getattr(row, "title", None)
        content  = getattr(row, "content_summary", None)
        date_str = getattr(row, "date", None)

        result = preprocessor.process_document(
            doc_id=doc_id,
            title=title,
            content=content,
            date_str=date_str,
        )
        # 리스트 컬럼 → 문자열 직렬화 (Parquet 호환)
        result["tokens_str"]    = " ".join(result["tokens"])
        result["all_dates_str"] = "|".join(result["all_dates"])
        result["transplanting_dates_str"] = "|".join(result["transplanting_dates"])
        # regions: [(지역명, 횟수, doc_id)] → "지역명:횟수" 형태로 직렬화
        result["regions_str"] = "|".join(
            f"{r[0]}:{r[1]}" for r in result["regions"]
        )
        # 원본 리스트 컬럼 제거 (Parquet 직렬화 문제 방지)
        del result["tokens"]
        del result["all_dates"]
        del result["transplanting_dates"]
        del result["regions"]

        records.append(result)

    # 출력 디렉토리 생성 및 Parquet 저장
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_parquet(output_parquet, index=False, engine="pyarrow")

    logger.info(f"전처리 완료: {len(out_df)}건 → {output_parquet}")
    return output_parquet


# ─────────────────────────────────────────
# 직접 실행 엔트리포인트 (python preprocessor.py)
# ─────────────────────────────────────────
if __name__ == "__main__":
    out_path = run_batch()
    print(f"\n[완료] 결과 파일: {out_path}")

    # 결과 미리보기 출력
    df = pd.read_parquet(out_path)
    print(f"\n== 처리 결과 미리보기 ({len(df)}건) ==")
    for _, row in df.iterrows():
        print(f"\n[문서 {row['doc_id']}] {row['title']}")
        print(f"  토큰 수    : {row['token_count']}")
        print(f"  주요 토큰  : {row['tokens_str'][:80]}")
        print(f"  지역명     : {row['regions_str']}")
        print(f"  날짜       : {row['all_dates_str']}")
        print(f"  모내기날짜 : {row['transplanting_dates_str']}")
