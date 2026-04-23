# -*- coding: utf-8 -*-
"""
농업 관련 데이터 수집 크롤러
=====================================================
수집 대상:
  1. 농촌진흥청 농사로 - 지역별 벼 모내기 적정시기 / 육묘일수 표
  2. 기상청 농업기상 정보 - 지역별 농업기상 특보 / 파종·이앙 적기 예보
  3. 공공데이터포털 지방보조금 API - 농업용 자재 보조사업 공고
  4. 학술논문 (RISS) - 수도용 상토 관련 논문 메타데이터

출력 파일:
  data/processed/agri_calendar.csv
  data/processed/agri_weather.csv
  data/processed/subsidy_schedule.csv
  data/processed/academic_papers.csv

오류 로그: logs/agri_crawl_errors.log
API 키 누락 안내: logs/api_keys_needed.txt
"""

import os
import sys
import time
import random
import logging
import traceback
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
# 이 파일 기준으로 프로젝트 루트를 계산 (crawlers/ 한 단계 위)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# 필요한 디렉토리가 없으면 자동 생성
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 로거 설정
# ---------------------------------------------------------------------------
LOG_FILE = LOGS_DIR / "agri_crawl_errors.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("agri_crawler")

# ---------------------------------------------------------------------------
# 공통 상수
# ---------------------------------------------------------------------------
# 요청 간격 (초) - 최솟값, 최댓값
REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.0

# 공통 HTTP 헤더 (User-Agent 설정)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
}

# HTTP 요청 타임아웃 (초)
REQUEST_TIMEOUT = 20


# ===========================================================================
# 유틸리티 함수
# ===========================================================================

def _random_sleep() -> None:
    """요청 간격을 랜덤하게 대기하여 서버 부하 및 차단을 방지한다."""
    delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
    time.sleep(delay)


def _log_error(source: str, exc: Exception) -> None:
    """에러 정보를 로그 파일에 기록한다."""
    logger.error(
        "[%s] 크롤링 실패: %s\n%s",
        source,
        exc,
        traceback.format_exc(),
    )


def _save_csv(df: pd.DataFrame, path: Path, source_name: str) -> None:
    """DataFrame을 CSV로 저장하고 결과를 로그에 남긴다."""
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("[%s] 저장 완료: %s  (%d행)", source_name, path, len(df))


def _note_api_key_needed(api_name: str) -> None:
    """API 키가 없는 경우 logs/api_keys_needed.txt에 안내 메시지를 기록한다."""
    note_file = LOGS_DIR / "api_keys_needed.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] API 키 필요: {api_name}\n"
    with open(note_file, "a", encoding="utf-8") as f:
        f.write(message)
    logger.warning("[API 키 누락] %s 키가 없습니다. %s 에 기록했습니다.", api_name, note_file)


def _empty_df(*columns) -> pd.DataFrame:
    """빈 DataFrame을 반환한다. 파이프라인이 중단되지 않도록 한다."""
    return pd.DataFrame(columns=list(columns))


# ===========================================================================
# 소스 1: 농촌진흥청 농사로 - 지역별 벼 모내기 적정시기 / 육묘일수
# ===========================================================================

NONGSARO_URL = (
    "https://www.nongsaro.go.kr/portal/ps/psb/psbl/cultivationLsList.ps"
    "?menuId=PS03912"
)

# 농사로 모내기 정보 페이지에서 파싱할 컬럼 정의
AGRI_CALENDAR_COLUMNS = [
    "지역",
    "모내기_적정시기_시작",
    "모내기_적정시기_종료",
    "육묘일수_일반",
    "육묘일수_조기",
    "비고",
    "수집일시",
]


def crawl_agri_calendar() -> pd.DataFrame:
    """
    농촌진흥청 농사로에서 지역별 벼 모내기 적정시기 표와 육묘일수를 수집한다.

    Returns:
        pd.DataFrame: 지역별 모내기 정보 (저장 경로: data/processed/agri_calendar.csv)
    """
    source_name = "농사로_모내기_적정시기"
    logger.info("[%s] 크롤링 시작: %s", source_name, NONGSARO_URL)
    save_path = PROCESSED_DIR / "agri_calendar.csv"

    try:
        _random_sleep()
        response = requests.get(
            NONGSARO_URL,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")
        records = []

        # ---------------------------------------------------------------
        # 파싱 전략 1: 페이지 내 <table> 태그 전수 탐색
        # 농사로 재배력 페이지는 지역별 표를 복수의 <table>로 구성한다.
        # ---------------------------------------------------------------
        tables = soup.find_all("table")
        logger.info("[%s] 발견된 <table> 수: %d", source_name, len(tables))

        for table in tables:
            # 테이블 내 <th> 텍스트로 모내기 관련 표 여부를 확인한다.
            headers_text = [
                th.get_text(strip=True) for th in table.find_all("th")
            ]
            # 모내기, 이앙, 육묘 키워드 중 하나라도 포함되면 해당 표로 간주한다.
            is_target_table = any(
                kw in " ".join(headers_text)
                for kw in ("모내기", "이앙", "육묘", "적정시기")
            )
            if not is_target_table:
                continue

            tbody = table.find("tbody")
            rows = tbody.find_all("tr") if tbody else table.find_all("tr")

            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue

                # 셀 수에 따라 컬럼 매핑을 유연하게 처리한다.
                record = {
                    "지역": cells[0] if len(cells) > 0 else "",
                    "모내기_적정시기_시작": cells[1] if len(cells) > 1 else "",
                    "모내기_적정시기_종료": cells[2] if len(cells) > 2 else "",
                    "육묘일수_일반": cells[3] if len(cells) > 3 else "",
                    "육묘일수_조기": cells[4] if len(cells) > 4 else "",
                    "비고": cells[5] if len(cells) > 5 else "",
                    "수집일시": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                records.append(record)

        # ---------------------------------------------------------------
        # 파싱 전략 2: 표가 없거나 파싱 결과가 빈 경우
        # 페이지 전체에서 지역명 + 날짜 패턴을 텍스트로 추출한다.
        # ---------------------------------------------------------------
        if not records:
            logger.warning(
                "[%s] 테이블 파싱 결과 없음. 텍스트 기반 추출을 시도합니다.", source_name
            )
            # 주요 벼 재배 지역 목록 (남부·중부·북부 구분)
            target_regions = [
                "경기", "강원", "충북", "충남", "전북", "전남",
                "경북", "경남", "제주", "서울", "인천", "대전",
                "광주", "대구", "부산", "울산", "세종",
            ]
            text_blocks = soup.get_text(separator="\n").split("\n")
            for i, line in enumerate(text_blocks):
                line = line.strip()
                for region in target_regions:
                    if region in line:
                        # 해당 줄과 다음 몇 줄을 묶어서 하나의 레코드로 구성한다.
                        context = " | ".join(
                            t.strip()
                            for t in text_blocks[i : i + 5]
                            if t.strip()
                        )
                        records.append(
                            {
                                "지역": region,
                                "모내기_적정시기_시작": "",
                                "모내기_적정시기_종료": "",
                                "육묘일수_일반": "",
                                "육묘일수_조기": "",
                                "비고": context,
                                "수집일시": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )
                        break

        if not records:
            logger.warning("[%s] 수집된 데이터가 없습니다. 빈 DataFrame을 반환합니다.", source_name)
            return _empty_df(*AGRI_CALENDAR_COLUMNS)

        df = pd.DataFrame(records)
        # 중복 행 제거 (지역 기준)
        df = df.drop_duplicates(subset=["지역"]).reset_index(drop=True)
        _save_csv(df, save_path, source_name)
        return df

    except Exception as exc:
        _log_error(source_name, exc)
        return _empty_df(*AGRI_CALENDAR_COLUMNS)


# ===========================================================================
# 소스 2: 기상청 농업기상 정보 - 특보 / 파종·이앙 적기 예보
# ===========================================================================

AGRI_WEATHER_URL = "https://weather.go.kr/w/agri"

# 농업기상 수집 컬럼 정의
AGRI_WEATHER_COLUMNS = [
    "지역",
    "특보_종류",
    "특보_내용",
    "파종_이앙_적기",
    "예보_날짜",
    "수집일시",
]


def crawl_agri_weather() -> pd.DataFrame:
    """
    기상청 농업기상 페이지에서 지역별 농업기상 특보와 파종·이앙 적기 예보를 수집한다.

    Returns:
        pd.DataFrame: 농업기상 정보 (저장 경로: data/processed/agri_weather.csv)
    """
    source_name = "기상청_농업기상"
    logger.info("[%s] 크롤링 시작: %s", source_name, AGRI_WEATHER_URL)
    save_path = PROCESSED_DIR / "agri_weather.csv"

    try:
        _random_sleep()
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)

        response = session.get(AGRI_WEATHER_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")
        records = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---------------------------------------------------------------
        # 파싱 전략 1: 특보 관련 섹션 탐색
        # 농업기상 특보는 보통 class에 "warn", "special", "alert" 등을 포함한다.
        # ---------------------------------------------------------------
        warn_keywords = ["warn", "special", "alert", "notice", "agri"]
        warn_sections = []
        for kw in warn_keywords:
            warn_sections += soup.find_all(
                True, class_=lambda c: c and kw in c.lower()
            )

        for section in warn_sections:
            # 지역명 추출 시도 (strong, span, dt 태그 우선)
            region_tag = section.find(["strong", "span", "dt", "th"])
            region = region_tag.get_text(strip=True) if region_tag else "전국"

            # 특보 내용 추출
            content_tag = section.find(["p", "dd", "td", "div"])
            content = content_tag.get_text(strip=True) if content_tag else ""

            if not content:
                continue

            records.append(
                {
                    "지역": region,
                    "특보_종류": "농업기상특보",
                    "특보_내용": content,
                    "파종_이앙_적기": "",
                    "예보_날짜": now_str[:10],
                    "수집일시": now_str,
                }
            )

        # ---------------------------------------------------------------
        # 파싱 전략 2: 테이블에서 파종·이앙 적기 정보 추출
        # ---------------------------------------------------------------
        for table in soup.find_all("table"):
            headers_text = [
                th.get_text(strip=True) for th in table.find_all("th")
            ]
            is_target = any(
                kw in " ".join(headers_text)
                for kw in ("파종", "이앙", "적기", "농업", "기상")
            )
            if not is_target:
                continue

            tbody = table.find("tbody")
            rows = tbody.find_all("tr") if tbody else table.find_all("tr")

            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) < 2:
                    continue
                records.append(
                    {
                        "지역": cells[0] if len(cells) > 0 else "",
                        "특보_종류": cells[1] if len(cells) > 1 else "",
                        "특보_내용": cells[2] if len(cells) > 2 else "",
                        "파종_이앙_적기": cells[3] if len(cells) > 3 else "",
                        "예보_날짜": cells[4] if len(cells) > 4 else now_str[:10],
                        "수집일시": now_str,
                    }
                )

        # ---------------------------------------------------------------
        # 파싱 전략 3: 텍스트 전체에서 파종·이앙 관련 문장 추출
        # ---------------------------------------------------------------
        if not records:
            logger.warning(
                "[%s] 구조화된 데이터 없음. 텍스트 기반 추출을 시도합니다.", source_name
            )
            text_lines = [
                line.strip()
                for line in soup.get_text(separator="\n").split("\n")
                if line.strip()
            ]
            agri_keywords = ["파종", "이앙", "모내기", "육묘", "농업기상", "특보", "적기"]
            for line in text_lines:
                if any(kw in line for kw in agri_keywords):
                    records.append(
                        {
                            "지역": "전국",
                            "특보_종류": "텍스트_추출",
                            "특보_내용": line,
                            "파종_이앙_적기": line if "적기" in line else "",
                            "예보_날짜": now_str[:10],
                            "수집일시": now_str,
                        }
                    )

        if not records:
            logger.warning("[%s] 수집된 데이터가 없습니다. 빈 DataFrame을 반환합니다.", source_name)
            return _empty_df(*AGRI_WEATHER_COLUMNS)

        df = pd.DataFrame(records)
        _save_csv(df, save_path, source_name)
        return df

    except Exception as exc:
        _log_error(source_name, exc)
        return _empty_df(*AGRI_WEATHER_COLUMNS)


# ===========================================================================
# 소스 3: 공공데이터포털 지방보조금 API - 농업용 자재 보조사업 공고
# ===========================================================================

# 공공데이터포털 지방보조금 API 엔드포인트
SUBSIDY_API_URL = "https://www.data.go.kr/dataset/15013191/openapi.do"

# 농업용 자재 관련 필터 키워드 (상토·육묘·농자재 등)
SUBSIDY_FILTER_KEYWORDS = ["상토", "육묘", "농자재", "농업용", "모판", "비료", "농약", "농업"]

# 보조금 공고 컬럼 정의
SUBSIDY_COLUMNS = [
    "공고번호",
    "사업명",
    "지자체명",
    "공고일",
    "계약_종료일",
    "예산액",
    "사업_내용",
    "키워드",
    "수집일시",
]

# 보조금 크롤링 대체 URL (API 키 없을 때 HTML 스크래핑 시도)
SUBSIDY_FALLBACK_URL = "https://www.data.go.kr/data/15013191/openapi.do"


def _filter_subsidy_records(records: list) -> list:
    """
    수집된 보조금 레코드 중 농업용 자재 관련 항목만 필터링한다.

    Args:
        records: 원본 레코드 목록

    Returns:
        list: 필터링된 레코드 목록
    """
    filtered = []
    for rec in records:
        # 사업명 또는 사업내용에 농업용 자재 관련 키워드가 포함되는지 확인한다.
        combined = " ".join([
            str(rec.get("사업명", "")),
            str(rec.get("사업_내용", "")),
        ])
        matched_kw = [kw for kw in SUBSIDY_FILTER_KEYWORDS if kw in combined]
        if matched_kw:
            rec["키워드"] = ",".join(matched_kw)
            filtered.append(rec)
    return filtered


def _crawl_subsidy_with_api(api_key: str) -> list:
    """
    공공데이터포털 지방보조금 API를 호출하여 보조사업 공고를 수집한다.

    Args:
        api_key: 공공데이터포털 API 인증키 (DATA_GO_KR_API_KEY)

    Returns:
        list: 수집된 레코드 목록 (필터링 전)
    """
    records = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    page_no = 1
    num_of_rows = 100  # 한 번에 가져올 항목 수
    max_pages = 10      # 최대 페이지 수 (무한 루프 방지)

    while page_no <= max_pages:
        params = {
            "serviceKey": api_key,
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            "type": "json",
        }

        _random_sleep()
        try:
            resp = requests.get(
                SUBSIDY_API_URL,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(
                "[공공데이터포털_보조금_API] 페이지 %d 요청 실패: %s", page_no, exc
            )
            break

        # API 응답 구조 파싱 (data.go.kr 공통 응답 형식 대응)
        try:
            items = (
                data.get("response", {})
                .get("body", {})
                .get("items", {})
                .get("item", [])
            )
            # items가 dict 단일 객체인 경우 리스트로 감싼다.
            if isinstance(items, dict):
                items = [items]
        except (AttributeError, TypeError):
            items = []

        if not items:
            # 더 이상 데이터가 없으면 루프를 종료한다.
            break

        for item in items:
            records.append(
                {
                    "공고번호": item.get("ancnmNo", ""),
                    "사업명": item.get("bsnNm", "") or item.get("businessName", ""),
                    "지자체명": item.get("lcgvNm", "") or item.get("localGovName", ""),
                    "공고일": item.get("ancnmDt", "") or item.get("announcementDate", ""),
                    "계약_종료일": item.get("ctrtEndDt", "") or item.get("contractEndDate", ""),
                    "예산액": item.get("bdgtAmt", "") or item.get("budgetAmount", ""),
                    "사업_내용": item.get("bsnCn", "") or item.get("businessContent", ""),
                    "키워드": "",
                    "수집일시": now_str,
                }
            )

        logger.info(
            "[공공데이터포털_보조금_API] 페이지 %d 처리 완료 (%d건)", page_no, len(items)
        )
        page_no += 1

    return records


def _crawl_subsidy_fallback() -> list:
    """
    API 키가 없을 때 공공데이터포털 보조금 페이지를 HTML 스크래핑으로 대체한다.
    실제 서비스에서는 API 키를 발급받아 사용하는 것을 권장한다.

    Returns:
        list: 수집된 레코드 목록 (필터링 전)
    """
    logger.info("[공공데이터포털_보조금_대체] HTML 스크래핑으로 대체 수집 시도")
    records = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 대체 수집용 검색 URL 목록 (농업 관련 보조금 검색)
    search_urls = [
        "https://www.data.go.kr/data/15013191/openapi.do",
        "https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15013191",
    ]

    for url in search_urls:
        try:
            _random_sleep()
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")

            # 테이블 형식의 목록 파싱 시도
            for table in soup.find_all("table"):
                rows = table.find_all("tr")
                for row in rows[1:]:  # 첫 번째 행은 헤더로 건너뜀
                    cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                    if len(cells) < 2:
                        continue
                    records.append(
                        {
                            "공고번호": cells[0] if len(cells) > 0 else "",
                            "사업명": cells[1] if len(cells) > 1 else "",
                            "지자체명": cells[2] if len(cells) > 2 else "",
                            "공고일": cells[3] if len(cells) > 3 else "",
                            "계약_종료일": cells[4] if len(cells) > 4 else "",
                            "예산액": cells[5] if len(cells) > 5 else "",
                            "사업_내용": " | ".join(cells),
                            "키워드": "",
                            "수집일시": now_str,
                        }
                    )

            if records:
                logger.info(
                    "[공공데이터포털_보조금_대체] %s 에서 %d건 수집", url, len(records)
                )
                break

        except Exception as exc:
            logger.warning("[공공데이터포털_보조금_대체] %s 실패: %s", url, exc)

    return records


def crawl_subsidy_schedule() -> pd.DataFrame:
    """
    공공데이터포털 지방보조금 API에서 농업용 자재 보조사업 공고와 계약 종료일을 수집한다.
    환경변수 DATA_GO_KR_API_KEY 가 없으면 크롤링 방식으로 대체한다.

    Returns:
        pd.DataFrame: 보조금 공고 정보 (저장 경로: data/processed/subsidy_schedule.csv)
    """
    source_name = "공공데이터포털_농업보조금"
    logger.info("[%s] 크롤링 시작", source_name)
    save_path = PROCESSED_DIR / "subsidy_schedule.csv"

    # 환경변수에서 API 키를 읽는다.
    api_key = os.environ.get("DATA_GO_KR_API_KEY", "").strip()

    try:
        if api_key:
            logger.info("[%s] API 키 확인 완료. API 방식으로 수집합니다.", source_name)
            raw_records = _crawl_subsidy_with_api(api_key)
        else:
            # API 키가 없으면 누락 사실을 기록하고 크롤링으로 대체한다.
            _note_api_key_needed("DATA_GO_KR_API_KEY (공공데이터포털 지방보조금)")
            raw_records = _crawl_subsidy_fallback()

        # 농업용 자재 관련 키워드로 필터링한다.
        filtered_records = _filter_subsidy_records(raw_records)
        logger.info(
            "[%s] 전체 %d건 중 필터링 후 %d건",
            source_name,
            len(raw_records),
            len(filtered_records),
        )

        if not filtered_records:
            logger.warning("[%s] 수집된 데이터가 없습니다. 빈 DataFrame을 반환합니다.", source_name)
            return _empty_df(*SUBSIDY_COLUMNS)

        df = pd.DataFrame(filtered_records, columns=SUBSIDY_COLUMNS)
        _save_csv(df, save_path, source_name)
        return df

    except Exception as exc:
        _log_error(source_name, exc)
        return _empty_df(*SUBSIDY_COLUMNS)


# ===========================================================================
# 소스 4: 학술논문 (RISS) - 수도용 상토 관련 논문 메타데이터
# ===========================================================================

RISS_SEARCH_URL = (
    "https://www.riss.kr/search/Search.do"
    "?searchGubun=simple&query=%EC%88%98%EB%8F%84%EC%9A%A9%EC%83%81%ED%86%A0"
)
# 위 URL의 query 파라미터는 "수도용상토"를 URL 인코딩한 값이다.

# 논문 메타데이터 컬럼 정의
ACADEMIC_PAPER_COLUMNS = [
    "제목",
    "저자",
    "발행연도",
    "발행기관",
    "초록",
    "키워드",
    "링크",
    "수집일시",
]

# Selenium 대기 최대 시간 (초)
SELENIUM_WAIT_TIMEOUT = 15


def _parse_riss_papers_from_soup(soup: BeautifulSoup) -> list:
    """
    BeautifulSoup 객체에서 RISS 검색 결과 목록을 파싱한다.

    Args:
        soup: 파싱된 HTML BeautifulSoup 객체

    Returns:
        list: 논문 레코드 목록
    """
    records = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # RISS 검색 결과 목록은 보통 <ul class="srchResultListW"> 안에 <li> 형태로 구성된다.
    result_list = soup.find("ul", class_=lambda c: c and "srchResult" in c)
    if result_list:
        items = result_list.find_all("li", recursive=False)
    else:
        # 클래스명이 변경된 경우를 대비하여 article, li 태그 전체를 탐색한다.
        items = soup.find_all(["article", "li"], class_=lambda c: c and "item" in (c or "").lower())

    for item in items:
        # 제목 파싱 - a 태그 우선
        title_tag = (
            item.find("a", class_=lambda c: c and "title" in (c or "").lower())
            or item.find("p", class_=lambda c: c and "title" in (c or "").lower())
            or item.find("h3")
            or item.find("h4")
            or item.find("a")
        )
        title = title_tag.get_text(strip=True) if title_tag else ""

        # 링크 파싱
        link = ""
        if title_tag and title_tag.name == "a":
            href = title_tag.get("href", "")
            link = "https://www.riss.kr" + href if href.startswith("/") else href

        # 저자 파싱
        author_tag = item.find(
            True, class_=lambda c: c and any(kw in (c or "") for kw in ["author", "writer", "저자"])
        )
        author = author_tag.get_text(strip=True) if author_tag else ""

        # 발행연도 파싱 - 숫자 4자리 패턴 탐색
        year = ""
        year_tag = item.find(
            True, class_=lambda c: c and any(kw in (c or "") for kw in ["year", "date", "년도"])
        )
        if year_tag:
            year = year_tag.get_text(strip=True)
        else:
            # 텍스트에서 연도 패턴(19xx, 20xx)을 직접 추출한다.
            import re
            item_text = item.get_text()
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", item_text)
            year = year_match.group(0) if year_match else ""

        # 발행기관 파싱
        publisher_tag = item.find(
            True,
            class_=lambda c: c and any(
                kw in (c or "") for kw in ["publisher", "organ", "기관", "학회"]
            ),
        )
        publisher = publisher_tag.get_text(strip=True) if publisher_tag else ""

        # 초록 파싱
        abstract_tag = item.find(
            True,
            class_=lambda c: c and any(
                kw in (c or "") for kw in ["abstract", "summary", "초록", "요약"]
            ),
        )
        abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""

        # 키워드 파싱
        keyword_tag = item.find(
            True,
            class_=lambda c: c and any(
                kw in (c or "") for kw in ["keyword", "tag", "키워드"]
            ),
        )
        keywords = keyword_tag.get_text(strip=True) if keyword_tag else ""

        # 제목이 없는 항목은 유효하지 않은 결과로 간주하고 건너뛴다.
        if not title:
            continue

        records.append(
            {
                "제목": title,
                "저자": author,
                "발행연도": year,
                "발행기관": publisher,
                "초록": abstract,
                "키워드": keywords,
                "링크": link,
                "수집일시": now_str,
            }
        )

    return records


def _crawl_riss_with_selenium() -> list:
    """
    Selenium을 사용하여 RISS 검색 결과를 수집한다 (JavaScript 렌더링 지원).

    Returns:
        list: 파싱된 논문 레코드 목록
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    logger.info("[RISS_Selenium] Selenium 드라이버 초기화 중...")

    # Chrome 옵션 설정 (헤드리스 모드)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,900")
    chrome_options.add_argument(f"user-agent={DEFAULT_HEADERS['User-Agent']}")
    # 불필요한 이미지·폰트 로딩 차단으로 속도를 높인다.
    prefs = {"profile.managed_default_content_settings.images": 2}
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)

    try:
        logger.info("[RISS_Selenium] 페이지 로딩: %s", RISS_SEARCH_URL)
        driver.get(RISS_SEARCH_URL)

        # 검색 결과 목록이 로딩될 때까지 대기한다.
        WebDriverWait(driver, SELENIUM_WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ul.srchResultListW, .result-list"))
        )

        _random_sleep()

        # 렌더링된 HTML을 BeautifulSoup으로 파싱한다.
        soup = BeautifulSoup(driver.page_source, "html.parser")
        records = _parse_riss_papers_from_soup(soup)
        logger.info("[RISS_Selenium] %d건 수집 완료", len(records))
        return records

    finally:
        # 드라이버를 반드시 종료한다.
        driver.quit()


def _crawl_riss_with_requests() -> list:
    """
    requests + BeautifulSoup을 사용하여 RISS 검색 결과를 수집한다 (Selenium 대체 수단).

    Returns:
        list: 파싱된 논문 레코드 목록
    """
    logger.info("[RISS_requests] requests+BeautifulSoup 방식으로 대체 수집 시작")

    _random_sleep()
    # RISS는 JavaScript로 결과를 렌더링하므로, 일부 결과만 정적 HTML에 포함될 수 있다.
    try:
        resp = requests.get(
            RISS_SEARCH_URL,
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        records = _parse_riss_papers_from_soup(soup)
        logger.info("[RISS_requests] %d건 수집 완료", len(records))
        return records
    except Exception as exc:
        logger.warning("[RISS_requests] 수집 실패: %s", exc)
        return []


def crawl_academic_papers() -> pd.DataFrame:
    """
    RISS에서 '수도용상토' 관련 학술논문의 제목·저자·연도·초록·키워드를 수집한다.
    Selenium이 사용 불가능하면 requests+BeautifulSoup으로 대체한다.

    Returns:
        pd.DataFrame: 학술논문 메타데이터 (저장 경로: data/processed/academic_papers.csv)
    """
    source_name = "RISS_학술논문"
    logger.info("[%s] 크롤링 시작: %s", source_name, RISS_SEARCH_URL)
    save_path = PROCESSED_DIR / "academic_papers.csv"

    records = []

    try:
        # Selenium 사용을 우선 시도한다.
        try:
            records = _crawl_riss_with_selenium()
        except ImportError:
            # selenium 패키지가 설치되지 않은 경우
            logger.warning("[%s] selenium 패키지를 찾을 수 없습니다. requests 방식으로 대체합니다.", source_name)
            records = _crawl_riss_with_requests()
        except Exception as sel_exc:
            # Selenium 실행 중 오류 발생 (드라이버 미설치, 버전 불일치 등)
            logger.warning(
                "[%s] Selenium 수집 실패: %s. requests 방식으로 대체합니다.",
                source_name,
                sel_exc,
            )
            records = _crawl_riss_with_requests()

        if not records:
            logger.warning("[%s] 수집된 데이터가 없습니다. 빈 DataFrame을 반환합니다.", source_name)
            return _empty_df(*ACADEMIC_PAPER_COLUMNS)

        df = pd.DataFrame(records, columns=ACADEMIC_PAPER_COLUMNS)
        # 제목 기준 중복 제거
        df = df.drop_duplicates(subset=["제목"]).reset_index(drop=True)
        _save_csv(df, save_path, source_name)
        return df

    except Exception as exc:
        _log_error(source_name, exc)
        return _empty_df(*ACADEMIC_PAPER_COLUMNS)


# ===========================================================================
# main: 4개 소스 순서대로 수집
# ===========================================================================

def main() -> None:
    """
    모든 농업 데이터 소스를 순서대로 수집한다.
    개별 소스에서 오류가 발생해도 나머지 소스는 계속 진행한다.
    """
    logger.info("=" * 60)
    logger.info("농업 데이터 크롤러 시작 - %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    results = {}

    # ------------------------------------------------------------------
    # 소스 1: 농촌진흥청 농사로 - 지역별 벼 모내기 적정시기 / 육묘일수
    # ------------------------------------------------------------------
    logger.info("\n[1/4] 농촌진흥청 농사로 모내기 적정시기 수집 중...")
    df_calendar = crawl_agri_calendar()
    results["agri_calendar"] = df_calendar
    logger.info("[1/4] 완료: %d건\n", len(df_calendar))

    # ------------------------------------------------------------------
    # 소스 2: 기상청 농업기상 정보 - 특보 / 파종·이앙 적기 예보
    # ------------------------------------------------------------------
    logger.info("[2/4] 기상청 농업기상 정보 수집 중...")
    df_weather = crawl_agri_weather()
    results["agri_weather"] = df_weather
    logger.info("[2/4] 완료: %d건\n", len(df_weather))

    # ------------------------------------------------------------------
    # 소스 3: 공공데이터포털 지방보조금 API - 농업용 자재 보조사업 공고
    # ------------------------------------------------------------------
    logger.info("[3/4] 공공데이터포털 농업보조금 수집 중...")
    df_subsidy = crawl_subsidy_schedule()
    results["subsidy_schedule"] = df_subsidy
    logger.info("[3/4] 완료: %d건\n", len(df_subsidy))

    # ------------------------------------------------------------------
    # 소스 4: RISS 학술논문 - 수도용 상토 관련 논문 메타데이터
    # ------------------------------------------------------------------
    logger.info("[4/4] RISS 학술논문 수집 중...")
    df_papers = crawl_academic_papers()
    results["academic_papers"] = df_papers
    logger.info("[4/4] 완료: %d건\n", len(df_papers))

    # ------------------------------------------------------------------
    # 수집 결과 요약 출력
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("수집 완료 요약")
    logger.info("=" * 60)
    for name, df in results.items():
        status = "성공" if len(df) > 0 else "데이터 없음"
        logger.info("  %-25s : %4d건  [%s]", name, len(df), status)
    logger.info("=" * 60)
    logger.info("출력 디렉토리: %s", PROCESSED_DIR)
    logger.info("오류 로그:     %s", LOG_FILE)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
