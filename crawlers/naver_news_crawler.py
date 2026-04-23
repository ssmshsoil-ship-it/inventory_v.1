"""
네이버 뉴스 크롤러
- 상토/육묘/농업 관련 키워드를 기반으로 네이버 뉴스를 수집한다.
- 네이버 검색 API 우선 사용, 없으면 웹 직접 크롤링으로 자동 전환한다.
- 결과는 SQLite DB + CSV 백업으로 저장한다.
"""

import os
import re
import csv
import json
import time
import random
import logging
import sqlite3
import hashlib
import requests

from datetime import datetime, date
from pathlib import Path
from difflib import SequenceMatcher
from urllib.parse import quote

from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────
# 환경변수 로드
# ────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# 키워드 그룹 정의
# ────────────────────────────────────────────────────────────────────────────
KEYWORD_GROUPS = {
    "A_상토관련": [
        "수도용상토", "벼모상토", "육묘상토", "상토부족", "상토불량",
        "상토민원", "모상토", "상토출고", "상토수요",
    ],
    "B_재배_농업": [
        "모내기시기", "이앙시기", "모내기일정", "벼이앙", "수도작재배",
        "못자리", "벼모기르기", "파종시기", "육묘방법",
    ],
    "C_농업행정_정책": [
        "지자체보조상토", "상토지원사업", "농협상토", "보조상토", "영농자재지원",
        "농협검수", "못자리상토지원사업", "벼육묘용", "상토보조",
    ],
}

# 전체 키워드 평탄화 (그룹명 포함 튜플 리스트)
ALL_KEYWORDS = [
    (group, kw)
    for group, keywords in KEYWORD_GROUPS.items()
    for kw in keywords
]

# ────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # 프로젝트 루트
DB_PATH = BASE_DIR / "data" / "master_db" / "news_corpus.db"
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"
ERROR_LOG_PATH = LOG_DIR / "crawl_errors.log"

# 필요한 디렉토리 생성
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# 로거 설정
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=str(ERROR_LOG_PATH),
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
error_logger = logging.getLogger("crawl_errors")

# ────────────────────────────────────────────────────────────────────────────
# 네이버 API 자격증명
# ────────────────────────────────────────────────────────────────────────────
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")
USE_API = bool(NAVER_CLIENT_ID and NAVER_CLIENT_SECRET)

# ────────────────────────────────────────────────────────────────────────────
# HTTP 요청 설정
# ────────────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://search.naver.com/",
}

# 요청 재시도 설정
MAX_RETRIES = 3
RETRY_WAIT = 3          # 재시도 대기 시간(초)
REQUEST_DELAY = (1, 2)  # 요청 간격 범위(초)

# ────────────────────────────────────────────────────────────────────────────
# 제목 유사도 판단
# ────────────────────────────────────────────────────────────────────────────
TITLE_SIMILARITY_THRESHOLD = 0.8  # 이 값 이상이면 중복으로 판단


def title_similarity(a: str, b: str) -> float:
    """두 제목 문자열의 유사도를 0~1 사이 값으로 반환한다."""
    return SequenceMatcher(None, a, b).ratio()


# ────────────────────────────────────────────────────────────────────────────
# SQLite 초기화
# ────────────────────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    """news_articles 테이블이 없으면 생성한다."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword       TEXT    NOT NULL,
            date          TEXT,
            title         TEXT    NOT NULL,
            press         TEXT,
            content_summary TEXT,
            url           TEXT    UNIQUE NOT NULL,
            crawled_at    TEXT    NOT NULL
        )
    """)
    # URL 인덱스 (중복 확인 성능 향상)
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_url ON news_articles(url)
    """)
    # 날짜/키워드 복합 인덱스
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_keyword_date
        ON news_articles(keyword, date)
    """)
    conn.commit()


# ────────────────────────────────────────────────────────────────────────────
# 기사 저장 (DB + CSV)
# ────────────────────────────────────────────────────────────────────────────
def save_article(conn: sqlite3.Connection, article: dict) -> bool:
    """
    기사 하나를 DB에 저장한다.
    URL 중복이면 False, 성공이면 True를 반환한다.
    """
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO news_articles
                (keyword, date, title, press, content_summary, url, crawled_at)
            VALUES
                (:keyword, :date, :title, :press, :content_summary, :url, :crawled_at)
        """, article)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # URL 중복
        return False


def save_to_csv(article: dict, keyword: str, yyyymm: str) -> None:
    """
    CSV 백업 파일에 기사를 추가한다.
    파일명: data/raw/news_raw_{keyword}_{YYYYMM}.csv
    """
    # 키워드에서 파일명으로 사용 불가한 문자 제거
    safe_keyword = re.sub(r'[\\/:*?"<>|]', "_", keyword)
    csv_path = RAW_DIR / f"news_raw_{safe_keyword}_{yyyymm}.csv"

    # 파일이 없으면 헤더도 함께 작성
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["keyword", "date", "title", "press", "content_summary", "url", "crawled_at"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(article)


# ────────────────────────────────────────────────────────────────────────────
# 중복 제목 처리
# ────────────────────────────────────────────────────────────────────────────
def is_similar_title_exists(conn: sqlite3.Connection, title: str, keyword: str) -> bool:
    """
    DB에 동일 키워드로 수집된 기사 중 제목 유사도가 임계값 이상인 항목이
    있는지 확인한다. 있으면 True(중복)를 반환한다.
    최신 것을 보관하기 위해 기존 유사 기사는 삭제 후 새로 저장한다.
    실제 삭제/교체 로직은 호출부(process_article)에서 담당한다.
    """
    cursor = conn.cursor()
    # 같은 키워드의 기사 제목만 비교 (성능 고려)
    cursor.execute(
        "SELECT id, title, date FROM news_articles WHERE keyword = ?",
        (keyword,),
    )
    rows = cursor.fetchall()
    for row_id, existing_title, existing_date in rows:
        if title_similarity(title, existing_title) >= TITLE_SIMILARITY_THRESHOLD:
            return True, row_id, existing_date
    return False, None, None


def process_article(conn: sqlite3.Connection, article: dict) -> bool:
    """
    단일 기사를 중복 처리 후 저장한다.

    처리 흐름:
    1. URL 중복이면 건너뛴다.
    2. 제목 유사도가 임계값 이상인 기존 기사가 있으면,
       현재 기사가 더 최신이면 기존을 삭제하고 새것을 저장한다.
       그렇지 않으면 건너뛴다.
    3. 중복 없으면 저장한다.
    반환값: 실제로 새 기사가 저장되면 True
    """
    cursor = conn.cursor()

    # ── 1. URL 중복 확인 ──────────────────────────────────────
    cursor.execute("SELECT 1 FROM news_articles WHERE url = ?", (article["url"],))
    if cursor.fetchone():
        return False

    # ── 2. 제목 유사도 중복 확인 ──────────────────────────────
    is_dup, dup_id, dup_date = is_similar_title_exists(conn, article["title"], article["keyword"])
    if is_dup:
        # 날짜 비교 (문자열 YYYY-MM-DD 형식이면 사전순 비교 가능)
        article_date = article.get("date") or ""
        if article_date > (dup_date or ""):
            # 현재 기사가 더 최신 → 기존 삭제 후 새것 저장
            cursor.execute("DELETE FROM news_articles WHERE id = ?", (dup_id,))
            conn.commit()
        else:
            # 기존이 더 최신이거나 날짜 불명 → 건너뜀
            return False

    # ── 3. 저장 ───────────────────────────────────────────────
    return save_article(conn, article)


# ────────────────────────────────────────────────────────────────────────────
# HTTP 요청 헬퍼 (재시도 포함)
# ────────────────────────────────────────────────────────────────────────────
def safe_get(url: str, params: dict = None, extra_headers: dict = None) -> requests.Response | None:
    """
    GET 요청을 실행한다. 실패 시 RETRY_WAIT 초 대기 후 최대 MAX_RETRIES 회 재시도.
    모든 시도가 실패하면 None을 반환하고 에러 로그를 기록한다.
    """
    headers = {**HEADERS}
    if extra_headers:
        headers.update(extra_headers)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            error_logger.error(
                "요청 실패 (시도 %d/%d) | URL: %s | 오류: %s",
                attempt, MAX_RETRIES, url, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
    return None


def random_delay() -> None:
    """요청 간격으로 랜덤 대기한다."""
    time.sleep(random.uniform(*REQUEST_DELAY))


# ────────────────────────────────────────────────────────────────────────────
# 날짜 파싱 유틸
# ────────────────────────────────────────────────────────────────────────────
def parse_date_str(raw: str) -> str:
    """
    다양한 날짜 형식을 'YYYY-MM-DD' 문자열로 정규화한다.
    파싱 실패 시 빈 문자열을 반환한다.
    """
    if not raw:
        return ""
    raw = raw.strip()

    # RFC 2822 형식 (API 응답): "Wed, 10 Jan 2024 09:00:00 +0900"
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # ISO 8601 / YYYY.MM.DD / YYYY-MM-DD 등
    patterns = [
        r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})",
        r"(\d{4})(\d{2})(\d{2})",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
            return f"{y}-{mo}-{d}"

    # "N시간 전", "N일 전" 등 상대 표현은 오늘 날짜로 처리
    if "전" in raw or "방금" in raw:
        return date.today().strftime("%Y-%m-%d")

    return ""


def date_to_yyyymm(date_str: str) -> str:
    """'YYYY-MM-DD' 문자열에서 'YYYYMM' 부분을 반환한다."""
    if date_str and len(date_str) >= 7:
        return date_str[:7].replace("-", "")
    return datetime.now().strftime("%Y%m")


# ────────────────────────────────────────────────────────────────────────────
# 네이버 검색 API 수집
# ────────────────────────────────────────────────────────────────────────────
API_BASE_URL = "https://openapi.naver.com/v1/search/news.json"
API_MAX_DISPLAY = 100   # 1회 요청 최대 결과 수
API_MAX_START = 1000    # API 허용 최대 start 값


def fetch_via_api(keyword: str, start_date: str, end_date: str) -> list[dict]:
    """
    네이버 뉴스 검색 API를 사용해 기사 목록을 수집한다.

    Args:
        keyword: 검색 키워드
        start_date: 수집 시작일 (YYYY-MM-DD)
        end_date:   수집 종료일 (YYYY-MM-DD)

    Returns:
        기사 딕셔너리 리스트 (keyword, date, title, press, content_summary, url, crawled_at)
    """
    api_headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }

    articles = []
    start = 1
    crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    while start <= API_MAX_START:
        params = {
            "query": keyword,
            "display": API_MAX_DISPLAY,
            "start": start,
            "sort": "date",  # 최신순
        }
        resp = safe_get(API_BASE_URL, params=params, extra_headers=api_headers)
        if resp is None:
            break

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            art_date = parse_date_str(item.get("pubDate", ""))

            # 수집 기간 필터링
            if art_date and art_date < start_date:
                # 날짜순 정렬이므로 이후 항목은 더 오래된 것 → 조기 종료
                return articles
            if art_date and art_date > end_date:
                continue

            # HTML 태그 제거
            title = re.sub(r"<[^>]+>", "", item.get("title", "")).strip()
            summary = re.sub(r"<[^>]+>", "", item.get("description", "")).strip()
            press = item.get("originallink", "").split("/")[2] if item.get("originallink") else ""

            articles.append({
                "keyword": keyword,
                "date": art_date,
                "title": title,
                "press": press,
                "content_summary": summary,
                "url": item.get("link", ""),
                "crawled_at": crawled_at,
            })

        # 전체 결과 수 확인
        total = data.get("total", 0)
        if start + API_MAX_DISPLAY > min(total, API_MAX_START):
            break

        start += API_MAX_DISPLAY
        random_delay()

    return articles


# ────────────────────────────────────────────────────────────────────────────
# 네이버 뉴스 직접 크롤링
# ────────────────────────────────────────────────────────────────────────────
CRAWL_BASE_URL = "https://search.naver.com/search.naver"
CRAWL_PAGE_SIZE = 10    # 페이지당 결과 수
CRAWL_MAX_PAGES = 100   # 키워드당 최대 페이지 수 (1000건 상한)


def _parse_crawl_page(html: str, keyword: str, crawled_at: str) -> list[dict]:
    """
    네이버 뉴스 검색 결과 HTML에서 기사 정보를 파싱한다.
    반환: 기사 딕셔너리 리스트
    """
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    # 뉴스 결과 영역 탐색 (뉴스 검색 결과 리스트)
    # 네이버 뉴스 검색 결과: .news_wrap, .list_news > .bx 등 다양한 구조 처리
    news_items = soup.select("ul.list_news > li.bx")
    if not news_items:
        # 대안 선택자
        news_items = soup.select(".news_wrap")

    for item in news_items:
        # 제목 및 URL
        title_tag = (
            item.select_one("a.news_tit")
            or item.select_one(".news_tit")
            or item.select_one("a[class*='tit']")
        )
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        url = title_tag.get("href", "")
        if not url:
            continue

        # 언론사
        press_tag = (
            item.select_one(".info_group a.press")
            or item.select_one(".press")
            or item.select_one("a[class*='press']")
        )
        press = press_tag.get_text(strip=True) if press_tag else ""

        # 날짜
        date_tag = (
            item.select_one(".info_group span.info")
            or item.select_one("span.date")
            or item.select_one("span[class*='date']")
        )
        raw_date = date_tag.get_text(strip=True) if date_tag else ""
        art_date = parse_date_str(raw_date)

        # 요약
        desc_tag = (
            item.select_one(".dsc_txt_wrap")
            or item.select_one(".dsc_txt")
            or item.select_one("a.dsc_txt_wrap")
        )
        summary = desc_tag.get_text(strip=True) if desc_tag else ""

        articles.append({
            "keyword": keyword,
            "date": art_date,
            "title": title,
            "press": press,
            "content_summary": summary,
            "url": url,
            "crawled_at": crawled_at,
        })

    return articles


def fetch_via_crawl(keyword: str, start_date: str, end_date: str) -> list[dict]:
    """
    네이버 뉴스 검색 페이지를 직접 크롤링해 기사 목록을 수집한다.

    URL 패턴:
        https://search.naver.com/search.naver?where=news&query={keyword}
            &start={offset}&ds={date}&de={date}&sort=1

    Args:
        keyword:    검색 키워드
        start_date: 수집 시작일 (YYYY-MM-DD)
        end_date:   수집 종료일 (YYYY-MM-DD)

    Returns:
        기사 딕셔너리 리스트
    """
    # 날짜 형식 변환: YYYY-MM-DD → YYYY.MM.DD (네이버 검색 파라미터 형식)
    ds = start_date.replace("-", ".")
    de = end_date.replace("-", ".")

    articles = []
    crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    encoded_keyword = quote(keyword)

    for page in range(CRAWL_MAX_PAGES):
        offset = page * CRAWL_PAGE_SIZE + 1

        url = (
            f"{CRAWL_BASE_URL}?where=news"
            f"&query={encoded_keyword}"
            f"&start={offset}"
            f"&ds={ds}&de={de}"
            f"&sort=1"  # 날짜순
        )

        resp = safe_get(url)
        if resp is None:
            break

        page_articles = _parse_crawl_page(resp.text, keyword, crawled_at)

        if not page_articles:
            # 더 이상 결과 없음
            break

        articles.extend(page_articles)

        # 다음 페이지가 없는지 확인 (네이버는 특정 페이지 이후 결과 미제공)
        soup = BeautifulSoup(resp.text, "html.parser")
        next_btn = soup.select_one("a.btn_next") or soup.select_one(".btn_next:not(.disabled)")
        if not next_btn or "disabled" in next_btn.get("class", []):
            break

        random_delay()

    return articles


# ────────────────────────────────────────────────────────────────────────────
# 키워드 단위 수집 진입점
# ────────────────────────────────────────────────────────────────────────────
def collect_keyword(
    conn: sqlite3.Connection,
    keyword: str,
    start_date: str,
    end_date: str,
) -> int:
    """
    단일 키워드에 대해 기사를 수집하고 DB/CSV에 저장한다.

    Returns:
        실제로 새로 저장된 기사 건수
    """
    print(f"  ▶ 키워드: [{keyword}] 수집 시작 ({'API' if USE_API else '크롤링'} 방식)")

    # 수집 방식 결정
    if USE_API:
        raw_articles = fetch_via_api(keyword, start_date, end_date)
    else:
        raw_articles = fetch_via_crawl(keyword, start_date, end_date)

    saved_count = 0
    for article in raw_articles:
        # 필수 필드 검증
        if not article.get("title") or not article.get("url"):
            continue

        # DB 저장 (중복 처리 포함)
        is_new = process_article(conn, article)
        if is_new:
            # CSV 백업 저장
            yyyymm = date_to_yyyymm(article.get("date", ""))
            save_to_csv(article, keyword, yyyymm)
            saved_count += 1

    print(f"     → 수집: {len(raw_articles)}건 | 신규 저장: {saved_count}건")
    return saved_count


# ────────────────────────────────────────────────────────────────────────────
# 전체 파이프라인 실행
# ────────────────────────────────────────────────────────────────────────────
def run_crawl(
    start_date: str = "2005-01-01",
    end_date: str = None,
    keyword_groups: list[str] = None,
) -> dict:
    """
    전체 키워드에 대해 뉴스 수집을 실행한다.

    Args:
        start_date:     수집 시작일 (YYYY-MM-DD), 기본값 2005-01-01
        end_date:       수집 종료일 (YYYY-MM-DD), 기본값 오늘
        keyword_groups: 수집할 그룹명 리스트 (예: ["A_상토관련"]),
                        None이면 전체 그룹 수집

    Returns:
        {그룹명: {키워드: 저장 건수}} 형태의 결과 딕셔너리
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"  네이버 뉴스 수집 시작")
    print(f"  수집 기간: {start_date} ~ {end_date}")
    print(f"  수집 방식: {'네이버 검색 API' if USE_API else '웹 직접 크롤링'}")
    print(f"  DB 경로: {DB_PATH}")
    print("=" * 60)

    conn = sqlite3.connect(str(DB_PATH))
    init_db(conn)

    results = {}
    total_saved = 0

    try:
        for group_name, keywords in KEYWORD_GROUPS.items():
            # 그룹 필터 적용
            if keyword_groups and group_name not in keyword_groups:
                continue

            print(f"\n[그룹: {group_name}]")
            results[group_name] = {}

            for keyword in keywords:
                try:
                    count = collect_keyword(conn, keyword, start_date, end_date)
                    results[group_name][keyword] = count
                    total_saved += count
                except Exception as exc:
                    error_logger.error(
                        "키워드 수집 중 예외 발생 | 키워드: %s | 오류: %s",
                        keyword, exc,
                    )
                    print(f"     [오류] {keyword} 수집 실패: {exc}")
                    results[group_name][keyword] = 0

                random_delay()

    finally:
        conn.close()

    # ── 수집 결과 요약 출력 ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  수집 완료 요약")
    print("=" * 60)
    for group_name, kw_counts in results.items():
        group_total = sum(kw_counts.values())
        print(f"\n  [{group_name}] 합계: {group_total}건")
        for kw, cnt in kw_counts.items():
            print(f"    - {kw}: {cnt}건")
    print(f"\n  ★ 전체 신규 저장 건수: {total_saved}건")
    print("=" * 60)

    return results


# ────────────────────────────────────────────────────────────────────────────
# main: 샘플 수집 (2019년 이후)
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    실행 진입점.
    2019-01-01 이후 데이터를 전체 키워드 그룹에 대해 샘플 수집한다.
    """
    run_crawl(
        start_date="2019-01-01",
        end_date=date.today().strftime("%Y-%m-%d"),
        keyword_groups=None,  # 전체 그룹 수집
    )


if __name__ == "__main__":
    main()
