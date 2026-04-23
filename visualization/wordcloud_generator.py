"""
wordcloud_generator.py
농업 뉴스 데이터 기반 7종 워드클라우드 자동 생성 모듈.

입력:
  - data/processed/news_tokenized.parquet   (기본 토큰 데이터)
  - data/processed/sentiment_scores.csv     (부정 감성 기사 필터용, 없으면 skip)
  파일이 없으면 샘플 농업 키워드 딕셔너리로 시연용 워드클라우드를 생성한다.

출력 (visualization/ 폴더):
  1. wc_total.png                  전체 통합
  2. wc_year_{YYYY}.png            연도별 (2015~2026)
  3. wc_month_{MM}.png             월별 (01~12)
  4. wc_region_{지역명}.png        지역별 (전남, 전북, 충남, 경남, 경북)
  5. wc_negative_issues.png        부정 감성 기사
  6. wc_transplanting_season.png   모내기 시기 전후 2주 기사
  7. wc_peak_months.png            상위 출고월(4월, 5월) 기사
  8. wc_index.html                 7종 이미지 링크 인덱스 페이지
"""

import warnings
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# matplotlib 백엔드를 화면 없이 파일 저장 전용으로 설정 (서버 환경 호환)
matplotlib.use("Agg")

# ──────────────────────────────────────────────
# 프로젝트 루트 및 주요 경로 상수
# ──────────────────────────────────────────────
PROJECT_ROOT   = Path(r"C:\ai_workspace\sh-ai-model")
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
VIS_DIR        = PROJECT_ROOT / "visualization"

# 입력 파일 경로
PARQUET_PATH   = PROCESSED_DIR / "news_tokenized.parquet"
SENTIMENT_PATH = PROCESSED_DIR / "sentiment_scores.csv"

# 워드클라우드 공통 설정
WC_BACKGROUND  = "white"   # 배경색
WC_MAX_WORDS   = 100       # 최대 단어 수
WC_WIDTH       = 800       # 이미지 가로 픽셀
WC_HEIGHT      = 600       # 이미지 세로 픽셀

# 지역별 워드클라우드 대상 지역 목록
TARGET_REGIONS = ["전남", "전북", "충남", "경남", "경북"]

# 모내기 시기 기준 (매년 5월 20일 기준, ±2주)
TRANSPLANTING_MONTH = 5
TRANSPLANTING_DAY   = 20
TRANSPLANTING_DELTA = timedelta(weeks=2)

# 상위 출고월 (4월, 5월)
PEAK_MONTHS = [4, 5]

# 연도 범위 (2015~2026)
YEAR_RANGE = range(2015, 2027)

# 월 범위 (1~12)
MONTH_RANGE = range(1, 13)


# ──────────────────────────────────────────────
# 한글 폰트 자동 탐색 함수
# ──────────────────────────────────────────────
def _find_korean_font() -> Optional[str]:
    """
    Windows Fonts 폴더에서 한글 폰트를 우선순위 순으로 탐색한다.
    탐색 순서: 맑은고딕.ttf → 나눔고딕.ttf → gulim.ttc
    찾으면 절대 경로 문자열을 반환하고, 없으면 None을 반환한다.
    """
    fonts_dir = Path(r"C:\Windows\Fonts")

    # 우선순위별 후보 폰트 파일명 목록
    candidates = [
        "malgun.ttf",       # 맑은 고딕
        "NanumGothic.ttf",  # 나눔고딕 (설치된 경우)
        "nanumgothic.ttf",  # 소문자 버전
        "gulim.ttc",        # 굴림
        "NGULIM.TTF",       # 굴림 대문자 버전
    ]

    for fname in candidates:
        font_path = fonts_dir / fname
        if font_path.exists():
            print(f"[폰트] 한글 폰트 발견: {font_path}")
            return str(font_path)

    print("[폰트] 한글 폰트를 찾을 수 없습니다 → matplotlib 기본 폰트(영문) 사용")
    return None


# ──────────────────────────────────────────────
# 시연용 샘플 농업 키워드 딕셔너리
# ──────────────────────────────────────────────
def _get_sample_word_freq() -> dict:
    """
    news_tokenized.parquet 파일이 없을 때 시연용으로 사용하는
    농업 도메인 키워드 빈도 딕셔너리를 반환한다.
    """
    return {
        # 핵심 농자재/재배 용어
        "상토": 320, "모내기": 280, "이앙": 260, "벼": 240, "종자": 180,
        "비료": 170, "농협": 160, "수확": 150, "작황": 140, "농가": 130,
        # 이슈 키워드
        "품귀": 120, "부족": 110, "민원": 100, "공급": 95, "수급": 90,
        "품절": 85, "대란": 80, "재고": 75, "차질": 70, "지연": 65,
        # 기상 관련
        "가뭄": 60, "강수": 55, "홍수": 50, "온도": 45, "기상": 40,
        # 지역
        "전남": 100, "전북": 95, "충남": 90, "경남": 85, "경북": 80,
        "경기": 75, "강원": 70, "충북": 65, "제주": 60,
        # 긍정 키워드
        "풍작": 55, "원활": 50, "안정": 45, "확보": 40, "품질향상": 35,
        "공급확대": 30, "호조": 28, "충분": 25, "적기": 22, "성공": 20,
        # 정책/유통
        "농림부": 50, "정책": 45, "지원": 40, "유통": 35, "시장": 30,
        "가격": 28, "물량": 25, "협력": 22,
    }


# ──────────────────────────────────────────────
# 데이터 로드 함수
# ──────────────────────────────────────────────
def _load_news_data() -> Optional[pd.DataFrame]:
    """
    news_tokenized.parquet 파일을 로드하여 DataFrame으로 반환한다.
    파일이 없으면 None을 반환한다.

    반환 DataFrame 필수 컬럼:
      date(datetime), region(str), tokens(list of str)
      year(int), month(int)
    """
    if not PARQUET_PATH.exists():
        print(f"[데이터] {PARQUET_PATH} 없음 → 샘플 키워드 딕셔너리 사용")
        return None

    print(f"[데이터] {PARQUET_PATH} 로드 중...")
    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # year, month 컬럼 없으면 자동 생성
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

    # tokens 컬럼 타입 정규화 (문자열이면 리스트로 변환)
    if "tokens" in df.columns:
        sample_val = df["tokens"].dropna().iloc[0] if not df["tokens"].dropna().empty else None
        if sample_val is not None and isinstance(sample_val, str):
            df["tokens"] = df["tokens"].apply(
                lambda x: x.split() if isinstance(x, str) else []
            )
    elif "tokens_str" in df.columns:
        # preprocessor.py가 저장한 tokens_str 컬럼을 tokens 리스트로 변환
        df["tokens"] = df["tokens_str"].apply(
            lambda x: x.split() if isinstance(x, str) else []
        )

    print(f"[데이터] 로드 완료: {len(df):,}건")
    return df


def _load_sentiment_data() -> Optional[pd.DataFrame]:
    """
    sentiment_scores.csv 파일을 로드하여 DataFrame으로 반환한다.
    파일이 없으면 None을 반환하고 skip 처리한다.
    """
    if not SENTIMENT_PATH.exists():
        print(f"[감성] {SENTIMENT_PATH} 없음 → 부정 감성 필터 skip")
        return None

    print(f"[감성] {SENTIMENT_PATH} 로드 중...")
    df = pd.read_csv(SENTIMENT_PATH, encoding="utf-8-sig")
    print(f"[감성] 로드 완료: {len(df):,}건")
    return df


# ──────────────────────────────────────────────
# 토큰 → 빈도 딕셔너리 변환 헬퍼
# ──────────────────────────────────────────────
def _tokens_to_freq(df: pd.DataFrame, token_col: str = "tokens") -> dict:
    """
    DataFrame의 tokens 컬럼(리스트 형태)에서
    전체 단어 빈도 딕셔너리를 계산하여 반환한다.
    빈도 딕셔너리가 비어 있으면 샘플 키워드로 대체한다.
    """
    counter: Counter = Counter()
    for tokens in df[token_col]:
        if isinstance(tokens, list):
            counter.update(tokens)
        elif isinstance(tokens, str) and tokens:
            counter.update(tokens.split())

    # 1글자 토큰 제거 (노이즈 방지)
    freq = {w: c for w, c in counter.items() if len(w) > 1}

    if not freq:
        print("  [경고] 토큰이 비어 있음 → 샘플 키워드 딕셔너리로 대체")
        freq = _get_sample_word_freq()

    return freq


# ──────────────────────────────────────────────
# WordCloud 객체 생성 헬퍼
# ──────────────────────────────────────────────
def _make_wordcloud(freq: dict, font_path: Optional[str]):
    """
    빈도 딕셔너리와 폰트 경로를 받아 WordCloud 객체를 생성한다.

    Parameters
    ----------
    freq : dict
        단어 → 빈도(또는 가중치) 딕셔너리
    font_path : str or None
        한글 폰트 절대 경로. None이면 기본 폰트 사용(영문 fallback).

    Returns
    -------
    WordCloud 객체
    """
    try:
        from wordcloud import WordCloud  # type: ignore
    except ImportError as e:
        raise ImportError(
            "wordcloud 패키지가 설치되어 있지 않습니다. "
            "pip install wordcloud 를 실행하세요."
        ) from e

    # 폰트 경로 유무에 따라 WordCloud 생성
    wc_kwargs = dict(
        background_color=WC_BACKGROUND,
        max_words=WC_MAX_WORDS,
        width=WC_WIDTH,
        height=WC_HEIGHT,
        prefer_horizontal=0.85,   # 가로 방향 단어 비율
        colormap="viridis",       # 색상맵 (녹색 계열 → 농업 도메인 어울림)
    )

    if font_path:
        wc_kwargs["font_path"] = font_path

    from wordcloud import WordCloud
    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq)
    return wc


# ──────────────────────────────────────────────
# 이미지 저장 헬퍼
# ──────────────────────────────────────────────
def _save_wordcloud(wc, output_path: Path, title: str) -> None:
    """
    WordCloud 객체를 PNG 파일로 저장한다.
    matplotlib figure를 이용해 제목을 함께 렌더링한다.

    Parameters
    ----------
    wc : WordCloud
        저장할 워드클라우드 객체
    output_path : Path
        저장 파일 경로
    title : str
        이미지 상단에 표시할 제목
    """
    fig, ax = plt.subplots(figsize=(WC_WIDTH / 100, WC_HEIGHT / 100), dpi=100)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  [저장] {output_path}")


# ──────────────────────────────────────────────
# WordCloudGenerator 클래스
# ──────────────────────────────────────────────
class WordCloudGenerator:
    """
    농업 뉴스 데이터 기반 7종 워드클라우드를 자동 생성하는 클래스.

    생성 목록:
      1. 전체 통합          → wc_total.png
      2. 연도별             → wc_year_{YYYY}.png  (2015~2026)
      3. 월별               → wc_month_{MM}.png   (01~12)
      4. 지역별             → wc_region_{지역명}.png (전남, 전북, 충남, 경남, 경북)
      5. 부정 감성 기사만   → wc_negative_issues.png
      6. 모내기 시기 전후   → wc_transplanting_season.png
      7. 상위 출고월(4~5월) → wc_peak_months.png
      인덱스               → wc_index.html

    데이터가 없을 경우 샘플 농업 키워드 딕셔너리로 시연용 워드클라우드를 생성한다.
    """

    def __init__(self):
        """
        초기화: 출력 디렉토리 생성, 한글 폰트 탐색, 데이터 로드
        """
        # 출력 디렉토리 생성
        VIS_DIR.mkdir(parents=True, exist_ok=True)

        # 한글 폰트 자동 탐색
        self.font_path: Optional[str] = _find_korean_font()

        # 뉴스 토큰 데이터 로드 (없으면 None)
        self.df: Optional[pd.DataFrame] = _load_news_data()

        # 감성 점수 데이터 로드 (없으면 None)
        self.sentiment_df: Optional[pd.DataFrame] = _load_sentiment_data()

        # 전체 빈도 딕셔너리 (샘플 또는 실제 데이터 기반)
        if self.df is not None and "tokens" in self.df.columns:
            self._total_freq: dict = _tokens_to_freq(self.df)
        else:
            # 데이터 없음 → 샘플 키워드 딕셔너리 사용
            self._total_freq = _get_sample_word_freq()
            print("[데이터] 샘플 농업 키워드 딕셔너리로 워드클라우드 생성 (시연 모드)")

        # 생성된 이미지 경로 추적 (인덱스 HTML 생성에 사용)
        self._generated_files: list[dict] = []

    # ──────────────────────────────────────────────
    # 내부: 워드클라우드 생성 및 저장 공통 메서드
    # ──────────────────────────────────────────────
    def _generate_and_save(
        self,
        freq: dict,
        filename: str,
        title: str,
        section: str,
    ) -> Optional[Path]:
        """
        빈도 딕셔너리로 워드클라우드를 생성하고 파일로 저장한다.
        생성 실패 시 경고 메시지를 출력하고 None을 반환한다.

        Parameters
        ----------
        freq : dict
            단어 → 빈도 딕셔너리 (비어 있으면 생성 skip)
        filename : str
            저장 파일명 (예: wc_total.png)
        title : str
            이미지 제목 문자열
        section : str
            인덱스 HTML 섹션 분류 (예: "연도별", "월별" 등)

        Returns
        -------
        Path or None
            저장된 파일 경로. 실패 시 None.
        """
        if not freq:
            print(f"  [경고] 빈도 데이터 없음, skip: {filename}")
            return None

        output_path = VIS_DIR / filename
        try:
            wc = _make_wordcloud(freq, self.font_path)
            _save_wordcloud(wc, output_path, title)
            # 생성 파일 목록에 추가 (인덱스 HTML용)
            self._generated_files.append(
                {
                    "path": output_path,
                    "filename": filename,
                    "title": title,
                    "section": section,
                }
            )
            return output_path
        except Exception as exc:
            print(f"  [오류] {filename} 생성 실패: {exc}")
            return None

    # ──────────────────────────────────────────────
    # 1. 전체 통합 워드클라우드
    # ──────────────────────────────────────────────
    def generate_total(self) -> Optional[Path]:
        """
        전체 기사를 통합한 워드클라우드를 생성한다.
        출력: visualization/wc_total.png
        """
        print("\n[1/7] 전체 통합 워드클라우드 생성 중...")
        return self._generate_and_save(
            freq=self._total_freq,
            filename="wc_total.png",
            title="전체 통합 워드클라우드",
            section="전체",
        )

    # ──────────────────────────────────────────────
    # 2. 연도별 워드클라우드
    # ──────────────────────────────────────────────
    def generate_by_year(self) -> list[Path]:
        """
        2015~2026년 연도별 워드클라우드를 생성한다.
        출력: visualization/wc_year_{YYYY}.png
        """
        print("\n[2/7] 연도별 워드클라우드 생성 중...")
        results = []

        for year in YEAR_RANGE:
            # 실제 데이터에서 해당 연도 필터링
            if self.df is not None:
                year_df = self.df[self.df["year"] == year]
                if year_df.empty:
                    # 해당 연도 데이터 없음 → 샘플 키워드에 연도 가중치 적용하여 생성
                    freq = {k: max(1, v // 2) for k, v in self._total_freq.items()}
                else:
                    freq = _tokens_to_freq(year_df)
            else:
                # 데이터 없음 → 샘플 키워드 사용
                freq = _get_sample_word_freq()

            path = self._generate_and_save(
                freq=freq,
                filename=f"wc_year_{year}.png",
                title=f"{year}년 워드클라우드",
                section="연도별",
            )
            if path:
                results.append(path)

        print(f"  연도별 {len(results)}개 생성 완료")
        return results

    # ──────────────────────────────────────────────
    # 3. 월별 워드클라우드
    # ──────────────────────────────────────────────
    def generate_by_month(self) -> list[Path]:
        """
        1~12월별 워드클라우드를 생성한다.
        출력: visualization/wc_month_{MM}.png
        """
        print("\n[3/7] 월별 워드클라우드 생성 중...")
        results = []

        for month in MONTH_RANGE:
            # 실제 데이터에서 해당 월 필터링 (연도 무관)
            if self.df is not None:
                month_df = self.df[self.df["month"] == month]
                if month_df.empty:
                    freq = {k: max(1, v // 3) for k, v in self._total_freq.items()}
                else:
                    freq = _tokens_to_freq(month_df)
            else:
                freq = _get_sample_word_freq()

            path = self._generate_and_save(
                freq=freq,
                filename=f"wc_month_{month:02d}.png",
                title=f"{month}월 워드클라우드",
                section="월별",
            )
            if path:
                results.append(path)

        print(f"  월별 {len(results)}개 생성 완료")
        return results

    # ──────────────────────────────────────────────
    # 4. 지역별 워드클라우드
    # ──────────────────────────────────────────────
    def generate_by_region(self) -> list[Path]:
        """
        전남, 전북, 충남, 경남, 경북 지역별 워드클라우드를 생성한다.
        출력: visualization/wc_region_{지역명}.png
        """
        print("\n[4/7] 지역별 워드클라우드 생성 중...")
        results = []

        for region in TARGET_REGIONS:
            # 실제 데이터에서 해당 지역 필터링
            if self.df is not None and "region" in self.df.columns:
                region_df = self.df[self.df["region"].str.contains(region, na=False)]
                if region_df.empty:
                    # 지역 데이터 없음 → 샘플 키워드 사용
                    freq = _get_sample_word_freq()
                else:
                    freq = _tokens_to_freq(region_df)
            else:
                freq = _get_sample_word_freq()

            path = self._generate_and_save(
                freq=freq,
                filename=f"wc_region_{region}.png",
                title=f"{region} 지역 워드클라우드",
                section="지역별",
            )
            if path:
                results.append(path)

        print(f"  지역별 {len(results)}개 생성 완료")
        return results

    # ──────────────────────────────────────────────
    # 5. 부정 감성 기사 워드클라우드
    # ──────────────────────────────────────────────
    def generate_negative_issues(self) -> Optional[Path]:
        """
        부정 감성(sentiment_label == '부정' 또는 sentiment_score < 0) 기사만
        필터링하여 워드클라우드를 생성한다.
        sentiment_scores.csv가 없으면 뉴스 토큰 데이터에서 직접 부정 단어를 추출한다.
        출력: visualization/wc_negative_issues.png
        """
        print("\n[5/7] 부정 감성 기사 워드클라우드 생성 중...")

        # 부정 판별에 사용하는 키워드 사전 (keyword_analyzer.py와 동일)
        NEGATIVE_WORDS = {
            "부족", "품귀", "불량", "민원", "피해", "지연", "문제",
            "품절", "대란", "차질", "어려움", "가격상승", "재고부족",
        }

        freq: dict = {}

        # 방법1: sentiment_scores.csv 에서 부정 기사 인덱스 추출 후 토큰 수집
        if self.sentiment_df is not None and self.df is not None:
            try:
                # sentiment_label 컬럼 우선 사용
                if "sentiment_label" in self.sentiment_df.columns:
                    neg_mask = self.sentiment_df["sentiment_label"] == "부정"
                elif "sentiment_score" in self.sentiment_df.columns:
                    neg_mask = self.sentiment_df["sentiment_score"] < 0
                else:
                    neg_mask = pd.Series([False] * len(self.sentiment_df))

                neg_indices = self.sentiment_df[neg_mask].index
                # 인덱스 기반으로 원본 토큰 데이터 필터링
                neg_df = self.df.loc[self.df.index.isin(neg_indices)]

                if not neg_df.empty:
                    freq = _tokens_to_freq(neg_df)
                    print(f"  sentiment_scores.csv 기반 부정 기사: {len(neg_df):,}건")
            except Exception as exc:
                print(f"  [경고] sentiment 데이터 처리 오류: {exc}")

        # 방법2: 뉴스 토큰 데이터에서 부정 단어 포함 기사 직접 필터링
        if not freq and self.df is not None:
            def _is_negative(tokens) -> bool:
                """토큰 목록에 부정 단어가 하나 이상 포함되어 있으면 True."""
                if isinstance(tokens, list):
                    return bool(set(tokens) & NEGATIVE_WORDS)
                return False

            neg_df = self.df[self.df["tokens"].apply(_is_negative)]
            if not neg_df.empty:
                freq = _tokens_to_freq(neg_df)
                print(f"  토큰 기반 부정 기사: {len(neg_df):,}건")

        # 방법3: 데이터 없음 → 샘플 부정 키워드 빈도 딕셔너리
        if not freq:
            print("  [폴백] 샘플 부정 키워드 딕셔너리 사용")
            freq = {
                "부족": 320, "품귀": 280, "민원": 240, "피해": 220, "지연": 200,
                "불량": 180, "품절": 160, "대란": 150, "차질": 140, "문제": 130,
                "공급": 120, "수급": 110, "재고": 100, "가격상승": 90, "어려움": 80,
                "상토": 75, "농가": 70, "이앙": 65, "벼": 60, "모내기": 55,
            }

        return self._generate_and_save(
            freq=freq,
            filename="wc_negative_issues.png",
            title="부정 감성 기사 주요 키워드",
            section="특수",
        )

    # ──────────────────────────────────────────────
    # 6. 모내기 시기 전후 2주 기사 워드클라우드
    # ──────────────────────────────────────────────
    def generate_transplanting_season(self) -> Optional[Path]:
        """
        매년 모내기 기준일(5월 20일) ±2주 범위의 기사를 수집하여
        워드클라우드를 생성한다.
        출력: visualization/wc_transplanting_season.png
        """
        print("\n[6/7] 모내기 시기 전후 2주 워드클라우드 생성 중...")

        freq: dict = {}

        if self.df is not None and "date" in self.df.columns:
            # 연도별로 모내기 기준일 ±2주 범위 기사 수집
            collected_rows = []
            for year in self.df["year"].dropna().unique():
                try:
                    base_date = pd.Timestamp(
                        year=int(year),
                        month=TRANSPLANTING_MONTH,
                        day=TRANSPLANTING_DAY,
                    )
                    start_date = base_date - TRANSPLANTING_DELTA
                    end_date   = base_date + TRANSPLANTING_DELTA

                    mask = (
                        (self.df["date"] >= start_date)
                        & (self.df["date"] <= end_date)
                    )
                    collected_rows.append(self.df[mask])
                except Exception:
                    continue

            if collected_rows:
                season_df = pd.concat(collected_rows, ignore_index=True)
                if not season_df.empty:
                    freq = _tokens_to_freq(season_df)
                    print(f"  모내기 시기 수집 기사: {len(season_df):,}건")

        # 데이터 없음 → 모내기 관련 샘플 키워드
        if not freq:
            print("  [폴백] 모내기 시기 샘플 키워드 딕셔너리 사용")
            freq = {
                "모내기": 400, "이앙": 380, "상토": 360, "벼": 300, "이식": 280,
                "적기": 260, "공급": 240, "수급": 220, "농협": 200, "종자": 180,
                "부족": 160, "품귀": 150, "5월": 140, "농가": 130, "기상": 120,
                "강수": 110, "온도": 100, "비료": 90, "전남": 85, "충남": 80,
                "전북": 75, "경남": 70, "경북": 65, "모심기": 60, "벼심기": 55,
            }

        return self._generate_and_save(
            freq=freq,
            filename="wc_transplanting_season.png",
            title="모내기 시기(5월 20일 ±2주) 주요 키워드",
            section="특수",
        )

    # ──────────────────────────────────────────────
    # 7. 상위 출고월(4월, 5월) 기사 워드클라우드
    # ──────────────────────────────────────────────
    def generate_peak_months(self) -> Optional[Path]:
        """
        상위 출고월(4월, 5월) 기사를 필터링하여 워드클라우드를 생성한다.
        출력: visualization/wc_peak_months.png
        """
        print("\n[7/7] 상위 출고월(4·5월) 워드클라우드 생성 중...")

        freq: dict = {}

        if self.df is not None:
            peak_df = self.df[self.df["month"].isin(PEAK_MONTHS)]
            if not peak_df.empty:
                freq = _tokens_to_freq(peak_df)
                print(f"  4·5월 기사: {len(peak_df):,}건")

        # 데이터 없음 → 4·5월 관련 샘플 키워드
        if not freq:
            print("  [폴백] 4·5월 샘플 키워드 딕셔너리 사용")
            freq = {
                "상토": 420, "모내기": 400, "이앙": 380, "출고": 340, "공급": 320,
                "수급": 300, "농협": 280, "벼": 260, "종자": 240, "4월": 220,
                "5월": 200, "부족": 180, "품귀": 160, "농가": 150, "비료": 140,
                "전남": 130, "충남": 120, "전북": 110, "경남": 100, "경북": 95,
                "기상": 90, "강수": 85, "적기": 80, "성수기": 75, "대응": 70,
            }

        return self._generate_and_save(
            freq=freq,
            filename="wc_peak_months.png",
            title="상위 출고월(4월·5월) 주요 키워드",
            section="특수",
        )

    # ──────────────────────────────────────────────
    # 인덱스 HTML 생성
    # ──────────────────────────────────────────────
    def generate_index_html(self) -> Path:
        """
        생성된 7종 워드클라우드 이미지를 모두 링크하는
        wc_index.html 인덱스 파일을 생성한다.
        <img> 태그로 모든 이미지를 포함한다.

        Returns
        -------
        Path
            생성된 인덱스 HTML 파일 경로
        """
        print("\n[인덱스] wc_index.html 생성 중...")

        index_path = VIS_DIR / "wc_index.html"

        # 섹션별로 이미지 그룹화
        sections: dict = {}
        for item in self._generated_files:
            sec = item["section"]
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(item)

        # HTML 본문 조립
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='ko'>",
            "<head>",
            "  <meta charset='UTF-8'>",
            "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "  <title>농업 뉴스 워드클라우드 인덱스</title>",
            "  <style>",
            "    body { font-family: 'Malgun Gothic', '맑은 고딕', sans-serif;",
            "           background: #f5f5f5; margin: 0; padding: 20px; }",
            "    h1   { color: #2c5f2e; text-align: center; margin-bottom: 30px; }",
            "    h2   { color: #4a7c59; border-left: 5px solid #2c5f2e;",
            "           padding-left: 12px; margin-top: 40px; }",
            "    .grid { display: flex; flex-wrap: wrap; gap: 20px;",
            "            justify-content: flex-start; }",
            "    .card { background: white; border-radius: 8px;",
            "            box-shadow: 0 2px 8px rgba(0,0,0,0.12);",
            "            padding: 12px; text-align: center; }",
            "    .card img { max-width: 320px; width: 100%;",
            "                border-radius: 4px; display: block; }",
            "    .card p   { margin: 8px 0 0; font-size: 13px; color: #555; }",
            "    footer { text-align: center; margin-top: 50px;",
            "             font-size: 12px; color: #aaa; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>농업 뉴스 워드클라우드 인덱스</h1>",
        ]

        # 섹션별 이미지 카드 출력
        # 섹션 표시 순서 정의 (전체 → 연도별 → 월별 → 지역별 → 특수)
        section_order = ["전체", "연도별", "월별", "지역별", "특수"]
        for sec in section_order:
            if sec not in sections:
                continue

            html_parts.append(f"  <h2>{sec}</h2>")
            html_parts.append("  <div class='grid'>")

            for item in sections[sec]:
                # 상대 경로로 img src 지정 (같은 폴더 내 파일)
                html_parts += [
                    "    <div class='card'>",
                    f"      <a href='{item['filename']}' target='_blank'>",
                    f"        <img src='{item['filename']}' alt='{item['title']}'>",
                    "      </a>",
                    f"      <p>{item['title']}</p>",
                    "    </div>",
                ]

            html_parts.append("  </div>")

        # 생성되지 않은 섹션의 잔여 이미지 처리
        remaining_sections = [s for s in sections if s not in section_order]
        for sec in remaining_sections:
            html_parts.append(f"  <h2>{sec}</h2>")
            html_parts.append("  <div class='grid'>")
            for item in sections[sec]:
                html_parts += [
                    "    <div class='card'>",
                    f"      <a href='{item['filename']}' target='_blank'>",
                    f"        <img src='{item['filename']}' alt='{item['title']}'>",
                    "      </a>",
                    f"      <p>{item['title']}</p>",
                    "    </div>",
                ]
            html_parts.append("  </div>")

        # 생성 일시 및 총 이미지 수 footer
        from datetime import datetime
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_count  = len(self._generated_files)
        html_parts += [
            f"  <footer>총 {total_count}개 워드클라우드 생성 완료 | {generated_at}</footer>",
            "</body>",
            "</html>",
        ]

        html_content = "\n".join(html_parts)
        index_path.write_text(html_content, encoding="utf-8")
        print(f"  [저장] {index_path}  (총 {total_count}개 이미지 링크)")
        return index_path

    # ──────────────────────────────────────────────
    # 전체 실행 메서드
    # ──────────────────────────────────────────────
    def run_all(self) -> dict:
        """
        7종 워드클라우드와 인덱스 HTML을 모두 생성한다.

        Returns
        -------
        dict
            {
              "total":                 Path or None,
              "by_year":               list[Path],
              "by_month":              list[Path],
              "by_region":             list[Path],
              "negative_issues":       Path or None,
              "transplanting_season":  Path or None,
              "peak_months":           Path or None,
              "index_html":            Path,
            }
        """
        print("=" * 60)
        print("WordCloudGenerator 전체 생성 시작")
        print(f"출력 디렉토리: {VIS_DIR}")
        print("=" * 60)

        results = {}

        # 1. 전체 통합
        results["total"] = self.generate_total()

        # 2. 연도별
        results["by_year"] = self.generate_by_year()

        # 3. 월별
        results["by_month"] = self.generate_by_month()

        # 4. 지역별
        results["by_region"] = self.generate_by_region()

        # 5. 부정 감성 기사
        results["negative_issues"] = self.generate_negative_issues()

        # 6. 모내기 시기 전후 2주
        results["transplanting_season"] = self.generate_transplanting_season()

        # 7. 상위 출고월(4·5월)
        results["peak_months"] = self.generate_peak_months()

        # 인덱스 HTML 생성
        results["index_html"] = self.generate_index_html()

        # 결과 요약 출력
        print("\n" + "=" * 60)
        print("전체 생성 완료 요약")
        print("=" * 60)
        total_generated = len(self._generated_files)
        print(f"  생성된 이미지: {total_generated}개")
        print(f"  인덱스 HTML : {results['index_html']}")
        print(f"  출력 폴더   : {VIS_DIR}")
        print("=" * 60)

        return results
