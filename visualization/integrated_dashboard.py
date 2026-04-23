# -*- coding: utf-8 -*-
"""
integrated_dashboard.py
────────────────────────────────────────────────────────────────────────────
Plotly 기반 4개 인터랙티브 차트를 생성하고 visualization/dashboard.html로
통합 출력하는 DashboardGenerator 클래스.

차트 목록:
  1. 3축 시계열  - 출고량 × 기온 × 뉴스 빈도
  2. 임계기온 돌파 이벤트 히트맵
  3. 지역별 모내기 시기 × 출고 상관관계 산점도
  4. 연도별 이슈 키워드 트렌드 히트맵
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# 프로젝트 루트 경로 (고정)
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\ai_workspace\sh-ai-model")


# ═══════════════════════════════════════════════════════════════════════════
#  DashboardGenerator
# ═══════════════════════════════════════════════════════════════════════════
class DashboardGenerator:
    """
    4개 Plotly 차트를 생성하고 단일 HTML 파일로 통합하는 클래스.

    사용 예시
    ----------
    gen = DashboardGenerator()
    gen.run()
    """

    # 차트에서 사용할 고정 색상 팔레트
    COLOR_BAR   = "#4C72B0"   # 출고량 막대 (파란색 계열)
    COLOR_TEMP  = "#DD4444"   # 기온 선 (빨간색)
    COLOR_NEWS  = "#2CA02C"   # 뉴스 점선 (녹색)

    # 지역별 모내기 적정 시기 (중간 날짜 기준, 월-일 형식)
    TRANSPLANT_SCHEDULE = {
        "경기/강원 중부": "06-05",
        "충청/강원 남부": "05-25",
        "경북/전북": "05-20",
        "경남/전남": "05-10",
        "제주": "04-25",
    }

    # 분석 대상 키워드 20개 (영농/농약/기상 관련)
    KEYWORDS_20 = [
        "모내기", "이앙", "제초제", "살충제", "살균제",
        "병해충", "가뭄", "폭염", "태풍", "집중호우",
        "친환경농업", "드론방제", "스마트팜", "농자재", "비료",
        "쌀값", "쌀 생산량", "농업직불금", "FTA", "기후변화",
    ]

    def __init__(self):
        # 경로 정의
        self.path_training   = PROJECT_ROOT / "data" / "processed" / "final_training_data.csv"
        self.path_weather    = PROJECT_ROOT / "data" / "raw" / "weather" / "daily_all_stations.csv"
        self.path_news_db    = PROJECT_ROOT / "data" / "master_db" / "news_corpus.db"
        self.path_keywords   = PROJECT_ROOT / "data" / "processed" / "keyword_trends.csv"
        self.output_dir      = PROJECT_ROOT / "visualization"
        self.output_html     = self.output_dir / "dashboard.html"

    # ──────────────────────────────────────────
    # 내부 유틸리티: 데이터 로드 / 샘플 생성
    # ──────────────────────────────────────────

    def _load_sales_daily(self) -> pd.DataFrame:
        """
        final_training_data.csv 또는 data/raw/sales/ 에서 일별 출고량을 로드.
        없으면 2019-01-01 ~ 2026-04-20 랜덤 샘플 생성.

        반환: date(datetime), daily_qty(float) 컬럼을 가진 DataFrame
        """
        if self.path_training.exists():
            try:
                df = pd.read_csv(self.path_training, encoding="utf-8", low_memory=False)
                # 컬럼명이 한글(UTF-8)로 저장되어 있으므로 인덱스로 접근
                # 인덱스 2 = date, 인덱스 11 = 출고수량
                date_col = df.columns[2]   # 'date'
                qty_col  = df.columns[11]  # '출고수량'

                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df[qty_col]  = pd.to_numeric(df[qty_col], errors="coerce")
                daily = (
                    df.dropna(subset=[date_col, qty_col])
                    .groupby(date_col)[qty_col]
                    .sum()
                    .reset_index()
                )
                daily.columns = ["date", "daily_qty"]
                # 기간 필터 (2019~2026)
                daily = daily[
                    (daily["date"] >= "2019-01-01") & (daily["date"] <= "2026-12-31")
                ]
                if len(daily) > 0:
                    print(f"[출고량] final_training_data.csv 로드 완료 ({len(daily):,}일)")
                    return daily
            except Exception as e:
                print(f"[출고량] CSV 로드 실패 ({e}), 랜덤 데이터 생성")
        else:
            print("[출고량] final_training_data.csv 없음 → 랜덤 샘플 생성")

        # ── 랜덤 샘플 생성 ──────────────────────────────
        return self._generate_random_sales()

    def _generate_random_sales(self) -> pd.DataFrame:
        """
        2019-01-01 ~ 2026-04-20 구간의 일별 출고량 랜덤 샘플 생성.
        계절성(봄-가을 피크), 연도별 완만한 성장, 주중/주말 패턴 반영.
        """
        np.random.seed(42)
        dates = pd.date_range("2019-01-01", "2026-04-20", freq="D")
        n = len(dates)

        # 기본 트렌드 (연간 1% 성장)
        trend = np.linspace(1.0, 1.07, n)

        # 계절 사인 파형 (농번기 4~10월 피크)
        doy = np.array([d.timetuple().tm_yday for d in dates])
        season = 1 + 0.6 * np.sin((doy - 60) * 2 * np.pi / 365)

        # 주중/주말 패턴 (주말은 약 20% 감소)
        weekday = np.array([0.8 if d.weekday() >= 5 else 1.0 for d in dates])

        base = 2000 * trend * season * weekday
        noise = np.random.lognormal(0, 0.3, n)
        qty = (base * noise).clip(min=0)

        return pd.DataFrame({"date": dates, "daily_qty": qty})

    def _load_weather_national(self) -> pd.DataFrame:
        """
        daily_all_stations.csv 에서 전국 일평균기온(avgTa 컬럼 평균)을 로드.
        없으면 랜덤 생성.

        반환: date(datetime), avg_temp(float) 컬럼을 가진 DataFrame
        """
        if self.path_weather.exists():
            try:
                df = pd.read_csv(
                    self.path_weather, encoding="utf-8", low_memory=False,
                    usecols=["tm", "avgTa"]
                )
                df["tm"] = pd.to_datetime(df["tm"], errors="coerce")
                df["avgTa"] = pd.to_numeric(df["avgTa"], errors="coerce")
                national = (
                    df.dropna(subset=["tm", "avgTa"])
                    .groupby("tm")["avgTa"]
                    .mean()
                    .reset_index()
                )
                national.columns = ["date", "avg_temp"]
                national = national[
                    (national["date"] >= "2019-01-01") & (national["date"] <= "2026-12-31")
                ]
                print(f"[기온] daily_all_stations.csv 로드 완료 ({len(national):,}일)")
                return national
            except Exception as e:
                print(f"[기온] CSV 로드 실패 ({e}), 랜덤 데이터 생성")
        else:
            print("[기온] daily_all_stations.csv 없음 → 랜덤 샘플 생성")

        return self._generate_random_weather()

    def _generate_random_weather(self) -> pd.DataFrame:
        """2019-01-01 ~ 2026-04-20 전국 일평균기온 랜덤 생성 (계절 반영)."""
        np.random.seed(43)
        dates = pd.date_range("2019-01-01", "2026-04-20", freq="D")
        doy = np.array([d.timetuple().tm_yday for d in dates])
        # 전국 연평균 약 13°C, 진폭 ±13°C 사인파
        temp = 13 + 13 * np.sin((doy - 80) * 2 * np.pi / 365)
        temp += np.random.normal(0, 2.5, len(dates))  # 일별 변동
        return pd.DataFrame({"date": dates, "avg_temp": temp})

    def _load_news_daily(self) -> pd.DataFrame:
        """
        news_corpus.db (news_articles 테이블) 에서 일별 기사 수 집계.
        없거나 데이터가 비어 있으면 랜덤 생성.

        반환: date(datetime), news_count(int) 컬럼을 가진 DataFrame
        """
        if self.path_news_db.exists():
            try:
                conn = sqlite3.connect(self.path_news_db)
                df = pd.read_sql(
                    "SELECT date, COUNT(*) AS news_count "
                    "FROM news_articles "
                    "WHERE date IS NOT NULL "
                    "GROUP BY date",
                    conn
                )
                conn.close()
                if len(df) > 0:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date"])
                    df = df[
                        (df["date"] >= "2019-01-01") & (df["date"] <= "2026-12-31")
                    ]
                    print(f"[뉴스] news_corpus.db 로드 완료 ({len(df):,}일)")
                    return df[["date", "news_count"]]
            except Exception as e:
                print(f"[뉴스] DB 로드 실패 ({e}), 랜덤 데이터 생성")

        print("[뉴스] DB 없음 또는 데이터 없음 → 랜덤 샘플 생성")
        return self._generate_random_news()

    def _generate_random_news(self) -> pd.DataFrame:
        """
        2019-01-01 ~ 2026-04-20 일별 뉴스 기사 수 랜덤 생성.
        농번기(3~6월, 9~10월) 기사 수 증가, 연도별 미세 트렌드 반영.
        """
        np.random.seed(44)
        dates = pd.date_range("2019-01-01", "2026-04-20", freq="D")
        month = np.array([d.month for d in dates])

        # 월별 기본 기사 수 (농번기 피크)
        monthly_base = {
            1: 8,  2: 9,  3: 18, 4: 25, 5: 30, 6: 22,
            7: 15, 8: 16, 9: 20, 10: 22, 11: 14, 12: 9,
        }
        base = np.array([monthly_base[m] for m in month], dtype=float)

        # 연도별 완만한 증가 (SNS/온라인 기사 증가 추세)
        year = np.array([d.year for d in dates])
        year_factor = 1 + (year - 2019) * 0.04
        noise = np.random.poisson(lam=base * year_factor).astype(float)
        return pd.DataFrame({"date": dates, "news_count": noise})

    def _load_keyword_trends(self) -> pd.DataFrame:
        """
        keyword_trends.csv 에서 연도별 키워드 빈도 로드.
        없으면 랜덤 생성.

        반환: (연도, 키워드) 멀티인덱스 또는 일반 DataFrame
              columns: year(int), keyword(str), count(int)
        """
        if self.path_keywords.exists():
            try:
                df = pd.read_csv(self.path_keywords, encoding="utf-8")
                # 컬럼 자동 감지: year, keyword, count 류
                year_col    = next((c for c in df.columns if "year" in c.lower() or "연도" in c), None)
                keyword_col = next((c for c in df.columns if "keyword" in c.lower() or "키워드" in c), None)
                count_col   = next((c for c in df.columns if "count" in c.lower() or "빈도" in c or "freq" in c.lower()), None)

                if year_col and keyword_col and count_col:
                    df = df.rename(columns={year_col: "year", keyword_col: "keyword", count_col: "count"})
                    df = df[["year", "keyword", "count"]]
                    print(f"[키워드] keyword_trends.csv 로드 완료 ({len(df):,}행)")
                    return df
            except Exception as e:
                print(f"[키워드] CSV 로드 실패 ({e}), 랜덤 데이터 생성")

        print("[키워드] keyword_trends.csv 없음 → 랜덤 샘플 생성")
        return self._generate_random_keywords()

    def _generate_random_keywords(self) -> pd.DataFrame:
        """
        2005~2026 연도별 키워드 20개 빈도 랜덤 생성.
        키워드별 트렌드 패턴(성장/하락/피크) 반영.
        """
        np.random.seed(45)
        years = list(range(2005, 2027))
        rows = []
        for kw_idx, kw in enumerate(self.KEYWORDS_20):
            # 키워드별 트렌드 패턴 부여
            if kw_idx < 5:       # 전통 농업 키워드: 서서히 감소
                trend = np.linspace(80, 40, len(years))
            elif kw_idx < 10:    # 기상 관련: 2010년대 중반 이후 급증
                peak = len(years) // 2
                trend = np.concatenate([
                    np.linspace(10, 60, peak),
                    np.linspace(60, 90, len(years) - peak),
                ])
            elif kw_idx < 15:    # 신기술 키워드: 2015년 이후 등장 급증
                trend = np.zeros(len(years))
                start = 10  # 2015년 인덱스
                trend[start:] = np.linspace(5, 70, len(years) - start)
            else:                # 정책/경제 키워드: 변동성 높음
                trend = np.random.uniform(20, 70, len(years))

            # 연도별 노이즈 추가
            counts = (trend + np.random.normal(0, 8, len(years))).clip(min=0).astype(int)
            for y, c in zip(years, counts):
                rows.append({"year": y, "keyword": kw, "count": int(c)})

        return pd.DataFrame(rows)

    # ──────────────────────────────────────────
    # 차트 생성 메서드
    # ──────────────────────────────────────────

    def _make_chart1_timeseries(
        self,
        sales: pd.DataFrame,
        weather: pd.DataFrame,
        news: pd.DataFrame,
    ) -> go.Figure:
        """
        [차트1] 3축 시계열 통합 분석
          - Y축1(좌): 일별 출고량  → 막대그래프, 파란색
          - Y축2(우): 일평균기온   → 선그래프, 빨간색
          - Y축3(우2): 뉴스 기사 수 → 점선, 녹색
        """
        # 공통 날짜 기준 병합 (outer 조인 후 결측은 0)
        merged = (
            sales.set_index("date")
            .join(weather.set_index("date"), how="outer")
            .join(news.set_index("date"), how="outer")
            .fillna({"daily_qty": 0, "news_count": 0})
            .reset_index()
            .sort_values("date")
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # ── Y축1: 출고량 막대 ─────────────────────
        fig.add_trace(
            go.Bar(
                x=merged["date"],
                y=merged["daily_qty"],
                name="일별 출고량 (개)",
                marker_color=self.COLOR_BAR,
                opacity=0.7,
                yaxis="y1",
                hovertemplate="%{x|%Y-%m-%d}<br>출고량: %{y:,.0f}개<extra></extra>",
            ),
            secondary_y=False,
        )

        # ── Y축2: 기온 선 ─────────────────────────
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["avg_temp"],
                name="일평균기온 (°C)",
                line=dict(color=self.COLOR_TEMP, width=1.5),
                yaxis="y2",
                hovertemplate="%{x|%Y-%m-%d}<br>기온: %{y:.1f}°C<extra></extra>",
            ),
            secondary_y=True,
        )

        # ── Y축3: 뉴스 기사 수 (점선) ─────────────
        # Plotly make_subplots에서 3축은 yaxis3 로 수동 추가
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["news_count"],
                name="뉴스 기사 수",
                line=dict(color=self.COLOR_NEWS, width=1.2, dash="dot"),
                yaxis="y3",
                hovertemplate="%{x|%Y-%m-%d}<br>기사 수: %{y}건<extra></extra>",
            ),
        )

        # ── 레이아웃 ──────────────────────────────
        fig.update_layout(
            title=dict(
                text="출고량 × 기온 × 뉴스 빈도 통합 분석 (2019-2026)",
                font=dict(size=18),
                x=0.5,
            ),
            xaxis=dict(
                title="날짜",
                rangeslider=dict(visible=True),  # 범위 슬라이더
                type="date",
            ),
            yaxis=dict(
                title="일별 출고량 (개)",
                titlefont=dict(color=self.COLOR_BAR),
                tickfont=dict(color=self.COLOR_BAR),
                side="left",
            ),
            yaxis2=dict(
                title="일평균기온 (°C)",
                titlefont=dict(color=self.COLOR_TEMP),
                tickfont=dict(color=self.COLOR_TEMP),
                overlaying="y",
                side="right",
            ),
            yaxis3=dict(
                title="뉴스 기사 수 (건)",
                titlefont=dict(color=self.COLOR_NEWS),
                tickfont=dict(color=self.COLOR_NEWS),
                overlaying="y",
                side="right",
                anchor="free",
                position=0.98,   # 가장 오른쪽에 배치
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
            height=550,
            plot_bgcolor="#F9F9F9",
            margin=dict(l=70, r=120, t=80, b=60),
        )

        return fig

    def _make_chart2_threshold_heatmap(
        self,
        sales: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> go.Figure:
        """
        [차트2] 임계기온 돌파 이벤트 분석 히트맵
          임계기온 [5, 8, 10, 13, 15]°C 돌파 후 7/14/21일 출고량 변화율 계산.
          변화율이 없는 경우(NaN) 0으로 처리.
        """
        thresholds = [5, 8, 10, 13, 15]   # 임계기온 목록 (°C)
        horizons   = [7, 14, 21]           # 반응 관찰 기간 (일)

        # 출고량/기온 일별 병합
        merged = (
            sales.set_index("date")
            .join(weather.set_index("date"), how="inner")
            .reset_index()
            .sort_values("date")
        )
        merged = merged.dropna(subset=["daily_qty", "avg_temp"])

        # 임계기온 × 관찰기간별 평균 변화율 계산
        heatmap_data = np.zeros((len(thresholds), len(horizons)))

        for i, thresh in enumerate(thresholds):
            # 해당 임계기온을 상향 돌파한 날 탐지 (이전 날 이하 → 오늘 초과)
            above = merged["avg_temp"] > thresh
            cross = above & (~above.shift(1, fill_value=False))  # 돌파 시점
            cross_dates = merged.loc[cross, "date"].tolist()

            for j, horizon in enumerate(horizons):
                changes = []
                for event_date in cross_dates:
                    # 이벤트 전 7일 평균 출고량
                    pre_mask  = (
                        (merged["date"] >= event_date - pd.Timedelta(days=7)) &
                        (merged["date"] <  event_date)
                    )
                    # 이벤트 후 horizon일 평균 출고량
                    post_mask = (
                        (merged["date"] > event_date) &
                        (merged["date"] <= event_date + pd.Timedelta(days=horizon))
                    )
                    pre_qty  = merged.loc[pre_mask,  "daily_qty"].mean()
                    post_qty = merged.loc[post_mask, "daily_qty"].mean()

                    if pd.notna(pre_qty) and pd.notna(post_qty) and pre_qty > 0:
                        change_pct = (post_qty - pre_qty) / pre_qty * 100
                        changes.append(change_pct)

                # 해당 셀 평균 변화율
                heatmap_data[i, j] = np.mean(changes) if changes else 0.0

        # ── Plotly 히트맵 ─────────────────────────
        x_labels = [f"+{h}일" for h in horizons]
        y_labels = [f"{t}°C 돌파" for t in thresholds]

        fig = go.Figure(
            go.Heatmap(
                z=heatmap_data,
                x=x_labels,
                y=y_labels,
                colorscale="RdBu",
                zmid=0,              # 0을 기준으로 색상 분기
                colorbar=dict(title="변화율 (%)"),
                text=np.round(heatmap_data, 1),
                texttemplate="%{text}%",
                hovertemplate="임계기온: %{y}<br>관찰기간: %{x}<br>출고 변화율: %{z:.1f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="임계기온 돌파 후 출고 반응 분석",
                font=dict(size=18),
                x=0.5,
            ),
            xaxis=dict(title="임계 돌파 후 관찰 기간"),
            yaxis=dict(title="임계기온"),
            height=420,
            margin=dict(l=120, r=80, t=80, b=60),
            plot_bgcolor="#F9F9F9",
        )

        return fig

    def _make_chart3_transplant_scatter(
        self,
        sales: pd.DataFrame,
    ) -> go.Figure:
        """
        [차트3] 지역별 모내기 시기 × 출고 상관관계 산점도
          X축: 모내기 적정 시기 기준 ±30일 구간 출고량 합계
          Y축: 연도별 해당 구간 출고량
          지역별 색상 구분, 연도별 데이터 포인트
        """
        # 연도 목록 (2019~2026)
        years = list(range(2019, 2027))

        # 지역별 색상 팔레트
        region_colors = {
            "경기/강원 중부": "#1F77B4",
            "충청/강원 남부": "#FF7F0E",
            "경북/전북":      "#2CA02C",
            "경남/전남":      "#D62728",
            "제주":           "#9467BD",
        }

        fig = go.Figure()

        for region, peak_mmdd in self.TRANSPLANT_SCHEDULE.items():
            color = region_colors[region]
            x_vals, y_vals, hover_texts = [], [], []

            for year in years:
                # 해당 연도의 모내기 중심 날짜 계산
                try:
                    center_date = pd.Timestamp(f"{year}-{peak_mmdd}")
                except Exception:
                    continue

                # 모내기 시기 ±30일 구간 출고량 합계
                window_start = center_date - pd.Timedelta(days=30)
                window_end   = center_date + pd.Timedelta(days=30)
                mask = (
                    (sales["date"] >= window_start) &
                    (sales["date"] <= window_end)
                )
                total_qty = sales.loc[mask, "daily_qty"].sum()

                if total_qty > 0:
                    # X축: 모내기 중심일(월-일 숫자 변환, 비교용)
                    x_val = int(peak_mmdd.replace("-", ""))
                    x_vals.append(x_val)
                    y_vals.append(total_qty)
                    hover_texts.append(
                        f"{region} {year}년<br>"
                        f"모내기 중심: {peak_mmdd}<br>"
                        f"±30일 출고량: {total_qty:,.0f}개"
                    )

            if x_vals:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name=region,
                        marker=dict(color=color, size=11, line=dict(width=1, color="white")),
                        text=hover_texts,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

        # X축 눈금을 실제 날짜 형식으로 변환
        tick_vals  = [int(v.replace("-", "")) for v in self.TRANSPLANT_SCHEDULE.values()]
        tick_texts = [
            f"{region}<br>({mmdd})"
            for region, mmdd in self.TRANSPLANT_SCHEDULE.items()
        ]

        fig.update_layout(
            title=dict(
                text="지역별 모내기 시기와 출고량 상관관계",
                font=dict(size=18),
                x=0.5,
            ),
            xaxis=dict(
                title="이앙 적정 시기 (MMDD)",
                tickvals=tick_vals,
                ticktext=[v.replace("<br>", "\n") for v in tick_texts],
            ),
            yaxis=dict(title="해당 구간 출고량 합계 (개)"),
            legend=dict(
                title="지역",
                orientation="v",
                xanchor="right",
                x=1.15,
            ),
            height=480,
            hovermode="closest",
            plot_bgcolor="#F9F9F9",
            margin=dict(l=80, r=140, t=80, b=80),
        )

        return fig

    def _make_chart4_keyword_heatmap(
        self,
        keyword_df: pd.DataFrame,
    ) -> go.Figure:
        """
        [차트4] 연도별 이슈 키워드 트렌드 히트맵
          X축: 연도 (2005~2026)
          Y축: 주요 키워드 20개
          색상: 등장 빈도
        """
        # 피벗 테이블 생성: 행=키워드, 열=연도
        pivot = keyword_df.pivot_table(
            index="keyword", columns="year", values="count", aggfunc="sum", fill_value=0
        )

        # 키워드 순서를 KEYWORDS_20 리스트 순서로 고정
        pivot = pivot.reindex(
            [kw for kw in self.KEYWORDS_20 if kw in pivot.index]
        )

        years    = pivot.columns.tolist()
        keywords = pivot.index.tolist()
        z_data   = pivot.values.tolist()

        fig = go.Figure(
            go.Heatmap(
                z=z_data,
                x=years,
                y=keywords,
                colorscale="YlOrRd",
                colorbar=dict(title="기사 빈도"),
                hovertemplate="연도: %{x}<br>키워드: %{y}<br>빈도: %{z:,}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="연도별 주요 키워드 빈도 변화",
                font=dict(size=18),
                x=0.5,
            ),
            xaxis=dict(
                title="연도",
                tickmode="linear",
                dtick=1,
                tickangle=-45,
            ),
            yaxis=dict(
                title="키워드",
                tickfont=dict(size=11),
                autorange="reversed",   # 위에서 아래 순서로 표시
            ),
            height=560,
            margin=dict(l=150, r=80, t=80, b=80),
            plot_bgcolor="#F9F9F9",
        )

        return fig

    # ──────────────────────────────────────────
    # HTML 통합 및 메인 실행
    # ──────────────────────────────────────────

    def _charts_to_html(self, figures: list) -> str:
        """
        Plotly Figure 리스트를 단일 HTML 문자열로 통합.
        plotly.offline.plot (include_plotlyjs 옵션 활용)으로
        첫 번째 차트에만 plotly.js 포함, 이후는 재사용.
        """
        html_parts = []

        # HTML 헤더
        html_parts.append(
            """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>영농 출고 통합 분석 대시보드 (2019-2026)</title>
  <style>
    body {
      font-family: 'Malgun Gothic', '맑은 고딕', sans-serif;
      background-color: #F0F2F5;
      margin: 0;
      padding: 20px;
    }
    h1 {
      text-align: center;
      color: #2C3E50;
      font-size: 24px;
      margin-bottom: 4px;
    }
    .subtitle {
      text-align: center;
      color: #7F8C8D;
      font-size: 13px;
      margin-bottom: 30px;
    }
    .chart-card {
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
      padding: 20px;
      margin-bottom: 30px;
    }
  </style>
</head>
<body>
  <h1>영농 출고 통합 분석 대시보드</h1>
  <p class="subtitle">데이터 기간: 2019 ~ 2026 | 생성: SH AI Model</p>
"""
        )

        for idx, fig in enumerate(figures):
            # 첫 번째 차트에만 plotly.js CDN 포함
            include_js = "cdn" if idx == 0 else False
            div_html = pyo.plot(
                fig,
                output_type="div",
                include_plotlyjs=include_js,
                config={
                    "displayModeBar": True,       # 툴바 표시
                    "scrollZoom": True,           # 스크롤 줌 허용
                    "displaylogo": False,         # Plotly 로고 숨김
                    "modeBarButtonsToRemove": ["sendDataToCloud"],
                },
            )
            html_parts.append(f'  <div class="chart-card">\n{div_html}\n  </div>\n')

        # HTML 푸터
        html_parts.append("</body>\n</html>")

        return "\n".join(html_parts)

    def run(self):
        """
        전체 파이프라인 실행:
          1. 데이터 로드 (없으면 랜덤 생성)
          2. 4개 차트 생성
          3. HTML 통합 후 저장
        """
        print("=" * 60)
        print("  영농 출고 통합 분석 대시보드 생성 시작")
        print("=" * 60)

        # ── 1. 데이터 로드 ───────────────────────
        print("\n[1/6] 출고량 데이터 로드...")
        sales = self._load_sales_daily()

        print("[2/6] 기온 데이터 로드...")
        weather = self._load_weather_national()

        print("[3/6] 뉴스 기사 수 로드...")
        news = self._load_news_daily()

        print("[4/6] 키워드 트렌드 로드...")
        keyword_df = self._load_keyword_trends()

        # ── 2. 차트 생성 ─────────────────────────
        print("\n[5/6] 차트 생성 중...")
        print("  - 차트1: 3축 시계열 생성")
        fig1 = self._make_chart1_timeseries(sales, weather, news)

        print("  - 차트2: 임계기온 돌파 이벤트 히트맵 생성")
        fig2 = self._make_chart2_threshold_heatmap(sales, weather)

        print("  - 차트3: 지역별 모내기 × 출고 산점도 생성")
        fig3 = self._make_chart3_transplant_scatter(sales)

        print("  - 차트4: 연도별 키워드 트렌드 히트맵 생성")
        fig4 = self._make_chart4_keyword_heatmap(keyword_df)

        # ── 3. HTML 통합 저장 ────────────────────
        print("\n[6/6] HTML 통합 파일 저장...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        html_content = self._charts_to_html([fig1, fig2, fig3, fig4])

        with open(self.output_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n완료! 저장 경로: {self.output_html}")
        print("=" * 60)


# ─────────────────────────────────────────────
# 직접 실행 시 대시보드 생성
# ─────────────────────────────────────────────
if __name__ == "__main__":
    gen = DashboardGenerator()
    gen.run()
