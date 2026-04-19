# -*- coding: utf-8 -*-
"""
출고 데이터 + 기상 데이터 → 주차별 학습 데이터셋 생성
실행: python src/build_dataset.py
"""
import math
import pandas as pd
from pathlib import Path
from config import DATA_DIR, PROCESSED_DIR, MASTER_DB, WEATHER_DIR, RAW_WEATHER_DIR, TRAINING_DATA, SALES_DATA_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ERP_HISTORICAL = PROCESSED_DIR / "erp_2019_2022.csv"


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="정제데이터")
    df = df[df["이상치_플래그"] == "정상"]
    df = df[df["대분류"].isin(["수도용상토", "원예용상토"])]
    df["ISO연도"] = df["ISO연도"].astype(int)
    df["ISO주차"] = df["ISO주차"].astype(int)
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["ISO연도", "ISO주차", "대분류"]).agg(
        출고수량=("출고수량", "sum"),
        출고량_L=("출고량_L", "sum"),
        거래건수=("출고수량", "count"),
    ).reset_index()

    pivot = grp.pivot_table(
        index=["ISO연도", "ISO주차"],
        columns="대분류",
        values=["출고수량", "출고량_L"],
        aggfunc="sum",
    ).reset_index()
    pivot.columns = ["_".join(c).strip("_") for c in pivot.columns]
    pivot = pivot.rename(columns={
        "ISO연도_": "ISO연도", "ISO주차_": "ISO주차",
        "출고수량_수도용상토": "수도용_포",
        "출고수량_원예용상토": "원예용_포",
        "출고량_L_수도용상토": "수도용_L",
        "출고량_L_원예용상토": "원예용_L",
    })
    pivot["수도용_포"] = pivot.get("수도용_포", 0).fillna(0)
    pivot["원예용_포"] = pivot.get("원예용_포", 0).fillna(0)
    pivot["총출고_포"] = pivot["수도용_포"] + pivot["원예용_포"]
    return pivot


def load_erp_historical(path: Path) -> pd.DataFrame:
    """2019-2022 ERP 전처리 결과 로드 (erp_2019_2022.csv)"""
    df = pd.read_csv(path, encoding="utf-8")
    df["ISO연도"] = df["ISO연도"].astype(int)
    df["ISO주차"] = df["ISO주차"].astype(int)
    df["수도용_포"] = df["수도용_포"].fillna(0).astype(int)
    df["원예용_포"] = df["원예용_포"].fillna(0).astype(int)
    df["총출고_포"] = df["수도용_포"] + df["원예용_포"]
    # master_db와 컬럼 정합: L 컬럼은 없으므로 0 처리
    df["수도용_L"] = 0.0
    df["원예용_L"] = 0.0
    return df


def merge_raw_weather_data(raw_weather_dir: Path) -> pd.DataFrame:
    """
    지정된 폴더의 모든 기상 데이터 CSV를 병합하고 전처리합니다.
    - 날짜 형식 'YYYY-MM-DD' 통일 및 중복 데이터 제거
    - '연도', '월', '주차' 컬럼 생성
    - 2019년부터 데이터가 끝나는 시점까지 모든 날짜에 대한 행 생성 (결측치는 NaN)
    
    Args:
        raw_weather_dir (Path): 원본 CSV 파일들이 있는 폴더 경로
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    print(f"\n- 원본 기상 데이터 병합 및 전처리 시작: {raw_weather_dir}")
    
    csv_files = sorted(list(raw_weather_dir.glob("*.csv")))
    if not csv_files:
        print(f"  [경고] '{raw_weather_dir}' 폴더에 병합할 CSV 파일이 없습니다.")
        return pd.DataFrame()

    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding="utf-8")
            df_list.append(df)
            print(f"  [OK] 로드: {file.name} ({len(df)} 행)")
        except Exception as e:
            print(f"  [오류] {file.name} 파일 로드 실패 - {e}")

    if not df_list:
        print("  [경고] 성공적으로 로드된 데이터가 없습니다.")
        return pd.DataFrame()
    
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"  - 병합 완료: 총 {len(merged_df)} 행")

    # 날짜 컬럼('일시')을 datetime으로 변환 (시간 정보는 제거)
    if '일시' not in merged_df.columns:
        raise ValueError("병합된 데이터에 '일시' 컬럼이 없습니다. 날짜 컬럼명을 확인해주세요.")
    merged_df['date'] = pd.to_datetime(merged_df['일시']).dt.normalize()

    # 중복 데이터 제거 (날짜와 지점 기준)
    if '지점' not in merged_df.columns:
        raise ValueError("병합된 데이터에 '지점' 컬럼이 없습니다. 관측소 ID 컬럼명을 확인해주세요.")
        
    print(f"  - 중복 제거 전: {len(merged_df)} 행")
    merged_df = merged_df.sort_values(by=['date', '지점']).drop_duplicates(subset=['date', '지점'], keep='first')
    print(f"  - 중복 제거 후: {len(merged_df)} 행")

    # 2019년부터 데이터 끝까지 모든 날짜에 대한 플레이스홀더 생성
    if merged_df.empty:
        print("  [경고] 유효 데이터가 없어 시계열을 생성할 수 없습니다.")
        return pd.DataFrame()
        
    min_date = pd.to_datetime('2019-01-01')
    max_date = merged_df['date'].max()
    all_stations = sorted(merged_df['지점'].unique())
    
    print(f"  - 데이터 기간: {merged_df['date'].min().date()} ~ {max_date.date()}")
    print(f"  - 전체 시계열 생성: {min_date.date()} ~ {max_date.date()} (총 {len(all_stations)}개 관측 지점)")

    # 데카르트 곱으로 모든 날짜-지점 조합 생성
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    placeholder_df = pd.DataFrame(
        pd.MultiIndex.from_product(
            [full_date_range, all_stations], 
            names=['date', '지점']
        ).to_frame(index=False)
    )
    
    # 원본 데이터와 left-join하여 빈 기간을 NaN으로 채움
    final_df = pd.merge(placeholder_df, merged_df, on=['date', '지점'], how='left')
    
    # 날짜 기반 피처 생성
    final_df['연도'] = final_df['date'].dt.year
    final_df['월'] = final_df['date'].dt.month
    final_df['주차'] = final_df['date'].dt.isocalendar().week.astype(int)

    # 요청에 따라 날짜 컬럼을 'YYYY-MM-DD' 형식의 문자열로 변경
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    print(f"  [OK] 전처리 완료: 최종 {len(final_df)} 행")
    
    return final_df


def analyze_region_data(sales_dir: Path):
    """
    지정된 폴더의 모든 판매 데이터(Excel)를 로드하여 지역 관련 컬럼의
    고유값과 빈도수를 분석하고 출력합니다.
    """
    print(f"\n- 지역 컬럼 데이터 분석 시작: {sales_dir}")
    
    # .xlsx와 .xls 파일을 모두 찾음
    excel_files = sorted(list(sales_dir.glob("*.xlsx"))) + sorted(list(sales_dir.glob("*.xls")))
    if not excel_files:
        print(f"  [경고] '{sales_dir}' 폴더에 분석할 Excel 파일이 없습니다.")
        return

    df_list = []
    for file in excel_files:
        try:
            # 첫 번째 시트만 읽음
            df = pd.read_excel(file) 
            df_list.append(df)
            print(f"  [OK] 로드: {file.name} ({len(df)} 행)")
        except Exception as e:
            print(f"  [오류] {file.name} 파일 로드 실패 - {e}")
            
    if not df_list:
        print("  [경고] 분석할 데이터를 로드하지 못했습니다.")
        return
        
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"\n  [OK] 총 {len(excel_files)}개 파일, {len(merged_df)} 행 데이터 병합 완료.")

    # 지역 관련 컬럼 찾기
    region_col = None
    possible_cols = ['지역', '지역(임)', '시군명'] 
    for col in possible_cols:
        if col in merged_df.columns:
            region_col = col
            break
    
    if not region_col:
        print(f"  [오류] 데이터에서 지역 관련 컬럼({', '.join(possible_cols)})을 찾을 수 없습니다.")
        print(f"  -> 사용 가능한 컬럼: {list(merged_df.columns)}")
        return
        
    print(f"  [OK] 분석 대상 컬럼: '{region_col}'")
    
    # NaN 값 및 공백 처리
    cleaned_series = merged_df[region_col].fillna('N/A').astype(str).str.strip()

    print(f"\n--- '{region_col}' 컬럼 고유값 및 빈도수 ---")
    value_counts = cleaned_series.value_counts()
    
    # 모든 결과를 출력하도록 pandas 옵션 설정
    with pd.option_context('display.max_rows', None):
        print(value_counts)
    
    print(f"\n분석 완료: 총 {len(value_counts)}개의 고유값이 발견되었습니다.")


def load_weekly_weather(weather_dir: Path) -> pd.DataFrame:
    """
    통합 기상 데이터 로드 (2019~2026년)
    
    우선순위:
    1. data/weather/weekly_features.csv (weather_data_collector.py 생성)
    2. data/weather_processed.csv (대체 경로)
    """
    # 1순위: weekly_features.csv
    weekly_path = weather_dir / "weekly_features.csv"
    
    if weekly_path.exists():
        print(f"  [OK] 통합 기상 데이터 로드: {weekly_path}")
        weather = pd.read_csv(weekly_path, encoding="utf-8")
        
        # 전국 평균 (stnId == 999) 또는 전체 평균
        if 'stnId' in weather.columns:
            # 전국 평균 데이터만 사용
            weekly = weather[weather['stnId'] == 999].copy()
            if len(weekly) == 0:
                # 전국 평균이 없으면 모든 지점 평균 계산
                weekly = weather.groupby(["year", "week"]).agg(
                    avg_temp=("avg_temp", "mean"),
                    min_temp=("min_temp", "mean"),
                    max_temp=("max_temp", "mean"),
                    total_rain=("total_rain", "mean"),
                    rain_days=("rain_days", "mean"),
                    cold_stress_days=("cold_stress_days", "mean"),
                    warm_days=("warm_days", "mean"),
                    temp_anomaly=("temp_anomaly", "mean"),
                    cum_temp_ytd=("cum_temp_ytd", "mean"),
                ).reset_index()
        else:
            weekly = weather.copy()
        
        # 벌교(보성 165번) 지역 기상 - 있으면 추가
        if 'stnId' in weather.columns and 165 in weather['stnId'].values:
            boseong = weather[weather["stnId"] == 165][
                ["year", "week", "avg_temp", "total_rain", "cold_stress_days", "temp_anomaly"]
            ].copy()
            boseong.columns = [
                "year", "week",
                "boseong_avg_temp", "boseong_rain",
                "boseong_cold_stress", "boseong_temp_anomaly",
            ]
            weekly = weekly.merge(boseong, on=["year", "week"], how="left")
        else:
            # 보성 데이터가 없으면 전국 평균으로 대체
            weekly["boseong_avg_temp"] = weekly["avg_temp"]
            weekly["boseong_rain"] = weekly["total_rain"]
            weekly["boseong_cold_stress"] = weekly["cold_stress_days"]
            weekly["boseong_temp_anomaly"] = weekly["temp_anomaly"]
    else:
        # 2순위: weather_processed.csv
        processed_path = Path("data/weather_processed.csv")
        if processed_path.exists():
            print(f"  [OK] 대체 기상 데이터 로드: {processed_path}")
            weekly = pd.read_csv(processed_path, encoding="utf-8")
            # 보성 데이터 없으면 전국 평균으로 대체
            if "boseong_avg_temp" not in weekly.columns:
                weekly["boseong_avg_temp"] = weekly["avg_temp"]
                weekly["boseong_rain"] = weekly.get("total_rain", 0)
                weekly["boseong_cold_stress"] = weekly.get("cold_stress_days", 0)
                weekly["boseong_temp_anomaly"] = weekly.get("temp_anomaly", 0)
        else:
            raise FileNotFoundError(
                f"기상 데이터를 찾을 수 없습니다.\n"
                f"다음 중 하나를 실행하세요:\n"
                f"  1. python src/weather_data_collector.py (2019~2026 통합 데이터 생성)\n"
                f"  2. {weekly_path} 또는 {processed_path} 파일 확인"
            )

    weekly = weekly.rename(columns={"year": "ISO연도", "week": "ISO주차"})
    weekly["ISO연도"] = weekly["ISO연도"].astype(int)
    weekly["ISO주차"] = weekly["ISO주차"].astype(int)
    
    print(f"  [OK] 기상 데이터: {len(weekly)}주 ({weekly['ISO연도'].min()}~{weekly['ISO연도'].max()}년)")
    
    return weekly


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ISO연도", "ISO주차"]).reset_index(drop=True)

    for target in ["수도용_포", "원예용_포"]:
        for lag in [1, 2, 3, 4]:
            df[f"{target}_lag{lag}"] = df[target].shift(lag)
        df[f"{target}_lag52"] = df[target].shift(52)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    w = df["ISO주차"]
    df["sin_week"]  = (2 * math.pi * w / 52).apply(math.sin)
    df["cos_week"]  = (2 * math.pi * w / 52).apply(math.cos)
    df["is_spring"] = w.between(10, 22).astype(int)
    df["is_peak"]   = w.between(13, 18).astype(int)
    return df


def build(save=True) -> pd.DataFrame:
    print("1. master_db 로드 및 집계...")
    master = load_master(MASTER_DB)
    weekly_master = aggregate_weekly(master)

    if ERP_HISTORICAL.exists():
        print("1b. ERP 2019-2022 로드 및 병합...")
        weekly_hist = load_erp_historical(ERP_HISTORICAL)
        # master_db 기간과 중복 제거 (master_db 우선)
        master_years = set(weekly_master["ISO연도"].unique())
        weekly_hist = weekly_hist[~weekly_hist["ISO연도"].isin(master_years)]
        weekly_out = pd.concat([weekly_hist, weekly_master], ignore_index=True)
        weekly_out = weekly_out.sort_values(["ISO연도", "ISO주차"]).reset_index(drop=True)
        print(f"   병합 후: {len(weekly_hist)}주(2019-22) + {len(weekly_master)}주(master) = {len(weekly_out)}주")
    else:
        weekly_out = weekly_master

    print("2. 기상 데이터 로드...")
    weather = load_weekly_weather(WEATHER_DIR)

    print("3. 조인...")
    dataset = weekly_out.merge(weather, on=["ISO연도", "ISO주차"], how="left")

    print("4. lag 피처 생성...")
    dataset = add_lag_features(dataset)

    print("5. 캘린더 피처 생성...")
    dataset = add_calendar_features(dataset)

    print(f"완료: {len(dataset)}행 × {len(dataset.columns)}컬럼")
    print(f"기간: {dataset['ISO연도'].min()}년 {dataset['ISO주차'].iloc[0]}주 ~ "
          f"{dataset['ISO연도'].max()}년 {dataset['ISO주차'].iloc[-1]}주")

    if save:
        dataset.to_csv(TRAINING_DATA, index=False, encoding="utf-8-sig")
        print(f"저장: {TRAINING_DATA}")

    return dataset


if __name__ == "__main__":
    # build() # 기존 빌드 함수는 잠시 주석 처리
    analyze_region_data(SALES_DATA_DIR)
