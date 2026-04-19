"""
출고 데이터 + 기상 데이터 → 주차별 학습 데이터셋 생성
실행: python src/build_dataset.py
"""
import math
import pandas as pd
from pathlib import Path
from config import DATA_DIR, PROCESSED_DIR, MASTER_DB, WEATHER_DIR, RAW_WEATHER_DIR, TRAINING_DATA

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
    df = pd.read_csv(path)
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
    지정된 폴더의 모든 기상 데이터 CSV 파일을 하나의 DataFrame으로 병합합니다.
    
    Args:
        raw_weather_dir (Path): 원본 CSV 파일들이 있는 폴더 경로
    
    Returns:
        pd.DataFrame: 병합된 데이터프레임
    """
    print(f"\n- 원본 기상 데이터 병합 시작: {raw_weather_dir}")
    
    csv_files = sorted(list(raw_weather_dir.glob("*.csv")))
    if not csv_files:
        print(f"  ! 경고: '{raw_weather_dir}' 폴더에 병합할 CSV 파일이 없습니다.")
        return pd.DataFrame()

    df_list = []
    for file in csv_files:
        try:
            # API로 수집한 데이터는 보통 utf-8이지만, 문제가 발생하면 'cp949' 시도
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"  ✓ 로드: {file.name} ({len(df)} 행)")
        except Exception as e:
            print(f"  ! 오류: {file.name} 파일 로드 실패 - {e}")

    if not df_list:
        print("  ! 경고: 성공적으로 로드된 데이터가 없습니다.")
        return pd.DataFrame()
    
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"  ✓ 병합 완료: 총 {len(merged_df)} 행")
    
    return merged_df


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
        print(f"  ✓ 통합 기상 데이터 로드: {weekly_path}")
        weather = pd.read_csv(weekly_path)
        
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
            print(f"  ✓ 대체 기상 데이터 로드: {processed_path}")
            weekly = pd.read_csv(processed_path)
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
    
    print(f"  ✓ 기상 데이터: {len(weekly)}주 ({weekly['ISO연도'].min()}~{weekly['ISO연도'].max()}년)")
    
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
    build()
