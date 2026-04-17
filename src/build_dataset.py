"""
출고 데이터 + 기상 데이터 → 주차별 학습 데이터셋 생성
실행: python src/build_dataset.py
"""
import math
import pandas as pd
from pathlib import Path
from config import DATA_DIR, PROCESSED_DIR, MASTER_DB, WEATHER_DIR, TRAINING_DATA

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


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


def load_weekly_weather(weather_dir: Path) -> pd.DataFrame:
    weekly_path = weather_dir / "weekly_features.csv"
    if True:
        weather = pd.read_csv(weekly_path)
        # 전국 평균
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

        # 벌교(보성 165번) 지역 기상
        boseong = weather[weather["stnId"] == 165][
            ["year", "week", "avg_temp", "total_rain", "cold_stress_days", "temp_anomaly"]
        ].copy()
        boseong.columns = [
            "year", "week",
            "boseong_avg_temp", "boseong_rain",
            "boseong_cold_stress", "boseong_temp_anomaly",
        ]
        weekly = weekly.merge(boseong, on=["year", "week"], how="left")

    weekly = weekly.rename(columns={"year": "ISO연도", "week": "ISO주차"})
    weekly["ISO연도"] = weekly["ISO연도"].astype(int)
    weekly["ISO주차"] = weekly["ISO주차"].astype(int)
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
    weekly_out = aggregate_weekly(master)

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
