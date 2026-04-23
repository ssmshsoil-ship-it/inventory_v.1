#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2019-2022년 기상청 ASOS 일별 데이터 수집 및 기존 파일에 병합

기존 data/raw/weather/ 의 각 지점 파일에 2019-2022년 데이터를 앞에 추가합니다.
Usage: python src/collect_weather_2019_2022.py
"""
import pandas as pd
import requests
import time
from pathlib import Path
from urllib.parse import unquote
import sys
import io

# Windows 콘솔 UTF-8 출력 설정
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(r"C:\ai_workspace\sh-ai-model")
WEATHER_DIR  = PROJECT_ROOT / "data" / "raw" / "weather"
API_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
SKIP_FILES = {"daily_all_stations.csv", "weekly_features.csv"}

TARGET_YEARS = [2019, 2020, 2021, 2022]


# ── API 키 로드 ──────────────────────────────────────────────────────────
def load_api_key():
    env_path = PROJECT_ROOT / ".env"
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "KMA_API_KEY":
                return value.strip()
    return None


# ── API 호출: 특정 지점, 특정 연도 ──────────────────────────────────────
def fetch_year(api_key, stn_id, year):
    params = {
        "serviceKey": api_key,
        "pageNo":     "1",
        "numOfRows":  "999",
        "dataType":   "JSON",
        "dataCd":     "ASOS",
        "dateCd":     "DAY",
        "startDt":    f"{year}0101",
        "endDt":      f"{year}1231",
        "stnIds":     str(stn_id),
    }
    try:
        r = requests.get(API_URL, params=params, timeout=30)
        if r.status_code == 403:
            print("[FAIL] 403 Forbidden - API 키 확인 필요")
            return None
        if r.status_code != 200:
            print(f"[FAIL] HTTP {r.status_code}")
            return None
        data = r.json()
        hdr = data.get("response", {}).get("header", {})
        if hdr.get("resultCode") != "00":
            print(f"[FAIL] API: {hdr.get('resultMsg')}")
            return None
        items = data["response"]["body"].get("items") or {}
        item_list = items.get("item", [])
        if not item_list:
            return None
        if isinstance(item_list, dict):
            item_list = [item_list]
        return pd.DataFrame(item_list)
    except Exception as e:
        print(f"[ERR] {e}")
        return None


# ── 파생 컬럼 추가 ──────────────────────────────────────────────────────
def add_derived_columns(df, ref_columns):
    """기존 파일과 동일한 파생 컬럼 추가"""
    df = df.copy()
    tm = pd.to_datetime(df["tm"])
    df["year"]      = tm.dt.year
    df["month"]     = tm.dt.month
    df["week"]      = tm.dt.isocalendar().week.astype(int)
    df["doy"]       = tm.dt.dayofyear
    df["is_spring"] = df["month"].isin([2, 3, 4, 5]).astype(int)

    minTa = pd.to_numeric(df.get("minTa", pd.Series(dtype=float)), errors="coerce")
    maxTa = pd.to_numeric(df.get("maxTa", pd.Series(dtype=float)), errors="coerce")
    sumRn = pd.to_numeric(df.get("sumRn", pd.Series(dtype=float)), errors="coerce").fillna(0)

    df["cold_stress"] = ((df["is_spring"] == 1) & (minTa < 0)).astype(int)
    df["warm_day"]    = (maxTa >= 25).astype(int)
    df["rain_day"]    = (sumRn > 0).astype(int)

    # 기존 파일에 없는 컬럼은 NaN으로 채우고 순서 맞춤
    for col in ref_columns:
        if col not in df.columns:
            df[col] = None
    return df[ref_columns]


# ── 메인 ────────────────────────────────────────────────────────────────
def main():
    api_key = load_api_key()
    if not api_key:
        print("[ERROR] .env 파일에서 KMA_API_KEY를 찾을 수 없습니다.")
        sys.exit(1)
    print(f"[OK] API 키 로드 완료 ({api_key[:10]}...)\n")

    station_files = sorted(
        [f for f in WEATHER_DIR.glob("daily_*.csv") if f.name not in SKIP_FILES]
    )
    print(f"총 {len(station_files)}개 파일 처리 예정")
    print("=" * 60)

    success = skip_cnt = fail = 0

    for i, csv_file in enumerate(station_files, 1):
        # 기존 데이터 로드
        try:
            existing = pd.read_csv(csv_file, encoding="utf-8-sig")
        except Exception as e:
            print(f"[{i}] {csv_file.name} 로드 실패: {e}")
            fail += 1
            continue

        stn_id = str(int(existing["stnId"].iloc[0]))
        stn_nm = existing["stnNm"].iloc[0]
        ref_columns = list(existing.columns)

        # 이미 2019년 이전 데이터가 있으면 건너뜀
        min_year = pd.to_datetime(existing["tm"]).dt.year.min()
        if min_year <= 2019:
            print(f"[{i:2d}] {stn_nm}({stn_id}) - 이미 {min_year}년 데이터 있음, 건너뜀")
            skip_cnt += 1
            continue

        print(f"[{i:2d}/{len(station_files)}] {stn_nm} ({stn_id})", end="  ")

        # 2019-2022년 수집
        year_dfs = []
        for year in TARGET_YEARS:
            df_yr = fetch_year(api_key, stn_id, year)
            if df_yr is not None and len(df_yr) > 0:
                year_dfs.append(df_yr)
                print(f"{year}OK({len(df_yr)})", end=" ", flush=True)
            else:
                print(f"{year}NG", end=" ", flush=True)
            time.sleep(0.3)
        print()

        if not year_dfs:
            print(f"  [FAIL] 데이터 수집 실패")
            fail += 1
            continue

        new_data = pd.concat(year_dfs, ignore_index=True)
        new_data = add_derived_columns(new_data, ref_columns)

        # 병합 (신규 2019-2022 + 기존 2023-2026), 날짜 정렬, 중복 제거
        combined = pd.concat([new_data, existing], ignore_index=True)
        combined = (
            combined
            .sort_values("tm")
            .drop_duplicates(subset=["tm"], keep="last")
            .reset_index(drop=True)
        )

        combined.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"  [DONE] +{len(new_data)}행 추가 -> 총 {len(combined)}행 저장")
        success += 1

    # ── daily_all_stations.csv 재빌드 ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"결과: 성공 {success}개 / 건너뜀 {skip_cnt}개 / 실패 {fail}개")
    print("\n[INFO] daily_all_stations.csv 재빌드 중...")

    all_dfs = []
    for f in sorted(station_files):
        try:
            all_dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
        except Exception:
            pass

    if all_dfs:
        combined_all = pd.concat(all_dfs, ignore_index=True)
        out_path = WEATHER_DIR / "daily_all_stations.csv"
        combined_all.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [DONE] {len(combined_all):,}행 저장: {out_path}")

    print("\n[완료] 다음 단계: src/train.py 재실행으로 2019-2022 데이터 포함 모델 학습")


if __name__ == "__main__":
    main()
