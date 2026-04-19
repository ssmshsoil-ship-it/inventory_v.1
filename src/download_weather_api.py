# -*- coding: utf-8 -*-
"""
기상청 ASOS 일자료 API를 사용하여 2019년부터 2022년까지의 전국 기상 데이터를 수집합니다.
실행 전, KMA_API_KEY 환경 변수에 기상청 API 인증키(일반 인증키, Decoding)를 설정해야 합니다.
(Windows: set KMA_API_KEY=your_api_key_here)

실행: python src/download_weather_api.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- 설정 ---
# API 키 (환경 변수에서 로드)
API_KEY = os.getenv('KMA_API_KEY')

# 데이터 수집 기간
START_YEAR = 2019
END_YEAR = 2022

# 저장 경로
OUTPUT_DIR = Path("data/raw/weather")
OUTPUT_FILE = OUTPUT_DIR / f"weather_{START_YEAR}_{END_YEAR}.csv"
FAILED_LOG_FILE = Path("failed_log.txt")

# API 정보
API_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
STATIONS = 'ALL'  # 전국 모든 지점
ITEMS_PER_PAGE = 700 # 한 페이지에 가져올 데이터 수 (최대 999)
# --- ---

def log_failed_period(start_date: str, end_date: str):
    """실패한 날짜 범위를 로그 파일에 기록합니다."""
    with open(FAILED_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"Failed to fetch data for period: {start_date} to {end_date}\n")

def fetch_weather_data_for_period(start_date: str, end_date: str) -> list:
    """지정된 기간의 전국 ASOS 기상 데이터를 API로 요청하고, 실패 시 재시도합니다."""
    all_data = []
    page_no = 1
    
    params = {
        'serviceKey': API_KEY,
        'pageNo': page_no,
        'numOfRows': ITEMS_PER_PAGE,
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'DAY',
        'startDt': start_date,
        'endDt': end_date,
        'stnIds': STATIONS,
    }

    while True:
        max_retries = 3
        response_data = None
        for attempt in range(max_retries):
            try:
                params['pageNo'] = page_no
                response = requests.get(API_URL, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                result_code = data['response']['header']['resultCode']
                result_msg = data['response']['header']['resultMsg']

                # DB_ERROR(03) 등 재시도 가능한 오류
                if result_code in ['03', '21']: # DB_ERROR, DEADLINE_EXCEEDED
                    raise requests.exceptions.RequestException(f"API Error ({result_code}): {result_msg}")

                if result_code != '00':
                    print(f"  [복구 불가능한 API 오류] {result_msg} (기간: {start_date}~{end_date})")
                    log_failed_period(start_date, end_date)
                    return []

                response_data = data
                break # 성공 시 재시도 루프 탈출

            except requests.exceptions.RequestException as e:
                print(f"  [네트워크/DB 오류] {e} (시도 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print("  -> 최대 재시도 실패. 이 기간의 수집을 중단합니다.")
                    log_failed_period(start_date, end_date)
                    return []
        
        # --- 성공 응답 처리 ---
        items = response_data['response']['body']['items'].get('item', [])
        if not items:
            break

        all_data.extend(items)
        
        total_count = response_data['response']['body']['totalCount']
        if page_no * ITEMS_PER_PAGE >= total_count:
            break
            
        page_no += 1
        time.sleep(1) # 과도한 요청 방지를 위한 1초 대기
            
    return all_data


def main():
    """지정한 연도 범위의 기상 데이터를 월 단위로 수집하여 CSV로 저장합니다."""
    if not API_KEY:
        print("[오류] KMA_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("스크립트를 실행하기 전에 API 키를 설정해주세요.")
        print("예: set KMA_API_KEY=your_api_key_here")
        return

    print(f"기상 데이터 수집을 시작합니다 ({START_YEAR}년 ~ {END_YEAR}년)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_data = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start_of_month = datetime(year, month, 1)
            
            if month == 12:
                end_of_month = datetime(year, 12, 31)
            else:
                end_of_month = datetime(year, month + 1, 1) - pd.Timedelta(days=1)

            start_str = start_of_month.strftime('%Y%m%d')
            end_str = end_of_month.strftime('%Y%m%d')

            print(f"- {year}년 {month}월 데이터 수집 중... ({start_str} ~ {end_str})")
            monthly_data = fetch_weather_data_for_period(start_str, end_str)
            
            if monthly_data:
                full_data.extend(monthly_data)
                print(f"  [OK] {len(monthly_data)}건 수집 완료.")
            else:
                print(f"  [정보] 해당 기간에 수집된 데이터가 없습니다.")
    
    if not full_data:
        print("\n[완료] 수집된 데이터가 없어 파일을 생성하지 않습니다.")
        return

    print("\n데이터프레임 변환 및 전처리를 시작합니다...")
    df = pd.DataFrame(full_data)

    # API 원본 컬럼명 중 필요한 것만 선택 후, 기존 데이터와 컬럼명 통일
    required_cols_original = ['stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn']
    cols_to_select = [col for col in required_cols_original if col in df.columns]
    df = df[cols_to_select]
    df = df.rename(columns={'stnId': 'stn_id', 'tm': 'date'})
    
    print(f"  [OK] 컬럼 선택 및 이름 변경 완료.")
    
    # 데이터 타입 변환 및 날짜 형식 맞추기
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    numeric_cols = ['stn_id', 'avgTa', 'minTa', 'maxTa', 'sumRn']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  [OK] 데이터 타입 변환 완료.")

    if 'sumRn' in df.columns:
        df['sumRn'] = df['sumRn'].fillna(0)
        print(f"  [OK] 강수량(sumRn) 결측치를 0으로 처리.")

    df = df.sort_values(by=['stn_id', 'date']).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[성공] 모든 데이터 수집이 완료되었습니다.")
    print(f" -> 최종 데이터: {len(df)}행, {len(df.columns)}컬럼")
    print(f" -> 저장 경로: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
