# -*- coding: utf-8 -*-
"""
개별 판매 데이터, 고객-지역 매핑 정보, 일별 기상 데이터를 통합하고
모델 학습을 위한 최종 피처 데이터셋을 생성합니다.
실행: python src/integrate_features.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_SALES_DIR, CUSTOMER_MAP_CSV, RAW_WEATHER_DIR, FINAL_TRAINING_DATA
from build_dataset import merge_raw_weather_data # build_dataset.py의 함수 재사용

# 표준 지역(도) -> 대표 기상 관측소 지점번호 매핑
# 이 매핑은 분석을 통해 더 정교하게 만들 수 있습니다.
PROVINCE_TO_STATION_MAP = {
    '경기': 119,  # 수원
    '강원': 101,  # 춘천
    '충북': 131,  # 청주
    '충남': 133,  # 대전
    '전북': 146,  # 전주
    '전남': 156,  # 광주
    '경북': 143,  # 대구
    '경남': 159,  # 부산
    '제주': 184,  # 제주
    '본사': 156,  # 본사는 주 고객층이 많은 '전남'으로 우선 매핑
    '미분류': 156, # 미분류 건도 '전남'으로 우선 매핑
}


def load_all_sales_data(sales_dir: Path, customer_map_path: Path) -> pd.DataFrame:
    """판매 데이터를 로드하고, 고객 마스터맵을 적용하여 표준 지역명을 부여합니다."""
    print(f"\n- 1. 판매 데이터 로드 및 지역 매핑 시작: {sales_dir}")
    if not customer_map_path.exists():
        raise FileNotFoundError(f"고객-지역 매핑 파일이 없습니다: {customer_map_path}")
    
    customer_map = pd.read_csv(customer_map_path)
    
    excel_files = sorted(list(sales_dir.glob("*.xlsx"))) + sorted(list(sales_dir.glob("*.xls")))
    if not excel_files:
        raise FileNotFoundError(f"판매 데이터 파일이 없습니다: {sales_dir}")

    df_list = []
    for file in excel_files:
        engine = 'openpyxl' if file.suffix == '.xlsx' else 'xlrd'
        df = pd.read_excel(file, engine=engine)
        # '고객' 또는 유사 컬럼명을 '고객명'으로 통일
        rename_map = {col: '고객명' for col in df.columns if '고객' in col or '거래처' in col}
        df = df.rename(columns=rename_map)
        df_list.append(df)
    
    sales_df = pd.concat(df_list, ignore_index=True)
    # 고객명을 기준으로 지역 정보(province)를 병합
    merged_df = pd.merge(sales_df, customer_map[['고객명', 'province']], on='고객명', how='left')
    merged_df['province'] = merged_df['province'].fillna('미분류')
    print(f"  [OK] 총 {len(sales_df)}건의 판매 데이터에 지역 정보 매핑 완료.")
    return merged_df


def integrate_and_engineer_features():
    """데이터 통합 및 특성 공학 전체 파이프라인"""
    # 1. 판매 데이터 로드 및 고객-지역 매핑
    sales_df = load_all_sales_data(RAW_SALES_DIR, CUSTOMER_MAP_CSV)

    # 2. 기상 데이터 결합 준비
    print("\n- 2. 기상 데이터 결합 준비")
    # '출고일자' 또는 유사 컬럼을 'date'로 통일하고 datetime 형식으로 변환
    sales_df = sales_df.rename(columns={'출고일자': 'date', '일자': 'date'})
    
    # 다양한 날짜 형식(하이픈, 슬래시 등)을 유연하게 처리
    original_rows = len(sales_df)
    sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce', format='mixed')
    
    # 날짜 변환 실패(NaT) 행이 있다면 제거
    sales_df = sales_df.dropna(subset=['date'])
    removed_rows = original_rows - len(sales_df)
    if removed_rows > 0:
        print(f"  [경고] 유효하지 않은 날짜 형식으로 인해 {removed_rows}개 행을 제거했습니다.")
        
    sales_df['date'] = sales_df['date'].dt.normalize()
    # 매핑 테이블을 이용해 '지점' (기상 관측소 ID) 컬럼 추가
    sales_df['지점'] = sales_df['province'].map(PROVINCE_TO_STATION_MAP)
    print("  [OK] 판매 데이터에 기상 관측소 ID 매핑 완료.")
    
    # 3. 일별 기상 데이터 로드
    weather_df = merge_raw_weather_data(RAW_WEATHER_DIR)
    # merge를 위해 날짜 형식 통일
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.normalize()
    # 필요한 기상 컬럼만 선택
    weather_cols = ['date', '지점', '평균기온(°C)', '일강수량(mm)']
    weather_df = weather_df[weather_cols].rename(columns={
        '평균기온(°C)': 'avg_temp', '일강수량(mm)': 'rainfall'
    })
    print("\n- 3. 일별 기상 데이터 로드 완료")

    # 4. 판매 데이터와 기상 데이터 결합
    final_df = pd.merge(sales_df, weather_df, on=['date', '지점'], how='left')
    print("\n- 4. 판매 + 기상 데이터 결합 완료")

    # 5. 특성 공학
    print("\n- 5. 특성 공학(Feature Engineering) 시작")
    final_df = final_df.sort_values(by=['지점', 'date']).reset_index(drop=True)
    
    # 그룹별(지점별)로 계산해야 정확함
    final_df['temp_change_weekly'] = final_df.groupby('지점')['avg_temp'].diff(7)
    final_df['rain_sum_3d'] = final_df.groupby('지점')['rainfall'].rolling(window=3).sum().reset_index(0,drop=True)
    
    final_df['is_peak_season'] = final_df['date'].dt.month.isin([3, 4]).astype(int)
    
    # 결측치 발생 가능 (rolling, diff 초기값) -> 0으로 채우기
    final_df.fillna({
        'temp_change_weekly': 0, 'rain_sum_3d': 0
    }, inplace=True)
    print("  [OK] 파생 변수 생성 완료: '전주 대비 기온 변화', '최근 3일 누적 강수량', '피크 시즌 여부'")

    # 6. 최종 데이터 저장
    print(f"\n- 6. 최종 학습 데이터셋 저장")
    try:
        final_df.to_csv(FINAL_TRAINING_DATA, index=False, encoding='utf-8-sig')
        print(f"  [OK] 저장 완료: {FINAL_TRAINING_DATA}")
        print(f"  -> 최종 데이터: {final_df.shape[0]}행, {final_df.shape[1]}컬럼")
        print(f"  -> 결과 (일부 컬럼):\n{final_df[['date', '고객명', 'province', 'avg_temp', 'rain_sum_3d', 'is_peak_season']].head().to_string()}")
    except Exception as e:
        print(f"  [오류] 파일 저장 실패: {e}")


if __name__ == "__main__":
    integrate_and_engineer_features()
