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

    # 기상 데이터가 확보된 2023~2026년 데이터만 사용하도록 필터링
    original_rows = len(sales_df)
    sales_df = sales_df[sales_df['date'].dt.year.between(2023, 2026)].copy()
    filtered_rows = original_rows - len(sales_df)
    if filtered_rows > 0:
        print(f"  [OK] 기상 데이터가 없는 기간(2019-2022)의 데이터 {filtered_rows}건을 학습에서 제외합니다.")
        
    sales_df['date'] = sales_df['date'].dt.normalize()
    # 매핑 테이블을 이용해 'stn_id' (기상 관측소 ID) 컬럼 추가
    sales_df['stn_id'] = sales_df['province'].map(PROVINCE_TO_STATION_MAP)
    print("  [OK] 판매 데이터에 기상 관측소 ID 매핑 완료.")
    
    # 3. 일별 기상 데이터 로드
    weather_df = merge_raw_weather_data(RAW_WEATHER_DIR)
    
    if weather_df.empty:
        print("\n- 3. [경고] 기상 데이터 없이 판매 데이터만으로 일단 통합을 진행합니다.")
        final_df = sales_df.copy()
        # 특성 공학에서 사용할 컬럼들을 NaN으로 추가
        final_df['avg_temp'] = np.nan
        final_df['precip'] = np.nan
        print("\n- 4. 판매 + 기상 데이터 결합 완료 (기상 데이터 없음)")
    else:
        # merge를 위해 날짜 형식 통일
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.normalize()
        
        print(f"  [진단] 원본 기상 데이터 컬럼: {weather_df.columns.tolist()}")
        # 표준 영문 컬럼명으로 유연하게 변경하고 필요한 컬럼만 선택
        rename_map = {}
        for col in weather_df.columns:
            lower_col = col.lower()
            # 기온 관련 컬럼명 표준화 (avg_temp)
            if col in ['평균기온(°C)', '평균기온', '평균 기온'] or '기온' in col or 'temp' in lower_col:
                rename_map[col] = 'avg_temp'
            # 강수량 관련 컬럼명 표준화 (precip)
            elif '강수량' in col or 'rain' in lower_col:
                rename_map[col] = 'precip'

        weather_df.rename(columns=rename_map, inplace=True)
        if rename_map:
            print(f"  [OK] 기상 데이터 컬럼명 표준화 완료. 변경사항: {rename_map}")
        
        required_cols = ['date', 'stn_id', 'avg_temp', 'precip']
        # 존재하는 컬럼만 선택
        weather_df = weather_df[[col for col in required_cols if col in weather_df.columns]]
        print("\n- 3. 일별 기상 데이터 로드 및 정제 완료")

        # 4. 판매 데이터와 기상 데이터 결합
        print("\n- 4. 판매 + 기상 데이터 결합 준비...")
        print("  - sales_df 키 (상위 5개):")
        print(sales_df[['date', 'stn_id']].head().to_string())
        print(f"  - sales_df key dtypes: date({sales_df['date'].dtype}), stn_id({sales_df['stn_id'].dtype})")
        
        if 'stn_id' in weather_df.columns and 'date' in weather_df.columns:
            print("  - weather_df 키 (상위 5개):")
            print(weather_df[['date', 'stn_id']].head().to_string())
            print(f"  - weather_df key dtypes: date({weather_df['date'].dtype}), stn_id({weather_df['stn_id'].dtype})")
            print(f"  [진단] Merge 직전 weather_df에 'avg_temp' 존재 여부: {'avg_temp' in weather_df.columns}")
        else:
            print(f"  - [경고] weather_df에 조인 키가 부족합니다. 현재 컬럼: {weather_df.columns.tolist()}")

        final_df = pd.merge(sales_df, weather_df, on=['date', 'stn_id'], how='left')

        # 결합 성공 여부 확인
        if 'avg_temp' in final_df.columns:
            merged_with_weather = final_df['avg_temp'].notna().sum()
            if merged_with_weather > 0:
                print(f"  [OK] 총 {len(final_df)}개 판매 데이터 중 {merged_with_weather}건에 기상 데이터가 결합되었습니다.")
            else:
                print(f"  [경고] 기상 데이터가 결합되지 않았습니다. 날짜 또는 stn_id가 일치하지 않을 수 있습니다.")
        else:
            print("  [경고] 기상 데이터 결합 후 'avg_temp' 컬럼이 생성되지 않았습니다. merge 로직이나 컬럼명을 확인하세요.")

        print("\n- 4. 판매 + 기상 데이터 결합 완료")

        # 기상 데이터 결합률 확인
        if 'avg_temp' in final_df.columns:
            nan_count = final_df['avg_temp'].isna().sum()
            total_count = len(final_df)
            if total_count > 0:
                nan_rate = (nan_count / total_count) * 100
                merge_rate = 100 - nan_rate
                print(f"  [확인] 기상 데이터 누락 행: {nan_count}건 / {total_count}건 ({nan_rate:.2f}%)")
                print(f"  [확인] 기상 데이터 결합률: {merge_rate:.2f}%")
                if merge_rate < 95:
                    print("  [경고] 기상 데이터 결합률이 95% 미만입니다. 고객-지역 매핑 또는 기상 데이터 자체를 점검해야 합니다.")
        else:
            print("  [오류] 'avg_temp' 컬럼이 없어 기상 데이터 결합률을 확인할 수 없습니다.")
            print(f"  -> final_df에 있는 실제 컬럼명: {final_df.columns.tolist()}")

    # 5. 특성 공학
    print("\n- 5. 특성 공학(Feature Engineering) 시작")
    # stn_id 기준으로 정렬
    final_df = final_df.sort_values(by=['stn_id', 'date']).reset_index(drop=True)
    
    # 그룹별(지점별)로 계산해야 정확함
    if 'avg_temp' in final_df.columns:
        final_df['temp_change_weekly'] = final_df.groupby('stn_id')['avg_temp'].diff(7)
    if 'precip' in final_df.columns:
        final_df['precip_sum_3d'] = final_df.groupby('stn_id')['precip'].rolling(window=3).sum().reset_index(0,drop=True)
    
    final_df['is_peak_season'] = final_df['date'].dt.month.isin([3, 4]).astype(int)
    
    # 결측치 발생 가능 (rolling, diff 초기값) -> 0으로 채우기
    final_df.fillna({
        'temp_change_weekly': 0, 'precip_sum_3d': 0
    }, inplace=True)
    print("  [OK] 파생 변수 생성 완료: '전주 대비 기온 변화', '최근 3일 누적 강수량', '피크 시즌 여부'")

    # 6. 최종 데이터 저장
    print(f"\n- 6. 최종 학습 데이터셋 저장")
    try:
        final_df.to_csv(FINAL_TRAINING_DATA, index=False, encoding='utf-8-sig')
        print(f"  [OK] 저장 완료: {FINAL_TRAINING_DATA}")
        print(f"  -> 최종 데이터: {final_df.shape[0]}행, {final_df.shape[1]}컬럼")
        
        # 출력할 컬럼이 존재하는지 확인하여 KeyError 방지
        display_cols = ['date', '고객명', 'province', 'avg_temp', 'precip', 'precip_sum_3d', 'is_peak_season']
        existing_display_cols = [col for col in display_cols if col in final_df.columns]
        print(f"  -> 결과 (일부 컬럼):\n{final_df[existing_display_cols].head().to_string()}")
    except Exception as e:
        print(f"  [오류] 파일 저장 실패: {e}")


if __name__ == "__main__":
    integrate_and_engineer_features()
