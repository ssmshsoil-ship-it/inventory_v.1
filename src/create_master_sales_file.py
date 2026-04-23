# -*- coding: utf-8 -*-
"""
'data/raw/sales' 디렉토리의 모든 엑셀 파일을 병합하여
하나의 마스터 영업 데이터 파일(master_v1.1.xlsx)을 생성합니다.

실행: python src/create_master_sales_file.py
"""

import pandas as pd
from pathlib import Path

# 컬럼명 통일을 위한 매핑 (유사 이름 처리)
COLUMN_MAP = {
    '출고일자': ['출고일자', '출고 일자', '일자'],
    '고객': ['고객', '고객사', '고객명'],
    '품명': ['품명', '품목', '품목명'],
    '규격': ['규격'],
    '출고수량': ['출고수량', '수량', '출고 수량'],
    '품번': ['품번', '품목코드'],
    '비고': ['비고', '메모']
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임의 컬럼명을 표준 컬럼명으로 변경합니다."""
    rename_map = {}
    df_columns = df.columns.tolist()
    for standard_name, possible_names in COLUMN_MAP.items():
        found_col = next((col for col in possible_names if col in df_columns), None)
        if found_col and found_col != standard_name:
            rename_map[found_col] = standard_name
    
    if rename_map:
        print(f"    - 컬럼명 통일: {rename_map}")
        return df.rename(columns=rename_map)
    return df

def create_master_sales_file():
    """'data/raw/sales'의 엑셀 파일을 병합하여 마스터 파일을 생성합니다."""
    
    # 1. 경로 설정
    sales_dir = Path("data/raw/sales")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not sales_dir.exists():
        print(f"[오류] 영업 데이터 디렉토리를 찾을 수 없습니다: {sales_dir}")
        print(" -> 'data/raw/sales' 경로에 엑셀 파일이 있는지 확인해주세요.")
        return

    # 2. 모든 엑셀 파일 찾기 (.xls와 .xlsx 확장자 모두 포함)
    excel_files = list(sales_dir.glob("*.xlsx")) + list(sales_dir.glob("*.xls"))
    if not excel_files:
        print(f"[오류] '{sales_dir}' 디렉토리에서 엑셀 파일을 찾을 수 없습니다.")
        return
        
    print(f"총 {len(excel_files)}개의 엑셀 파일을 병합합니다.")

    # 3. 모든 파일을 읽어 하나의 데이터프레임으로 합치기
    all_dfs = []
    for file in excel_files:
        try:
            # .xlsx 파일은 'openpyxl' 엔진으로 읽는 것을 권장합니다.
            df = pd.read_excel(file, engine='openpyxl' if file.suffix == '.xlsx' else None)
            df = standardize_columns(df)  # 컬럼명 표준화
            all_dfs.append(df)
            print(f"  - '{file.name}' 로드 완료.")
        except Exception as e:
            print(f"[경고] '{file.name}' 파일을 읽는 중 오류가 발생했습니다: {e}")
            
    if not all_dfs:
        print("[오류] 유효한 데이터를 읽어오지 못했습니다. 파일 내용을 확인해주세요.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n병합 완료. 총 {len(master_df)}개 행이 생성되었습니다.")
    
    # 4. 데이터 기본 정제 및 동적 파일명 생성
    # - '출고일자'는 날짜 타입으로, '출고수량'은 숫자 타입으로 변환합니다.
    if '출고일자' in master_df.columns:
        master_df['출고일자'] = pd.to_datetime(master_df['출고일자'], errors='coerce')
        master_df.dropna(subset=['출고일자'], inplace=True)
        
    if '출고수량' in master_df.columns:
        master_df['출고수량'] = pd.to_numeric(master_df['출고수량'], errors='coerce').fillna(0)

    # - 데이터 기간을 바탕으로 파일명 동적 생성
    output_filename = "master_v1.1.xlsx" # 기본 파일명
    if '출고일자' in master_df.columns and not master_df.empty:
        min_date = master_df['출고일자'].min().strftime('%Y%m%d')
        max_date = master_df['출고일자'].max().strftime('%Y%m%d')
        output_filename = f"{min_date}-{max_date} master_v1.1.xlsx"
        print(f"  - 데이터 기간을 바탕으로 동적 파일명 생성: {output_filename}")
    else:
        print("[경고] 유효한 '출고일자'가 없어 기본 파일명을 사용합니다.")
        
    output_path = output_dir / output_filename
    print(f"병합된 전체 컬럼 목록: {master_df.columns.tolist()}")

    # 5. 요청된 컬럼 순서로 재정렬하고, 해당 컬럼만 선택
    desired_columns = ['출고일자', '고객', '품명', '규격', '출고수량', '품번', '비고']
    
    # 원본 데이터에 존재하는 컬럼만 필터링하여 순서대로 정렬
    final_columns = [col for col in desired_columns if col in master_df.columns]
    
    missing_cols = set(desired_columns) - set(final_columns)
    if missing_cols:
        print(f"[경고] 요청된 컬럼 중 일부가 원본 데이터에 없습니다: {list(missing_cols)}")

    final_df = master_df[final_columns].copy()

    # 출고일자를 'YYYY-MM-DD' 형식으로 변경 (시간 정보 제거)
    if '출고일자' in final_df.columns:
        final_df['출고일자'] = final_df['출고일자'].dt.date

    print(f"\n최종 선택 및 재정렬된 컬럼: {final_df.columns.tolist()}")

    # 6. 결과 저장
    try:
        final_df.to_excel(output_path, index=False, engine='openpyxl')
        print("\n" + "="*50)
        print("  마스터 데이터 파일 생성 완료!")
        print(f"  - 파일 경로: {output_path}")
        print(f"  - 총 행 수: {len(final_df)}")
        print(f"  - 컬럼 수: {len(final_df.columns)}")
        print("="*50)
    except Exception as e:
        print(f"[오류] 최종 파일을 저장하는 중 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    create_master_sales_file()
