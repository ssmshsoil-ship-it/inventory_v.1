# -*- coding: utf-8 -*-
"""
'data/raw/sales' 디렉토리의 모든 엑셀 파일을 병합하여
하나의 마스터 영업 데이터 파일(master_v1.1.xlsx)을 생성합니다.

실행: python src/create_master_sales_file.py
"""

import pandas as pd
from pathlib import Path

def create_master_sales_file():
    """'data/raw/sales'의 엑셀 파일을 병합하여 마스터 파일을 생성합니다."""
    
    # 1. 경로 설정
    sales_dir = Path("data/raw/sales")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = "20190101-20260420 master_v1.1.xlsx"
    output_path = output_dir / output_filename
    
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
            all_dfs.append(df)
            print(f"  - '{file.name}' 로드 완료.")
        except Exception as e:
            print(f"[경고] '{file.name}' 파일을 읽는 중 오류가 발생했습니다: {e}")
            
    if not all_dfs:
        print("[오류] 유효한 데이터를 읽어오지 못했습니다. 파일 내용을 확인해주세요.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n병합 완료. 총 {len(master_df)}개 행이 생성되었습니다.")
    print(f"병합된 전체 컬럼 목록: {master_df.columns.tolist()}")

    # 4. 요청된 컬럼 순서로 재정렬하고, 해당 컬럼만 선택
    desired_columns = ['출고일자', '고객', '품명', '규격', '출고수량', '품번', '비고']
    
    # 원본 데이터에 존재하는 컬럼만 필터링하여 순서대로 정렬
    final_columns = [col for col in desired_columns if col in master_df.columns]
    
    missing_cols = set(desired_columns) - set(final_columns)
    if missing_cols:
        print(f"[경고] 요청된 컬럼 중 일부가 원본 데이터에 없습니다: {list(missing_cols)}")

    final_df = master_df[final_columns]
    print(f"\n최종 선택 및 재정렬된 컬럼: {final_df.columns.tolist()}")

    # 5. 결과 저장
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
