# -*- coding: utf-8 -*-
"""
unique_customers.csv 파일을 읽어 고객명에서 지역 정보를 추출하고,
정제된 고객-지역 매핑 마스터 파일을 생성합니다.
실행: python src/clean_region_data.py
"""
import pandas as pd
import re
from config import PROCESSED_DIR, UNIQUE_CUSTOMERS_CSV, CUSTOMER_MAP_CSV

# 시/군 -> 도 단위 매핑 정의
PROVINCE_MAP = {
    '전남': ['보성', '고흥', '화순', '나주', '함평', '영암', '해남', '순천', '광주', '장흥', '강진', '무안', '목포', '영광', '담양', '곡성', '구례', '완도', '진도'],
    '전북': ['익산', '전주', '김제', '군산', '정읍', '남원', '완주', '고창', '부안', '임실', '순창', '진안', '무주', '장수'],
    '경남': ['함안', '창원', '밀양', '진주', '합천', '의령', '거창', '함양', '산청', '하동', '남해', '사천', '고성', '거제', '통영', '김해', '양산', '부산', '울산'],
    '경북': ['상주', '안동', '의성', '구미', '경주', '대구', '김천', '영천', '경산', '청도', '고령', '성주', '칠곡', '군위', '예천', '문경', '영주', '봉화', '울진', '영덕', '포항', '청송', '영양'],
    '충남': ['대전', '천안', '공주', '논산', '당진', '서산', '보령', '아산', '계룡', '금산', '부여', '서천', '청양', '홍성', '예산', '태안'],
    '충북': ['청주', '충주', '제천', '음성', '진천', '보은', '옥천', '영동', '증평', '괴산', '단양'],
    '경기': ['수원', '화성', '이천', '평택', '안성', '용인', '성남', '고양', '부천', '안양', '광명', '과천', '오산', '시흥', '군포', '의왕', '하남', '광주', '남양주', '구리', '의정부', '동두천', '양주', '파주', '김포', '연천', '가평', '양평', '여주'],
    '강원': ['철원', '춘천', '원주', '강릉', '동해', '태백', '속초', '삼척', '홍천', '횡성', '영월', '평창', '정선', '화천', '양구', '인제', '고성', '양양'],
    '제주': ['제주', '서귀포'],
}

def clean_customer_data(input_path, output_path):
    """
    고유 고객 목록을 정제하여 지역(시/군, 도) 정보를 매핑하고 저장합니다.
    """
    print(f"- 입력 파일 로드: {input_path}")
    if not input_path.exists():
        print(f"[오류] 입력 파일 '{input_path}'을(를) 찾을 수 없습니다.")
        print(" -> 먼저 python src/build_dataset.py 를 실행하여 파일을 생성하세요.")
        return

    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"  [OK] {len(df)}개의 고유 고객 데이터 로드 완료.")

    # 1. 키워드 추출: 정규표현식으로 시/군 명칭 추출
    all_cities = [city for cities in PROVINCE_MAP.values() for city in cities]
    # 긴 이름이 짧은 이름을 포함하는 경우(예: 광주, 광주시)를 위해 긴 이름부터 매칭
    all_cities.sort(key=len, reverse=True) 
    city_pattern = '|'.join(all_cities)
    
    df['city'] = df['고객명'].str.extract(f'({city_pattern})', expand=False)
    print(f"- 고객명에서 시/군 키워드 추출 완료. (총 {df['city'].notna().sum()}건 매칭)")

    # 2. 광역 단위 매핑
    # city -> province 역방향 맵 생성
    city_to_province_map = {city: province for province, cities in PROVINCE_MAP.items() for city in cities}
    df['province'] = df['city'].map(city_to_province_map)
    print("- 시/군 -> 도/광역시 단위 매핑 완료.")

    # 3. 예외 처리
    # (주) 성화, (주)대동산업 등은 '본사' 또는 '기타'로 분류
    headquarters_keywords = ['성화', '대동산업']
    hq_pattern = '|'.join(headquarters_keywords)
    
    is_hq = df['고객명'].str.contains(hq_pattern, na=False)
    df.loc[is_hq, 'province'] = '본사'
    df.loc[is_hq, 'city'] = '본사'
    print(f"- 예외 처리: '본사' 키워드로 {is_hq.sum()}건 분류 완료.")

    # 매핑되지 않은 나머지 항목들은 '미분류'로 처리
    df[['city', 'province']] = df[['city', 'province']].fillna('미분류')
    print("- 매핑되지 않은 데이터 '미분류' 처리 완료.")

    # 4. 출력
    # 컬럼 순서 정리
    df = df[['고객명', '빈도수', 'province', 'city']]
    
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 정제된 고객-지역 매핑 파일을 저장했습니다.")
        print(f" -> 경로: {output_path}")
        print(f" -> 결과 (상위 5건):\n{df.head().to_string()}")

    except Exception as e:
        print(f"\n[오류] 파일 저장 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    clean_customer_data(UNIQUE_CUSTOMERS_CSV, CUSTOMER_MAP_CSV)
