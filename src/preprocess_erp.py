"""
2019~2022년 ERP 출고 데이터 전처리
sh(성화) + dd(대동산업→성화) XLS 파일 → ISO주차별 집계 CSV
실행: python src/preprocess_erp.py
"""
import pandas as pd
from pathlib import Path
from config import DATA_DIR, PROCESSED_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022]

# 품명 → 대분류 분류 규칙
def classify_product(품명: str) -> str | None:
    p = str(품명)
    # 명시적 제외 (비료, 황토, 시료, 기타 비상토류)
    if any(k in p for k in ['비료', '시료', '황토', '마사', '씨앗', '종자', '퇴비',
                             '포인트', '바이오 미라클', '바이오미라클', '황금들']):
        return None
    # 수도용상토 (명시적 키워드 우선)
    if '수도용' in p:
        return '수도용상토'
    if any(k in p for k in ['상토나라', '경량상토', '준중량상토', '입상상토']):
        return '수도용상토'
    # 유기농상토 계열 → 수도용상토 (성화 수도용 유기농 라인)
    if '유기농상토' in p or '자연유기농상토' in p:
        return '수도용상토'
    # 원예용상토
    if '원예용' in p:
        return '원예용상토'
    if any(k in p for k in ['딸기', '토백이', '배지', '조경용', '탑스마트', '탑-스마트']):
        return '원예용상토'
    return None


def load_erp_files() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        for company in ['sh', 'dd']:
            path = DATA_DIR / f"{year}{company}.xls"
            if not path.exists():
                print(f"  [경고] 파일 없음: {path}")
                continue
            df = pd.read_excel(path)
            df['_year'] = year
            df['_company'] = company
            frames.append(df)
            print(f"  {path.name}: {len(df)}행 로드")

    all_df = pd.concat(frames, ignore_index=True)
    print(f"\n전체 로드: {len(all_df)}행")
    return all_df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 날짜 파싱 → ISO 연도/주차
    df['출고일자'] = pd.to_datetime(df['출고일자'], format='%Y/%m/%d', errors='coerce')
    invalid_dates = df['출고일자'].isna().sum()
    if invalid_dates > 0:
        print(f"  [경고] 날짜 파싱 실패: {invalid_dates}건 제외")
    df = df.dropna(subset=['출고일자']).copy()

    iso = df['출고일자'].dt.isocalendar()
    df['ISO연도'] = iso['year'].astype(int)
    df['ISO주차'] = iso['week'].astype(int)

    # 2. 음수 수량(반품/정정) 제거
    neg = (df['출고수량'] < 0).sum()
    print(f"  음수 수량 제거: {neg}건")
    df = df[df['출고수량'] > 0]

    # 3. 단위 = 포 만 유지
    non_po = (df['단위'] != '포').sum()
    print(f"  단위 != 포 제거: {non_po}건")
    df = df[df['단위'] == '포']

    # 4. 품명 → 대분류 매핑
    df['대분류'] = df['품명'].apply(classify_product)
    unclassified = df['대분류'].isna().sum()
    print(f"  분류 불가(제외): {unclassified}건")
    df = df.dropna(subset=['대분류'])

    print(f"  최종 유효 행수: {len(df)}")
    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['ISO연도', 'ISO주차', '대분류'])['출고수량'].sum().reset_index()
    pivot = grp.pivot_table(
        index=['ISO연도', 'ISO주차'],
        columns='대분류',
        values='출고수량',
        aggfunc='sum',
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={
        '수도용상토': '수도용_포',
        '원예용상토': '원예용_포',
    })
    for col in ['수도용_포', '원예용_포']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot['수도용_포'] = pivot['수도용_포'].fillna(0).astype(int)
    pivot['원예용_포'] = pivot['원예용_포'].fillna(0).astype(int)

    # ISO주차 53주 처리 (해당 주를 1월 첫주로 이동)
    pivot = pivot.sort_values(['ISO연도', 'ISO주차']).reset_index(drop=True)
    return pivot


def run():
    print("=== ERP 2019-2022 전처리 시작 ===\n")

    print("[1] 파일 로드...")
    df = load_erp_files()

    print("\n[2] 전처리...")
    df_clean = preprocess(df)

    print("\n[3] 주차별 집계...")
    weekly = aggregate_weekly(df_clean)

    print(f"\n결과: {len(weekly)}주 × {len(weekly.columns)}컬럼")
    print(f"기간: {weekly['ISO연도'].min()}년 {weekly['ISO주차'].min()}주 ~ "
          f"{weekly['ISO연도'].max()}년 {weekly['ISO주차'].max()}주")
    print("\n연도별 합계:")
    for yr in sorted(weekly['ISO연도'].unique()):
        sub = weekly[weekly['ISO연도'] == yr]
        print(f"  {yr}: 수도용 {sub['수도용_포'].sum():,}포 / 원예용 {sub['원예용_포'].sum():,}포")

    out_path = PROCESSED_DIR / 'erp_2019_2022.csv'
    weekly.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n저장: {out_path}")


if __name__ == '__main__':
    run()
