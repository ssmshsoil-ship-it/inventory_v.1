"""
상토 수요 예측을 위한 데이터 전처리 모듈

주요 기능:
- 휴일 효과 변수 생성 (파종기 중 공휴일/주말 전후)
- 누적 기온 계산 (최근 3주간 평균 기온)
- 지역 가중치 적용 (성화 공장 주요 납품처)
- 원가 데이터 통합 (cost 엑셀 파일)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class SoilDataPreprocessor:
    """상토 수요 예측 데이터 전처리 클래스"""
    
    # 성화 공장 주요 납품처 지역 (가중치 높게 적용)
    MAJOR_DELIVERY_REGIONS = {
        '경기': 0.3,
        '충남': 0.25,
        '충북': 0.2,
        '전북': 0.15,
        '강원': 0.1
    }
    
    # 파종기 월
    PLANTING_MONTHS = [2, 3, 4, 5]
    
    # 한국 공휴일 (2022-2025년)
    HOLIDAYS = {
        2022: [
            '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02',  # 신정, 설날
            '2022-03-01', '2022-03-09',  # 삼일절, 대선
            '2022-05-05', '2022-05-08',  # 어린이날, 석가탄신일
            '2022-06-01', '2022-06-06',  # 지방선거, 현충일
            '2022-08-15',  # 광복절
            '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12',  # 추석
            '2022-10-03', '2022-10-09',  # 개천절, 한글날
            '2022-12-25',  # 성탄절
        ],
        2023: [
            '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',  # 신정, 설날
            '2023-03-01',  # 삼일절
            '2023-05-05', '2023-05-27',  # 어린이날, 석가탄신일
            '2023-06-06',  # 현충일
            '2023-08-15',  # 광복절
            '2023-09-28', '2023-09-29', '2023-09-30',  # 추석
            '2023-10-03', '2023-10-09',  # 개천절, 한글날
            '2023-12-25',  # 성탄절
        ],
        2024: [
            '2024-01-01',  # 신정
            '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',  # 설날
            '2024-03-01',  # 삼일절
            '2024-04-10',  # 총선
            '2024-05-05', '2024-05-06', '2024-05-15',  # 어린이날, 대체공휴일, 석가탄신일
            '2024-06-06',  # 현충일
            '2024-08-15',  # 광복절
            '2024-09-16', '2024-09-17', '2024-09-18',  # 추석
            '2024-10-03', '2024-10-09',  # 개천절, 한글날
            '2024-12-25',  # 성탄절
        ],
        2025: [
            '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30',  # 신정, 설날
            '2025-03-01', '2025-03-03',  # 삼일절, 대체공휴일
            '2025-05-05', '2025-05-06',  # 어린이날, 석가탄신일
            '2025-06-06',  # 현충일
            '2025-08-15',  # 광복절
            '2025-10-03', '2025-10-05', '2025-10-06', '2025-10-07', '2025-10-08',  # 개천절, 추석
            '2025-10-09',  # 한글날
            '2025-12-25',  # 성탄절
        ],
    }
    
    # 농번기 주요 공휴일 (상토 수요에 큰 영향)
    MAJOR_FARMING_HOLIDAYS = {
        '설날': '농번기 시작 전 대량 주문',
        '삼일절': '파종 준비 시작',
        '식목일': '파종 본격화',
    }
    
    def __init__(self, df):
        """
        Args:
            df: 날짜, 출고량, 기온, 지역 정보가 포함된 DataFrame
        """
        self.df = df.copy()
        self._validate_data()
        
    def _validate_data(self):
        """필수 컬럼 확인"""
        required_cols = ['date']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
        
        # 날짜 컬럼을 datetime으로 변환
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
    
    def add_holiday_features(self):
        """
        휴일 효과 변수 추가
        - is_holiday: 공휴일 여부
        - is_weekend: 주말 여부
        - days_to_holiday: 다음 휴일까지 남은 일수
        - days_from_holiday: 휴일 이후 경과 일수
        - is_planting_season: 파종기 여부
        - holiday_effect: 파종기 중 휴일 전후 효과 (전 3일, 후 3일)
        """
        # 공휴일 여부 (연도별)
        all_holidays = []
        for year, holidays in self.HOLIDAYS.items():
            all_holidays.extend(holidays)
        holidays = pd.to_datetime(all_holidays)
        self.df['is_holiday'] = self.df['date'].isin(holidays)
        
        # 주말 여부 (토요일=5, 일요일=6)
        self.df['is_weekend'] = self.df['date'].dt.dayofweek.isin([5, 6])
        
        # 파종기 여부
        self.df['is_planting_season'] = self.df['date'].dt.month.isin(self.PLANTING_MONTHS)
        
        # 휴일(공휴일 + 주말) 통합
        self.df['is_holiday_or_weekend'] = self.df['is_holiday'] | self.df['is_weekend']
        
        # 다음 휴일까지 남은 일수 계산
        self.df['days_to_holiday'] = self._calculate_days_to_next_holiday()
        
        # 휴일 이후 경과 일수 계산
        self.df['days_from_holiday'] = self._calculate_days_from_last_holiday()
        
        # 파종기 중 휴일 전후 효과 (전 5일, 후 3일)
        # 농번기 시작 전 휴일에 주문이 몰리는 패턴 반영
        self.df['holiday_effect'] = 0
        mask_planting = self.df['is_planting_season']
        
        # 휴일 전 5일 (주문 증가 기간)
        self.df.loc[mask_planting & (self.df['days_to_holiday'] <= 5) & (self.df['days_to_holiday'] >= 1), 'holiday_effect'] = 2
        
        # 휴일 당일
        self.df.loc[mask_planting & (self.df['days_to_holiday'] == 0), 'holiday_effect'] = 1
        
        # 휴일 후 3일 (주문 감소 기간)
        self.df.loc[mask_planting & (self.df['days_from_holiday'] <= 3) & (self.df['days_from_holiday'] >= 1), 'holiday_effect'] = -1
        
        # 설날/추석 등 주요 명절 전후 효과 강화
        self.df['is_major_holiday'] = self._check_major_holidays()
        self.df.loc[mask_planting & self.df['is_major_holiday'] & (self.df['days_to_holiday'] <= 7), 'holiday_effect'] = 3
        
        return self
    
    def _calculate_days_to_next_holiday(self):
        """다음 휴일까지 남은 일수 계산"""
        days_to_holiday = []
        
        for date in self.df['date']:
            if self.df.loc[self.df['date'] == date, 'is_holiday_or_weekend'].values[0]:
                days_to_holiday.append(0)
            else:
                # 다음 휴일 찾기
                future_holidays = self.df[
                    (self.df['date'] > date) & 
                    (self.df['is_holiday_or_weekend'])
                ]['date']
                
                if len(future_holidays) > 0:
                    next_holiday = future_holidays.iloc[0]
                    days = (next_holiday - date).days
                    days_to_holiday.append(days)
                else:
                    days_to_holiday.append(999)  # 휴일이 없으면 큰 값
        
        return days_to_holiday
    
    def _check_major_holidays(self):
        """주요 명절(설날, 추석) 여부 확인"""
        is_major = []
        
        for date in self.df['date']:
            year = date.year
            month = date.month
            day = date.day
            
            # 설날 (음력 1월 1일 전후) - 대략 1월 말~2월 중순
            # 추석 (음력 8월 15일 전후) - 대략 9월 중순~10월 초
            is_seollal = (month in [1, 2]) and (15 <= day <= 28 or day <= 15)
            is_chuseok = (month in [9, 10]) and (1 <= day <= 20)
            
            is_major.append(is_seollal or is_chuseok)
        
        return is_major
    
    def _calculate_days_from_last_holiday(self):
        """마지막 휴일 이후 경과 일수 계산"""
        days_from_holiday = []
        
        for date in self.df['date']:
            if self.df.loc[self.df['date'] == date, 'is_holiday_or_weekend'].values[0]:
                days_from_holiday.append(0)
            else:
                # 이전 휴일 찾기
                past_holidays = self.df[
                    (self.df['date'] < date) & 
                    (self.df['is_holiday_or_weekend'])
                ]['date']
                
                if len(past_holidays) > 0:
                    last_holiday = past_holidays.iloc[-1]
                    days = (date - last_holiday).days
                    days_from_holiday.append(days)
                else:
                    days_from_holiday.append(999)  # 휴일이 없으면 큰 값
        
        return days_from_holiday
    
    def add_cumulative_temperature(self, temp_column='temperature'):
        """
        누적 기온 변수 추가
        - temp_3week_avg: 최근 3주간 평균 기온
        - temp_3week_cumsum: 최근 3주간 기온 누적합
        - temp_3week_growing_degree: 최근 3주간 생장온도 누적 (기준온도 5도 이상)
        
        Args:
            temp_column: 기온 데이터가 있는 컬럼명
        """
        if temp_column not in self.df.columns:
            raise ValueError(f"기온 컬럼 '{temp_column}'이 없습니다.")
        
        # 최근 3주(21일) 이동 평균
        self.df['temp_3week_avg'] = self.df[temp_column].rolling(window=21, min_periods=1).mean()
        
        # 최근 3주 누적합
        self.df['temp_3week_cumsum'] = self.df[temp_column].rolling(window=21, min_periods=1).sum()
        
        # 생장온도 누적 (기준온도 5도 이상만 누적)
        growing_degree = self.df[temp_column].apply(lambda x: max(0, x - 5))
        self.df['temp_3week_growing_degree'] = growing_degree.rolling(window=21, min_periods=1).sum()
        
        return self
    
    def add_regional_weighted_temperature(self, region_column='region', temp_column='temperature'):
        """
        지역 가중치를 적용한 기온 변수 추가
        성화 공장 주요 납품처 지역의 기온에 더 높은 가중치 적용
        
        Args:
            region_column: 지역 정보가 있는 컬럼명
            temp_column: 기온 데이터가 있는 컬럼명
        """
        if region_column not in self.df.columns:
            # 지역 컬럼이 없으면 전국 평균으로 처리
            print(f"경고: '{region_column}' 컬럼이 없습니다. 지역 가중치를 적용하지 않습니다.")
            self.df['weighted_temperature'] = self.df[temp_column]
            return self
        
        if temp_column not in self.df.columns:
            raise ValueError(f"기온 컬럼 '{temp_column}'이 없습니다.")
        
        # 지역별 가중치 적용
        self.df['region_weight'] = self.df[region_column].map(self.MAJOR_DELIVERY_REGIONS).fillna(0.05)
        
        # 가중 기온 계산
        self.df['weighted_temperature'] = self.df[temp_column] * self.df['region_weight']
        
        # 날짜별 가중 평균 기온 (여러 지역 데이터가 있는 경우)
        if self.df.groupby('date').size().max() > 1:
            weighted_temp_by_date = self.df.groupby('date').apply(
                lambda x: (x['weighted_temperature'].sum() / x['region_weight'].sum())
            ).reset_index(name='daily_weighted_temp')
            
            self.df = self.df.merge(weighted_temp_by_date, on='date', how='left')
        else:
            self.df['daily_weighted_temp'] = self.df['weighted_temperature']
        
        # 가중 기온의 3주 누적
        self.df = self.df.sort_values('date')
        self.df['weighted_temp_3week_avg'] = self.df.groupby(region_column)['weighted_temperature'].transform(
            lambda x: x.rolling(window=21, min_periods=1).mean()
        )
        
        return self
    
    def add_cost_features(self, cost_file_path='data/cost.xlsx'):
        """
        원가 데이터 추가
        
        Args:
            cost_file_path: cost 엑셀 파일 경로
        """
        cost_path = Path(cost_file_path)
        
        if not cost_path.exists():
            print(f"경고: cost 파일을 찾을 수 없습니다: {cost_file_path}")
            print("원가 데이터 없이 진행합니다.")
            return self
        
        try:
            # cost 엑셀 파일 로드
            cost_df = pd.read_excel(cost_path)
            print(f"원가 데이터 로드 완료: {len(cost_df)}행")
            
            # 날짜 컬럼 찾기 (일반적인 날짜 컬럼명들)
            date_cols = [col for col in cost_df.columns if any(
                keyword in col.lower() for keyword in ['date', '날짜', '일자', 'day']
            )]
            
            if date_cols:
                date_col = date_cols[0]
                cost_df[date_col] = pd.to_datetime(cost_df[date_col])
                
                # 원가 관련 컬럼 찾기
                cost_cols = [col for col in cost_df.columns if any(
                    keyword in col.lower() for keyword in ['cost', '원가', '단가', 'price', '가격']
                )]
                
                if cost_cols:
                    # 날짜 기준으로 병합
                    merge_cols = [date_col] + cost_cols
                    self.df = self.df.merge(
                        cost_df[merge_cols],
                        left_on='date',
                        right_on=date_col,
                        how='left'
                    )
                    
                    # 결측치 forward fill (이전 값으로 채우기)
                    for col in cost_cols:
                        if col in self.df.columns:
                            self.df[col] = self.df[col].fillna(method='ffill')
                    
                    print(f"원가 피처 추가 완료: {cost_cols}")
                else:
                    print("경고: 원가 관련 컬럼을 찾을 수 없습니다.")
            else:
                print("경고: 날짜 컬럼을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"원가 데이터 로드 중 오류 발생: {e}")
            print("원가 데이터 없이 진행합니다.")
        
        return self
    
    def preprocess_all(self, temp_column='temperature', region_column='region', cost_file_path='data/cost.xlsx'):
        """
        모든 전처리 단계를 한 번에 실행
        
        Args:
            temp_column: 기온 데이터가 있는 컬럼명
            region_column: 지역 정보가 있는 컬럼명
            cost_file_path: cost 엑셀 파일 경로
        
        Returns:
            전처리된 DataFrame
        """
        self.add_holiday_features()
        self.add_cumulative_temperature(temp_column)
        self.add_regional_weighted_temperature(region_column, temp_column)
        self.add_cost_features(cost_file_path)
        
        return self.df
    
    def get_feature_importance_info(self):
        """생성된 피처들의 설명 반환"""
        features_info = {
            '휴일 효과 변수': [
                'is_holiday: 공휴일 여부',
                'is_weekend: 주말 여부',
                'is_planting_season: 파종기(2~5월) 여부',
                'days_to_holiday: 다음 휴일까지 남은 일수',
                'days_from_holiday: 휴일 이후 경과 일수',
                'is_major_holiday: 주요 명절(설날/추석) 여부',
                'holiday_effect: 파종기 중 휴일 전후 효과',
                '  - 3: 주요 명절 전 7일 (대량 주문 기간)',
                '  - 2: 일반 휴일 전 5일 (주문 증가)',
                '  - 1: 휴일 당일',
                '  - 0: 평일',
                '  - -1: 휴일 후 3일 (주문 감소)'
            ],
            '누적 기온 변수': [
                'temp_3week_avg: 최근 3주간 평균 기온',
                'temp_3week_cumsum: 최근 3주간 기온 누적합',
                'temp_3week_growing_degree: 최근 3주간 생장온도 누적 (기준 5도)'
            ],
            '지역 가중 기온 변수': [
                'region_weight: 지역별 가중치 (주요 납품처 높음)',
                'weighted_temperature: 가중치 적용 기온',
                'daily_weighted_temp: 날짜별 가중 평균 기온',
                'weighted_temp_3week_avg: 가중 기온의 3주 이동평균'
            ],
            '원가 변수': [
                'cost 엑셀 파일에서 로드된 원가 관련 컬럼들',
                '(파일 내용에 따라 자동으로 추가됨)'
            ]
        }
        
        return features_info


# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'temperature': np.random.normal(15, 10, len(dates)),
        'region': np.random.choice(['경기', '충남', '충북', '전북', '강원', '기타'], len(dates)),
        'shipment': np.random.randint(100, 1000, len(dates))
    })
    
    # 전처리 실행
    preprocessor = SoilDataPreprocessor(sample_data)
    processed_df = preprocessor.preprocess_all(temp_column='temperature', region_column='region')
    
    # 결과 확인
    print("전처리된 데이터 샘플:")
    print(processed_df.head(10))
    print("\n생성된 컬럼:")
    print(processed_df.columns.tolist())
    print("\n피처 설명:")
    for category, features in preprocessor.get_feature_importance_info().items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  - {feature}")
