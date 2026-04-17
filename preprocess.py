"""
상토 수요 예측을 위한 데이터 전처리 모듈

주요 기능:
- 휴일 효과 변수 생성 (파종기 중 공휴일/주말 전후)
- 누적 기온 계산 (최근 3주간 평균 기온)
- 지역 가중치 적용 (성화 공장 주요 납품처)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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
    
    # 한국 공휴일 (예시 - 실제로는 외부 라이브러리나 API 사용 권장)
    HOLIDAYS_2024 = [
        '2024-01-01',  # 신정
        '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',  # 설날
        '2024-03-01',  # 삼일절
        '2024-04-10',  # 총선
        '2024-05-05',  # 어린이날
        '2024-05-06',  # 대체공휴일
        '2024-05-15',  # 석가탄신일
        '2024-06-06',  # 현충일
        '2024-08-15',  # 광복절
        '2024-09-16', '2024-09-17', '2024-09-18',  # 추석
        '2024-10-03',  # 개천절
        '2024-10-09',  # 한글날
        '2024-12-25',  # 성탄절
    ]
    
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
        # 공휴일 여부
        holidays = pd.to_datetime(self.HOLIDAYS_2024)
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
        
        # 파종기 중 휴일 전후 효과 (전 3일, 후 3일)
        self.df['holiday_effect'] = 0
        mask_planting = self.df['is_planting_season']
        
        # 휴일 전 3일
        self.df.loc[mask_planting & (self.df['days_to_holiday'] <= 3) & (self.df['days_to_holiday'] >= 0), 'holiday_effect'] = 1
        
        # 휴일 후 3일
        self.df.loc[mask_planting & (self.df['days_from_holiday'] <= 3) & (self.df['days_from_holiday'] >= 0), 'holiday_effect'] = -1
        
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
    
    def preprocess_all(self, temp_column='temperature', region_column='region'):
        """
        모든 전처리 단계를 한 번에 실행
        
        Args:
            temp_column: 기온 데이터가 있는 컬럼명
            region_column: 지역 정보가 있는 컬럼명
        
        Returns:
            전처리된 DataFrame
        """
        self.add_holiday_features()
        self.add_cumulative_temperature(temp_column)
        self.add_regional_weighted_temperature(region_column, temp_column)
        
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
                'holiday_effect: 파종기 중 휴일 전후 효과 (-1: 휴일 후 3일, 0: 평일, 1: 휴일 전 3일)'
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
