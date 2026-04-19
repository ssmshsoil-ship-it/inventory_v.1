# -*- coding: utf-8 -*-
"""
(주)성와의 재고 관리 최적화 시뮬레이터 (v4)

주요 기능:
1. sh_delivery_v3.cbm 모델을 사용한 2027년 수요 예측
2. 재고 유통기한(6개월)을 고려한 재가공 비용 리스크 계산
3. 대리점 긴급 요청 등 변수를 반영한 시뮬레이션
4. 경쟁 심화 지역 식별 및 재고 관리 가이드 제안
5. '오늘 생산량 제언' 및 '재고 신선도 경고' 리포트 생성

실행: python src/inventory_management_v4.py
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from pathlib import Path
from datetime import datetime, timedelta

# --- 기본 설정 ---
# 설정값들은 외부 config 파일로 분리하는 것을 권장합니다.
MODEL_PATH = Path("models/sh_delivery_v3.cbm")
HISTORICAL_DATA_PATH = Path("data/processed/final_training_data.csv")
REPORTS_DIR = Path("reports")

# 재고 관련 파라미터
SHELF_LIFE_DAYS = 180  # 재고 유통기한 (6개월)
REPROCESSING_COST_PER_UNIT = 1500 # 단위당 재가공 비용 (가정)
SAFETY_STOCK_WEEKS = 4 # 최소 안전 재고 (주 단위)

# 시뮬레이션 기간
SIMULATION_YEAR = 2027

class InventoryOptimizerV4:
    """재고 관리 최적화 시뮬레이터 클래스"""

    def __init__(self):
        print("재고 관리 최적화 시뮬레이터 v4를 시작합니다.")
        self.model = self._load_model()
        self.historical_data = self._load_historical_data()
        self.competitive_regions = self._analyze_competition()
        # 현재 재고 데이터는 별도 파일(예: current_inventory.csv)에서 로드해야 합니다.
        # 여기서는 시뮬레이션을 위해 가상의 초기 재고를 생성합니다.
        self.current_inventory = self._initialize_inventory()

    def _load_model(self) -> CatBoostRegressor:
        """sh_delivery_v3.cbm 모델을 로드합니다."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
        print(f"[OK] 예측 모델 로드 완료: {MODEL_PATH}")
        model = CatBoostRegressor()
        model.load_model(MODEL_PATH)
        return model

    def _load_historical_data(self) -> pd.DataFrame:
        """과거 출고 데이터(final_training_data.csv)를 로드합니다."""
        if not HISTORICAL_DATA_PATH.exists():
            raise FileNotFoundError(f"과거 데이터 파일이 없습니다: {HISTORICAL_DATA_PATH}")
        print(f"[OK] 과거 데이터 로드 완료: {HISTORICAL_DATA_PATH}")
        df = pd.read_csv(HISTORICAL_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _analyze_competition(self) -> list:
        """
        과거 8년 데이터(가용 데이터 기준)에서 지역별 출고량의 변동폭을 분석하여
        '경쟁 심화 지역'을 식별합니다.
        """
        print("\n- 경쟁 심화 지역 분석 시작...")
        # 'province'와 'week'를 기준으로 주별 출고수량 집계
        # 여기서는 가용 데이터 전체를 사용합니다.
        weekly_sales = self.historical_data.groupby(['province', pd.Grouper(key='date', freq='W-MON')])['출고수량'].sum().reset_index()
        
        # 지역별 출고량 표준편차 계산
        volatility = weekly_sales.groupby('province')['출고수량'].std().sort_values(ascending=False)
        
        # 변동성이 상위 25% 이상인 지역을 '경쟁 심화 지역'으로 분류
        threshold = volatility.quantile(0.75)
        competitive_regions = volatility[volatility >= threshold].index.tolist()
        
        print(f"  [분석 결과] 경쟁 심화 지역 (변동성 상위 25%): {competitive_regions}")
        print(f"  -> 해당 지역은 재고를 더 타이트하게 관리할 것을 권장합니다.")
        return competitive_regions

    def _prepare_future_features(self) -> pd.DataFrame:
        """2027년 수요 예측을 위한 특징(feature) 데이터프레임을 생성합니다."""
        print("\n- 2027년 예측용 데이터 생성 시작...")
        # 중요: 2027년 기상 데이터는 예측치를 사용해야 합니다.
        # 여기서는 가장 최근 년도(2026년)의 데이터를 2027년의 근사치로 사용합니다.
        
        # 1. 2027년 날짜 범위 생성
        dates_2027 = pd.date_range(start=f'{SIMULATION_YEAR}-01-01', end=f'{SIMULATION_YEAR}-12-31', freq='D')
        
        # 2. 과거 데이터에서 지역/품목 등 조합 추출
        # 예측에 필요한 모든 범주형 조합을 가져옵니다.
        categorical_cols = ['고객명', '품목', 'province', 'stn_id']
        combinations = self.historical_data[categorical_cols].drop_duplicates().dropna()
        
        # 3. 날짜와 조합을 기준으로 2027년 데이터프레임 생성
        future_df = pd.DataFrame(dates_2027, columns=['date'])
        future_df['_key'] = 1
        combinations['_key'] = 1
        future_df = pd.merge(future_df, combinations, on='_key').drop('_key', axis=1)

        # 4. 2026년 기상 데이터를 2027년에 매핑
        last_year_weather = self.historical_data[self.historical_data['date'].dt.year == (SIMULATION_YEAR - 1)]
        weather_cols = ['avg_temp', 'min_temp', 'max_temp', 'precip']
        
        # 날짜의 '월-일'을 키로 사용하여 매핑
        last_year_weather['month_day'] = last_year_weather['date'].dt.strftime('%m-%d')
        future_df['month_day'] = future_df['date'].dt.strftime('%m-%d')
        
        # 기상 데이터는 지점(stn_id)별로 매핑되어야 함
        weather_to_map = last_year_weather[['stn_id', 'month_day'] + weather_cols].drop_duplicates(subset=['stn_id', 'month_day'])
        
        future_df = pd.merge(future_df, weather_to_map, on=['stn_id', 'month_day'], how='left')
        
        # 기상 데이터 결측치는 전/후 값으로 채움
        future_df[weather_cols] = future_df.groupby('stn_id')[weather_cols].transform(lambda x: x.ffill().bfill())
        future_df = future_df.drop(columns=['month_day'])

        # 5. 모델 학습에 사용된 모든 피처 생성
        # train_model.py 스크립트를 참조하여 동일한 피처를 생성해야 합니다.
        features = self.model.feature_names_
        
        # 날짜 관련 피처
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['week'] = future_df['date'].dt.isocalendar().week
        future_df['dayofweek'] = future_df['date'].dt.dayofweek
        
        # 기상 관련 파생 피처
        future_df = future_df.sort_values(by=['stn_id', 'date']).reset_index(drop=True)
        future_df['temp_change_weekly'] = future_df.groupby('stn_id')['avg_temp'].diff(7).fillna(0)
        future_df['precip_sum_3d'] = future_df.groupby('stn_id')['precip'].rolling(window=3).sum().reset_index(0,drop=True).fillna(0)
        future_df['is_peak_season'] = future_df['date'].dt.month.isin([3, 4]).astype(int)

        # 학습에 사용된 모든 피처가 있는지 확인하고, 없으면 0이나 'missing'으로 채움
        for col in features:
            if col not in future_df.columns:
                # 데이터 타입에 따라 적절한 기본값 설정
                if self.historical_data[col].dtype == 'object':
                    future_df[col] = 'missing'
                else:
                    future_df[col] = 0

        print(f"  [OK] {len(future_df)}건의 2027년 예측용 데이터 생성 완료.")
        return future_df[features] # 모델에 필요한 피처만 반환

    def predict_demand(self, agency_request_multiplier: dict = None) -> pd.DataFrame:
        """
        2027년 수요를 예측하고, 대리점 긴급 요청 가중치를 적용합니다.
        
        Args:
            agency_request_multiplier (dict): {'지역명': 가중치} 형태. 예: {'경기': 1.2}
        """
        features_2027 = self._prepare_future_features()
        print("\n- 2027년 수요 예측 시작...")
        
        predictions = self.model.predict(features_2027)
        predicted_demand = features_2027.copy()
        predicted_demand['predicted_demand'] = np.maximum(0, predictions) # 예측값은 0 이상

        # 대리점 긴급 요청 가중치 적용
        if agency_request_multiplier:
            print(f"  [시뮬레이션] 대리점 긴급 요청 가중치 적용: {agency_request_multiplier}")
            for region, multiplier in agency_request_multiplier.items():
                mask = (predicted_demand['province'] == region)
                predicted_demand.loc[mask, 'predicted_demand'] *= multiplier
        
        # 일별, 지역별, 품목별 수요 집계
        daily_summary = predicted_demand.groupby(['date', 'province', '품목'])['predicted_demand'].sum().reset_index()
        print("  [OK] 일별/지역별/품목별 수요 예측 완료.")
        return daily_summary

    def _initialize_inventory(self):
        """초기 재고 상태를 가상으로 생성합니다."""
        # 실제로는 DB나 ERP에서 현재 재고 데이터를 가져와야 합니다.
        # 품목별로 4주치 평균 수요를 초기 재고로 가정합니다.
        avg_weekly_demand = self.historical_data.groupby('품목')['출고수량'].sum() / self.historical_data['date'].dt.to_period('W').nunique()
        initial_stock = (avg_weekly_demand * SAFETY_STOCK_WEEKS).to_dict()
        
        # 재고 신선도 추적을 위해 (수량, 생산일) 형태로 저장
        today = datetime.now().date()
        inventory = {
            item: [(qty, today)] for item, qty in initial_stock.items()
        }
        print(f"\n- 가상 초기 재고 생성 완료 (총 {len(inventory)}개 품목).")
        return inventory

    def run_simulation(self, demand_forecast: pd.DataFrame):
        """재고 시뮬레이션을 실행하고 일별 리포트를 생성합니다."""
        print("\n- 재고 시뮬레이션 및 리포트 생성 시작...")
        
        # 오늘 날짜를 기준으로 리포트 생성
        today = datetime.now().date()
        
        # --- 1. 재고 신선도 경고 리포트 ---
        freshness_warnings = []
        expired_risk_cost = 0
        
        for item, stock_batches in self.current_inventory.items():
            for qty, production_date in stock_batches:
                age = (today - production_date).days
                if age > SHELF_LIFE_DAYS:
                    # 유통기한 초과
                    risk_cost = qty * REPROCESSING_COST_PER_UNIT
                    freshness_warnings.append(f"  - [만료] 품목: {item}, 수량: {qty}, 생산일: {production_date}, 비용 리스크: {risk_cost:,.0f}원")
                    expired_risk_cost += risk_cost
                elif age > SHELF_LIFE_DAYS - 30:
                    # 유통기한 임박 (30일 이내)
                    freshness_warnings.append(f"  - [주의] 품목: {item}, 수량: {qty}, 생산일: {production_date}, 남은 기간: {SHELF_LIFE_DAYS - age}일")

        # --- 2. 오늘 생산량 제언 리포트 ---
        production_suggestions = []
        
        # 향후 N주간의 예상 수요 집계
        future_demand_period = demand_forecast[demand_forecast['date'].between(today, today + timedelta(weeks=SAFETY_STOCK_WEEKS))]
        demand_by_item = future_demand_period.groupby('품목')['predicted_demand'].sum()
        
        for item, current_stock_qty in {k: sum(q for q, d in v) for k, v in self.current_inventory.items()}.items():
            target_stock = demand_by_item.get(item, 0)
            
            # 경쟁 심화 지역의 주력 품목은 안전 재고를 더 타이트하게 관리
            # (이 로직은 더 정교화될 수 있습니다)
            is_competitive_item = self.historical_data[self.historical_data['품목'] == item]['province'].isin(self.competitive_regions).any()
            
            safety_stock_multiplier = 0.8 if is_competitive_item else 1.0
            required_production = (target_stock * safety_stock_multiplier) - current_stock_qty
            
            if required_production > 0:
                suggestion = f"  - 품목: {item}, 현재고: {current_stock_qty:,.0f}, 목표 재고: {target_stock:,.0f}, 제언량: {required_production:,.0f}"
                if is_competitive_item:
                    suggestion += " (경쟁 지역 주력 품목, 재고 타이트하게 관리)"
                production_suggestions.append(suggestion)

        # --- 최종 리포트 출력 ---
        print("\n" + "="*50)
        print(f" (주)성화 재고 최적화 리포트 (기준일: {today})")
        print("="*50)
        
        print("\n[ 품목별 재고 신선도 경고 ]")
        if freshness_warnings:
            for warning in freshness_warnings:
                print(warning)
            print(f"\n  >> 총 재가공 비용 리스크: {expired_risk_cost:,.0f}원")
        else:
            print("  - 현재 유통기한 만료 또는 임박 재고 없음.")
            
        print("\n[ 오늘 생산량 제언 ]")
        if production_suggestions:
            for suggestion in production_suggestions:
                print(suggestion)
        else:
            print("  - 현재 모든 품목의 재고가 충분하여 추가 생산 불필요.")
        print("="*50)


if __name__ == "__main__":
    optimizer = InventoryOptimizerV4()
    
    # 시뮬레이션 실행 (대리점 긴급 요청 가중치 예시)
    # 실제 운영 시에는 이 값을 GUI나 설정 파일에서 받아올 수 있습니다.
    emergency_requests = {
        '경기': 1.2, # 경기도 20% 추가 요청
        '전남': 1.1  # 전남 10% 추가 요청
    }
    
    future_demand = optimizer.predict_demand(agency_request_multiplier=emergency_requests)
    
    # 시뮬레이션 결과 기반 리포트 생성
    optimizer.run_simulation(future_demand)
