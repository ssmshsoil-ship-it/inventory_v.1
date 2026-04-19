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
        self._resolve_column_names() # 주요 컬럼명 동적 확인
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

    def _resolve_column_names(self):
        """historical_data에서 주요 컬럼명('품목', '출고수량' 등)을 동적으로 찾습니다."""
        print("\n- 주요 데이터 컬럼명 확인...")
        
        # 품목 컬럼 찾기
        item_candidates = ['품목', '품명']
        for col in item_candidates:
            if col in self.historical_data.columns:
                self.item_col = col
                break
        else:
            raise KeyError(f"품목/품명 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {self.historical_data.columns.tolist()}")

        # 출고수량 컬럼 찾기
        qty_candidates = ['출고수량', '수량']
        for col in qty_candidates:
            if col in self.historical_data.columns:
                self.qty_col = col
                break
        else:
            raise KeyError(f"출고수량/수량 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {self.historical_data.columns.tolist()}")
        
        # 지역(province) 컬럼 찾기
        province_candidates = ['province', '지역']
        for col in province_candidates:
            if col in self.historical_data.columns:
                self.province_col = col
                break
        else:
            raise KeyError(f"province/지역 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {self.historical_data.columns.tolist()}")
        
        print(f"  [OK] 품목 컬럼: '{self.item_col}', 수량 컬럼: '{self.qty_col}', 지역 컬럼: '{self.province_col}'")

    def _analyze_competition(self) -> list:
        """
        과거 8년 데이터(가용 데이터 기준)에서 지역별 출고량의 변동폭을 분석하여
        '경쟁 심화 지역'을 식별합니다.
        """
        print("\n- 경쟁 심화 지역 분석 시작...")
        # 'province'와 'week'를 기준으로 주별 출고수량 집계
        # 여기서는 가용 데이터 전체를 사용합니다.
        weekly_sales = self.historical_data.groupby([self.province_col, pd.Grouper(key='date', freq='W-MON')])[self.qty_col].sum().reset_index()
        
        # 지역별 출고량 표준편차 계산
        volatility = weekly_sales.groupby(self.province_col)[self.qty_col].std().sort_values(ascending=False)
        
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
        categorical_cols = ['고객명', self.item_col, self.province_col, 'stn_id']
        combinations = self.historical_data[categorical_cols].drop_duplicates().dropna()
        
        # 3. 날짜와 조합을 기준으로 2027년 데이터프레임 생성
        future_df = pd.DataFrame(dates_2027, columns=['date'])
        future_df['_key'] = 1
        combinations['_key'] = 1
        future_df = pd.merge(future_df, combinations, on='_key').drop('_key', axis=1)

        # 4. 2026년 기상 데이터를 2027년에 매핑
        last_year_weather = self.historical_data[self.historical_data['date'].dt.year == (SIMULATION_YEAR - 1)].copy()
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
                print(f"  [경고] 예측용 데이터에 '{col}' 피처가 없어 기본값으로 채웁니다.")

        print(f"  [OK] {len(future_df)}건의 2027년 예측용 데이터 생성 완료.")
        return future_df # 모델링에 필요한 모든 컬럼 반환

    def predict_demand(self, agency_request_multiplier: dict = None) -> pd.DataFrame:
        """
        2027년 수요를 예측하고, 대리점 긴급 요청 가중치를 적용합니다.
        
        Args:
            agency_request_multiplier (dict): {'지역명': 가중치} 형태. 예: {'경기': 1.2}
        """
        future_df = self._prepare_future_features()
        print("\n- 2027년 수요 예측 시작...")
        
        features_for_prediction = self.model.feature_names_
        predictions = self.model.predict(future_df[features_for_prediction])

        # 만약 예측값의 합이 비정상적으로 낮으면, Backup 로직 가동
        if predictions.sum() < 1:
            print("  [경고] 모델 예측값이 0에 가깝습니다. 피처 불일치 가능성이 있습니다. Backup 로직을 가동합니다.")
            
            # 작년(2026년) 데이터를 기반으로 예측값 생성
            last_year_data = self.historical_data[self.historical_data['date'].dt.year == (SIMULATION_YEAR - 1)].copy()
            last_year_data['month_day'] = last_year_data['date'].dt.strftime('%m-%d')
            
            # 품목별/월-일별 평균 출고량을 계산
            backup_demand_map = last_year_data.groupby([self.item_col, 'month_day'])[self.qty_col].mean().to_dict()
            
            future_df['month_day'] = future_df['date'].dt.strftime('%m-%d')
            # 튜플 키 (품목, 월-일)로 매핑
            future_df['predicted_demand'] = future_df.apply(
                lambda row: backup_demand_map.get((row[self.item_col], row['month_day']), 0), axis=1
            )
            future_df = future_df.drop(columns=['month_day'])
            predicted_demand = future_df.copy()
        else:
            predicted_demand = future_df.copy()
            predicted_demand['predicted_demand'] = np.maximum(0, predictions) # 예측값은 0 이상

        # 대리점 긴급 요청 가중치 적용
        if agency_request_multiplier:
            print(f"  [시뮬레이션] 대리점 긴급 요청 가중치 적용: {agency_request_multiplier}")
            for region, multiplier in agency_request_multiplier.items():
                mask = (predicted_demand[self.province_col] == region)
                predicted_demand.loc[mask, 'predicted_demand'] *= multiplier
        
        # 일별, 지역별, 품목별 수요 집계
        print(f"  [진단] Groupby 직전 컬럼: {predicted_demand.columns.tolist()}")
        daily_summary = predicted_demand.groupby(['date', self.province_col, self.item_col])['predicted_demand'].sum().reset_index()
        print("  [OK] 일별/지역별/품목별 수요 예측 완료.")
        return daily_summary

    def _initialize_inventory(self):
        """초기 재고 상태를 가상으로 생성합니다."""
        # 실제로는 DB나 ERP에서 현재 재고 데이터를 가져와야 합니다.
        # 테스트를 위해, 품목별 과거 평균 주간 수요의 50%를 초기 재고로 가정합니다.
        print("\n- 가상 초기 재고 생성 시작 (과거 평균의 50% 수준)...")
        avg_weekly_demand = self.historical_data.groupby(self.item_col)[self.qty_col].sum() / self.historical_data['date'].dt.to_period('W').nunique()
        
        # 안전 재고의 50% 수준으로 초기 재고 설정
        initial_stock = (avg_weekly_demand * SAFETY_STOCK_WEEKS * 0.5).to_dict()
        
        # 재고 신선도 추적을 위해 (수량, 생산일) 형태로 저장 (생산일 랜덤화)
        today = pd.to_datetime(datetime.now().date())
        inventory = {}
        for item, qty in initial_stock.items():
            if qty > 0:
                # 재고를 2개 배치로 나누고, 생산일을 최근 90일 내로 랜덤 설정
                qty1 = int(qty * 0.5)
                qty2 = int(qty) - qty1
                prod_date1 = today - timedelta(days=np.random.randint(0, 90))
                prod_date2 = today - timedelta(days=np.random.randint(0, 90))
                inventory[item] = []
                if qty1 > 0: inventory[item].append((qty1, prod_date1))
                if qty2 > 0: inventory[item].append((qty2, prod_date2))

        print(f"  [OK] 가상 초기 재고 생성 완료 (총 {len(inventory)}개 품목).")
        return inventory

    def run_simulation(self, demand_forecast: pd.DataFrame, agency_request_multiplier: dict = None):
        """재고 시뮬레이션을 실행하고 실무자용 리포트를 생성합니다."""
        print("\n- 재고 시뮬레이션 및 리포트 생성 시작...")
        
        # 시뮬레이션 기준일을 예측 시작일(2027-01-01)로 설정
        today = pd.to_datetime(f'{SIMULATION_YEAR}-01-01')
        print(f"  [정보] 시뮬레이션 기준일: {today.strftime('%Y-%m-%d')}")
        
        # --- 1. 기초 데이터 생성 ---
        report_data = []
        future_start_date = today
        future_end_date = today + timedelta(weeks=SAFETY_STOCK_WEEKS)
        future_demand_period = demand_forecast[demand_forecast['date'].between(future_start_date, future_end_date)]
        demand_by_item = future_demand_period.groupby(self.item_col)['predicted_demand'].sum()
        
        all_items = set(self.current_inventory.keys()) | set(demand_by_item.index)

        for item in sorted(list(all_items)):
            current_stock_qty = sum(q for q, d in self.current_inventory.get(item, []))
            forecasted_demand = demand_by_item.get(item, 0)
            
            stock_batches = self.current_inventory.get(item, [])
            avg_age = 0
            if current_stock_qty > 0:
                weighted_age_sum = sum((today - prod_date).days * qty for qty, prod_date in stock_batches)
                avg_age = weighted_age_sum / current_stock_qty
            
            is_competitive_item = self.historical_data[self.historical_data[self.item_col] == item][self.province_col].isin(self.competitive_regions).any()
            safety_stock_multiplier = 0.8 if is_competitive_item else 1.0
            
            recommended_stock = forecasted_demand * safety_stock_multiplier
            production_suggestion = max(0, recommended_stock - current_stock_qty)
            
            report_data.append({
                "품목": item, "예상 수요(4주)": forecasted_demand, "권장 재고": recommended_stock,
                "현재 재고": current_stock_qty, "생산 제언": production_suggestion, "리스크 지수(일)": avg_age,
            })
        report_df = pd.DataFrame(report_data)

        # --- 2. 최종 리포트 출력 ---
        print("\n" + "="*70)
        print(f" (주)성화 재고 최적화 리포트 (기준일: {today.strftime('%Y-%m-%d')})")
        print("="*70)

        # 2-1. 지역별 특이사항
        print("\n[ 지역별 특이사항 (대리점 요청 적용) ]")
        if agency_request_multiplier:
            for region, multiplier in agency_request_multiplier.items():
                last_year_start = future_start_date.replace(year=future_start_date.year - 1)
                last_year_end = future_end_date.replace(year=future_end_date.year - 1)
                
                demand_this_year = future_demand_period[future_demand_period[self.province_col] == region]['predicted_demand'].sum()
                demand_last_year = self.historical_data[
                    (self.historical_data['date'].between(last_year_start, last_year_end)) &
                    (self.historical_data[self.province_col] == region)
                ][self.qty_col].sum()
                
                if demand_last_year > 0:
                    change_pct = (demand_this_year - demand_last_year) / demand_last_year * 100
                    print(f"  - {region}: 향후 4주 수요 {demand_this_year:,.0f}개, 작년 동기 대비 {change_pct:+.1f}% (가중치 {multiplier} 적용됨)")
                else:
                    print(f"  - {region}: 향후 4주 수요 {demand_this_year:,.0f}개 (작년 동기 데이터 없음)")
        else:
            print("  - 대리점 요청 등 특별 가중치 미적용.")

        # 2-2. 봄 시즌 핵심 품목 관리
        print("\n[ 봄 시즌(3-5월) 핵심 품목 Top 10 생산 계획 ]")
        spring_demand = demand_forecast[demand_forecast['date'].dt.month.isin([3, 4, 5])]
        spring_demand_by_item = spring_demand.groupby(self.item_col)['predicted_demand'].sum()
        top_10_spring_items = spring_demand_by_item.nlargest(10).index
        
        top_10_df = report_df[report_df['품목'].isin(top_10_spring_items)].copy()
        top_10_df = top_10_df.sort_values("생산 제언", ascending=False)

        if not top_10_df.empty:
            for col in ["예상 수요(4주)", "권장 재고", "생산 제언"]:
                top_10_df[col] = top_10_df[col].map('{:,.0f}'.format)
            print(top_10_df[['품목', '예상 수요(4주)', '권장 재고', '생산 제언']].to_string(index=False))
        else:
            print("  - 봄 시즌 핵심 품목 정보를 계산할 수 없습니다.")
        
        # 2-3. 재고 과잉 경고
        print("\n[ !!재고 과잉 경고 (유기산 리스크)!! ]")
        overstock_df = report_df[(report_df['리스크 지수(일)'] > 90) & (report_df['현재 재고'] > 0)].copy()
        overstock_df = overstock_df.sort_values("리스크 지수(일)", ascending=False)

        if not overstock_df.empty:
            print("  - 아래 품목들은 평균 보유 기간이 90일을 초과하여 우선 출하 또는 재가공 검토가 필요합니다.")
            overstock_df["리스크 지수(일)"] = overstock_df["리스크 지수(일)"].map('{:.1f}일'.format)
            print(overstock_df[['품목', '현재 재고', '리스크 지수(일)']].to_string(index=False))
        else:
            print("  - 현재 재고 과잉으로 판단되는 품목이 없습니다.")

        print("\n[ 전체 품목 생산 제언 (필요량 많은 순) ]")
        production_needed_df = report_df[report_df['생산 제언'] > 0].copy()
        production_needed_df = production_needed_df.sort_values('생산 제언', ascending=False)
        
        if not production_needed_df.empty:
            for col in ["예상 수요(4주)", "권장 재고", "현재 재고", "생산 제언"]:
                production_needed_df[col] = production_needed_df[col].map('{:,.0f}'.format)
            production_needed_df["리스크 지수(일)"] = production_needed_df["리스크 지수(일)"].map('{:.1f}'.format)
            print(production_needed_df.to_string(index=False))
        else:
            print("  - 현재 모든 품목의 재고가 충분하여 추가 생산이 필요한 항목은 없습니다.")
            
        print("="*70)


if __name__ == "__main__":
    optimizer = InventoryOptimizerV4()
    
    # 시뮬레이션 실행 (대리점 긴급 요청 가중치 예시)
    # 실제 운영 시에는 이 값을 GUI나 설정 파일에서 받아올 수 있습니다.
    emergency_requests = {
        '충남': 1.2, # 충남 20% 추가 요청
        '전남': 1.1  # 전남 10% 추가 요청
    }
    
    future_demand = optimizer.predict_demand(agency_request_multiplier=emergency_requests)
    
    # 시뮬레이션 결과 기반 리포트 생성
    optimizer.run_simulation(future_demand, agency_request_multiplier=emergency_requests)
