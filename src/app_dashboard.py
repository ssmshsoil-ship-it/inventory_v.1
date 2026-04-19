# -*- coding: utf-8 -*-
"""
(주)성와의 재고 관리 최적화를 위한 Streamlit 기반 인터랙티브 대시보드

v3 예측 모델과 재고 관리 로직을 연동하여,
슬라이더를 통해 외부 변수를 조절하면 2027년 예측치가 실시간으로 업데이트됩니다.

실행: streamlit run src/app_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from pathlib import Path
from datetime import datetime, timedelta

# --- 1. 기본 설정 및 페이지 구성 ---
st.set_page_config(layout="wide", page_title="(주)성화 재고관리 AI 대시보드")

# --- 2. 전역 설정값 ---
MODEL_PATH = Path("models/sh_delivery_v3.cbm")
HISTORICAL_DATA_PATH = Path("data/processed/final_training_data.csv")
SIMULATION_YEAR = 2027
SHELF_LIFE_DAYS = 180
REPROCESSING_COST_PER_UNIT = 1500
SAFETY_STOCK_WEEKS = 4

# --- 3. 핵심 기능 함수 (Streamlit 캐시 적용) ---

@st.cache_resource
def load_model():
    """v3 모델을 로드하고 캐시합니다."""
    if not MODEL_PATH.exists():
        st.error(f"모델 파일({MODEL_PATH})을 찾을 수 없습니다. `train_model.py`를 실행했는지 확인하세요.")
        st.stop()
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_historical_data():
    """과거 데이터를 로드하고, 주요 컬럼명을 동적으로 확인하여 캐시합니다."""
    if not HISTORICAL_DATA_PATH.exists():
        st.error(f"과거 데이터({HISTORICAL_DATA_PATH})를 찾을 수 없습니다. `integrate_features.py`를 실행했는지 확인하세요.")
        st.stop()
    
    df = pd.read_csv(HISTORICAL_DATA_PATH, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    item_col = next((col for col in ['품목', '품명'] if col in df.columns), None)
    qty_col = next((col for col in ['출고수량', '수량'] if col in df.columns), None)
    province_col = next((col for col in ['province', '지역'] if col in df.columns), None)

    if not all([item_col, qty_col, province_col]):
        st.error("데이터에서 '품목', '출고수량', 'province' 컬럼 중 하나 이상을 찾을 수 없습니다.")
        st.stop()
        
    return df, item_col, qty_col, province_col

@st.cache_data
def analyze_competition(_historical_data, province_col, qty_col):
    """경쟁 심화 지역을 분석하고 캐시합니다."""
    weekly_sales = _historical_data.groupby([province_col, pd.Grouper(key='date', freq='W-MON')])[qty_col].sum().reset_index()
    volatility = weekly_sales.groupby(province_col)[qty_col].std().sort_values(ascending=False)
    threshold = volatility.quantile(0.75)
    return volatility[volatility >= threshold].index.tolist()

@st.cache_data
def initialize_inventory(_historical_data, item_col, qty_col):
    """가상의 초기 재고를 생성하고 캐시합니다."""
    avg_weekly_demand = _historical_data.groupby(item_col)[qty_col].sum() / _historical_data['date'].dt.to_period('W').nunique()
    initial_stock = (avg_weekly_demand * SAFETY_STOCK_WEEKS * 0.5).to_dict()
    
    today = pd.to_datetime(datetime.now().date())
    inventory = {}
    for item, qty in initial_stock.items():
        if qty > 0:
            qty1 = int(qty * 0.5); qty2 = int(qty) - qty1
            prod_date1 = today - timedelta(days=np.random.randint(0, 90))
            prod_date2 = today - timedelta(days=np.random.randint(0, 90))
            inventory[item] = []
            if qty1 > 0: inventory[item].append((qty1, prod_date1))
            if qty2 > 0: inventory[item].append((qty2, prod_date2))
    return inventory

@st.cache_data
def prepare_future_features(_historical_data, _model, item_col, province_col):
    """2027년 예측용 피처 데이터프레임을 생성하고 캐시합니다."""
    dates_2027 = pd.date_range(start=f'{SIMULATION_YEAR}-01-01', end=f'{SIMULATION_YEAR}-12-31', freq='D')
    
    # 예측 단위(품목/지역/관측소) 조합 추출 (수량 뻥튀기 방지)
    base_cols = [item_col, province_col, 'stn_id']
    combinations = _historical_data[base_cols].drop_duplicates().dropna()
    
    future_df = pd.DataFrame(dates_2027, columns=['date'])
    future_df['_key'] = 1; combinations['_key'] = 1
    future_df = pd.merge(future_df, combinations, on='_key').drop('_key', axis=1)

    # 모델이 '고객명' 피처를 사용하므로, 대표값으로 설정
    future_df['고객명'] = 'dummy_customer'

    # 2026년 기상 데이터를 2027년에 매핑 (중복 조인 방지)
    weather_cols = ['avg_temp', 'min_temp', 'max_temp', 'precip']
    weather_source = _historical_data[['date', 'stn_id'] + weather_cols].drop_duplicates()
    last_year_weather = weather_source[weather_source['date'].dt.year == (SIMULATION_YEAR - 1)].copy()

    last_year_weather['month_day'] = last_year_weather['date'].dt.strftime('%m-%d')
    future_df['month_day'] = future_df['date'].dt.strftime('%m-%d')
    
    weather_to_map = last_year_weather[['stn_id', 'month_day'] + weather_cols].drop_duplicates(subset=['stn_id', 'month_day'])
    future_df = pd.merge(future_df, weather_to_map, on=['stn_id', 'month_day'], how='left')
    future_df[weather_cols] = future_df.groupby('stn_id')[weather_cols].transform(lambda x: x.ffill().bfill())
    future_df = future_df.drop(columns=['month_day'])

    features = _model.feature_names_
    future_df['year'] = future_df['date'].dt.year
    future_df['month'] = future_df['date'].dt.month
    future_df['week'] = future_df['date'].dt.isocalendar().week
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    future_df = future_df.sort_values(by=['stn_id', 'date']).reset_index(drop=True)
    future_df['temp_change_weekly'] = future_df.groupby('stn_id')['avg_temp'].diff(7).fillna(0)
    future_df['precip_sum_3d'] = future_df.groupby('stn_id')['precip'].rolling(window=3).sum().reset_index(0,drop=True).fillna(0)
    future_df['is_peak_season'] = future_df['date'].dt.month.isin([3, 4]).astype(int)

    for col in features:
        if col not in future_df.columns:
            if _historical_data[col].dtype == 'object': future_df[col] = 'missing'
            else: future_df[col] = 0
    return future_df

# --- 4. 메인 대시보드 함수 ---
def main():
    st.title("📈 (주)성화 재고 관리 AI 대시보드 (2027년 예측)")
    
    with st.spinner("AI 모델 및 데이터를 로드하는 중입니다..."):
        model = load_model()
        historical_data, item_col, qty_col, province_col = load_historical_data()
        competitive_regions = analyze_competition(historical_data, province_col, qty_col)
        current_inventory = initialize_inventory(historical_data, item_col, qty_col)
        future_df_base = prepare_future_features(historical_data, model, item_col, province_col)

    st.sidebar.header("⚙️ 시뮬레이션 조건 설정")
    st.sidebar.write("대리점의 긴급 요청 등 외부 변수를 조절하여 수요 예측 및 재고 변화를 시뮬레이션합니다.")
    provinces = sorted(historical_data[province_col].unique())
    agency_request_multiplier = {}
    with st.sidebar.expander("지역별 수요 가중치 조절", expanded=True):
        for p in provinces:
            default_value = 1.2 if p == '충남' else 1.1 if p == '전남' else 1.0
            agency_request_multiplier[p] = st.slider(f"{p} 수요 가중치", 0.5, 2.5, default_value, 0.1)

    with st.spinner("실시간으로 2027년 수요를 다시 예측하고 리포트를 생성하는 중입니다..."):
        predictions = model.predict(future_df_base[model.feature_names_])
        predicted_demand = future_df_base.copy()
        predicted_demand['predicted_demand'] = np.maximum(0, predictions)

        # 예측값 상한선 설정 (단위 검증)
        max_daily_qty = historical_data.groupby(['date', item_col])[qty_col].sum().max()
        clipping_upper_bound = max_daily_qty * 2
        predicted_demand['predicted_demand'] = predicted_demand['predicted_demand'].clip(upper=clipping_upper_bound)

        for region, multiplier in agency_request_multiplier.items():
            if multiplier != 1.0:
                mask = (predicted_demand[province_col] == region)
                predicted_demand.loc[mask, 'predicted_demand'] *= multiplier
        demand_forecast = predicted_demand.groupby(['date', province_col, item_col])['predicted_demand'].sum().reset_index()

        # --- 리포트 생성 로직 ---
        today = pd.to_datetime(f'{SIMULATION_YEAR}-01-01')
        future_start_date = today
        future_end_date = today + timedelta(weeks=SAFETY_STOCK_WEEKS)
        future_demand_period = demand_forecast[demand_forecast['date'].between(future_start_date, future_end_date)]
        demand_by_item = future_demand_period.groupby(item_col)['predicted_demand'].sum()
        all_items = set(current_inventory.keys()) | set(demand_by_item.index)
        
        report_data = []
        for item in sorted(list(all_items)):
            current_stock_qty = sum(q for q, d in current_inventory.get(item, []))
            forecasted_demand = demand_by_item.get(item, 0)
            stock_batches = current_inventory.get(item, [])
            avg_age = sum((today - d).days * q for q, d in stock_batches) / current_stock_qty if current_stock_qty > 0 else 0
            is_competitive_item = historical_data[historical_data[item_col] == item][province_col].isin(competitive_regions).any()
            safety_stock_multiplier = 0.8 if is_competitive_item else 1.0
            recommended_stock = forecasted_demand * safety_stock_multiplier
            production_suggestion = max(0, recommended_stock - current_stock_qty)
            report_data.append({
                "품목": item, "예상 수요(4주)": forecasted_demand, "권장 재고": recommended_stock,
                "현재 재고": current_stock_qty, "생산 제언": production_suggestion, "리스크 지수(일)": avg_age,
            })
        report_df = pd.DataFrame(report_data)

        # 수치형 컬럼들의 타입을 강제로 통일하여 포맷팅 오류 방지
        numeric_cols = ["예상 수요(4주)", "권장 재고", "현재 재고", "생산 제언", "리스크 지수(일)"]
        for col in numeric_cols:
            if col in report_df.columns:
                report_df[col] = pd.to_numeric(report_df[col], errors='coerce').fillna(0)

    # --- 대시보드 출력 ---
    st.header(f"📇 재고 최적화 리포트 (기준일: {today.strftime('%Y-%m-%d')})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 지역별 특이사항")
        if agency_request_multiplier:
            for region, multiplier in agency_request_multiplier.items():
                if multiplier != 1.0:
                    last_year_start = future_start_date.replace(year=future_start_date.year - 1)
                    last_year_end = future_end_date.replace(year=future_end_date.year - 1)
                    demand_this_year = future_demand_period[future_demand_period[province_col] == region]['predicted_demand'].sum()
                    
                    if demand_this_year > 1_000_000:
                        st.warning(f"{region}의 4주 예측 수요({demand_this_year:,.0f}개)가 100만 개를 초과하여 비정상적으로 보입니다.")
                    
                    demand_last_year = historical_data[(historical_data['date'].between(last_year_start, last_year_end)) & (historical_data[province_col] == region)][qty_col].sum()
                    
                    delta_text = "-"
                    if pd.notna(demand_this_year) and pd.notna(demand_last_year) and demand_last_year >= 100:
                        change_pct = (demand_this_year - demand_last_year) / demand_last_year * 100
                        delta_text = f"{change_pct:+.1f} %"
                        
                    st.metric(label=f"{region} 수요 변화 (작년 동기 대비)", value=f"{demand_this_year:,.0f} 개", delta=delta_text)
    
    with col2:
        st.subheader("⚠️ 재고 과잉 경고 (유기산 리스크)")
        overstock_df = report_df[(report_df['리스크 지수(일)'] > 90) & (report_df['현재 재고'] > 0)].copy()
        if not overstock_df.empty:
            overstock_df = overstock_df.sort_values("리스크 지수(일)", ascending=False)
            st.dataframe(overstock_df[['품목', '현재 재고', '리스크 지수(일)']].style.format({'현재 재고': '{:,.0f}', '리스크 지수(일)': '{:.1f}일'}, na_rep='-'), use_container_width=True)
        else:
            st.info("현재 재고 과잉으로 판단되는 품목이 없습니다.")

    st.subheader("📈 봄 시즌(3-5월) 핵심 품목 Top 10 생산 계획")
    spring_demand = demand_forecast[demand_forecast['date'].dt.month.isin([3, 4, 5])]
    spring_demand_by_item = spring_demand.groupby(item_col)['predicted_demand'].sum()
    top_10_spring_items = spring_demand_by_item.nlargest(10).index
    top_10_df = report_df[report_df['품목'].isin(top_10_spring_items)].sort_values("생산 제언", ascending=False)
        
    st.dataframe(top_10_df[['품목', '예상 수요(4주)', '권장 재고', '생산 제언']].style.format(formatter='{:,.0f}', na_rep='-'), use_container_width=True)

    with st.expander("📄 전체 품목 생산 제언 보기 (필요량 많은 순)"):
        production_needed_df = report_df[report_df['생산 제언'] > 0].sort_values('생산 제언', ascending=False)
        if not production_needed_df.empty:
            st.dataframe(production_needed_df.style.format({
                "예상 수요(4주)": '{:,.0f}', "권장 재고": '{:,.0f}', "현재 재고": '{:,.0f}', 
                "생산 제언": '{:,.0f}', "리스크 지수(일)": '{:.1f}'
            }, na_rep='-'), use_container_width=True)
        else:
            st.success("현재 모든 품목의 재고가 충분하여 추가 생산이 필요한 항목은 없습니다.")

# --- 5. 앱 실행 ---
if __name__ == "__main__":
    main()
