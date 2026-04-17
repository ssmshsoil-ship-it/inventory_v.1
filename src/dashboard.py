"""
상토 수요 예측 및 배차 관리 대시보드 (Streamlit)

실행: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import json

# 페이지 설정
st.set_page_config(
    page_title="성화 상토 수요 예측 시스템",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_forecast_data():
    """예측 데이터 로드"""
    forecast_path = Path('output/forecast_2027.xlsx')
    if not forecast_path.exists():
        return None, None
    
    weekly = pd.read_excel(forecast_path, sheet_name='2027년 주간 예측')
    monthly = pd.read_excel(forecast_path, sheet_name='월별 요약')
    
    return weekly, monthly


def load_dispatch_data():
    """배차 계획 데이터 로드"""
    dispatch_path = Path('output/dispatch_plan.xlsx')
    if not dispatch_path.exists():
        return None, None
    
    dispatch = pd.read_excel(dispatch_path, sheet_name='배차 계획')
    monthly = pd.read_excel(dispatch_path, sheet_name='월별 요약')
    
    return dispatch, monthly


def load_notifications():
    """기사 알림 데이터 로드"""
    notification_path = Path('output/driver_notifications.json')
    if not notification_path.exists():
        return []
    
    with open(notification_path, 'r', encoding='utf-8') as f:
        notifications = json.load(f)
    
    return notifications


def main():
    """메인 대시보드"""
    
    # 헤더
    st.markdown('<div class="main-header">🚚 성화 상토 수요 예측 및 배차 관리 시스템</div>', 
                unsafe_allow_html=True)
    
    # 사이드바
    st.sidebar.title("📊 메뉴")
    menu = st.sidebar.radio(
        "선택하세요:",
        ["📈 수요 예측", "🚛 배차 계획", "📢 기사 알림", "💰 비용 분석", "⚙️ 설정"]
    )
    
    # 데이터 로드
    weekly_forecast, monthly_forecast = load_forecast_data()
    dispatch_plan, dispatch_monthly = load_dispatch_data()
    notifications = load_notifications()
    
    # 메뉴별 화면
    if menu == "📈 수요 예측":
        show_forecast_page(weekly_forecast, monthly_forecast)
    
    elif menu == "🚛 배차 계획":
        show_dispatch_page(dispatch_plan, dispatch_monthly)
    
    elif menu == "📢 기사 알림":
        show_notifications_page(notifications)
    
    elif menu == "💰 비용 분석":
        show_cost_analysis_page(dispatch_plan, dispatch_monthly)
    
    elif menu == "⚙️ 설정":
        show_settings_page()


def show_forecast_page(weekly_forecast, monthly_forecast):
    """수요 예측 페이지"""
    st.header("📈 2027년 상토 수요 예측")
    
    if weekly_forecast is None:
        st.warning("⚠️ 예측 데이터가 없습니다. 먼저 `python src/predict_2027.py`를 실행하세요.")
        return
    
    # 주요 지표
    col1, col2, col3, col4 = st.columns(4)
    
    total_shipment = weekly_forecast['예상 출고량(포)'].sum()
    avg_weekly = weekly_forecast['예상 출고량(포)'].mean()
    max_week = weekly_forecast.loc[weekly_forecast['예상 출고량(포)'].idxmax(), '주차']
    max_shipment = weekly_forecast['예상 출고량(포)'].max()
    
    with col1:
        st.metric("연간 총 예상 출고량", f"{total_shipment:,.0f} 포")
    with col2:
        st.metric("주간 평균 출고량", f"{avg_weekly:,.0f} 포")
    with col3:
        st.metric("최대 출고 주차", max_week)
    with col4:
        st.metric("최대 출고량", f"{max_shipment:,.0f} 포")
    
    # 주간 출고량 그래프
    st.subheader("주간 출고량 예측")
    fig = px.bar(
        weekly_forecast,
        x='주차',
        y='예상 출고량(포)',
        title='2027년 주간 상토 출고량 예측',
        color='파종기 여부',
        color_discrete_map={1: 'green', 0: 'steelblue'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # 월별 요약
    st.subheader("월별 출고량 요약")
    if monthly_forecast is not None:
        fig2 = px.bar(
            monthly_forecast,
            x='월',
            y='월간 총 출고량(포)',
            title='2027년 월별 상토 출고량',
            text='월간 총 출고량(포)'
        )
        fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # 데이터 테이블
    with st.expander("📋 상세 데이터 보기"):
        st.dataframe(weekly_forecast, use_container_width=True)


def show_dispatch_page(dispatch_plan, dispatch_monthly):
    """배차 계획 페이지"""
    st.header("🚛 배차 계획")
    
    if dispatch_plan is None:
        st.warning("⚠️ 배차 계획 데이터가 없습니다. 먼저 `python src/dispatch.py`를 실행하세요.")
        return
    
    # 주요 지표
    col1, col2, col3, col4 = st.columns(4)
    
    total_trucks = dispatch_plan['총 트럭 대수'].sum()
    avg_trucks = dispatch_plan['총 트럭 대수'].mean()
    total_cost = dispatch_plan['예상 배송비(원)'].sum()
    max_trucks_week = dispatch_plan.loc[dispatch_plan['총 트럭 대수'].idxmax(), '주차']
    
    with col1:
        st.metric("연간 총 트럭 대수", f"{total_trucks:,.0f} 대")
    with col2:
        st.metric("주간 평균 트럭", f"{avg_trucks:.1f} 대")
    with col3:
        st.metric("연간 총 배송비", f"{total_cost:,.0f} 원")
    with col4:
        st.metric("최대 트럭 필요 주차", max_trucks_week)
    
    # 트럭 대수 그래프
    st.subheader("주간 트럭 배차 계획")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='25톤', x=dispatch_plan['주차'], y=dispatch_plan['25톤 트럭']))
    fig.add_trace(go.Bar(name='11톤', x=dispatch_plan['주차'], y=dispatch_plan['11톤 트럭']))
    fig.add_trace(go.Bar(name='5톤', x=dispatch_plan['주차'], y=dispatch_plan['5톤 트럭']))
    
    fig.update_layout(
        barmode='stack',
        title='주간 트럭 배차 현황 (톤수별)',
        xaxis_title='주차',
        yaxis_title='트럭 대수',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 월별 배차 요약
    if dispatch_monthly is not None:
        st.subheader("월별 배차 요약")
        col1, col2 = st.columns(2)
        
        with col1:
            fig2 = px.bar(
                dispatch_monthly,
                x='월',
                y='월간 총 트럭 대수',
                title='월별 총 트럭 대수',
                text='월간 총 트럭 대수'
            )
            fig2.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = px.bar(
                dispatch_monthly,
                x='월',
                y='월간 총 배송비(원)',
                title='월별 총 배송비',
                text='월간 총 배송비(원)'
            )
            fig3.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
    
    # 데이터 테이블
    with st.expander("📋 상세 배차 계획 보기"):
        st.dataframe(dispatch_plan, use_container_width=True)


def show_notifications_page(notifications):
    """기사 알림 페이지"""
    st.header("📢 기사 사전 알림")
    
    if not notifications:
        st.warning("⚠️ 알림 데이터가 없습니다.")
        return
    
    # 필터
    col1, col2 = st.columns([1, 3])
    with col1:
        priority_filter = st.selectbox(
            "우선순위 필터",
            ["전체", "high", "normal"]
        )
    
    # 필터링
    filtered_notifications = notifications
    if priority_filter != "전체":
        filtered_notifications = [n for n in notifications if n['priority'] == priority_filter]
    
    st.info(f"📋 총 {len(filtered_notifications)}개의 알림")
    
    # 알림 카드 표시
    for notif in filtered_notifications:
        priority_color = "🔴" if notif['priority'] == 'high' else "🟢"
        
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                st.markdown(f"### {priority_color} {notif['delivery_week']}")
                st.write(f"**알림일**: {notif['notification_date']}")
                st.write(f"**배송일**: {notif['delivery_date']}")
            
            with col2:
                st.write(f"**총 출고량**: {notif['total_shipment']:,} 포")
                st.write(f"**필요 트럭**: {notif['total_trucks']} 대")
                st.write(f"**메시지**: {notif['message']}")
            
            with col3:
                st.write("**트럭 배분**")
                for truck_type, count in notif['truck_allocation'].items():
                    if count > 0:
                        st.write(f"- {truck_type}: {count}대")
            
            st.divider()


def show_cost_analysis_page(dispatch_plan, dispatch_monthly):
    """비용 분석 페이지"""
    st.header("💰 배송 비용 분석")
    
    if dispatch_plan is None:
        st.warning("⚠️ 배차 계획 데이터가 없습니다.")
        return
    
    # 총 비용 분석
    total_cost = dispatch_plan['예상 배송비(원)'].sum()
    avg_cost_per_week = dispatch_plan['예상 배송비(원)'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("연간 총 배송비", f"{total_cost:,.0f} 원")
    with col2:
        st.metric("주간 평균 배송비", f"{avg_cost_per_week:,.0f} 원")
    
    # 주간 비용 추이
    st.subheader("주간 배송비 추이")
    fig = px.line(
        dispatch_plan,
        x='주차',
        y='예상 배송비(원)',
        title='주간 배송비 변화',
        markers=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 월별 비용
    if dispatch_monthly is not None:
        st.subheader("월별 배송비")
        fig2 = px.pie(
            dispatch_monthly,
            values='월간 총 배송비(원)',
            names='월',
            title='월별 배송비 비중'
        )
        st.plotly_chart(fig2, use_container_width=True)


def show_settings_page():
    """설정 페이지"""
    st.header("⚙️ 시스템 설정")
    
    st.subheader("트럭 사양 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.number_input("5톤 트럭 적재량 (포)", value=500, step=50)
        st.number_input("5톤 트럭 km당 비용 (원)", value=1500, step=100)
    
    with col2:
        st.number_input("11톤 트럭 적재량 (포)", value=1100, step=50)
        st.number_input("11톤 트럭 km당 비용 (원)", value=2500, step=100)
    
    with col3:
        st.number_input("25톤 트럭 적재량 (포)", value=2500, step=100)
        st.number_input("25톤 트럭 km당 비용 (원)", value=4000, step=100)
    
    st.subheader("알림 설정")
    st.number_input("사전 알림 일수", value=3, min_value=1, max_value=7)
    
    st.subheader("데이터 새로고침")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 예측 모델 재실행"):
            st.info("예측 모델을 재실행합니다...")
            # 실제로는 subprocess로 실행
    
    with col2:
        if st.button("🔄 배차 계획 재생성"):
            st.info("배차 계획을 재생성합니다...")


if __name__ == "__main__":
    main()
