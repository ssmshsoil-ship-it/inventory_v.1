"""
배차 자동화 모듈

예측된 출고량을 기반으로 필요한 트럭 대수를 계산하고
기사에게 사전 알림을 보내는 시스템
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json


class DispatchOptimizer:
    """배차 최적화 클래스"""
    
    # 트럭 사양 정보
    TRUCK_SPECS = {
        '5톤': {'capacity': 500, 'cost_per_km': 1500},
        '11톤': {'capacity': 1100, 'cost_per_km': 2500},
        '25톤': {'capacity': 2500, 'cost_per_km': 4000},
    }
    
    # 주요 배송지 정보 (예시)
    DELIVERY_LOCATIONS = {
        '경기': {'distance_km': 50, 'priority': 1},
        '충남': {'distance_km': 120, 'priority': 2},
        '충북': {'distance_km': 100, 'priority': 2},
        '전북': {'distance_km': 180, 'priority': 3},
        '강원': {'distance_km': 150, 'priority': 3},
    }
    
    def __init__(self):
        """초기화"""
        self.dispatch_plan = None
        
    def calculate_truck_requirements(self, weekly_forecast_df):
        """
        주간 예측 출고량을 기반으로 필요한 트럭 대수 계산
        
        Args:
            weekly_forecast_df: 주간 예측 DataFrame (weekly_shipment 컬럼 포함)
        
        Returns:
            배차 계획 DataFrame
        """
        dispatch_plan = []
        
        for _, row in weekly_forecast_df.iterrows():
            week_label = row['week_label']
            total_shipment = row['weekly_shipment']
            week_start = row['week_start']
            
            # 트럭 대수 계산 (25톤 우선, 나머지는 11톤, 5톤 순)
            remaining = total_shipment
            trucks_25t = int(remaining / self.TRUCK_SPECS['25톤']['capacity'])
            remaining -= trucks_25t * self.TRUCK_SPECS['25톤']['capacity']
            
            trucks_11t = int(remaining / self.TRUCK_SPECS['11톤']['capacity'])
            remaining -= trucks_11t * self.TRUCK_SPECS['11톤']['capacity']
            
            trucks_5t = int(np.ceil(remaining / self.TRUCK_SPECS['5톤']['capacity']))
            
            total_trucks = trucks_25t + trucks_11t + trucks_5t
            
            # 배차 계획 추가
            dispatch_plan.append({
                'week_label': week_label,
                'week_start': week_start,
                'total_shipment': int(total_shipment),
                'trucks_25t': trucks_25t,
                'trucks_11t': trucks_11t,
                'trucks_5t': trucks_5t,
                'total_trucks': total_trucks,
                'notification_date': week_start - timedelta(days=3),  # 3일 전 알림
            })
        
        self.dispatch_plan = pd.DataFrame(dispatch_plan)
        return self.dispatch_plan
    
    def calculate_dispatch_cost(self, dispatch_plan_df, region='경기'):
        """
        배차 비용 계산
        
        Args:
            dispatch_plan_df: 배차 계획 DataFrame
            region: 배송 지역
        
        Returns:
            비용이 추가된 DataFrame
        """
        if region not in self.DELIVERY_LOCATIONS:
            region = '경기'  # 기본값
        
        distance = self.DELIVERY_LOCATIONS[region]['distance_km']
        
        dispatch_plan_df['cost_25t'] = (
            dispatch_plan_df['trucks_25t'] * 
            self.TRUCK_SPECS['25톤']['cost_per_km'] * 
            distance * 2  # 왕복
        )
        
        dispatch_plan_df['cost_11t'] = (
            dispatch_plan_df['trucks_11t'] * 
            self.TRUCK_SPECS['11톤']['cost_per_km'] * 
            distance * 2
        )
        
        dispatch_plan_df['cost_5t'] = (
            dispatch_plan_df['trucks_5t'] * 
            self.TRUCK_SPECS['5톤']['cost_per_km'] * 
            distance * 2
        )
        
        dispatch_plan_df['total_cost'] = (
            dispatch_plan_df['cost_25t'] + 
            dispatch_plan_df['cost_11t'] + 
            dispatch_plan_df['cost_5t']
        )
        
        return dispatch_plan_df
    
    def generate_driver_notifications(self, dispatch_plan_df, output_path='output/driver_notifications.json'):
        """
        기사 알림 데이터 생성
        
        Args:
            dispatch_plan_df: 배차 계획 DataFrame
            output_path: 알림 데이터 저장 경로
        
        Returns:
            알림 데이터 리스트
        """
        notifications = []
        
        for _, row in dispatch_plan_df.iterrows():
            notification = {
                'notification_date': row['notification_date'].strftime('%Y-%m-%d'),
                'delivery_week': row['week_label'],
                'delivery_date': row['week_start'].strftime('%Y-%m-%d'),
                'total_shipment': int(row['total_shipment']),
                'truck_allocation': {
                    '25톤': int(row['trucks_25t']),
                    '11톤': int(row['trucks_11t']),
                    '5톤': int(row['trucks_5t']),
                },
                'total_trucks': int(row['total_trucks']),
                'message': f"{row['week_label']} 배송 예정: 총 {int(row['total_shipment']):,}포 ({int(row['total_trucks'])}대 필요)",
                'priority': 'high' if row['total_trucks'] > 10 else 'normal',
            }
            notifications.append(notification)
        
        # JSON 파일로 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notifications, f, ensure_ascii=False, indent=2)
        
        print(f"\n기사 알림 데이터 저장 완료: {output_path}")
        return notifications
    
    def optimize_dispatch_schedule(self, dispatch_plan_df):
        """
        배차 일정 최적화 (피크 시즌 분산)
        
        Args:
            dispatch_plan_df: 배차 계획 DataFrame
        
        Returns:
            최적화된 배차 계획 DataFrame
        """
        # 파종기(2~5월) 피크 시즌 감지
        dispatch_plan_df['month'] = pd.to_datetime(dispatch_plan_df['week_start']).dt.month
        dispatch_plan_df['is_peak_season'] = dispatch_plan_df['month'].isin([2, 3, 4, 5])
        
        # 피크 시즌에 트럭 대수가 많으면 경고
        dispatch_plan_df['needs_optimization'] = (
            dispatch_plan_df['is_peak_season'] & 
            (dispatch_plan_df['total_trucks'] > 15)
        )
        
        # 최적화 제안
        optimization_suggestions = []
        for _, row in dispatch_plan_df[dispatch_plan_df['needs_optimization']].iterrows():
            suggestion = {
                'week': row['week_label'],
                'current_trucks': int(row['total_trucks']),
                'suggestion': '사전 출고 또는 분산 배송 검토 필요',
                'alternative_dates': [
                    (row['week_start'] - timedelta(days=7)).strftime('%Y-%m-%d'),
                    (row['week_start'] + timedelta(days=7)).strftime('%Y-%m-%d'),
                ]
            }
            optimization_suggestions.append(suggestion)
        
        if optimization_suggestions:
            print(f"\n⚠️  최적화 필요 주차: {len(optimization_suggestions)}개")
            for sug in optimization_suggestions:
                print(f"  - {sug['week']}: {sug['current_trucks']}대 → {sug['suggestion']}")
        
        return dispatch_plan_df, optimization_suggestions
    
    def export_to_excel(self, dispatch_plan_df, output_path='output/dispatch_plan.xlsx'):
        """
        배차 계획을 엑셀로 저장
        
        Args:
            dispatch_plan_df: 배차 계획 DataFrame
            output_path: 출력 파일 경로
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 출력용 DataFrame 준비
        output_df = dispatch_plan_df[[
            'week_label', 'week_start', 'total_shipment',
            'trucks_25t', 'trucks_11t', 'trucks_5t', 'total_trucks',
            'total_cost', 'notification_date'
        ]].copy()
        
        # 컬럼명 한글화
        output_df.columns = [
            '주차', '배송 시작일', '총 출고량(포)',
            '25톤 트럭', '11톤 트럭', '5톤 트럭', '총 트럭 대수',
            '예상 배송비(원)', '기사 알림일'
        ]
        
        # 엑셀 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='배차 계획', index=False)
            
            # 월별 요약 시트
            monthly_summary = dispatch_plan_df.copy()
            monthly_summary['month'] = pd.to_datetime(monthly_summary['week_start']).dt.month
            monthly = monthly_summary.groupby('month').agg({
                'total_shipment': 'sum',
                'total_trucks': 'sum',
                'total_cost': 'sum'
            }).reset_index()
            monthly.columns = ['월', '월간 총 출고량(포)', '월간 총 트럭 대수', '월간 총 배송비(원)']
            monthly.to_excel(writer, sheet_name='월별 요약', index=False)
        
        print(f"배차 계획 엑셀 저장 완료: {output_path}")
    
    def generate_summary_report(self, dispatch_plan_df):
        """
        배차 계획 요약 보고서 생성
        
        Args:
            dispatch_plan_df: 배차 계획 DataFrame
        
        Returns:
            요약 통계 딕셔너리
        """
        summary = {
            'total_weeks': len(dispatch_plan_df),
            'total_shipment': int(dispatch_plan_df['total_shipment'].sum()),
            'total_trucks': int(dispatch_plan_df['total_trucks'].sum()),
            'total_cost': int(dispatch_plan_df['total_cost'].sum()),
            'avg_trucks_per_week': dispatch_plan_df['total_trucks'].mean(),
            'max_trucks_week': dispatch_plan_df.loc[dispatch_plan_df['total_trucks'].idxmax(), 'week_label'],
            'max_trucks_count': int(dispatch_plan_df['total_trucks'].max()),
        }
        
        print("\n" + "=" * 60)
        print("배차 계획 요약")
        print("=" * 60)
        print(f"총 주차 수: {summary['total_weeks']}주")
        print(f"연간 총 출고량: {summary['total_shipment']:,}포")
        print(f"연간 총 트럭 대수: {summary['total_trucks']:,}대")
        print(f"연간 총 배송비: {summary['total_cost']:,}원")
        print(f"주간 평균 트럭 대수: {summary['avg_trucks_per_week']:.1f}대")
        print(f"최대 트럭 필요 주차: {summary['max_trucks_week']} ({summary['max_trucks_count']}대)")
        print("=" * 60)
        
        return summary


def run_dispatch_optimization(forecast_file='output/forecast_2027.xlsx'):
    """
    배차 최적화 전체 파이프라인 실행
    
    Args:
        forecast_file: 예측 결과 엑셀 파일 경로
    """
    print("=" * 60)
    print("배차 자동화 시스템 시작")
    print("=" * 60)
    
    # 1. 예측 데이터 로드
    if not Path(forecast_file).exists():
        print(f"오류: 예측 파일을 찾을 수 없습니다: {forecast_file}")
        print("먼저 python src/predict_2027.py를 실행하세요.")
        return
    
    weekly_forecast = pd.read_excel(forecast_file, sheet_name='2027년 주간 예측')
    weekly_forecast.columns = ['week_label', 'week_start', 'week_end', 
                                'weekly_shipment', 'avg_temperature',
                                'is_planting_season', 'holiday_effect']
    weekly_forecast['week_start'] = pd.to_datetime(weekly_forecast['week_start'])
    
    print(f"예측 데이터 로드 완료: {len(weekly_forecast)}주")
    
    # 2. 배차 최적화 실행
    optimizer = DispatchOptimizer()
    
    # 트럭 대수 계산
    dispatch_plan = optimizer.calculate_truck_requirements(weekly_forecast)
    
    # 배송 비용 계산
    dispatch_plan = optimizer.calculate_dispatch_cost(dispatch_plan, region='경기')
    
    # 배차 일정 최적화
    dispatch_plan, optimization_suggestions = optimizer.optimize_dispatch_schedule(dispatch_plan)
    
    # 3. 기사 알림 생성
    notifications = optimizer.generate_driver_notifications(dispatch_plan)
    
    # 4. 엑셀 저장
    optimizer.export_to_excel(dispatch_plan)
    
    # 5. 요약 보고서
    summary = optimizer.generate_summary_report(dispatch_plan)
    
    print("\n" + "=" * 60)
    print("배차 자동화 완료!")
    print("=" * 60)
    
    return dispatch_plan, notifications, summary


if __name__ == "__main__":
    # 배차 최적화 실행
    dispatch_plan, notifications, summary = run_dispatch_optimization()
