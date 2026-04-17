"""
2027년 상토 수요 예측 스크립트

학습된 모델을 사용하여 2027년의 상토 출고량을 예측하고 엑셀 파일과 그래프로 저장합니다.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from catboost import CatBoostRegressor

# 한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from preprocess import SoilDataPreprocessor


class SoilDemandPredictor:
    """2027년 상토 수요 예측 클래스"""
    
    def __init__(self, model_path='models/best_model.cbm'):
        """
        Args:
            model_path: 학습된 모델 파일 경로
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """학습된 모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        self.model = CatBoostRegressor()
        self.model.load_model(self.model_path)
        print(f"모델 로드 완료: {self.model_path}")
    
    def create_2027_data(self, temperature_scenario='normal'):
        """
        2027년 예측용 데이터 생성
        
        Args:
            temperature_scenario: 기온 시나리오
                - 'normal': 평년 수준
                - 'warm': 평년보다 2도 높음
                - 'cold': 평년보다 2도 낮음
        
        Returns:
            2027년 날짜별 DataFrame
        """
        print(f"\n2027년 데이터 생성 중... (시나리오: {temperature_scenario})")
        
        # 2027년 전체 날짜 생성
        dates = pd.date_range('2027-01-01', '2027-12-31', freq='D')
        
        # 기온 시나리오 설정
        base_temp = 15
        seasonal_variation = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        random_variation = np.random.normal(0, 2, len(dates))
        
        if temperature_scenario == 'warm':
            adjustment = 2
        elif temperature_scenario == 'cold':
            adjustment = -2
        else:
            adjustment = 0
        
        temperature = base_temp + seasonal_variation + random_variation + adjustment
        
        # 지역 데이터 (주요 납품처 중심)
        regions = ['경기', '충남', '충북', '전북', '강원']
        
        # 각 날짜별로 지역 데이터 생성
        data_list = []
        for date in dates:
            for region in regions:
                # 지역별 약간의 기온 차이
                region_temp_diff = np.random.normal(0, 1)
                data_list.append({
                    'date': date,
                    'temperature': temperature[dates.get_loc(date)] + region_temp_diff,
                    'region': region
                })
        
        df = pd.DataFrame(data_list)
        
        print(f"생성된 데이터: {len(df)}행 (날짜: {len(dates)}일 × 지역: {len(regions)}개)")
        
        return df
    
    def preprocess_2027_data(self, df):
        """
        2027년 데이터 전처리
        
        Args:
            df: 원본 DataFrame
        
        Returns:
            전처리된 DataFrame
        """
        print("데이터 전처리 중...")
        
        # 전처리 수행
        preprocessor = SoilDataPreprocessor(df)
        processed_df = preprocessor.preprocess_all(temp_column='temperature', region_column='region')
        
        # 추가 피처 생성
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['week'] = processed_df['date'].dt.isocalendar().week
        processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
        processed_df['day_of_year'] = processed_df['date'].dt.dayofyear
        
        print(f"전처리 완료: {len(processed_df)}행, {len(processed_df.columns)}개 컬럼")
        
        return processed_df
    
    def predict(self, df):
        """
        상토 출고량 예측
        
        Args:
            df: 전처리된 DataFrame
        
        Returns:
            예측 결과가 추가된 DataFrame
        """
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        print("\n예측 수행 중...")
        
        # 피처 준비 (날짜와 지역 제외)
        exclude_cols = ['date', 'region']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 모델이 학습한 피처만 선택
        model_features = self.model.feature_names_
        available_features = [f for f in model_features if f in feature_cols]
        
        X = df[available_features].fillna(0)
        
        # 예측
        predictions = self.model.predict(X)
        
        # 음수 예측값 제거
        predictions = np.maximum(predictions, 0)
        
        # 결과 추가
        df['predicted_shipment'] = predictions.astype(int)
        
        print(f"예측 완료: 평균 {predictions.mean():.0f}포, 최대 {predictions.max():.0f}포")
        
        return df
    
    def aggregate_weekly(self, df):
        """
        주간 단위로 집계
        
        Args:
            df: 예측 결과 DataFrame
        
        Returns:
            주간 집계 DataFrame
        """
        print("\n주간 단위 집계 중...")
        
        # ISO 주차 기준으로 그룹화
        weekly = df.groupby(['year', 'week']).agg({
            'date': 'min',  # 주의 시작일
            'predicted_shipment': 'sum',  # 주간 총 출고량
            'temperature': 'mean',  # 주간 평균 기온
            'is_planting_season': 'max',  # 파종기 여부
            'holiday_effect': 'max'  # 휴일 효과
        }).reset_index()
        
        # 주의 종료일 추가
        weekly['week_end'] = weekly['date'] + timedelta(days=6)
        
        # 컬럼명 정리
        weekly.rename(columns={
            'date': 'week_start',
            'predicted_shipment': 'weekly_shipment',
            'temperature': 'avg_temperature'
        }, inplace=True)
        
        # 주차 정보 추가
        weekly['week_label'] = weekly.apply(
            lambda x: f"{int(x['year'])}년 {int(x['week'])}주차", axis=1
        )
        
        print(f"주간 집계 완료: {len(weekly)}주")
        
        return weekly
    
    def save_to_excel(self, weekly_df, output_path='output/forecast_2027.xlsx'):
        """
        예측 결과를 엑셀 파일로 저장
        
        Args:
            weekly_df: 주간 집계 DataFrame
            output_path: 출력 파일 경로
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 출력용 DataFrame 준비
        output_df = weekly_df[[
            'week_label', 'week_start', 'week_end',
            'weekly_shipment', 'avg_temperature',
            'is_planting_season', 'holiday_effect'
        ]].copy()
        
        # 컬럼명 한글화
        output_df.columns = [
            '주차', '시작일', '종료일',
            '예상 출고량(포)', '평균 기온(°C)',
            '파종기 여부', '휴일 효과'
        ]
        
        # 엑셀 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='2027년 주간 예측', index=False)
            
            # 월별 요약 시트 추가
            monthly_summary = weekly_df.copy()
            monthly_summary['month'] = pd.to_datetime(monthly_summary['week_start']).dt.month
            monthly = monthly_summary.groupby('month').agg({
                'weekly_shipment': 'sum',
                'avg_temperature': 'mean'
            }).reset_index()
            monthly.columns = ['월', '월간 총 출고량(포)', '평균 기온(°C)']
            monthly.to_excel(writer, sheet_name='월별 요약', index=False)
        
        print(f"\n엑셀 파일 저장 완료: {output_path}")
    
    def plot_forecast(self, weekly_df, output_path='output/forecast_2027.png'):
        """
        예측 결과 그래프 생성
        
        Args:
            weekly_df: 주간 집계 DataFrame
            output_path: 그래프 저장 경로
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # 1. 주간 출고량 예측
        axes[0].bar(weekly_df['week'], weekly_df['weekly_shipment'], 
                    color='steelblue', alpha=0.7)
        axes[0].set_xlabel('주차')
        axes[0].set_ylabel('예상 출고량 (포)')
        axes[0].set_title('2027년 주간 상토 출고량 예측', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 파종기 강조
        planting_weeks = weekly_df[weekly_df['is_planting_season'] == 1]
        if len(planting_weeks) > 0:
            axes[0].axvspan(planting_weeks['week'].min() - 0.5, 
                           planting_weeks['week'].max() + 0.5,
                           alpha=0.2, color='green', label='파종기')
            axes[0].legend()
        
        # 2. 기온과 출고량 관계
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(weekly_df['week'], weekly_df['avg_temperature'], 
                color='red', marker='o', label='평균 기온', linewidth=2)
        ax2_twin.bar(weekly_df['week'], weekly_df['weekly_shipment'], 
                    alpha=0.3, color='steelblue', label='출고량')
        
        ax2.set_xlabel('주차')
        ax2.set_ylabel('평균 기온 (°C)', color='red')
        ax2_twin.set_ylabel('예상 출고량 (포)', color='steelblue')
        ax2.set_title('기온과 출고량 관계', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2_twin.tick_params(axis='y', labelcolor='steelblue')
        ax2.grid(True, alpha=0.3)
        
        # 범례 통합
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 3. 월별 총 출고량
        monthly = weekly_df.copy()
        monthly['month'] = pd.to_datetime(monthly['week_start']).dt.month
        monthly_sum = monthly.groupby('month')['weekly_shipment'].sum().reset_index()
        
        axes[2].bar(monthly_sum['month'], monthly_sum['weekly_shipment'], 
                   color='darkgreen', alpha=0.7)
        axes[2].set_xlabel('월')
        axes[2].set_ylabel('월간 총 출고량 (포)')
        axes[2].set_title('2027년 월별 상토 출고량 예측', fontsize=14, fontweight='bold')
        axes[2].set_xticks(range(1, 13))
        axes[2].set_xticklabels([f'{m}월' for m in range(1, 13)])
        axes[2].grid(True, alpha=0.3)
        
        # 파종기 월 강조
        for month in [2, 3, 4, 5]:
            axes[2].axvspan(month - 0.4, month + 0.4, alpha=0.2, color='green')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"그래프 저장 완료: {output_path}")
        plt.close()
    
    def run_forecast(self, temperature_scenario='normal'):
        """
        전체 예측 파이프라인 실행
        
        Args:
            temperature_scenario: 기온 시나리오 ('normal', 'warm', 'cold')
        """
        print("=" * 60)
        print("2027년 상토 수요 예측 시작")
        print("=" * 60)
        
        # 1. 2027년 데이터 생성
        df_2027 = self.create_2027_data(temperature_scenario)
        
        # 2. 전처리
        processed_df = self.preprocess_2027_data(df_2027)
        
        # 3. 예측
        predicted_df = self.predict(processed_df)
        
        # 4. 주간 집계
        weekly_df = self.aggregate_weekly(predicted_df)
        
        # 5. 엑셀 저장
        self.save_to_excel(weekly_df)
        
        # 6. 그래프 저장
        self.plot_forecast(weekly_df)
        
        # 7. 요약 통계
        print("\n" + "=" * 60)
        print("예측 결과 요약")
        print("=" * 60)
        print(f"연간 총 예상 출고량: {weekly_df['weekly_shipment'].sum():,.0f} 포")
        print(f"주간 평균 출고량: {weekly_df['weekly_shipment'].mean():,.0f} 포")
        print(f"최대 주간 출고량: {weekly_df['weekly_shipment'].max():,.0f} 포 ({weekly_df.loc[weekly_df['weekly_shipment'].idxmax(), 'week_label']})")
        
        # 파종기 출고량
        planting_season = weekly_df[weekly_df['is_planting_season'] == 1]
        if len(planting_season) > 0:
            print(f"\n파종기(2~5월) 총 출고량: {planting_season['weekly_shipment'].sum():,.0f} 포")
            print(f"파종기 비중: {planting_season['weekly_shipment'].sum() / weekly_df['weekly_shipment'].sum() * 100:.1f}%")
        
        print("\n" + "=" * 60)
        print("예측 완료!")
        print("=" * 60)
        
        return weekly_df


if __name__ == "__main__":
    # 예측 실행
    predictor = SoilDemandPredictor()
    
    # 기본 시나리오 (평년 수준)
    print("\n[시나리오 1: 평년 수준 기온]")
    weekly_forecast = predictor.run_forecast(temperature_scenario='normal')
    
    # 추가 시나리오 실행 (선택사항)
    # print("\n[시나리오 2: 평년보다 따뜻한 기온]")
    # predictor.run_forecast(temperature_scenario='warm')
    
    # print("\n[시나리오 3: 평년보다 추운 기온]")
    # predictor.run_forecast(temperature_scenario='cold')
