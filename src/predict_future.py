"""
2027년 상토 수요 예측기 (실무용)

학습된 모델을 사용하여 2027년 주차별 수요를 예측하고
트럭 배차 계획까지 포함한 실무용 보고서를 생성합니다.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from preprocess import SoilDataPreprocessor
from dispatch import DispatchOptimizer


class FutureDemandPredictor:
    """2027년 실무용 수요 예측 클래스"""
    
    def __init__(self, model_path='models/best_model.cbm', historical_data_path=None):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            historical_data_path: 과거 기온 데이터 경로 (평년값 계산용)
        """
        self.model_path = model_path
        self.historical_data_path = historical_data_path
        self.model = None
        self.avg_temperature_by_week = None
        
        self.load_model()
        self.calculate_normal_temperature()
    
    def load_model(self):
        """학습된 모델 로드"""
        if not os.path.exists(self.model_path):
            print(f"경고: 모델 파일을 찾을 수 없습니다: {self.model_path}")
            print("샘플 모델을 사용합니다.")
            return
        
        try:
            self.model = CatBoostRegressor()
            self.model.load_model(self.model_path)
            print(f"✓ 모델 로드 완료: {self.model_path}")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("샘플 예측을 사용합니다.")
    
    def calculate_normal_temperature(self):
        """
        과거 4년치 데이터에서 주차별 평년 기온 계산
        데이터가 없으면 표준 기온 패턴 사용
        """
        if self.historical_data_path and os.path.exists(self.historical_data_path):
            try:
                # 실제 데이터에서 평년값 계산
                df = pd.read_csv(self.historical_data_path)
                df['date'] = pd.to_datetime(df['date'])
                df['week'] = df['date'].dt.isocalendar().week
                
                # 주차별 평균 기온 계산
                self.avg_temperature_by_week = df.groupby('week')['temperature'].mean().to_dict()
                print(f"✓ 과거 데이터에서 평년 기온 계산 완료")
                return
            except Exception as e:
                print(f"과거 데이터 로드 실패: {e}")
        
        # 표준 기온 패턴 사용 (한국 평년 기온 근사)
        print("✓ 표준 평년 기온 패턴 사용")
        self.avg_temperature_by_week = {}
        for week in range(1, 53):
            # 정현파 기반 계절 패턴 (1월 초 = 주차 1)
            day_of_year = week * 7
            temp = 12.5 + 12.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            self.avg_temperature_by_week[week] = temp
    
    def create_2027_weekly_data(self):
        """
        2027년 주차별 데이터 생성 (평년 기온 사용)
        
        Returns:
            2027년 주차별 DataFrame
        """
        print("\n2027년 주차별 데이터 생성 중...")
        
        # 2027년 전체 날짜 생성
        dates = pd.date_range('2027-01-01', '2027-12-31', freq='D')
        
        # 주요 납품처 지역
        regions = ['경기', '충남', '충북', '전북', '강원']
        
        # 날짜별 데이터 생성
        data_list = []
        for date in dates:
            week_num = date.isocalendar().week
            
            # 평년 기온 가져오기
            base_temp = self.avg_temperature_by_week.get(week_num, 15.0)
            
            for region in regions:
                # 지역별 약간의 기온 차이 (±1도)
                region_temp_diff = np.random.normal(0, 0.5)
                
                data_list.append({
                    'date': date,
                    'temperature': base_temp + region_temp_diff,
                    'region': region
                })
        
        df = pd.DataFrame(data_list)
        print(f"✓ 생성 완료: {len(df)}행 (365일 × {len(regions)}개 지역)")
        
        return df
    
    def preprocess_data(self, df):
        """
        데이터 전처리
        
        Args:
            df: 원본 DataFrame
        
        Returns:
            전처리된 DataFrame
        """
        print("데이터 전처리 중...")
        
        try:
            preprocessor = SoilDataPreprocessor(df)
            processed_df = preprocessor.preprocess_all(
                temp_column='temperature',
                region_column='region',
                cost_file_path='data/cost.xlsx'
            )
            
            # 추가 피처 생성
            processed_df['year'] = processed_df['date'].dt.year
            processed_df['month'] = processed_df['date'].dt.month
            processed_df['week'] = processed_df['date'].dt.isocalendar().week
            processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
            processed_df['day_of_year'] = processed_df['date'].dt.dayofyear
            
            print(f"✓ 전처리 완료: {len(processed_df)}행, {len(processed_df.columns)}개 컬럼")
            return processed_df
            
        except Exception as e:
            print(f"전처리 중 오류: {e}")
            return df
    
    def predict_demand(self, df):
        """
        수요 예측
        
        Args:
            df: 전처리된 DataFrame
        
        Returns:
            예측 결과가 추가된 DataFrame
        """
        print("\n수요 예측 중...")
        
        if self.model is None:
            # 모델이 없으면 간단한 규칙 기반 예측
            print("⚠ 모델 없음 - 규칙 기반 예측 사용")
            base_demand = 500
            seasonal_factor = df['month'].apply(
                lambda m: 2.0 if m in [2, 3, 4, 5] else 0.5
            )
            temp_factor = (df['temperature'] - 10) * 5
            df['predicted_shipment'] = (base_demand * seasonal_factor + temp_factor).clip(lower=0).astype(int)
        else:
            # 모델 예측
            exclude_cols = ['date', 'region']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # 모델이 학습한 피처만 선택
            model_features = self.model.feature_names_
            available_features = [f for f in model_features if f in feature_cols]
            
            X = df[available_features].fillna(0)
            predictions = self.model.predict(X)
            predictions = np.maximum(predictions, 0)
            
            df['predicted_shipment'] = predictions.astype(int)
        
        print(f"✓ 예측 완료: 평균 {df['predicted_shipment'].mean():.0f}포/일")
        return df
    
    def aggregate_weekly(self, df):
        """
        주차별 집계
        
        Args:
            df: 예측 결과 DataFrame
        
        Returns:
            주차별 집계 DataFrame
        """
        print("\n주차별 집계 중...")
        
        # ISO 주차 기준 그룹화
        weekly = df.groupby(['year', 'week']).agg({
            'date': 'min',
            'predicted_shipment': 'sum',
            'temperature': 'mean',
            'is_planting_season': 'max',
            'holiday_effect': 'max'
        }).reset_index()
        
        # 주 종료일 추가
        weekly['week_end'] = weekly['date'] + timedelta(days=6)
        
        # 컬럼명 정리
        weekly.rename(columns={
            'date': 'week_start',
            'predicted_shipment': 'weekly_shipment',
            'temperature': 'avg_temperature'
        }, inplace=True)
        
        # 주차 라벨
        weekly['week_label'] = weekly['week'].apply(lambda w: f"{int(w)}주차")
        
        print(f"✓ 주차별 집계 완료: {len(weekly)}주")
        return weekly
    
    def calculate_truck_requirements(self, weekly_df):
        """
        주차별 트럭 배차 계산
        
        Args:
            weekly_df: 주차별 집계 DataFrame
        
        Returns:
            트럭 배차 정보가 추가된 DataFrame
        """
        print("\n트럭 배차 계산 중...")
        
        optimizer = DispatchOptimizer()
        
        # 트럭 대수 계산
        for idx, row in weekly_df.iterrows():
            total_shipment = row['weekly_shipment']
            
            # 25톤 우선 배차
            remaining = total_shipment
            trucks_25t = int(remaining / 2500)
            remaining -= trucks_25t * 2500
            
            # 11톤 배차
            trucks_11t = int(remaining / 1100)
            remaining -= trucks_11t * 1100
            
            # 5톤 배차
            trucks_5t = int(np.ceil(remaining / 500)) if remaining > 0 else 0
            
            weekly_df.loc[idx, '25톤_트럭'] = trucks_25t
            weekly_df.loc[idx, '11톤_트럭'] = trucks_11t
            weekly_df.loc[idx, '5톤_트럭'] = trucks_5t
            weekly_df.loc[idx, '총_트럭_대수'] = trucks_25t + trucks_11t + trucks_5t
        
        # 정수형 변환
        for col in ['25톤_트럭', '11톤_트럭', '5톤_트럭', '총_트럭_대수']:
            weekly_df[col] = weekly_df[col].astype(int)
        
        print(f"✓ 트럭 배차 계산 완료")
        return weekly_df
    
    def save_to_excel(self, weekly_df, output_path='reports/2027_Sunghwa_Forecast.xlsx'):
        """
        예측 결과를 실무용 엑셀 파일로 저장
        
        Args:
            weekly_df: 주차별 집계 DataFrame
            output_path: 출력 파일 경로
        """
        print(f"\n엑셀 파일 저장 중: {output_path}")
        
        # 출력 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 출력용 DataFrame 준비
        output_df = weekly_df[[
            'week_label', 'week_start', 'week_end',
            'weekly_shipment', 'avg_temperature',
            '25톤_트럭', '11톤_트럭', '5톤_트럭', '총_트럭_대수',
            'is_planting_season', 'holiday_effect'
        ]].copy()
        
        # 컬럼명 한글화
        output_df.columns = [
            '주차', '시작일', '종료일',
            '예상_출고량(포)', '평균_기온(℃)',
            '25톤_트럭(대)', '11톤_트럭(대)', '5톤_트럭(대)', '총_트럭(대)',
            '파종기_여부', '휴일_효과'
        ]
        
        # 날짜 포맷 변경
        output_df['시작일'] = pd.to_datetime(output_df['시작일']).dt.strftime('%Y-%m-%d')
        output_df['종료일'] = pd.to_datetime(output_df['종료일']).dt.strftime('%Y-%m-%d')
        
        # 엑셀 저장
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 주차별 상세 시트
            output_df.to_excel(writer, sheet_name='주차별_예측', index=False)
            
            # 월별 요약 시트
            monthly_df = weekly_df.copy()
            monthly_df['month'] = pd.to_datetime(monthly_df['week_start']).dt.month
            monthly_summary = monthly_df.groupby('month').agg({
                'weekly_shipment': 'sum',
                'avg_temperature': 'mean',
                '총_트럭_대수': 'sum'
            }).reset_index()
            monthly_summary.columns = ['월', '월간_총_출고량(포)', '평균_기온(℃)', '월간_총_트럭(대)']
            monthly_summary['월'] = monthly_summary['월'].apply(lambda m: f"{int(m)}월")
            monthly_summary.to_excel(writer, sheet_name='월별_요약', index=False)
            
            # 파종기 요약 시트
            planting_season = weekly_df[weekly_df['is_planting_season'] == 1].copy()
            if len(planting_season) > 0:
                planting_summary = pd.DataFrame({
                    '구분': ['파종기 (2~5월)', '비파종기', '연간 합계'],
                    '총_출고량(포)': [
                        planting_season['weekly_shipment'].sum(),
                        weekly_df[weekly_df['is_planting_season'] == 0]['weekly_shipment'].sum(),
                        weekly_df['weekly_shipment'].sum()
                    ],
                    '총_트럭(대)': [
                        planting_season['총_트럭_대수'].sum(),
                        weekly_df[weekly_df['is_planting_season'] == 0]['총_트럭_대수'].sum(),
                        weekly_df['총_트럭_대수'].sum()
                    ],
                    '비중(%)': [
                        planting_season['weekly_shipment'].sum() / weekly_df['weekly_shipment'].sum() * 100,
                        weekly_df[weekly_df['is_planting_season'] == 0]['weekly_shipment'].sum() / weekly_df['weekly_shipment'].sum() * 100,
                        100.0
                    ]
                })
                planting_summary.to_excel(writer, sheet_name='파종기_요약', index=False)
            
            # 요약 통계 시트
            summary_stats = pd.DataFrame({
                '항목': [
                    '연간 총 예상 출고량',
                    '주간 평균 출고량',
                    '최대 주간 출고량',
                    '최소 주간 출고량',
                    '연간 총 트럭 대수',
                    '주간 평균 트럭',
                    '최대 트럭 필요 주차'
                ],
                '값': [
                    f"{weekly_df['weekly_shipment'].sum():,.0f} 포",
                    f"{weekly_df['weekly_shipment'].mean():,.0f} 포",
                    f"{weekly_df['weekly_shipment'].max():,.0f} 포",
                    f"{weekly_df['weekly_shipment'].min():,.0f} 포",
                    f"{weekly_df['총_트럭_대수'].sum():,.0f} 대",
                    f"{weekly_df['총_트럭_대수'].mean():.1f} 대",
                    f"{weekly_df.loc[weekly_df['총_트럭_대수'].idxmax(), 'week_label']} ({weekly_df['총_트럭_대수'].max():.0f}대)"
                ]
            })
            summary_stats.to_excel(writer, sheet_name='요약_통계', index=False)
        
        print(f"✓ 엑셀 파일 저장 완료: {output_path}")
    
    def print_summary(self, weekly_df):
        """
        예측 결과 요약 출력
        
        Args:
            weekly_df: 주차별 집계 DataFrame
        """
        print("\n" + "=" * 70)
        print("2027년 성화 상토 수요 예측 결과")
        print("=" * 70)
        
        total_shipment = weekly_df['weekly_shipment'].sum()
        avg_weekly = weekly_df['weekly_shipment'].mean()
        max_week = weekly_df.loc[weekly_df['weekly_shipment'].idxmax(), 'week_label']
        max_shipment = weekly_df['weekly_shipment'].max()
        
        total_trucks = weekly_df['총_트럭_대수'].sum()
        avg_trucks = weekly_df['총_트럭_대수'].mean()
        max_trucks_week = weekly_df.loc[weekly_df['총_트럭_대수'].idxmax(), 'week_label']
        max_trucks = weekly_df['총_트럭_대수'].max()
        
        print(f"\n📦 출고량 예측")
        print(f"  • 연간 총 예상 출고량: {total_shipment:>15,.0f} 포")
        print(f"  • 주간 평균 출고량:   {avg_weekly:>15,.0f} 포")
        print(f"  • 최대 출고 주차:     {max_week:>15s} ({max_shipment:,.0f}포)")
        
        print(f"\n🚛 트럭 배차 계획")
        print(f"  • 연간 총 트럭 대수:   {total_trucks:>15,.0f} 대")
        print(f"  • 주간 평균 트럭:     {avg_trucks:>15.1f} 대")
        print(f"  • 최대 트럭 필요 주차: {max_trucks_week:>15s} ({max_trucks:.0f}대)")
        
        # 파종기 분석
        planting_season = weekly_df[weekly_df['is_planting_season'] == 1]
        if len(planting_season) > 0:
            planting_shipment = planting_season['weekly_shipment'].sum()
            planting_ratio = planting_shipment / total_shipment * 100
            
            print(f"\n🌱 파종기 (2~5월) 분석")
            print(f"  • 파종기 총 출고량:   {planting_shipment:>15,.0f} 포")
            print(f"  • 전체 대비 비중:     {planting_ratio:>15.1f} %")
            print(f"  • 파종기 주차 수:     {len(planting_season):>15d} 주")
        
        print("\n" + "=" * 70)
    
    def run(self):
        """전체 예측 파이프라인 실행"""
        print("=" * 70)
        print("2027년 성화 상토 수요 예측 시스템 시작")
        print("=" * 70)
        
        try:
            # 1. 2027년 데이터 생성
            df_2027 = self.create_2027_weekly_data()
            
            # 2. 전처리
            processed_df = self.preprocess_data(df_2027)
            
            # 3. 수요 예측
            predicted_df = self.predict_demand(processed_df)
            
            # 4. 주차별 집계
            weekly_df = self.aggregate_weekly(predicted_df)
            
            # 5. 트럭 배차 계산
            weekly_df = self.calculate_truck_requirements(weekly_df)
            
            # 6. 엑셀 저장
            self.save_to_excel(weekly_df)
            
            # 7. 요약 출력
            self.print_summary(weekly_df)
            
            print("\n✅ 예측 완료!")
            print("📄 결과 파일: reports/2027_Sunghwa_Forecast.xlsx")
            
            return weekly_df
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """메인 함수"""
    # 예측기 실행
    predictor = FutureDemandPredictor(
        model_path='models/best_model.cbm',
        historical_data_path=None  # 과거 데이터 경로 (있으면 지정)
    )
    
    result = predictor.run()
    
    if result is not None:
        print("\n" + "=" * 70)
        print("다음 단계:")
        print("  1. reports/2027_Sunghwa_Forecast.xlsx 파일을 확인하세요")
        print("  2. 주차별 예측 탭에서 상세 예측 결과를 확인하세요")
        print("  3. 월별 요약 탭에서 월간 계획을 수립하세요")
        print("  4. 파종기 요약 탭에서 성수기 대비 계획을 세우세요")
        print("=" * 70)


if __name__ == "__main__":
    main()
