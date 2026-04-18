"""
기상청 API를 사용한 기상 데이터 수집 및 분석 모듈

기상청 공공데이터포털 API를 통해 과거 20년치 기상 데이터를 수집하고
상토 수요 예측에 필요한 형태로 전처리합니다.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import warnings
warnings.filterwarnings('ignore')


class WeatherDataCollector:
    """기상청 API 기상 데이터 수집 클래스"""
    
    # 기상청 API 엔드포인트
    API_BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    
    # 전국 주요 관측 지점 (ASOS 지점)
    STATIONS = {
        '서울': '108',
        '인천': '112',
        '수원': '119',
        '강릉': '105',
        '춘천': '101',
        '청주': '131',
        '대전': '133',
        '전주': '146',
        '광주': '156',
        '목포': '165',
        '여수': '168',
        '부산': '159',
        '울산': '152',
        '대구': '143',
        '포항': '138',
        '제주': '184',
        '천안': '232',
        '보성': '258',  # 성화 공장 위치
    }
    
    def __init__(self, api_key):
        """
        Args:
            api_key: 기상청 API 인증키
        """
        self.api_key = api_key
        print(f"✓ API 키 설정 완료")
    
    def fetch_daily_data(self, station_code, start_date, end_date):
        """
        특정 지점의 일별 기상 데이터 수집
        
        Args:
            station_code: 관측 지점 코드
            start_date: 시작 날짜 (YYYYMMDD)
            end_date: 종료 날짜 (YYYYMMDD)
        
        Returns:
            DataFrame
        """
        params = {
            'serviceKey': self.api_key,
            'pageNo': '1',
            'numOfRows': '999',
            'dataType': 'JSON',
            'dataCd': 'ASOS',
            'dateCd': 'DAY',
            'startDt': start_date,
            'endDt': end_date,
            'stnIds': station_code,
        }
        
        try:
            response = requests.get(self.API_BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data and 'body' in data['response']:
                    body = data['response']['body']
                    
                    if 'items' in body and body['items']:
                        items = body['items']['item']
                        
                        # 단일 항목인 경우 리스트로 변환
                        if isinstance(items, dict):
                            items = [items]
                        
                        df = pd.DataFrame(items)
                        return df
                    else:
                        return None
                else:
                    print(f"  ⚠️  응답 구조 오류: {station_code}")
                    return None
            else:
                print(f"  ⚠️  HTTP 오류 {response.status_code}: {station_code}")
                return None
                
        except Exception as e:
            print(f"  ❌ API 호출 오류: {e}")
            return None
    
    def collect_historical_data(self, years=20, output_path='data/weather_historical.csv'):
        """
        과거 N년치 전국 기상 데이터 수집
        
        Args:
            years: 수집할 연도 수
            output_path: 저장 경로
        
        Returns:
            DataFrame
        """
        print(f"\n{'='*70}")
        print(f"기상 데이터 수집 시작 (최근 {years}년, 전국 {len(self.STATIONS)}개 지점)")
        print(f"{'='*70}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        print(f"수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        all_data = []
        total_stations = len(self.STATIONS)
        
        for idx, (station_name, station_code) in enumerate(self.STATIONS.items(), 1):
            print(f"\n[{idx}/{total_stations}] 📍 {station_name} ({station_code}) 데이터 수집 중...")
            
            # 1년 단위로 나누어 수집 (API 제한 고려)
            current_date = start_date
            station_data = []
            
            while current_date < end_date:
                year_end = min(current_date + timedelta(days=365), end_date)
                
                start_str = current_date.strftime('%Y%m%d')
                end_str = year_end.strftime('%Y%m%d')
                
                print(f"  수집 중: {start_str} ~ {end_str}", end=' ')
                
                df = self.fetch_daily_data(station_code, start_str, end_str)
                
                if df is not None and len(df) > 0:
                    df['station_name'] = station_name
                    df['station_code'] = station_code
                    station_data.append(df)
                    print(f"✓ {len(df)}일")
                else:
                    print("✗ 데이터 없음")
                
                current_date = year_end + timedelta(days=1)
                time.sleep(0.3)  # API 호출 제한 고려
            
            if station_data:
                station_df = pd.concat(station_data, ignore_index=True)
                all_data.append(station_df)
                print(f"  ✅ {station_name} 총 {len(station_df)}일 데이터 수집 완료")
        
        if not all_data:
            print("\n❌ 수집된 데이터가 없습니다.")
            return None
        
        # 데이터 통합
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*70}")
        print(f"✅ 데이터 수집 완료!")
        print(f"{'='*70}")
        print(f"총 데이터: {len(combined_df):,}행")
        print(f"저장 위치: {output_path}")
        print(f"지점 수: {len(self.STATIONS)}개")
        print(f"기간: {years}년")
        
        return combined_df
    
    def preprocess_weather_data(self, raw_data_path='data/weather_historical.csv',
                                output_path='data/weather_processed.csv'):
        """
        수집된 기상 데이터 전처리
        
        Args:
            raw_data_path: 원본 데이터 경로
            output_path: 전처리된 데이터 저장 경로
        
        Returns:
            전처리된 DataFrame
        """
        print(f"\n{'='*70}")
        print("기상 데이터 전처리 시작")
        print(f"{'='*70}")
        
        # 데이터 로드
        df = pd.read_csv(raw_data_path, encoding='utf-8-sig')
        print(f"원본 데이터: {len(df):,}행")
        
        # 날짜 변환
        df['date'] = pd.to_datetime(df['tm'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # 컬럼명 정리 및 숫자 변환
        column_mapping = {
            'avgTa': 'avg_temp',
            'minTa': 'min_temp',
            'maxTa': 'max_temp',
            'sumRn': 'rainfall',
            'avgWs': 'wind_speed',
            'maxWs': 'max_wind_speed',
            'avgRhm': 'humidity',
            'avgPv': 'vapor_pressure',
            'avgPa': 'pressure',
            'ssDur': 'sunshine_duration',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
        
        # 결측치 처리
        numeric_cols = ['avg_temp', 'min_temp', 'max_temp', 'rainfall', 
                       'wind_speed', 'max_wind_speed', 'humidity']
        
        for col in numeric_cols:
            if col in df.columns:
                # 지점별 평균으로 결측치 채우기
                df[col] = df.groupby('station_name')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        # 파생 변수 생성
        if 'max_temp' in df.columns and 'min_temp' in df.columns:
            df['temp_range'] = df['max_temp'] - df['min_temp']
        
        if 'rainfall' in df.columns:
            df['is_rainy'] = (df['rainfall'] > 0).astype(int)
        
        if 'avg_temp' in df.columns:
            # 생장온도 (기준 5도)
            df['growing_degree'] = df['avg_temp'].apply(lambda x: max(0, x - 5))
            
            # 냉해 위험 (0도 이하)
            df['cold_stress'] = (df['min_temp'] < 0).astype(int)
        
        print(f"\n전처리 완료: {len(df):,}행")
        
        # 일별 전국 평균 계산
        daily_avg = df.groupby('date').agg({
            'avg_temp': 'mean',
            'min_temp': 'mean',
            'max_temp': 'mean',
            'rainfall': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'temp_range': 'mean',
            'is_rainy': 'sum',
            'growing_degree': 'mean',
            'cold_stress': 'sum',
        }).reset_index()
        
        daily_avg['year'] = daily_avg['date'].dt.year
        daily_avg['month'] = daily_avg['date'].dt.month
        daily_avg['week'] = daily_avg['date'].dt.isocalendar().week
        
        # 주차별 집계
        weekly_avg = daily_avg.groupby(['year', 'week']).agg({
            'avg_temp': 'mean',
            'min_temp': 'mean',
            'max_temp': 'mean',
            'rainfall': 'sum',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'temp_range': 'mean',
            'is_rainy': 'sum',
            'growing_degree': 'sum',
            'cold_stress': 'sum',
        }).reset_index()
        
        weekly_avg.columns = ['year', 'week', 'avg_temp', 'min_temp', 'max_temp',
                             'total_rainfall', 'avg_humidity', 'avg_wind_speed',
                             'avg_temp_range', 'rainy_days', 'growing_degree_sum',
                             'cold_stress_days']
        
        # 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        weekly_avg.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 전처리 완료!")
        print(f"주차별 데이터: {len(weekly_avg):,}행")
        print(f"저장 위치: {output_path}")
        
        return weekly_avg
    
    def analyze_weather_patterns(self, processed_data_path='data/weather_processed.csv',
                                output_path='data/weather_analysis.json'):
        """
        기상 패턴 분석 및 통계
        
        Args:
            processed_data_path: 전처리된 데이터 경로
            output_path: 분석 결과 저장 경로
        
        Returns:
            분석 결과 딕셔너리
        """
        print(f"\n{'='*70}")
        print("기상 패턴 분석")
        print(f"{'='*70}")
        
        df = pd.read_csv(processed_data_path, encoding='utf-8-sig')
        
        # 주차별 평년값 계산 (20년 평균)
        normal_by_week = df.groupby('week').agg({
            'avg_temp': 'mean',
            'min_temp': 'mean',
            'max_temp': 'mean',
            'total_rainfall': 'mean',
            'rainy_days': 'mean',
            'growing_degree_sum': 'mean',
            'cold_stress_days': 'mean',
        }).round(2)
        
        # 파종기(2~5월, 대략 주차 5~22) 분석
        planting_season = df[df['week'].between(5, 22)]
        
        # 연도별 트렌드
        yearly_trend = df.groupby('year').agg({
            'avg_temp': 'mean',
            'total_rainfall': 'sum',
            'cold_stress_days': 'sum',
        }).round(2)
        
        analysis = {
            '수집_정보': {
                '총_데이터_수': len(df),
                '수집_기간': f"{df['year'].min()}년 ~ {df['year'].max()}년",
                '총_연도_수': int(df['year'].max() - df['year'].min() + 1),
            },
            '전체_기간_통계': {
                '평균_기온': round(df['avg_temp'].mean(), 2),
                '최저_기온': round(df['min_temp'].min(), 2),
                '최고_기온': round(df['max_temp'].max(), 2),
                '평균_강수량': round(df['total_rainfall'].mean(), 2),
                '평균_습도': round(df['avg_humidity'].mean(), 2),
            },
            '파종기_2~5월_통계': {
                '평균_기온': round(planting_season['avg_temp'].mean(), 2),
                '평균_주간_강수량': round(planting_season['total_rainfall'].mean(), 2),
                '평균_비오는_날': round(planting_season['rainy_days'].mean(), 2),
                '평균_생장온도': round(planting_season['growing_degree_sum'].mean(), 2),
                '평균_냉해일수': round(planting_season['cold_stress_days'].mean(), 2),
            },
            '주차별_평년값': normal_by_week.to_dict('index'),
            '연도별_트렌드': yearly_trend.to_dict('index'),
        }
        
        # 결과 출력
        print("\n📊 분석 결과:")
        print(f"\n[수집 정보]")
        print(f"  총 데이터: {analysis['수집_정보']['총_데이터_수']:,}주")
        print(f"  수집 기간: {analysis['수집_정보']['수집_기간']}")
        print(f"  총 연도: {analysis['수집_정보']['총_연도_수']}년")
        
        print(f"\n[전체 기간 통계]")
        for key, value in analysis['전체_기간_통계'].items():
            print(f"  {key}: {value}")
        
        print(f"\n[파종기 (2~5월) 통계]")
        for key, value in analysis['파종기_2~5월_통계'].items():
            print(f"  {key}: {value}")
        
        print(f"\n[기온 트렌드]")
        recent_5years = yearly_trend.tail(5)
        print(f"  최근 5년 평균 기온: {recent_5years['avg_temp'].mean():.2f}°C")
        print(f"  20년 전 대비 변화: {yearly_trend['avg_temp'].iloc[-1] - yearly_trend['avg_temp'].iloc[0]:+.2f}°C")
        
        # JSON 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # numpy 타입을 Python 기본 타입으로 변환
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        analysis_serializable = convert_to_serializable(analysis)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_serializable, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 분석 결과 저장: {output_path}")
        
        return analysis
    
    def run_full_pipeline(self, years=20):
        """
        전체 파이프라인 실행
        
        Args:
            years: 수집할 연도 수
        
        Returns:
            분석 결과
        """
        # 1. 데이터 수집
        raw_data = self.collect_historical_data(years=years)
        
        if raw_data is None:
            print("\n❌ 데이터 수집 실패")
            return None
        
        # 2. 데이터 전처리
        processed_data = self.preprocess_weather_data()
        
        # 3. 패턴 분석
        analysis = self.analyze_weather_patterns()
        
        print(f"\n{'='*70}")
        print("✅ 전체 파이프라인 완료!")
        print(f"{'='*70}")
        print("\n생성된 파일:")
        print("  1. data/weather_historical.csv - 원본 기상 데이터 (전국 일별)")
        print("  2. data/weather_processed.csv - 전처리된 주차별 데이터")
        print("  3. data/weather_analysis.json - 분석 결과 및 평년값")
        
        return analysis


def main():
    """메인 함수"""
    print("=" * 70)
    print("기상청 API 기상 데이터 수집 시스템")
    print("=" * 70)
    
    # API 키 설정
    API_KEY = "048234d69b91cf5b6c18b1381151060d5c5bb1b1dd26b0fcf26d777d7e63fa24"
    
    # 수집기 생성
    collector = WeatherDataCollector(api_key=API_KEY)
    
    # 전체 파이프라인 실행 (최근 20년)
    analysis = collector.run_full_pipeline(years=20)
    
    if analysis:
        print("\n" + "=" * 70)
        print("다음 단계:")
        print("  1. data/weather_processed.csv를 확인하세요")
        print("  2. data/weather_analysis.json에서 평년값을 확인하세요")
        print("  3. 이 데이터를 src/predict_future.py에 통합하세요")
        print("  4. src/train.py를 실행하여 모델을 재학습하세요")
        print("=" * 70)


if __name__ == "__main__":
    main()
