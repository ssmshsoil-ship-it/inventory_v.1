# -*- coding: utf-8 -*-
from pathlib import Path
import os
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정 (절대 경로)
ROOT = Path(r"C:\ai_workspace\sh-ai-model")

# .env 파일 절대 경로로 명시 (어떤 폴더에서 실행하든 동일한 경로 사용)
ENV_FILE_PATH = r"C:\ai_workspace\sh-ai-model\.env"

# .env 파일 로드 (BOM 이슈 방지를 위해 수동 파싱 후 load_dotenv 실행)
if os.path.exists(ENV_FILE_PATH):
    try:
        # utf-8-sig 인코딩으로 BOM을 자동 제거하며 파일을 읽음
        with open(ENV_FILE_PATH, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                # 주석이나 빈 줄, '='가 없는 줄은 건너뜀
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                # 환경 변수에 직접 등록
                os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"   [경고] .env 파일 수동 파싱 중 오류 발생: {e}")

    # load_dotenv를 override=True 옵션과 함께 실행하여 수동 등록된 값을 유지하거나 덮어쓰도록 함
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)
else:
    print(f"[경고] .env 파일을 찾을 수 없습니다: {ENV_FILE_PATH}")

DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"

# API 키 로드
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# API 키 확인 및 검증
if WEATHER_API_KEY and len(WEATHER_API_KEY) > 10:
    print(f"[OK] [Gemini] API Key Successfully Loaded: {WEATHER_API_KEY[:4]}****")
else:
    print(f"[오류] API 키를 찾을 수 없거나 유효하지 않습니다!")
    print(f"\n해결 방법:")
    print(f"1. 프로젝트 루트 폴더의 .env 파일을 확인하세요. (경로: {ENV_FILE_PATH})")
    print(f"2. 다음 형식으로 API 키가 정확히 입력되었는지 확인하세요:")
    print(f"   WEATHER_API_KEY=your_actual_api_key_here")
    print(f"3. 키 값의 앞뒤에 불필요한 공백이 없는지 확인하세요.")
    print(f"\n[오류] API 키가 없으면 기상 데이터를 수집할 수 없으므로 프로그램을 종료합니다.")
    
    # API 키가 없으면 프로그램 종료
    import sys
    sys.exit(1)

# 원본 데이터
MASTER_DB     = DATA_DIR / "master_db_v.0.xlsx"
WEATHER_DIR   = DATA_DIR / "weather"
RAW_WEATHER_DIR = DATA_DIR / "raw" / "weather"  # 원본 기상 데이터 폴더
RAW_SALES_DIR = DATA_DIR / "raw" / "sales"    # 원본 판매 데이터 폴더
COST_FILE     = DATA_DIR / "cost.xls"
SALES_DATA_DIR = DATA_DIR / "2019-2026" # 2019-2026년 상세 판매 데이터

# 처리된 데이터
TRAINING_DATA = PROCESSED_DIR / "training_dataset_v1.csv"
WEEKLY_WEATHER = PROCESSED_DIR / "weekly_weather.csv"
UNIQUE_CUSTOMERS_CSV = PROCESSED_DIR / "unique_customers.csv"
CUSTOMER_MAP_CSV = PROCESSED_DIR / "customer_master_map.csv"
FINAL_TRAINING_DATA = PROCESSED_DIR / "final_training_data.csv"

# 모델 타겟
TARGETS = ["수도용_포", "원예용_포"]

# 모델 및 리포트 경로
MODEL_PATH = MODELS_DIR / "sh_delivery_v1.cbm"
PREDICTION_CHART_PATH = REPORTS_DIR / "prediction_result.png"

# 피크 시즌 (모내기)
PEAK_WEEKS = (13, 18)
SPRING_WEEKS = (10, 22)

# 학습/검증 분리 기준
# 2019~2025년 학습, 2026년 검증
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
VAL_YEAR    = 2026
