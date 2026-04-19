from pathlib import Path
import os
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정 (절대 경로)
ROOT = Path(r"C:\ai_workspace\sh-ai-model")

# .env 파일 절대 경로로 명시 (어떤 폴더에서 실행하든 동일한 경로 사용)
ENV_FILE_PATH = r"C:\ai_workspace\sh-ai-model\.env"

# .env 파일 존재 여부 확인 및 로드
if os.path.exists(ENV_FILE_PATH):
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)
    print(f"✓ .env 파일 로드 성공: {ENV_FILE_PATH}")
else:
    print(f"⚠️  .env 파일을 찾을 수 없습니다: {ENV_FILE_PATH}")
    print(f"⚠️  다음 위치에 .env 파일을 생성하세요: {ENV_FILE_PATH}")

DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"

# API 키 로드
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# API 키 확인 및 검증
if WEATHER_API_KEY and len(WEATHER_API_KEY) > 10:
    print(f"✓ API Key Loaded: {WEATHER_API_KEY[:4]}****... (총 {len(WEATHER_API_KEY)}자)")
    print(f"✅ 기상 데이터 수집 준비 완료!")
else:
    print("⚠️  API 키를 찾을 수 없거나 유효하지 않습니다.")
    print(f"⚠️  {ENV_FILE_PATH} 파일에 WEATHER_API_KEY를 설정하세요.")
    WEATHER_API_KEY = None

# 원본 데이터
MASTER_DB     = DATA_DIR / "master_db_v.0.xlsx"
WEATHER_DIR   = DATA_DIR / "weather"
COST_FILE     = DATA_DIR / "cost.xls"

# 처리된 데이터
TRAINING_DATA = PROCESSED_DIR / "training_dataset_v1.csv"
WEEKLY_WEATHER = PROCESSED_DIR / "weekly_weather.csv"

# 모델 타겟
TARGETS = ["수도용_포", "원예용_포"]

# 피크 시즌 (모내기)
PEAK_WEEKS = (13, 18)
SPRING_WEEKS = (10, 22)

# 학습/검증 분리 기준
# 2019~2025년 학습, 2026년 검증
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
VAL_YEAR    = 2026
