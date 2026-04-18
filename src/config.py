from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"

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
