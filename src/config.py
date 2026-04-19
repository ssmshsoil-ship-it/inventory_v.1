from pathlib import Path
import os
from dotenv import load_dotenv

# 프로젝트 루트 경로 설정 (절대 경로)
ROOT = Path(r"C:\ai_workspace\sh-ai-model")

# .env 파일 절대 경로로 명시 (어떤 폴더에서 실행하든 동일한 경로 사용)
ENV_FILE_PATH = r"C:\ai_workspace\sh-ai-model\.env"

# .env 파일 존재 여부 확인 및 로드
print(f"\n{'='*70}")
print("🔍 환경 변수 로드 디버깅")
print(f"{'='*70}")
print(f"1. .env 파일 경로: {ENV_FILE_PATH}")
print(f"2. .env 파일 존재 여부: {os.path.exists(ENV_FILE_PATH)}")

if os.path.exists(ENV_FILE_PATH):
    # .env 파일을 직접 읽어서 BOM 제거 및 환경 변수 강제 등록
    print(f"3. .env 파일 직접 파싱 중 (BOM 제거)...")
    try:
        with open(ENV_FILE_PATH, 'r', encoding='utf-8-sig') as f:  # utf-8-sig로 BOM 자동 제거
            lines = f.readlines()
            
        for line in lines:
            # 공백 및 BOM 제거
            line = line.strip()
            
            # 주석이나 빈 줄 건너뛰기
            if not line or line.startswith('#'):
                continue
            
            # KEY=VALUE 형식 파싱
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 환경 변수에 강제 등록
                os.environ[key] = value
                print(f"   ✓ 환경 변수 등록: {key} = {value[:4]}****... (총 {len(value)}자)")
                
    except Exception as e:
        print(f"   ⚠️  파일 읽기 오류: {e}")
    
    # load_dotenv도 실행 (이중 안전장치)
    load_dotenv(dotenv_path=ENV_FILE_PATH, override=True, verbose=False)
    print(f"4. ✓ load_dotenv(override=True) 실행 완료")
    
    # 로드 직후 환경 변수 확인
    loaded_key = os.getenv('WEATHER_API_KEY')
    print(f"5. os.getenv('WEATHER_API_KEY') 최종 확인:")
    if loaded_key:
        print(f"   ✓ 키 로드 성공: {loaded_key[:4]}****... (총 {len(loaded_key)}자)")
    else:
        print(f"   ✗ 키가 None입니다!")
        print(f"   💡 .env 파일에 'WEATHER_API_KEY=your_key' 형식으로 작성되었는지 확인하세요")
else:
    print(f"3. ⚠️  .env 파일을 찾을 수 없습니다")
    print(f"   다음 위치에 .env 파일을 생성하세요: {ENV_FILE_PATH}")

print(f"{'='*70}\n")

DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"

# API 키 로드
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# API 키 확인 및 검증
print(f"\n{'='*70}")
print("🔑 API 키 검증")
print(f"{'='*70}")
print(f"찾고 있는 환경 변수명: 'WEATHER_API_KEY'")
print(f"로드된 값: {WEATHER_API_KEY if WEATHER_API_KEY else 'None'}")

if WEATHER_API_KEY and len(WEATHER_API_KEY) > 10:
    print(f"✓ API Key Loaded: {WEATHER_API_KEY[:4]}****... (총 {len(WEATHER_API_KEY)}자)")
    print(f"✅ 기상 데이터 수집 준비 완료!")
    print(f"{'='*70}\n")
else:
    print(f"❌ API 키를 찾을 수 없거나 유효하지 않습니다!")
    print(f"\n해결 방법:")
    print(f"1. {ENV_FILE_PATH} 파일을 열어주세요")
    print(f"2. 다음 형식으로 작성되었는지 확인하세요:")
    print(f"   WEATHER_API_KEY=your_actual_api_key_here")
    print(f"3. 앞뒤 공백이 없는지 확인하세요")
    print(f"4. 줄바꿈이 올바른지 확인하세요 (Windows: CRLF)")
    print(f"{'='*70}\n")
    
    # API 키가 없으면 프로그램 종료
    import sys
    print("❌ API 키 없이는 실행할 수 없습니다. 프로그램을 종료합니다.")
    sys.exit(1)

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
