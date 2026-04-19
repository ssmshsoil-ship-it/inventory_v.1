# -*- coding: utf-8 -*-
"""
final_training_data.csv를 사용하여 배송량 예측 모델을 학습하고,
성능 평가 후 모델과 결과 차트를 저장합니다.
실행: python src/train_model.py
"""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from config import FINAL_TRAINING_DATA, MODELS_DIR, REPORTS_DIR, TRAIN_YEARS, VAL_YEAR, MODEL_PATH, PREDICTION_CHART_PATH

def set_korean_font():
    """matplotlib에서 한글 폰트를 설정합니다."""
    font_path = None
    if Path('C:/Windows/Fonts/malgun.ttf').exists():
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    elif Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf').exists():
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    if font_path:
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rc('axes', unicode_minus=False) # 마이너스 폰트 깨짐 방지
        print(f"[OK] 한글 폰트가 '{font_name}'으로 설정되었습니다.")
    else:
        print("[경고] 한글 폰트를 찾을 수 없습니다. 차트의 한글이 깨질 수 있습니다.")

def train_delivery_model():
    """배송량 예측 모델 학습 파이프라인"""
    print("- 1. 학습 데이터 로드")
    if not FINAL_TRAINING_DATA.exists():
        print(f"[오류] 학습 데이터 파일이 없습니다: {FINAL_TRAINING_DATA}")
        print(" -> 먼저 python src/integrate_features.py 를 실행하여 데이터를 생성하세요.")
        return

    df = pd.read_csv(FINAL_TRAINING_DATA, encoding='utf-8-sig', low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  [OK] 데이터 로드 완료: {df.shape[0]}행, {df.shape[1]}컬럼")

    # 예측 대상 컬럼(Target) 가정
    # 요청에 명시되지 않아, 판매 데이터의 핵심 지표인 '출고수량'을 목표 변수로 가정합니다.
    target_col = '출고수량'
    if target_col not in df.columns:
        print(f"[오류] 예측 대상 컬럼 '{target_col}'을 데이터에서 찾을 수 없습니다.")
        return
    
    print("\n- 2. 데이터 전처리 (결측치 처리 및 타입 변환)")
    # 타겟 변수 NaN이 있는 행은 학습에 사용할 수 없으므로 제거
    original_rows = len(df)
    df.dropna(subset=[target_col], inplace=True)
    if len(df) < original_rows:
        print(f"  [OK] 타겟 변수('{target_col}')에 결측치가 있는 {original_rows - len(df)}개 행 제거.")

    # 학습 피처 정의
    features_to_exclude = [
        target_col, 'date', '공급가', '단가', '합계액', '공급가액', '부가세'
    ]
    features = [col for col in df.columns if col not in features_to_exclude]
    print("  [OK] 금액 관련 피처를 학습에서 제외합니다.")
    categorical_features = df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df[features].select_dtypes(include=np.number).columns.tolist()

    # 범주형 피처: 결측치를 'missing'으로 채우고 str 타입으로 강제 변환
    for col in categorical_features:
        df[col] = df[col].fillna('missing').astype(str)
    print(f"  [OK] 범주형 피처({len(categorical_features)}개) 전처리 완료.")

    # 수치형 피처: 결측치를 0으로 채우기 (기온, 강수량 등)
    df[numerical_features] = df[numerical_features].fillna(0)
    print(f"  [OK] 수치형 피처({len(numerical_features)}개) 결측치를 0으로 처리 완료.")

    print("\n- 3. 데이터 분할 및 피처 준비 (학습: ~{VAL_YEAR-1}년, 검증: {VAL_YEAR}년)")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['dayofweek'] = df['date'].dt.dayofweek
    
    train_df = df[df['year'].isin(TRAIN_YEARS)]
    val_df = df[df['year'] == VAL_YEAR]
    
    # 이미 전처리 단계에서 정의된 피처 리스트 사용
    print(f"  [OK] 자동 인식된 범주형 피처: {categorical_features}")

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_val = val_df[features]
    y_val = val_df[target_col]
    print(f"  [OK] 학습 데이터: {len(X_train)}건, 검증 데이터: {len(X_val)}건")

    print("\n- 4. CatBoost 모델 학습 시작")
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        eval_metric='MAE',
        cat_features=categorical_features,
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        use_best_model=True
    )

    print("\n- 5. 모델 성능 평가")
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"  [결과] 검증 데이터 평균 절대 오차(MAE): {mae:.2f}")
    print(f"  [참고] 검증 데이터 RMSE: {rmse:.2f}")

    print("\n- 6. 피처 중요도 분석")
    feature_importances = pd.Series(model.get_feature_importance(), index=features)
    top_20_features = feature_importances.sort_values(ascending=False).head(20)
    print("  [상위 20개 피처 중요도]")
    print(top_20_features.to_string())

    print("\n- 7. 예측 결과 차트 생성")
    val_results = val_df[['date']].copy()
    val_results['actual'] = y_val
    val_results['predicted'] = preds
    daily_results = val_results.groupby('date').sum().reset_index()

    plt.figure(figsize=(15, 6))
    plt.plot(daily_results['date'], daily_results['actual'], label='실제 출고량')
    plt.plot(daily_results['date'], daily_results['predicted'], label='예측 출고량', alpha=0.7)
    plt.title(f'{VAL_YEAR}년 출고량 실제값 vs. 예측값')
    plt.xlabel('날짜')
    plt.ylabel('일일 총 출고량')
    plt.legend()
    plt.grid(True)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig("reports/prediction_v2.png")
    print(f"  [OK] 차트 저장 완료: reports/prediction_v2.png")

    print("\n- 8. 모델 저장")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model("models/sh_delivery_v2.cbm")
    print(f"  [OK] 모델 저장 완료: models/sh_delivery_v2.cbm")


if __name__ == "__main__":
    set_korean_font()
    train_delivery_model()
