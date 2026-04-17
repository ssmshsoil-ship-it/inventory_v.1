"""
CatBoost + LightGBM 학습 및 비교
실행: python src/train.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from config import TRAINING_DATA, MODELS_DIR, TARGETS, TRAIN_YEARS, VAL_YEAR

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 피처 정의 ────────────────────────────────────────────────
WEATHER_FEATURES = [
    "avg_temp", "min_temp", "max_temp",
    "total_rain", "rain_days",
    "cold_stress_days", "warm_days", "temp_anomaly",
    "boseong_avg_temp", "boseong_rain",
    "boseong_cold_stress", "boseong_temp_anomaly",
]
CALENDAR_FEATURES = ["sin_week", "cos_week", "is_spring", "is_peak", "ISO주차"]
CAT_FEATURES      = ["ISO연도"]


def get_features(target: str) -> list[str]:
    other = [t for t in TARGETS if t != target]
    lag_self  = [f"{target}_lag{i}" for i in [1, 2, 3, 4, 52]]
    lag_other = [f"{t}_lag1" for t in other]
    return WEATHER_FEATURES + CALENDAR_FEATURES + CAT_FEATURES + lag_self + lag_other


def load_data(target: str):
    df = pd.read_csv(TRAINING_DATA, encoding="utf-8-sig")
    df["ISO연도"] = df["ISO연도"].astype(int)

    features = get_features(target)
    available = [f for f in features if f in df.columns]
    df = df.dropna(subset=[target])

    train = df[df["ISO연도"].isin(TRAIN_YEARS)]
    val   = df[df["ISO연도"] == VAL_YEAR]

    X_tr = train[available].fillna(0)
    y_tr = train[target]
    X_val = val[available].fillna(0)
    y_val = val[target]

    return X_tr, y_tr, X_val, y_val, available


def eval_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    # 0값 제외하고 MAPE 계산
    mask = y_true > 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if mask.sum() > 0 else float("nan")
    print(f"  {label:12s} MAE={mae:>10,.0f}포  MAPE={mape:5.1f}%")
    return {"MAE": mae, "MAPE": mape}


# ─── CatBoost ─────────────────────────────────────────────────
def tune_catboost(X_tr, y_tr, X_val, y_val, cat_cols, n_trials=30):
    def objective(trial):
        params = {
            "iterations":       trial.suggest_int("iterations", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth":            trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1, 10),
            "random_strength":  trial.suggest_float("random_strength", 0, 2),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": 42,
            "verbose": False,
        }
        cat_idx = [X_tr.columns.tolist().index(c) for c in cat_cols if c in X_tr.columns]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, cat_features=cat_idx,
                  eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        return mean_absolute_error(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_catboost(X_tr, y_tr, X_val, y_val, cat_cols, params):
    cat_idx = [X_tr.columns.tolist().index(c) for c in cat_cols if c in X_tr.columns]
    full_params = {**params, "loss_function": "MAE", "eval_metric": "MAE",
                   "random_seed": 42, "verbose": False}
    model = CatBoostRegressor(**full_params)
    model.fit(X_tr, y_tr, cat_features=cat_idx,
              eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    return model


# ─── LightGBM ─────────────────────────────────────────────────
def tune_lightgbm(X_tr, y_tr, X_val, y_val, n_trials=30):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 30),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
            "objective": "mae",
            "metric": "mae",
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
        return mean_absolute_error(y_val, model.predict(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_lightgbm(X_tr, y_tr, X_val, y_val, params):
    full_params = {**params, "objective": "mae", "metric": "mae",
                   "random_state": 42, "verbose": -1}
    model = lgb.LGBMRegressor(**full_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    return model


# ─── 메인 ─────────────────────────────────────────────────────
def run(n_trials=30):
    results = {}

    for target in TARGETS:
        print(f"\n{'='*50}")
        print(f"타겟: {target}")
        print(f"{'='*50}")

        X_tr, y_tr, X_val, y_val, features = load_data(target)
        cat_cols = [c for c in CAT_FEATURES if c in features]

        print(f"학습: {len(X_tr)}행 / 검증: {len(X_val)}행 / 피처: {len(features)}개")

        # CatBoost
        print(f"\n[CatBoost] Optuna 튜닝 중... ({n_trials} trials)")
        cb_params = tune_catboost(X_tr, y_tr, X_val, y_val, cat_cols, n_trials)
        cb_model  = train_catboost(X_tr, y_tr, X_val, y_val, cat_cols, cb_params)
        cb_metrics = eval_metrics(y_val, cb_model.predict(X_val), "CatBoost")
        model_key = "sudo" if "수도용" in target else "orye"
        cb_model.save_model(str(MODELS_DIR / f"catboost_{model_key}.cbm"))

        # LightGBM
        print(f"\n[LightGBM] Optuna 튜닝 중... ({n_trials} trials)")
        lgb_params = tune_lightgbm(X_tr, y_tr, X_val, y_val, n_trials)
        lgb_model  = train_lightgbm(X_tr, y_tr, X_val, y_val, lgb_params)
        lgb_metrics = eval_metrics(y_val, lgb_model.predict(X_val), "LightGBM")
        lgb_model.booster_.save_model(str(MODELS_DIR / f"lightgbm_{model_key}.bin"))

        # 피처 중요도 (CatBoost 기준)
        fi = pd.Series(
            cb_model.get_feature_importance(),
            index=X_tr.columns
        ).sort_values(ascending=False).head(10)
        print(f"\n[피처 중요도 Top10 - CatBoost]")
        for fname, score in fi.items():
            print(f"  {fname:30s} {score:.1f}")

        results[target] = {
            "catboost": cb_metrics, "lightgbm": lgb_metrics,
            "best": "catboost" if cb_metrics["MAE"] < lgb_metrics["MAE"] else "lightgbm"
        }

    # 최종 요약
    print(f"\n{'='*50}")
    print("최종 결과 요약")
    print(f"{'='*50}")
    for target, res in results.items():
        winner = res["best"].upper()
        print(f"\n{target}")
        print(f"  CatBoost  MAE={res['catboost']['MAE']:>10,.0f}  MAPE={res['catboost']['MAPE']:.1f}%")
        print(f"  LightGBM  MAE={res['lightgbm']['MAE']:>10,.0f}  MAPE={res['lightgbm']['MAPE']:.1f}%")
        print(f"  → 승자: {winner}")

    return results


if __name__ == "__main__":
    run(n_trials=30)
"""
상토 수요 예측 모델 학습 스크립트

CatBoostRegressor를 사용하여 상토 출고량을 예측하는 모델을 학습합니다.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from preprocess import SoilDataPreprocessor


class SoilDemandTrainer:
    """상토 수요 예측 모델 학습 클래스"""
    
    def __init__(self, data_path=None):
        """
        Args:
            data_path: 학습 데이터 경로 (None이면 샘플 데이터 생성)
        """
        self.data_path = data_path
        self.model = None
        self.train_data = None
        self.test_data = None
        self.cat_features = []
        
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        if self.data_path and os.path.exists(self.data_path):
            # 실제 데이터 로드
            df = pd.read_csv(self.data_path)
            print(f"데이터 로드 완료: {len(df)}행")
        else:
            # 샘플 데이터 생성 (3년치)
            print("샘플 데이터 생성 중...")
            dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
            
            # 계절성과 트렌드를 반영한 샘플 데이터
            df = pd.DataFrame({
                'date': dates,
                'temperature': 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 3, len(dates)),
                'region': np.random.choice(['경기', '충남', '충북', '전북', '강원', '기타'], len(dates)),
            })
            
            # 파종기(2~5월)에 출고량 증가 패턴 반영
            base_shipment = 500
            seasonal_effect = 300 * np.sin(2 * np.pi * (df['date'].dt.month - 2) / 12)
            seasonal_effect = np.where(df['date'].dt.month.isin([2, 3, 4, 5]), seasonal_effect + 400, seasonal_effect)
            temperature_effect = (df['temperature'] - 10) * 10
            random_noise = np.random.normal(0, 100, len(dates))
            
            df['shipment'] = base_shipment + seasonal_effect + temperature_effect + random_noise
            df['shipment'] = df['shipment'].clip(lower=0).astype(int)
        
        # 전처리 수행
        print("데이터 전처리 중...")
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
    
    def split_train_test(self, df, test_period_days=365):
        """
        학습/테스트 데이터 분할 (최근 1년을 테스트용으로)
        
        Args:
            df: 전처리된 DataFrame
            test_period_days: 테스트 기간 (일 단위, 기본 365일)
        """
        df = df.sort_values('date').reset_index(drop=True)
        
        # 최근 1년을 테스트 데이터로
        split_date = df['date'].max() - timedelta(days=test_period_days)
        
        train_df = df[df['date'] <= split_date].copy()
        test_df = df[df['date'] > split_date].copy()
        
        print(f"\n데이터 분할:")
        print(f"  학습 데이터: {len(train_df)}행 ({train_df['date'].min()} ~ {train_df['date'].max()})")
        print(f"  테스트 데이터: {len(test_df)}행 ({test_df['date'].min()} ~ {test_df['date'].max()})")
        
        self.train_data = train_df
        self.test_data = test_df
        
        return train_df, test_df
    
    def prepare_features(self, df, target_col='shipment'):
        """
        학습용 피처 준비
        
        Args:
            df: DataFrame
            target_col: 타겟 변수 컬럼명
        
        Returns:
            X, y, feature_names
        """
        # 제외할 컬럼
        exclude_cols = ['date', target_col]
        
        # 피처 선택
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # 범주형 변수 자동 감지
        self.cat_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                self.cat_features.append(col)
            elif col in ['month', 'week', 'day_of_week', 'year', 'region', 
                        'is_holiday', 'is_weekend', 'is_planting_season', 
                        'is_holiday_or_weekend', 'holiday_effect']:
                self.cat_features.append(col)
        
        print(f"\n피처 정보:")
        print(f"  전체 피처 수: {len(feature_cols)}")
        print(f"  범주형 피처: {self.cat_features}")
        
        return X, y, feature_cols
    
    def train_model(self, X_train, y_train, X_test=None, y_test=None):
        """
        CatBoost 모델 학습
        
        Args:
            X_train: 학습 피처
            y_train: 학습 타겟
            X_test: 검증 피처 (선택)
            y_test: 검증 타겟 (선택)
        """
        print("\n모델 학습 시작...")
        
        # CatBoost 모델 설정
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            eval_metric='MAE',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            cat_features=self.cat_features
        )
        
        # 학습
        if X_test is not None and y_test is not None:
            eval_set = (X_test, y_test)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                use_best_model=True,
                plot=False
            )
        else:
            self.model.fit(X_train, y_train, plot=False)
        
        print("모델 학습 완료!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        모델 평가
        
        Args:
            X_test: 테스트 피처
            y_test: 테스트 타겟
        
        Returns:
            평가 지표 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 예측
        y_pred = self.model.predict(X_test)
        
        # 평가 지표 계산
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print("\n=== 모델 성능 평가 ===")
        print(f"MAE (평균 절대 오차): {mae:.2f}")
        print(f"RMSE (평균 제곱근 오차): {rmse:.2f}")
        print(f"MAPE (평균 절대 백분율 오차): {mape:.2f}%")
        
        return metrics, y_pred
    
    def plot_predictions(self, y_test, y_pred, save_path='results/prediction_comparison.png'):
        """
        실제 값과 예측 값 비교 그래프 생성
        
        Args:
            y_test: 실제 값
            y_pred: 예측 값
            save_path: 그래프 저장 경로
        """
        # 결과 디렉토리 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 시계열 비교
        test_dates = self.test_data['date'].values
        axes[0].plot(test_dates, y_test.values, label='실제 값', color='blue', alpha=0.7)
        axes[0].plot(test_dates, y_pred, label='예측 값', color='red', alpha=0.7)
        axes[0].set_xlabel('날짜')
        axes[0].set_ylabel('출고량')
        axes[0].set_title('상토 출고량 예측 결과 (시계열)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 산점도
        axes[1].scatter(y_test, y_pred, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                     'r--', lw=2, label='완벽한 예측선')
        axes[1].set_xlabel('실제 값')
        axes[1].set_ylabel('예측 값')
        axes[1].set_title('실제 값 vs 예측 값')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n그래프 저장 완료: {save_path}")
        plt.close()
    
    def plot_feature_importance(self, save_path='results/feature_importance.png'):
        """
        피처 중요도 시각화
        
        Args:
            save_path: 그래프 저장 경로
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 피처 중요도 추출
        feature_importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_
        
        # 상위 20개 피처만 표시
        top_n = min(20, len(feature_names))
        indices = np.argsort(feature_importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), feature_importance[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('중요도')
        plt.title(f'피처 중요도 (상위 {top_n}개)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"피처 중요도 그래프 저장 완료: {save_path}")
        plt.close()
    
    def save_model(self, save_path='models/catboost_model.cbm'):
        """
        모델 저장
        
        Args:
            save_path: 모델 저장 경로
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model(save_path)
        print(f"\n모델 저장 완료: {save_path}")
        
        # 최적 모델도 별도로 저장
        best_model_path = 'models/best_model.cbm'
        self.model.save_model(best_model_path)
        print(f"최적 모델 저장 완료: {best_model_path}")
    
    def run_full_pipeline(self, data_path=None):
        """전체 학습 파이프라인 실행"""
        print("=" * 60)
        print("상토 수요 예측 모델 학습 파이프라인 시작")
        print("=" * 60)
        
        # 1. 데이터 로드 및 전처리
        df = self.load_and_preprocess_data()
        
        # 2. 학습/테스트 분할
        train_df, test_df = self.split_train_test(df)
        
        # 3. 피처 준비
        X_train, y_train, feature_names = self.prepare_features(train_df)
        X_test, y_test, _ = self.prepare_features(test_df)
        
        # 4. 모델 학습
        self.train_model(X_train, y_train, X_test, y_test)
        
        # 5. 모델 평가
        metrics, y_pred = self.evaluate_model(X_test, y_test)
        
        # 6. 시각화
        self.plot_predictions(y_test, y_pred)
        self.plot_feature_importance()
        
        # 7. 모델 저장
        self.save_model()
        
        print("\n" + "=" * 60)
        print("학습 파이프라인 완료!")
        print("=" * 60)
        
        return metrics


if __name__ == "__main__":
    # 학습 실행
    trainer = SoilDemandTrainer()
    metrics = trainer.run_full_pipeline()
    
    print("\n최종 성능 요약:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.2f}")
