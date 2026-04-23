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
