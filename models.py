"""
models.py
=========
Train, evaluate and compare regression models for band gap prediction.

Models: Linear Regression, Random Forest, Gradient Boosting, XGBoost, SVR, MLP.
Metrics: MAE, RMSE, R².

Public API
----------
    build_models() -> dict[str, BaseEstimator]
    train_evaluate(X_train, y_train, X_test, y_test) -> pd.DataFrame
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import MODEL_PARAMS, RANDOM_STATE


def build_models() -> dict:
    """
    Instantiate all regression models with config hyper-parameters.

    Returns
    -------
    dict  {name: sklearn-compatible estimator}
    """
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LinearRegression(**MODEL_PARAMS["LinearRegression"])),
        ]),
        "RandomForest": RandomForestRegressor(
            **MODEL_PARAMS["RandomForest"]
        ),
        "GradientBoosting": GradientBoostingRegressor(
            **MODEL_PARAMS["GradientBoosting"]
        ),
        "XGBoost": XGBRegressor(
            **MODEL_PARAMS["XGBoost"]
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(**MODEL_PARAMS["SVR"])),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(**MODEL_PARAMS["MLP"])),
        ]),
    }
    return models


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² for a prediction set."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """
    Train every model and evaluate on both train and test sets.

    Returns
    -------
    results_df : pd.DataFrame
        Rows = models, columns = test MAE, RMSE, R²,
        train MAE, RMSE, R², and train_time_s.
    trained : dict
        {name: fitted estimator}
    """
    models = build_models()
    rows = []
    trained = {}

    for name, model in models.items():
        print(f"  Training {name} …", end=" ", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # Test set evaluation
        y_pred_test = model.predict(X_test)
        test_metrics = evaluate(y_test, y_pred_test)

        # Train set evaluation
        y_pred_train = model.predict(X_train)
        train_metrics = evaluate(y_train, y_pred_train)

        row = {
            "model": name,
            "MAE": test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "R2": test_metrics["R2"],
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "train_time_s": round(train_time, 2),
        }
        rows.append(row)
        trained[name] = model

        print(f"R²={test_metrics['R2']:.4f}  MAE={test_metrics['MAE']:.4f}  "
              f"Train-R²={train_metrics['R2']:.4f}  ({train_time:.1f}s)")

    results_df = pd.DataFrame(rows).set_index("model")
    results_df = results_df.sort_values("R2", ascending=False)
    return results_df, trained


def get_best_model(trained: dict, results_df: pd.DataFrame):
    """Return (name, model) of the model with highest R²."""
    best_name = results_df["R2"].idxmax()
    return best_name, trained[best_name]


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_reference_data
    from feature_engineering import engineer_features, get_feature_columns
    from sklearn.model_selection import train_test_split
    from config import TEST_SIZE, TARGET_COLUMN

    ref = load_reference_data()
    ref = engineer_features(ref)
    feats = get_feature_columns(ref)

    X = ref[feats].values
    y = ref[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results, trained = train_evaluate(X_train, y_train, X_test, y_test)
    print("\n", results)
