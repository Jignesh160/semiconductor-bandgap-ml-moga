"""
explainability.py
=================
SHAP-based model interpretability and uncertainty quantification.

Public API
----------
    shap_analysis(model, X, feature_names) -> shap.Explanation
    plot_shap_summary(shap_values, X, feature_names, save_path)
    plot_shap_dependence(shap_values, X, feature_names, top_n, save_dir)
    bootstrap_uncertainty(model_cls, model_params, X_train, y_train, X_test,
                          n_bootstraps, confidence) -> dict
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from config import (
    SHAP_MAX_SAMPLES,
    SHAP_TOP_FEATURES,
    UQ_N_BOOTSTRAPS,
    UQ_CONFIDENCE,
    FIGURES_DIR,
    PLOT_DPI,
)


# ──────────────────────────────────────────────────────────────────────
# SHAP analysis
# ──────────────────────────────────────────────────────────────────────

def shap_analysis(model, X: np.ndarray, feature_names: list[str]):
    """
    Compute SHAP values for the given model.

    Uses TreeExplainer for tree-based models, KernelExplainer otherwise.
    Sub-samples X to SHAP_MAX_SAMPLES for speed.

    Returns
    -------
    shap.Explanation
    """
    n = min(SHAP_MAX_SAMPLES, X.shape[0])
    idx = np.random.RandomState(42).choice(X.shape[0], n, replace=False)
    X_sample = X[idx]

    model_type = type(model).__name__
    # Unwrap Pipeline if needed
    inner = model
    if hasattr(model, "named_steps"):
        # Pipeline: apply scaler, get final estimator
        step_names = list(model.named_steps.keys())
        if "scaler" in model.named_steps:
            X_sample = model.named_steps["scaler"].transform(X_sample)
        inner = model.named_steps[step_names[-1]]
        model_type = type(inner).__name__

    tree_types = {
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "XGBRegressor",
    }

    if model_type in tree_types:
        explainer = shap.TreeExplainer(inner)
        sv = explainer(X_sample)
    elif model_type in {"LinearRegression", "Ridge", "Lasso", "ElasticNet"}:
        explainer = shap.LinearExplainer(inner, X_sample)
        sv = explainer(X_sample)
    else:
        # KernelExplainer for non-linear / SVR
        background = shap.sample(X_sample, min(50, n))
        explainer = shap.KernelExplainer(inner.predict, background)
        raw = explainer.shap_values(X_sample)
        sv = shap.Explanation(
            values=raw,
            base_values=np.full(n, explainer.expected_value),
            data=X_sample,
            feature_names=feature_names,
        )

    sv.feature_names = feature_names
    return sv


def plot_shap_summary(
    shap_values,
    X: np.ndarray,
    feature_names: list[str],
    save_path: str | None = None,
):
    """SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X[:shap_values.shape[0]],
        feature_names=feature_names,
        show=False,
        max_display=SHAP_TOP_FEATURES,
    )
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "shap_summary.png")
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary plot saved → {path}")


def plot_shap_bar(
    shap_values,
    feature_names: list[str],
    save_path: str | None = None,
):
    """SHAP mean-absolute bar plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(
        shap_values,
        max_display=SHAP_TOP_FEATURES,
        show=False,
    )
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, "shap_bar.png")
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Bar plot saved → {path}")


# ──────────────────────────────────────────────────────────────────────
# Uncertainty quantification — bootstrap prediction intervals
# ──────────────────────────────────────────────────────────────────────

def bootstrap_uncertainty(
    model_cls,
    model_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_bootstraps: int = UQ_N_BOOTSTRAPS,
    confidence: float = UQ_CONFIDENCE,
) -> dict:
    """
    Estimate prediction uncertainty via bootstrap aggregation.

    For each bootstrap resample of (X_train, y_train), train a fresh
    model and predict on X_test.  The ensemble of predictions gives
    a per-sample mean and confidence interval.

    Parameters
    ----------
    model_cls : type
        e.g. ``RandomForestRegressor``
    model_params : dict
        Constructor kwargs.
    X_train, y_train : arrays
    X_test : array
    n_bootstraps : int
    confidence : float  (e.g. 0.90)

    Returns
    -------
    dict with keys 'mean', 'lower', 'upper', 'std'  (all arrays of len X_test)
    """
    rng = np.random.default_rng(42)
    n_train = X_train.shape[0]
    preds = np.empty((n_bootstraps, X_test.shape[0]))

    for i in range(n_bootstraps):
        idx = rng.choice(n_train, n_train, replace=True)
        m = model_cls(**model_params)
        m.fit(X_train[idx], y_train[idx])
        preds[i] = m.predict(X_test)

    alpha = (1 - confidence) / 2
    return {
        "mean": preds.mean(axis=0),
        "lower": np.quantile(preds, alpha, axis=0),
        "upper": np.quantile(preds, 1 - alpha, axis=0),
        "std": preds.std(axis=0),
    }


def plot_uncertainty(
    y_test: np.ndarray,
    uq: dict,
    save_path: str | None = None,
    n_show: int = 200,
):
    """Plot predicted vs true with confidence intervals."""
    n = min(n_show, len(y_test))
    idx = np.argsort(y_test)[:n]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        range(n), uq["lower"][idx], uq["upper"][idx],
        alpha=0.25, color="steelblue", label=f"{int(UQ_CONFIDENCE*100)}% CI",
    )
    ax.plot(range(n), uq["mean"][idx], "o-", ms=3, color="steelblue",
            label="Predicted mean")
    ax.plot(range(n), y_test[idx], "x", ms=4, color="crimson",
            label="True")
    ax.set_xlabel("Sample (sorted by true band gap)")
    ax.set_ylabel("Band gap (eV)")
    ax.set_title("Uncertainty Quantification — Bootstrap Prediction Intervals")
    ax.legend()
    plt.tight_layout()

    path = save_path or os.path.join(FIGURES_DIR, "uncertainty.png")
    plt.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"[UQ] Uncertainty plot saved → {path}")
