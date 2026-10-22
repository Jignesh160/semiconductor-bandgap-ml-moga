"""
main_pipeline.py
================
End-to-end orchestration of the semiconductor band gap prediction and
optimisation framework.

Stages
------
1. Load & explore JARVIS reference data
2. Monte Carlo synthetic dataset generation & validation
3. Feature engineering
4. Model training & comparison (on reference data)
5. Training on combined (reference + synthetic) data
6. SHAP explainability
7. Uncertainty quantification
8. Multi-objective optimisation (MOGA)
9. Summary report
"""

import os
import sys
import warnings
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Project modules
from config import (
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    MODEL_PARAMS, ALL_DESCRIPTORS,
    UQ_N_BOOTSTRAPS, UQ_CONFIDENCE, MC_RANDOM_SEED,
)
from data_loader import load_reference_data
from data_integrity import split_real_records
from monte_carlo import generate_synthetic, validate_synthetic
from feature_engineering import engineer_features, get_feature_columns
from models import train_evaluate, get_best_model, evaluate
from explainability import (
    shap_analysis, plot_shap_summary, plot_shap_bar,
    bootstrap_uncertainty, plot_uncertainty,
)
from moga import run_moga, extract_pareto
from visualisation import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_pairwise,
    plot_mc_marginals,
    plot_mc_correlation_diff,
    plot_model_comparison,
    plot_predictions_vs_true,
    plot_pareto_front,
    plot_pareto_bandgap_dist,
    plot_train_accuracy,
    plot_test_accuracy,
    plot_train_vs_test_comparison,
)


def _divider(title: str):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main():
    # ──────────────────────────────────────────────────────────────────
    # Stage 1: Load reference data
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 1 — Load JARVIS Reference Data")
    ref_df = load_reference_data(force_reload=False)
    print(f"  Shape: {ref_df.shape}")
    print(f"  Band gap range: [{ref_df[TARGET_COLUMN].min():.3f}, "
          f"{ref_df[TARGET_COLUMN].max():.3f}] eV")
    print(f"  Band gap mean:  {ref_df[TARGET_COLUMN].mean():.3f} eV")

    # Exploratory plots
    plot_target_distribution(ref_df, tag="reference")
    plot_correlation_matrix(ref_df, tag="reference")
    try:
        plot_pairwise(ref_df)
    except Exception as e:
        print(f"  [warn] Pairplot skipped: {e}")

    # Split verified real records before fitting the sampler or any model.
    ref_train, ref_test = split_real_records(
        ref_df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"  Real records — train: {len(ref_train):,}; test: {len(ref_test):,}")

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: Fit and sample MCS from training records only
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 2 — Training-Only Monte Carlo Sampling")
    syn_df = generate_synthetic(ref_train)
    syn_df["record_type"] = "synthetic_mcs"
    syn_df["generator"] = "Gaussian copula fitted to real training partition"
    syn_df["generator_seed"] = MC_RANDOM_SEED
    syn_df.to_csv(os.path.join(DATA_DIR, "synthetic_semiconductors.csv"),
                  index=False)

    val = validate_synthetic(ref_train, syn_df)
    val["marginals"].to_csv(
        os.path.join(RESULTS_DIR, "mc_validation_marginals.csv"), index=False)
    mc_cols = [c for c in ALL_DESCRIPTORS + [TARGET_COLUMN]
               if c in ref_train.columns and c in syn_df.columns]
    plot_mc_marginals(ref_train, syn_df, mc_cols)
    plot_mc_correlation_diff(val["corr_diff"])
    plot_target_distribution(syn_df, tag="synthetic")

    # ──────────────────────────────────────────────────────────────────
    # Stage 3: Deterministic feature engineering
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 3 — Feature Engineering")
    train_eng = engineer_features(ref_train)
    test_eng = engineer_features(ref_test)
    syn_eng = engineer_features(syn_df)
    feature_cols = get_feature_columns(train_eng)
    print(f"  Feature columns ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"    • {f}")

    # ──────────────────────────────────────────────────────────────────
    # Stage 4: Real-only baseline on the locked real test set
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 4 — Real-Only Model Training")
    X_train = train_eng[feature_cols].values
    y_train = train_eng[TARGET_COLUMN].values
    X_test = test_eng[feature_cols].values
    y_test = test_eng[TARGET_COLUMN].values
    print(f"  Train: {len(y_train):,}   Untouched real test: {len(y_test):,}")

    results_ref, trained_ref = train_evaluate(X_train, y_train, X_test, y_test)
    results_ref.to_csv(os.path.join(RESULTS_DIR, "model_comparison_reference.csv"))
    print("\n  Reference-data model comparison:")
    print(results_ref.to_string())

    plot_model_comparison(results_ref)

    best_name, best_model = get_best_model(trained_ref, results_ref)
    print(f"\n  ★ Best model (reference): {best_name}  "
          f"(R²={results_ref.loc[best_name, 'R2']:.4f})")

    y_pred_best = best_model.predict(X_test)
    plot_predictions_vs_true(y_test, y_pred_best, best_name)

    # ── NEW: Train and Test accuracy figures ──
    print("\n  Generating train/test accuracy figures …")
    plot_train_accuracy(results_ref, trained_ref, X_train, y_train)
    plot_test_accuracy(results_ref, trained_ref, X_test, y_test)
    plot_train_vs_test_comparison(results_ref)

    # ──────────────────────────────────────────────────────────────────
    # Stage 5: Real training rows + MCS rows, same locked real test set
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 5 — Controlled Real-Only vs Real+MCS Comparison")
    X_synthetic = syn_eng[feature_cols].values
    y_synthetic = syn_eng[TARGET_COLUMN].values
    X_augmented_train = np.concatenate([X_train, X_synthetic], axis=0)
    y_augmented_train = np.concatenate([y_train, y_synthetic], axis=0)
    results_comb, trained_comb = train_evaluate(
        X_augmented_train, y_augmented_train, X_test, y_test)
    results_comb.to_csv(os.path.join(RESULTS_DIR, "model_comparison_combined.csv"))
    print("\n  Combined-data model comparison:")
    print(results_comb.to_string())

    best_name_c, best_model_c = get_best_model(trained_comb, results_comb)
    print(f"\n  ★ Best model (combined): {best_name_c}  "
          f"(R²={results_comb.loc[best_name_c, 'R2']:.4f})")

    # ──────────────────────────────────────────────────────────────────
    # Stage 6: SHAP explainability (best model on reference split)
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 6 — SHAP Explainability")
    print(f"  Computing SHAP values for {best_name} …")
    sv = shap_analysis(best_model, X_test, feature_cols)
    plot_shap_summary(sv, X_test, feature_cols)
    plot_shap_bar(sv, feature_cols)

    # ──────────────────────────────────────────────────────────────────
    # Stage 7: Uncertainty quantification
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 7 — Uncertainty Quantification")

    # Determine model class & params for bootstrapping
    if best_name in MODEL_PARAMS:
        params = MODEL_PARAMS[best_name]
    else:
        params = {}

    # Map name to class
    _cls_map = {
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
    }
    if best_name in _cls_map:
        model_cls = _cls_map[best_name]
    else:
        # Default to RF for bootstrap UQ
        print(f"  [info] Using RandomForest for bootstrap UQ "
              f"(best model was {best_name})")
        model_cls = RandomForestRegressor
        params = MODEL_PARAMS["RandomForest"]

    print(f"  Running {UQ_N_BOOTSTRAPS} bootstrap iterations …")
    uq = bootstrap_uncertainty(
        model_cls, params,
        X_train, y_train, X_test,
        n_bootstraps=UQ_N_BOOTSTRAPS,
        confidence=UQ_CONFIDENCE,
    )
    plot_uncertainty(y_test, uq)

    coverage = np.mean((y_test >= uq["lower"]) & (y_test <= uq["upper"]))
    mean_width = np.mean(uq["upper"] - uq["lower"])
    print(f"  {int(UQ_CONFIDENCE*100)}% CI coverage: {coverage:.3f}")
    print(f"  Mean interval width: {mean_width:.3f} eV")
    print(f"  Mean prediction std: {uq['std'].mean():.3f} eV")

    # ──────────────────────────────────────────────────────────────────
    # Stage 8: Multi-Objective Optimisation (MOGA)
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 8 — Multi-Objective Genetic Algorithm (MOGA)")
    moga_result = run_moga(best_model, feature_cols, train_eng, verbose=False)

    pareto_df = extract_pareto(moga_result, train_eng, best_model, feature_cols)
    pareto_path = os.path.join(RESULTS_DIR, "pareto_solutions.csv")
    pareto_df.to_csv(pareto_path, index=False)
    print(f"  Pareto solutions saved → {pareto_path}")
    print(f"  Predicted band gap range in Pareto set: "
          f"[{pareto_df['predicted_bandgap'].min():.3f}, "
          f"{pareto_df['predicted_bandgap'].max():.3f}] eV")

    plot_pareto_front(moga_result, pareto_df=pareto_df)
    plot_pareto_bandgap_dist(pareto_df)

    # ──────────────────────────────────────────────────────────────────
    # Stage 9: Summary report
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 9 — Summary")
    summary = {
        "reference_records": int(len(ref_df)),
        "synthetic_records": int(len(syn_df)),
        "real_train_records": int(len(train_eng)),
        "real_test_records": int(len(test_eng)),
        "augmented_test_is_same_real_test": True,
        "synthetic_generator_fit_partition": "real training records only",
        "target_units": "eV",
        "features_used": len(feature_cols),
        "best_model_reference": best_name,
        "best_R2_reference": float(results_ref.loc[best_name, "R2"]),
        "best_MAE_reference": float(results_ref.loc[best_name, "MAE"]),
        "best_RMSE_reference": float(results_ref.loc[best_name, "RMSE"]),
        "best_train_R2_reference": float(results_ref.loc[best_name, "train_R2"]),
        "best_model_combined": best_name_c,
        "best_R2_combined": float(results_comb.loc[best_name_c, "R2"]),
        "best_MAE_combined": float(results_comb.loc[best_name_c, "MAE"]),
        "best_RMSE_combined": float(results_comb.loc[best_name_c, "RMSE"]),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "monte_carlo_seed": MC_RANDOM_SEED,
        "uq_coverage": float(coverage),
        "uq_mean_width_eV": float(mean_width),
        "uq_calibration_method": "bootstrap out-of-bag residuals",
        "uq_calibration_residual_count": int(uq["calibration_residual_count"]),
        "mc_max_corr_error": val["max_corr_err"],
        "pareto_front_size": int(len(pareto_df)),
    }

    summary_path = os.path.join(RESULTS_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Pipeline summary saved → {summary_path}")

    print("\n  Key results:")
    for k, v in summary.items():
        print(f"    {k:30s} : {v}")

    print(f"\n{'='*70}")
    print("  ✓  Pipeline complete. All figures in: " + FIGURES_DIR)
    print("  ✓  All results in: " + RESULTS_DIR)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
