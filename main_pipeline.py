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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Project modules
from config import (
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    DATA_DIR, RESULTS_DIR, FIGURES_DIR,
    MODEL_PARAMS, ALL_DESCRIPTORS,
    UQ_N_BOOTSTRAPS, UQ_CONFIDENCE,
)
from data_loader import load_reference_data
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

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: Monte Carlo synthetic data
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 2 — Monte Carlo Synthetic Data Generation")
    syn_df = generate_synthetic(ref_df)
    syn_df.to_csv(os.path.join(DATA_DIR, "synthetic_semiconductors.csv"),
                  index=False)

    val = validate_synthetic(ref_df, syn_df)
    val["marginals"].to_csv(
        os.path.join(RESULTS_DIR, "mc_validation_marginals.csv"), index=False)

    # MC validation plots
    mc_cols = [c for c in ALL_DESCRIPTORS + [TARGET_COLUMN]
               if c in ref_df.columns and c in syn_df.columns]
    plot_mc_marginals(ref_df, syn_df, mc_cols)
    plot_mc_correlation_diff(val["corr_diff"])

    plot_target_distribution(syn_df, tag="synthetic")

    # ──────────────────────────────────────────────────────────────────
    # Stage 3: Feature engineering
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 3 — Feature Engineering")
    ref_eng = engineer_features(ref_df)
    syn_eng = engineer_features(syn_df)
    feature_cols = get_feature_columns(ref_eng)
    print(f"  Feature columns ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"    • {f}")

    # ──────────────────────────────────────────────────────────────────
    # Stage 4: Model training & comparison (reference data only)
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 4 — Model Training (Reference Data)")
    X_ref = ref_eng[feature_cols].values
    y_ref = ref_eng[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_ref, y_ref, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")

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
    # Stage 5: Train on combined (reference + synthetic)
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 5 — Training on Combined Data")
    # Ensure synthetic has same feature columns
    for col in feature_cols:
        if col not in syn_eng.columns:
            syn_eng[col] = 0.0

    combined = pd.concat([ref_eng, syn_eng], ignore_index=True)
    X_comb = combined[feature_cols].values
    y_comb = combined[TARGET_COLUMN].values

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_comb, y_comb, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"  Combined train: {X_tr_c.shape[0]:,}   Test: {X_te_c.shape[0]:,}")

    results_comb, trained_comb = train_evaluate(X_tr_c, y_tr_c, X_te_c, y_te_c)
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
    moga_result = run_moga(best_model, feature_cols, ref_eng, verbose=False)

    pareto_df = extract_pareto(moga_result, ref_eng, best_model, feature_cols)
    pareto_path = os.path.join(RESULTS_DIR, "pareto_solutions.csv")
    pareto_df.to_csv(pareto_path, index=False)
    print(f"  Pareto solutions saved → {pareto_path}")
    print(f"  Predicted band gap range in Pareto set: "
          f"[{pareto_df['predicted_bandgap'].min():.3f}, "
          f"{pareto_df['predicted_bandgap'].max():.3f}] eV")

    plot_pareto_front(moga_result)
    plot_pareto_bandgap_dist(pareto_df)

    # ──────────────────────────────────────────────────────────────────
    # Stage 9: Summary report
    # ──────────────────────────────────────────────────────────────────
    _divider("Stage 9 — Summary")
    summary = {
        "reference_records": int(len(ref_df)),
        "synthetic_records": int(len(syn_df)),
        "features_used": len(feature_cols),
        "best_model_reference": best_name,
        "best_R2_reference": float(results_ref.loc[best_name, "R2"]),
        "best_MAE_reference": float(results_ref.loc[best_name, "MAE"]),
        "best_RMSE_reference": float(results_ref.loc[best_name, "RMSE"]),
        "best_train_R2_reference": float(results_ref.loc[best_name, "train_R2"]),
        "best_model_combined": best_name_c,
        "best_R2_combined": float(results_comb.loc[best_name_c, "R2"]),
        "uq_coverage": float(coverage),
        "uq_mean_width_eV": float(mean_width),
        "mc_max_corr_error": val["max_corr_err"],
        "pareto_front_size": int(len(moga_result.F)),
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
