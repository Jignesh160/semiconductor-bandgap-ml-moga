"""
visualisation.py
================
Publication-quality plots for every stage of the pipeline.

All figures are saved to FIGURES_DIR (``./figures/``).
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR, PLOT_DPI, PLOT_FIGSIZE, PLOT_STYLE, TARGET_COLUMN

try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8")


# ──────────────────────────────────────────────────────────────────────
# 1.  Reference data exploration
# ──────────────────────────────────────────────────────────────────────

def plot_target_distribution(df: pd.DataFrame, tag: str = "reference"):
    """Histogram + KDE of the band gap distribution."""
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    sns.histplot(df[TARGET_COLUMN], bins=80, kde=True, ax=ax,
                 color="steelblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Band gap (eV)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Band Gap Distribution — {tag}", fontsize=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"bandgap_dist_{tag}.png")
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_correlation_matrix(df: pd.DataFrame, tag: str = "reference"):
    """Heatmap of Pearson correlations between descriptors."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
                linewidths=0.3, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"Feature Correlation Matrix — {tag}", fontsize=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"correlation_{tag}.png")
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_pairwise(df: pd.DataFrame, features: list[str] | None = None):
    """Pair plot of the top descriptors coloured by band gap quartile."""
    if features is None:
        features = ["mean_atomic_mass", "mean_covalent_radius",
                     "density", "formation_energy_peratom", TARGET_COLUMN]
    sub = df[features].dropna().copy()
    sub["bg_quartile"] = pd.qcut(sub[TARGET_COLUMN], 4,
                                  labels=["Q1", "Q2", "Q3", "Q4"])
    g = sns.pairplot(sub, hue="bg_quartile", palette="viridis",
                     diag_kind="kde", plot_kws={"alpha": 0.4, "s": 10})
    g.fig.suptitle("Pair Plot of Key Descriptors", y=1.02, fontsize=15)
    path = os.path.join(FIGURES_DIR, "pairplot.png")
    g.savefig(path, dpi=PLOT_DPI)
    plt.close(g.fig)
    print(f"[plot] {path}")


# ──────────────────────────────────────────────────────────────────────
# 2.  Monte Carlo validation
# ──────────────────────────────────────────────────────────────────────

def plot_mc_marginals(ref: pd.DataFrame, syn: pd.DataFrame, columns: list[str]):
    """Overlay histograms: reference vs synthetic for each column."""
    n = len(columns)
    cols_per_row = 3
    nrows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(nrows, cols_per_row, figsize=(5 * cols_per_row, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.hist(ref[col].dropna(), bins=60, density=True, alpha=0.5,
                label="Reference", color="steelblue", edgecolor="white")
        ax.hist(syn[col].dropna(), bins=60, density=True, alpha=0.5,
                label="Synthetic", color="coral", edgecolor="white")
        ax.set_title(col, fontsize=11)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Monte Carlo Validation: Marginal Distributions", fontsize=15, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "mc_marginals.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def plot_mc_correlation_diff(corr_diff: pd.DataFrame):
    """Heatmap of |Δcorrelation| between reference and synthetic."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_diff, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.3, ax=ax, vmin=0, vmax=0.15,
                cbar_kws={"shrink": 0.8, "label": "|Δr|"})
    ax.set_title("Monte Carlo Correlation Error  |r_ref − r_syn|", fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "mc_corr_diff.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ──────────────────────────────────────────────────────────────────────
# 3.  Model comparison
# ──────────────────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame):
    """Bar chart of MAE, RMSE, R² for every model."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["MAE", "RMSE", "R2"]
    colours = ["#2196F3", "#FF9800", "#4CAF50"]

    for ax, metric, colour in zip(axes, metrics, colours):
        results_df[metric].plot.barh(ax=ax, color=colour, edgecolor="white")
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(metric, fontsize=14)
        ax.invert_yaxis()

    fig.suptitle("Model Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def plot_predictions_vs_true(y_true, y_pred, model_name: str):
    """Scatter plot: predicted vs true band gap."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("True Band Gap (eV)", fontsize=13)
    ax.set_ylabel("Predicted Band Gap (eV)", fontsize=13)
    ax.set_title(f"{model_name} — Predicted vs True", fontsize=15)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"pred_vs_true_{model_name}.png")
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[plot] {path}")


# ──────────────────────────────────────────────────────────────────────
# 3b. Train vs Test Accuracy Figures
# ──────────────────────────────────────────────────────────────────────

def plot_train_accuracy(results_df: pd.DataFrame, trained: dict,
                        X_train: np.ndarray, y_train: np.ndarray):
    """
    Generate a multi-panel figure showing training accuracy for all models.
    Panel 1: Parity plots (predicted vs true) for each model on TRAIN data.
    Panel 2: Bar chart comparing train R², MAE, RMSE across models.
    """
    model_names = list(results_df.index)
    n_models = len(model_names)

    # ── Panel 1: Parity plots for each model on training data ──
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(model_names):
        ax = axes[i]
        model = trained[name]
        y_pred = model.predict(X_train)

        ax.scatter(y_train, y_pred, alpha=0.15, s=6, color="#4CAF50", rasterized=True)
        lo = min(y_train.min(), y_pred.min())
        hi = max(y_train.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")

        r2 = results_df.loc[name, "train_R2"]
        mae = results_df.loc[name, "train_MAE"]
        ax.set_title(f"{name}\nTrain R²={r2:.4f}  MAE={mae:.3f} eV", fontsize=12)
        ax.set_xlabel("True Band Gap (eV)", fontsize=10)
        ax.set_ylabel("Predicted Band Gap (eV)", fontsize=10)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Training Data Accuracy — All Models", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "accuracy_train.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")

    # ── Panel 2: Bar chart of train metrics ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    train_metrics = results_df[["train_MAE", "train_RMSE", "train_R2"]].copy()
    train_metrics.columns = ["MAE", "RMSE", "R²"]
    colours = ["#66BB6A", "#FFA726", "#42A5F5"]
    titles = ["Train MAE (eV)", "Train RMSE (eV)", "Train R²"]

    for ax, col, colour, title in zip(axes, train_metrics.columns, colours, titles):
        train_metrics[col].plot.barh(ax=ax, color=colour, edgecolor="white")
        ax.set_xlabel(col, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        # Add value labels
        for idx, v in enumerate(train_metrics[col]):
            ax.text(v + 0.005, idx, f"{v:.4f}", va="center", fontsize=9)

    fig.suptitle("Training Accuracy Metrics — All Models", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "accuracy_train_bars.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def plot_test_accuracy(results_df: pd.DataFrame, trained: dict,
                       X_test: np.ndarray, y_test: np.ndarray):
    """
    Generate a multi-panel figure showing test accuracy for all models.
    Panel 1: Parity plots (predicted vs true) for each model on TEST data.
    Panel 2: Bar chart comparing test R², MAE, RMSE across models.
    """
    model_names = list(results_df.index)
    n_models = len(model_names)

    # ── Panel 1: Parity plots for each model on test data ──
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(model_names):
        ax = axes[i]
        model = trained[name]
        y_pred = model.predict(X_test)

        ax.scatter(y_test, y_pred, alpha=0.15, s=6, color="#2196F3", rasterized=True)
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")

        r2 = results_df.loc[name, "R2"]
        mae = results_df.loc[name, "MAE"]
        ax.set_title(f"{name}\nTest R²={r2:.4f}  MAE={mae:.3f} eV", fontsize=12)
        ax.set_xlabel("True Band Gap (eV)", fontsize=10)
        ax.set_ylabel("Predicted Band Gap (eV)", fontsize=10)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Test Data Accuracy — All Models", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "accuracy_test.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")

    # ── Panel 2: Bar chart of test metrics ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    test_metrics = results_df[["MAE", "RMSE", "R2"]].copy()
    test_metrics.columns = ["MAE", "RMSE", "R²"]
    colours = ["#2196F3", "#FF9800", "#4CAF50"]
    titles = ["Test MAE (eV)", "Test RMSE (eV)", "Test R²"]

    for ax, col, colour, title in zip(axes, test_metrics.columns, colours, titles):
        test_metrics[col].plot.barh(ax=ax, color=colour, edgecolor="white")
        ax.set_xlabel(col, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        # Add value labels
        for idx, v in enumerate(test_metrics[col]):
            ax.text(v + 0.005, idx, f"{v:.4f}", va="center", fontsize=9)

    fig.suptitle("Test Accuracy Metrics — All Models", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "accuracy_test_bars.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


def plot_train_vs_test_comparison(results_df: pd.DataFrame):
    """
    Side-by-side comparison of train vs test R² and MAE for all models.
    Helps identify overfitting (large train-test gap).
    """
    model_names = list(results_df.index)
    x_pos = np.arange(len(model_names))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # R² comparison
    ax = axes[0]
    ax.bar(x_pos - bar_width/2, results_df["train_R2"], bar_width,
           label="Train R²", color="#66BB6A", edgecolor="white")
    ax.bar(x_pos + bar_width/2, results_df["R2"], bar_width,
           label="Test R²", color="#2196F3", edgecolor="white")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Train vs Test R²", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    # MAE comparison
    ax = axes[1]
    ax.bar(x_pos - bar_width/2, results_df["train_MAE"], bar_width,
           label="Train MAE", color="#66BB6A", edgecolor="white")
    ax.bar(x_pos + bar_width/2, results_df["MAE"], bar_width,
           label="Test MAE", color="#2196F3", edgecolor="white")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("MAE (eV)", fontsize=12)
    ax.set_title("Train vs Test MAE", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=10)

    fig.suptitle("Train vs Test Performance — Overfitting Check", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "train_vs_test_comparison.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ──────────────────────────────────────────────────────────────────────
# 4.  MOGA Pareto front
# ──────────────────────────────────────────────────────────────────────

def plot_pareto_front(result, pareto_df: pd.DataFrame | None = None):
    """Scatter of the Pareto front: Obj 1 (|Δbg|) vs Obj 2 (ehull)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if pareto_df is not None and len(pareto_df) > 0:
        x_vals = pareto_df["obj_bg_deviation"]
        y_vals = pareto_df["obj_ehull"]
        sc = ax.scatter(x_vals, y_vals, c=pareto_df["predicted_bandgap"], cmap="viridis",
                        s=55, edgecolors="k", linewidths=0.6, alpha=0.9)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Predicted Band Gap (eV)", fontsize=11)
    else:
        F = result.F
        ax.scatter(F[:, 0], F[:, 1], s=40, c="darkorange", edgecolors="k",
                   linewidths=0.5, alpha=0.8)
    ax.set_xlabel("|Band Gap − Target| (eV)", fontsize=13)
    ax.set_ylabel("Energy Above Hull (eV/atom)", fontsize=13)
    ax.set_title("MOGA Pareto Optimal Frontier", fontsize=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pareto_front.png")
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_pareto_bandgap_dist(pareto_df: pd.DataFrame):
    """Distribution of predicted band gaps in the Pareto set."""
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.hist(pareto_df["predicted_bandgap"], bins=30, color="teal",
            edgecolor="white", alpha=0.8)
    ax.set_xlabel("Predicted Band Gap (eV)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Pareto-Optimal Predicted Band Gap Distribution", fontsize=15)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pareto_bg_dist.png")
    fig.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"[plot] {path}")
