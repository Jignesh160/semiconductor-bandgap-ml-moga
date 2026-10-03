"""
Configuration constants for the semiconductor band gap prediction framework.

Centralises all tuneable knobs — file paths, feature lists, model hyper-
parameters and MOGA settings — so that every module reads from one place.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# 1.  Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 2.  JARVIS dataset
# ──────────────────────────────────────────────────────────────────────
JARVIS_DATASET = "dft_3d"
TARGET_COLUMN = "optb88vdw_bandgap"

# Raw descriptor columns pulled directly from JARVIS
RAW_DESCRIPTORS = [
    "formation_energy_peratom",
    "ehull",
    "density",
]

# Columns we derive from the Atoms object
DERIVED_DESCRIPTORS = [
    "mean_atomic_mass",
    "mean_covalent_radius",
    "a",          # lattice parameter a
    "b",          # lattice parameter b
    "c",          # lattice parameter c
    "volume",
    "n_atoms",
]

ALL_DESCRIPTORS = RAW_DESCRIPTORS + DERIVED_DESCRIPTORS

# ──────────────────────────────────────────────────────────────────────
# 3.  Physics-informed engineered features
# ──────────────────────────────────────────────────────────────────────
ENGINEERED_FEATURES = [
    "packing_fraction",           # n_atoms * mean_vol / volume
    "mass_density_ratio",         # mean_atomic_mass / density
    "lattice_anisotropy",         # max(a,b,c) / min(a,b,c)
    "formation_energy_x_ehull",   # formation_energy * ehull interaction
    "radius_over_lattice",        # mean_covalent_radius / mean(a,b,c)
    "volume_per_atom",            # volume / n_atoms
    "density_x_radius",           # density * mean_covalent_radius
    "ehull_log1p",                # log1p(ehull) — regularised stability
]

# ──────────────────────────────────────────────────────────────────────
# 4.  Monte Carlo generation
# ──────────────────────────────────────────────────────────────────────
MC_N_SYNTHETIC = 30_000       # number of synthetic samples to generate
MC_RANDOM_SEED = 42
MC_KDE_BANDWIDTH = "silverman"

# ──────────────────────────────────────────────────────────────────────
# 5.  Train / test split
# ──────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────
# 6.  Model hyper-parameters
# ──────────────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "LinearRegression": {},
    "RandomForest": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "GradientBoosting": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "SVR": {
        "kernel": "rbf",
        "C": 10.0,
        "epsilon": 0.1,
        "gamma": "scale",
        "max_iter": 3000,
    },
}

# ──────────────────────────────────────────────────────────────────────
# 7.  SHAP / Explainability
# ──────────────────────────────────────────────────────────────────────
SHAP_MAX_SAMPLES = 500        # subsample for SHAP computation speed
SHAP_TOP_FEATURES = 10        # number of top features to display

# ──────────────────────────────────────────────────────────────────────
# 8.  Uncertainty quantification
# ──────────────────────────────────────────────────────────────────────
UQ_N_BOOTSTRAPS = 15          # bootstrap iterations for prediction intervals
UQ_CONFIDENCE = 0.90          # confidence level for intervals

# ──────────────────────────────────────────────────────────────────────
# 9.  Multi-Objective Genetic Algorithm (MOGA)
# ──────────────────────────────────────────────────────────────────────
MOGA_POP_SIZE = 200
MOGA_N_GENERATIONS = 100
MOGA_CROSSOVER_PROB = 0.9
MOGA_MUTATION_PROB = 0.1

# Target band gap range for optimisation (eV)
MOGA_TARGET_BANDGAP_MIN = 1.0
MOGA_TARGET_BANDGAP_MAX = 2.0

# Stability constraint: ehull ≤ threshold
MOGA_EHULL_THRESHOLD = 0.1   # eV/atom — thermodynamically near-stable

# ──────────────────────────────────────────────────────────────────────
# 10. Plotting style
# ──────────────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 6)
