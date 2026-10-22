"""
monte_carlo.py
==============
Generate a synthetic materials dataset via a Monte Carlo procedure that
preserves the marginal distributions, inter-descriptor correlations and
physical bounds observed in the JARVIS reference population.

Strategy
--------
1. Fit a Gaussian copula (rank-transform → standard-normal → correlated
   multivariate normal → inverse-rank-transform) to the reference data.
2. Sample from the copula and map back to physical space using the
   empirical inverse CDFs (via interpolation).
3. Apply hard physical bounds (e.g., density > 0, formation energy bounds)
   and reject unphysical draws.
4. Validate: compare marginal means / stds and Pearson correlations of
   the synthetic vs. reference sets.

Public API
----------
    generate_synthetic(ref_df, n_samples, seed) -> pd.DataFrame
    validate_synthetic(ref_df, syn_df) -> dict
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import trange

from config import (
    ALL_DESCRIPTORS,
    TARGET_COLUMN,
    MC_N_SYNTHETIC,
    MC_RANDOM_SEED,
    DATA_DIR,
)

import os


# ──────────────────────────────────────────────────────────────────────
# Physical lower bounds. Samples map through training empirical marginals, so
# arbitrary upper caps would truncate observed training values and alter the
# modeled joint distribution.
# ──────────────────────────────────────────────────────────────────────
_LOWER_BOUNDS = {
    "ehull": 0.0,
    "density": 1e-12,
    "mean_atomic_mass": 0.0,
    "mean_atomic_number": 0.0,
    "mean_covalent_radius": 0.0,
    "a": 0.0,
    "b": 0.0,
    "c": 0.0,
    "volume": 0.0,
    "n_atoms": 1.0,
    "n_elements": 1.0,
    TARGET_COLUMN: 0.0,
}


def _rank_transform(x: np.ndarray) -> np.ndarray:
    """Convert each column to approximate standard-normal via rank CDF."""
    n = x.shape[0]
    u = np.empty_like(x)
    for j in range(x.shape[1]):
        ranks = stats.rankdata(x[:, j], method="average")
        u[:, j] = stats.norm.ppf(ranks / (n + 1))
    return u


def generate_synthetic(
    ref_df: pd.DataFrame,
    n_samples: int = MC_N_SYNTHETIC,
    seed: int = MC_RANDOM_SEED,
) -> pd.DataFrame:
    """
    Monte Carlo synthetic dataset via Gaussian copula.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference semiconductor dataset (from data_loader).
    n_samples : int
        Number of synthetic records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with the same descriptor columns + target.
    """
    columns = [c for c in ALL_DESCRIPTORS + [TARGET_COLUMN] if c in ref_df.columns]
    data = ref_df[columns].dropna().values.astype(float)
    col_names = [c for c in columns]

    rng = np.random.default_rng(seed)

    # Step 1: rank-transform reference data → normal space
    z_ref = _rank_transform(data)
    # Estimate correlation in normal space
    corr = np.corrcoef(z_ref, rowvar=False)
    # Ensure positive-definiteness
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr, 1.0)

    # Step 2: build empirical inverse CDFs via sorted data
    sorted_data = np.sort(data, axis=0)
    n_ref = sorted_data.shape[0]
    quantiles = np.linspace(0, 1, n_ref)

    # Step 3: sample from the copula
    accepted = []
    n_accepted = 0
    total_tried = 0
    target_n = n_samples
    max_draws = max(target_n * 1_000, 100_000)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pbar = trange(target_n, desc="Monte Carlo sampling")
        while n_accepted < target_n:
            needed = target_n - n_accepted
            batch = min(max(int(needed * 1.3), 1000), 50_000)
            z = rng.multivariate_normal(np.zeros(len(col_names)), corr, size=batch)
            u = stats.norm.cdf(z)  # uniform marginals

            # Map to physical space via empirical inverse CDF
            synth = np.empty_like(z)
            for j in range(len(col_names)):
                synth[:, j] = np.interp(u[:, j], quantiles, sorted_data[:, j])

            # Hard physical bounds rejection
            mask = np.ones(batch, dtype=bool)
            for j, col in enumerate(col_names):
                if col in _LOWER_BOUNDS:
                    mask &= synth[:, j] >= _LOWER_BOUNDS[col]

            good = synth[mask]
            if len(good) > 0:
                take = min(len(good), needed)
                accepted.append(good[:take])
                n_accepted += take
                pbar.update(take)
            total_tried += batch
            if total_tried > max_draws:
                pbar.close()
                raise RuntimeError(
                    "MCS rejection sampling accepted too few rows within "
                    f"{max_draws:,} draws. Check training data and configured bounds."
                )

        pbar.close()

    synth_all = np.vstack(accepted)[:target_n]

    df_synth = pd.DataFrame(synth_all, columns=col_names)
    # Round n_atoms to integer
    if "n_atoms" in df_synth.columns:
        df_synth["n_atoms"] = df_synth["n_atoms"].round().astype(int)

    print(f"[monte_carlo] Generated {len(df_synth):,} synthetic samples "
          f"(tried {total_tried:,} total).")
    return df_synth


def validate_synthetic(ref_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict:
    """
    Compare marginal statistics and correlations of the synthetic set
    against the reference set.

    Returns
    -------
    dict with keys:
        'marginals'   : pd.DataFrame  — mean/std/median comparison per column
        'corr_ref'    : pd.DataFrame  — reference Pearson correlation
        'corr_syn'    : pd.DataFrame  — synthetic Pearson correlation
        'corr_diff'   : pd.DataFrame  — |ref − syn| correlation difference
        'max_corr_err': float          — max absolute correlation error
    """
    columns = [c for c in ALL_DESCRIPTORS + [TARGET_COLUMN]
               if c in ref_df.columns and c in syn_df.columns]

    stats_rows = []
    for col in columns:
        r = ref_df[col].dropna()
        s = syn_df[col].dropna()
        stats_rows.append({
            "feature": col,
            "ref_mean": r.mean(), "syn_mean": s.mean(),
            "ref_std": r.std(),  "syn_std": s.std(),
            "ref_median": r.median(), "syn_median": s.median(),
            "mean_pct_err": abs(r.mean() - s.mean()) / (abs(r.mean()) + 1e-9) * 100,
        })
    marginals = pd.DataFrame(stats_rows)

    corr_ref = ref_df[columns].corr()
    corr_syn = syn_df[columns].corr()
    corr_diff = (corr_ref - corr_syn).abs()

    result = {
        "marginals": marginals,
        "corr_ref": corr_ref,
        "corr_syn": corr_syn,
        "corr_diff": corr_diff,
        "max_corr_err": float(corr_diff.max().max()),
    }

    print(f"[validate] Max absolute correlation error: "
          f"{result['max_corr_err']:.4f}")
    print("[validate] Marginal comparison:")
    print(marginals.to_string(index=False))
    return result


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_reference_data

    ref = load_reference_data()
    syn = generate_synthetic(ref)
    syn.to_csv(os.path.join(DATA_DIR, "synthetic_semiconductors.csv"), index=False)
    _ = validate_synthetic(ref, syn)
