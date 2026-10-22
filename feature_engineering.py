"""
feature_engineering.py
======================
Create physics-informed features from the raw descriptors.

The goal is to give the ML models physically meaningful interaction
terms and non-linear transforms that encode domain knowledge about
semiconductor band gap behaviour.

Public API
----------
    engineer_features(df) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from config import ALL_DESCRIPTORS, TARGET_COLUMN


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add physics-informed engineered features in-place and return
    the augmented DataFrame.

    New columns
    -----------
    packing_fraction       : n_atoms * (4/3 π r³_cov) / volume
    mass_density_ratio     : mean_atomic_mass / density
    lattice_anisotropy     : max(a,b,c) / min(a,b,c)
    formation_energy_x_ehull : formation_energy_peratom * ehull
    radius_over_lattice    : mean_covalent_radius / mean(a,b,c)
    volume_per_atom        : volume / n_atoms
    density_x_radius       : density * mean_covalent_radius
    ehull_log1p            : log(1 + ehull)
    inv_mean_mass          : 1 / mean_atomic_mass
    lattice_ratio_ab       : a / b
    lattice_ratio_ac       : a / c
    formation_energy_sq    : formation_energy_peratom²
    electronegativity_range: max_electronegativity - min_electronegativity
    mass_x_en              : mean_atomic_mass * mean_electronegativity
    en_over_radius         : mean_electronegativity / mean_covalent_radius
    density_x_volume       : density * volume (total mass proxy)
    formation_energy_abs   : |formation_energy_peratom|
    radius_sq              : mean_covalent_radius²
    z_over_radius          : mean_atomic_number / mean_covalent_radius
    density_over_mass      : density / mean_atomic_mass
    lattice_volume_ratio   : (a*b*c) / volume
    en_x_formation         : mean_electronegativity * formation_energy
    log_volume_per_atom    : log(volume / n_atoms)
    sqrt_density           : sqrt(density)
    """
    out = df.copy()

    # Packing fraction (approximation using mean covalent radius as sphere)
    if {"n_atoms", "mean_covalent_radius", "volume"}.issubset(out.columns):
        sphere_vol = (4 / 3) * np.pi * out["mean_covalent_radius"] ** 3
        out["packing_fraction"] = (out["n_atoms"] * sphere_vol) / out["volume"]

    # Mass-to-density ratio
    if {"mean_atomic_mass", "density"}.issubset(out.columns):
        positive_density = out["density"].where(out["density"] > 0)
        out["mass_density_ratio"] = out["mean_atomic_mass"] / positive_density

    # Lattice anisotropy
    if {"a", "b", "c"}.issubset(out.columns):
        abc = out[["a", "b", "c"]]
        out["lattice_anisotropy"] = abc.max(axis=1) / (abc.min(axis=1) + 1e-9)

    # Interaction: formation energy × energy above hull
    if {"formation_energy_peratom", "ehull"}.issubset(out.columns):
        out["formation_energy_x_ehull"] = (
            out["formation_energy_peratom"] * out["ehull"]
        )

    # Covalent radius relative to mean lattice parameter
    if {"mean_covalent_radius", "a", "b", "c"}.issubset(out.columns):
        mean_lat = (out["a"] + out["b"] + out["c"]) / 3
        out["radius_over_lattice"] = out["mean_covalent_radius"] / (mean_lat + 1e-9)

    # Volume per atom
    if {"volume", "n_atoms"}.issubset(out.columns):
        out["volume_per_atom"] = out["volume"] / (out["n_atoms"] + 1e-9)

    # Density × radius
    if {"density", "mean_covalent_radius"}.issubset(out.columns):
        out["density_x_radius"] = out["density"] * out["mean_covalent_radius"]

    # Log-transformed ehull (regularises the long tail)
    if "ehull" in out.columns:
        out["ehull_log1p"] = np.log1p(out["ehull"].clip(lower=0))

    # Inverse mean atomic mass — lighter atoms tend toward wider gaps
    if "mean_atomic_mass" in out.columns:
        out["inv_mean_mass"] = 1.0 / (out["mean_atomic_mass"] + 1e-9)

    # Lattice ratios
    if {"a", "b", "c"}.issubset(out.columns):
        out["lattice_ratio_ab"] = out["a"] / (out["b"] + 1e-9)
        out["lattice_ratio_ac"] = out["a"] / (out["c"] + 1e-9)

    # Squared formation energy — captures non-linear stability signal
    if "formation_energy_peratom" in out.columns:
        out["formation_energy_sq"] = out["formation_energy_peratom"] ** 2

    # ── NEW Magpie-inspired and physics features ──────────────────────

    # Electronegativity range (max - min) → drives ionic character
    if {"max_electronegativity", "min_electronegativity"}.issubset(out.columns):
        out["electronegativity_range"] = (
            out["max_electronegativity"] - out["min_electronegativity"]
        )

    # Mass × electronegativity interaction
    if {"mean_atomic_mass", "mean_electronegativity"}.issubset(out.columns):
        out["mass_x_en"] = out["mean_atomic_mass"] * out["mean_electronegativity"]

    # Electronegativity over radius — charge density proxy
    if {"mean_electronegativity", "mean_covalent_radius"}.issubset(out.columns):
        out["en_over_radius"] = (
            out["mean_electronegativity"] / (out["mean_covalent_radius"] + 1e-9)
        )

    # Density × volume = total mass in cell
    if {"density", "volume"}.issubset(out.columns):
        out["density_x_volume"] = out["density"] * out["volume"]

    # Absolute formation energy
    if "formation_energy_peratom" in out.columns:
        out["formation_energy_abs"] = out["formation_energy_peratom"].abs()

    # Radius squared — non-linear radius dependence
    if "mean_covalent_radius" in out.columns:
        out["radius_sq"] = out["mean_covalent_radius"] ** 2

    # Atomic number / radius — nuclear charge density
    if {"mean_atomic_number", "mean_covalent_radius"}.issubset(out.columns):
        out["z_over_radius"] = (
            out["mean_atomic_number"] / (out["mean_covalent_radius"] + 1e-9)
        )

    # Density / mass — packing efficiency proxy
    if {"density", "mean_atomic_mass"}.issubset(out.columns):
        out["density_over_mass"] = out["density"] / (out["mean_atomic_mass"] + 1e-9)

    # Lattice volume ratio — deviation from orthorhombic
    if {"a", "b", "c", "volume"}.issubset(out.columns):
        out["lattice_volume_ratio"] = (
            (out["a"] * out["b"] * out["c"]) / (out["volume"] + 1e-9)
        )

    # Electronegativity × formation energy interaction
    if {"mean_electronegativity", "formation_energy_peratom"}.issubset(out.columns):
        out["en_x_formation"] = (
            out["mean_electronegativity"] * out["formation_energy_peratom"]
        )

    # Log volume per atom — regularised size
    if {"volume", "n_atoms"}.issubset(out.columns):
        out["log_volume_per_atom"] = np.log1p(
            (out["volume"] / (out["n_atoms"] + 1e-9)).clip(lower=0)
        )

    # Square root of density
    if "density" in out.columns:
        out["sqrt_density"] = np.sqrt(out["density"].clip(lower=0))

    # Guard against any NaNs or infinities from invalid/undefined inputs.
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if verbose:
        n_new = len(out.columns) - len(df.columns)
        print(f"[feature_eng] Added {n_new} engineered features "
              f"→ total {len(out.columns)} columns.")
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (everything except meta & target)."""
    exclude = {TARGET_COLUMN, "jid", "formula"}
    return [c for c in df.columns if c not in exclude]


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_reference_data

    ref = load_reference_data()
    ref_eng = engineer_features(ref)
    print("\nFeature columns:")
    for c in get_feature_columns(ref_eng):
        print(f"  {c}")
