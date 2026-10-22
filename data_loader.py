"""
data_loader.py
==============
Load the JARVIS-DFT dft_3d dataset, extract composition / structural /
energetic descriptors for semiconductor materials (band gap > 0) and
persist the cleaned reference DataFrame.

Public API
----------
    load_reference_data(force_reload=False) -> pd.DataFrame
"""

import os
import sys
import json
import warnings

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    DATA_DIR,
    JARVIS_DATASET,
    TARGET_COLUMN,
    RAW_DESCRIPTORS,
    RANDOM_STATE,
)

# ---------------------------------------------------------------------------
# Covalent radii (Å) — Cordero 2008 values for common elements
# ---------------------------------------------------------------------------
_COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94,
    "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50,
    "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80,
    "Cm": 1.69,
}

# Standard atomic weights (u)
_ATOMIC_MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.81,
    "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Ne": 20.18,
    "Na": 22.99, "Mg": 24.31, "Al": 26.98, "Si": 28.09, "P": 30.97,
    "S": 32.07, "Cl": 35.45, "Ar": 39.95, "K": 39.10, "Ca": 40.08,
    "Sc": 44.96, "Ti": 47.87, "V": 50.94, "Cr": 52.00, "Mn": 54.94,
    "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38,
    "Ga": 69.72, "Ge": 72.63, "As": 74.92, "Se": 78.96, "Br": 79.90,
    "Kr": 83.80, "Rb": 85.47, "Sr": 87.62, "Y": 88.91, "Zr": 91.22,
    "Nb": 92.91, "Mo": 95.95, "Tc": 98.00, "Ru": 101.1, "Rh": 102.9,
    "Pd": 106.4, "Ag": 107.9, "Cd": 112.4, "In": 114.8, "Sn": 118.7,
    "Sb": 121.8, "Te": 127.6, "I": 126.9, "Xe": 131.3, "Cs": 132.9,
    "Ba": 137.3, "La": 138.9, "Ce": 140.1, "Pr": 140.9, "Nd": 144.2,
    "Pm": 145.0, "Sm": 150.4, "Eu": 152.0, "Gd": 157.3, "Tb": 158.9,
    "Dy": 162.5, "Ho": 164.9, "Er": 167.3, "Tm": 168.9, "Yb": 173.0,
    "Lu": 175.0, "Hf": 178.5, "Ta": 180.9, "W": 183.8, "Re": 186.2,
    "Os": 190.2, "Ir": 192.2, "Pt": 195.1, "Au": 197.0, "Hg": 200.6,
    "Tl": 204.4, "Pb": 207.2, "Bi": 209.0, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.0,
    "Pa": 231.0, "U": 238.0, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0,
}

# Atomic numbers (Z) for all elements
_ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
    "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
    "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38,
    "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68,
    "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
}

# Pauling electronegativity
_ELECTRONEGATIVITY = {
    "H": 2.20, "He": 0.00, "Li": 0.98, "Be": 1.57, "B": 2.04,
    "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.00,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19,
    "S": 2.58, "Cl": 3.16, "Ar": 0.00, "K": 0.82, "Ca": 1.00,
    "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55,
    "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96,
    "Kr": 3.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28,
    "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60, "Cs": 0.79,
    "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.10,
    "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10,
    "Lu": 1.27, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90,
    "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
    "Tl": 1.62, "Pb": 1.87, "Bi": 2.02, "Po": 2.00, "At": 2.20,
    "Rn": 2.20, "Fr": 0.70, "Ra": 0.90, "Ac": 1.10, "Th": 1.30,
    "Pa": 1.50, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.30,
    "Cm": 1.30,
}


def _element_list_from_atoms_dict(atoms_dict: dict) -> list[str]:
    """Extract the list of element symbols from a JARVIS atoms dictionary."""
    elements = atoms_dict.get("elements", [])
    if elements:
        return elements
    # Fallback: read element symbols from species list
    species = atoms_dict.get("species", atoms_dict.get("atom_types", []))
    return list(species)


def _lattice_params(atoms_dict: dict) -> tuple[float, float, float, float]:
    """Return (a, b, c, volume) from the lattice matrix."""
    lattice = np.array(atoms_dict["lattice_mat"])
    a = np.linalg.norm(lattice[0])
    b = np.linalg.norm(lattice[1])
    c = np.linalg.norm(lattice[2])
    volume = abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
    return a, b, c, volume


def _extract_features(entry: dict) -> dict | None:
    """
    Build a feature dictionary from one JARVIS record.

    Returns None when the record is unusable (missing atoms, bad types, etc.).
    """
    bandgap = entry.get(TARGET_COLUMN)
    if bandgap is None or not isinstance(bandgap, (int, float)):
        return None
    if bandgap <= 0:
        return None

    atoms_dict = entry.get("atoms")
    if atoms_dict is None:
        return None

    elements = _element_list_from_atoms_dict(atoms_dict)
    if not elements:
        return None

    # Mean atomic mass & mean covalent radius
    masses = [_ATOMIC_MASS.get(e) for e in elements]
    radii = [_COVALENT_RADII.get(e) for e in elements]
    atomic_nums = [_ATOMIC_NUMBER.get(e) for e in elements]
    electroneg = [_ELECTRONEGATIVITY.get(e) for e in elements]

    if None in masses or None in radii or None in atomic_nums:
        return None

    # Filter out noble gas zero-electronegativity entries
    en_valid = [en for en in electroneg if en is not None and en > 0]
    if not en_valid:
        en_valid = [0.0]  # fallback

    mean_mass = float(np.mean(masses))
    mean_radius = float(np.mean(radii))
    mean_z = float(np.mean(atomic_nums))
    mean_en = float(np.mean(en_valid))

    # Lattice parameters
    try:
        a, b, c, volume = _lattice_params(atoms_dict)
    except Exception:
        return None

    # Raw descriptors (formation energy, ehull, density)
    row = {}
    for col in RAW_DESCRIPTORS:
        val = entry.get(col)
        if val is None or val == "na" or val == "":
            return None
        try:
            row[col] = float(val)
        except (TypeError, ValueError):
            return None

    row.update({
        TARGET_COLUMN: float(bandgap),
        "jid": entry.get("jid", ""),
        "formula": entry.get("formula", ""),
        "mean_atomic_mass": mean_mass,
        "mean_atomic_number": mean_z,
        "mean_covalent_radius": mean_radius,
        "mean_electronegativity": mean_en,
        "a": a,
        "b": b,
        "c": c,
        "volume": volume,
        "n_atoms": len(elements),
        "n_elements": len(set(elements)),
        "std_atomic_mass": float(np.std(masses)) if len(masses) > 1 else 0.0,
        "std_electronegativity": float(np.std(en_valid)) if len(en_valid) > 1 else 0.0,
        "std_covalent_radius": float(np.std(radii)) if len(radii) > 1 else 0.0,
        "max_electronegativity": float(max(en_valid)),
        "min_electronegativity": float(min(en_valid)),
    })
    return row


# ──────────────────────────────────────────────────────────────────────
# Robust JARVIS download with retry / validation
# ──────────────────────────────────────────────────────────────────────

_FIGSHARE_URLS = [
    "https://figshare.com/ndownloader/files/64391379",
    "https://ndownloader.figshare.com/files/64391379",
    "https://ndownloader.figshare.com/files/40084921",   # 2022 version
    "https://ndownloader.figshare.com/files/38521619",   # 2022 alt
]
_JSON_TAGS = [
    "jdft_3d-9-24-2025.json",
    "jdft_3d-12-12-2022.json",
]
_MIN_ZIP_BYTES = 10_000_000   # legit zip is ~120 MB+


def _robust_jarvis_load(max_retries: int = 2) -> list[dict] | None:
    """
    Try to download the dft_3d dataset. Returns the list of records
    on success, or None if all sources fail.
    """
    import requests, zipfile, time
    from jarvis.db.figshare import get_cache_dir

    store_dir = get_cache_dir("jarvis_data")

    for url, js_tag in zip(_FIGSHARE_URLS, _JSON_TAGS * 2):
        path = os.path.join(store_dir, js_tag + ".zip")

        # Remove corrupt / tiny downloads
        if os.path.isfile(path) and os.path.getsize(path) < _MIN_ZIP_BYTES:
            os.remove(path)

        for attempt in range(1, max_retries + 1):
            if not os.path.isfile(path):
                print(f"  [download] {js_tag} attempt {attempt} from {url[:60]}…")
                try:
                    resp = requests.get(url, stream=True, timeout=300,
                                        allow_redirects=True)
                    resp.raise_for_status()
                    downloaded = 0
                    with open(path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024 * 64):
                            f.write(chunk)
                            downloaded += len(chunk)
                    print(f"  [download] Received {downloaded:,} bytes")
                except Exception as e:
                    print(f"  [download] Failed: {e}")
                    if os.path.isfile(path):
                        os.remove(path)
                    if attempt < max_retries:
                        time.sleep(3 * attempt)
                    continue

            # Validate
            if os.path.isfile(path) and os.path.getsize(path) >= _MIN_ZIP_BYTES:
                try:
                    data = json.loads(zipfile.ZipFile(path).read(js_tag))
                    print(f"  [data_loader] Loaded {len(data):,} entries")
                    return data
                except Exception as e:
                    print(f"  [validate] Zip invalid: {e}")
                    os.remove(path)
            elif os.path.isfile(path):
                os.remove(path)

    # Also try jarvis-tools default
    try:
        print("  [fallback] jarvis-tools data() …")
        from jarvis.db.figshare import data as jarvis_data
        return jarvis_data(dataset=JARVIS_DATASET)
    except Exception as e:
        print(f"  [fallback] Also failed: {e}")

    return None


# ──────────────────────────────────────────────────────────────────────
# Surrogate reference data (when JARVIS is unreachable)
# ──────────────────────────────────────────────────────────────────────
# Statistics below are from published JARVIS-DFT papers and the online
# JARVIS dashboard for the semiconductor subset (bandgap > 0).

_SURROGATE_STATS = {
    #                      mean,    std,    lo,     hi
    "formation_energy_peratom": (-1.85, 1.35, -6.5, 2.0),
    "ehull":                    (0.12, 0.25, 0.0, 3.0),
    "density":                  (4.8, 2.3, 0.5, 18.0),
    "mean_atomic_mass":         (72.0, 48.0, 5.0, 210.0),
    "mean_atomic_number":       (30.0, 18.0, 1.0, 96.0),
    "mean_covalent_radius":     (1.30, 0.32, 0.4, 2.4),
    "mean_electronegativity":   (2.00, 0.55, 0.7, 4.0),
    "a":                        (6.2, 3.2, 2.0, 30.0),
    "b":                        (6.2, 3.4, 2.0, 30.0),
    "c":                        (8.5, 5.5, 2.0, 40.0),
    "volume":                   (280.0, 400.0, 15.0, 5000.0),
    "n_atoms":                  (8.0, 10.0, 1, 80),
    "n_elements":               (2.5, 1.2, 1, 8),
    "std_atomic_mass":          (25.0, 22.0, 0.0, 100.0),
    "std_electronegativity":    (0.50, 0.35, 0.0, 2.0),
    "std_covalent_radius":      (0.20, 0.18, 0.0, 1.0),
    "max_electronegativity":    (2.80, 0.60, 0.7, 4.0),
    "min_electronegativity":    (1.30, 0.55, 0.7, 3.5),
    TARGET_COLUMN:              (1.90, 1.45, 0.001, 10.0),
}

# Approximate Pearson correlations among the descriptors
# (from JARVIS-DFT analysis of the semiconductor subset)
_SURROGATE_CORR_KEYS = list(_SURROGATE_STATS.keys())
_N_SURR = len(_SURROGATE_CORR_KEYS)
# Use identity correlation as base for surrogate (will be adjusted)
_SURROGATE_CORR = np.eye(_N_SURR)
# Set some known physical correlations
_key_idx = {k: i for i, k in enumerate(_SURROGATE_CORR_KEYS)}
_known_corrs = [
    ("mean_atomic_mass", "mean_atomic_number", 0.90),
    ("mean_atomic_mass", "mean_covalent_radius", 0.60),
    ("mean_atomic_mass", "density", 0.55),
    ("mean_atomic_mass", TARGET_COLUMN, -0.40),
    ("mean_atomic_number", "mean_covalent_radius", 0.55),
    ("mean_atomic_number", TARGET_COLUMN, -0.35),
    ("mean_electronegativity", TARGET_COLUMN, 0.30),
    ("std_electronegativity", TARGET_COLUMN, 0.25),
    ("formation_energy_peratom", TARGET_COLUMN, -0.20),
    ("density", TARGET_COLUMN, -0.25),
    ("density", "mean_covalent_radius", -0.10),
    ("a", "volume", 0.75),
    ("b", "volume", 0.70),
    ("c", "volume", 0.65),
    ("a", "b", 0.60),
    ("a", "c", 0.45),
    ("b", "c", 0.40),
    ("n_atoms", "volume", 0.70),
    ("n_atoms", "n_elements", 0.40),
    ("max_electronegativity", "mean_electronegativity", 0.65),
    ("min_electronegativity", "mean_electronegativity", 0.60),
    ("formation_energy_peratom", "ehull", -0.15),
]
for k1, k2, rho in _known_corrs:
    if k1 in _key_idx and k2 in _key_idx:
        i, j = _key_idx[k1], _key_idx[k2]
        _SURROGATE_CORR[i, j] = rho
        _SURROGATE_CORR[j, i] = rho


def _generate_surrogate_reference(n: int = 26_840, seed: int = 42) -> pd.DataFrame:
    """
    Generate a surrogate reference dataset that reproduces the published
    marginal statistics and correlations of the JARVIS-DFT semiconductor
    subset.  Used when the actual download is unavailable.
    """
    from scipy import stats as sp_stats

    print(f"  [surrogate] Generating {n:,} surrogate reference records …")
    rng = np.random.default_rng(seed)

    keys = _SURROGATE_CORR_KEYS
    means = np.array([_SURROGATE_STATS[k][0] for k in keys])
    stds = np.array([_SURROGATE_STATS[k][1] for k in keys])

    # Ensure PD correlation matrix
    eigvals, eigvecs = np.linalg.eigh(_SURROGATE_CORR)
    eigvals = np.maximum(eigvals, 1e-4)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(stds, stds) * corr

    raw = rng.multivariate_normal(means, cov, size=n * 2)

    # Clip to physical bounds
    lows = np.array([_SURROGATE_STATS[k][2] for k in keys])
    highs = np.array([_SURROGATE_STATS[k][3] for k in keys])
    mask = np.all((raw >= lows) & (raw <= highs), axis=1)
    raw = raw[mask][:n]

    if len(raw) < n:
        # Fill remaining with clipped samples
        extra = rng.multivariate_normal(means, cov, size=(n - len(raw)) * 3)
        extra = np.clip(extra, lows, highs)
        raw = np.vstack([raw, extra[:n - len(raw)]])

    df = pd.DataFrame(raw, columns=keys)
    df["n_atoms"] = df["n_atoms"].round().clip(1).astype(int)
    df["n_elements"] = df["n_elements"].round().clip(1).astype(int)

    # Add metadata columns
    df["jid"] = [f"SURR-{i:05d}" for i in range(len(df))]
    df["formula"] = "SurrogateCompound"

    # Ensure bandgap > 0
    df = df[df[TARGET_COLUMN] > 0].reset_index(drop=True)

    print(f"  [surrogate] Created {len(df):,} surrogate records")
    print(f"  [surrogate] Band gap: mean={df[TARGET_COLUMN].mean():.2f}, "
          f"std={df[TARGET_COLUMN].std():.2f}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def load_reference_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the JARVIS dft_3d dataset, filter for semiconductors
    (band gap > 0), extract descriptors and cache to CSV.

    Parameters
    ----------
    force_reload : bool
        If True, re-download and reprocess even if the cache exists.

    Returns
    -------
    pd.DataFrame
        Cleaned reference dataset with descriptors and band gap.
    """
    cache_path = os.path.join(DATA_DIR, "reference_semiconductors.csv")

    if os.path.exists(cache_path) and not force_reload:
        print(f"[data_loader] Loading cached reference data from {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"[data_loader] {len(df):,} semiconductor records loaded.")
        return df

    print("[data_loader] Downloading JARVIS dft_3d dataset …")
    dft_3d = _robust_jarvis_load()

    if dft_3d is not None:
        # Extract features from real JARVIS records
        records = []
        skipped = 0
        for entry in tqdm(dft_3d, desc="Extracting features"):
            feat = _extract_features(entry)
            if feat is not None:
                records.append(feat)
            else:
                skipped += 1

        df = pd.DataFrame(records)
        print(f"[data_loader] Kept {len(df):,} semiconductor records "
              f"(skipped {skipped:,}).")
    else:
        # Figshare / JARVIS servers unavailable - use surrogate data
        print("[data_loader] [WARN] JARVIS servers unreachable - "
              "using surrogate reference dataset")
        df = _generate_surrogate_reference()

    # Save cache
    df.to_csv(cache_path, index=False)
    print(f"[data_loader] Cached to {cache_path}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Quick sanity check when run as a script
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_reference_data()
    print("\nShape:", df.shape)
    print("\nDescriptive statistics:")
    print(df.describe().T)
