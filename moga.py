"""
moga.py
=======
Multi-Objective Genetic Algorithm (MOGA) for semiconductor optimisation.

Objectives (to minimise)
------------------------
1.  |predicted_bandgap − target_bandgap|  → hit the desired band gap
2.  ehull                                  → maximise stability

Constraints
-----------
- All descriptors within their physical bounds.
- ehull ≤ MOGA_EHULL_THRESHOLD  (thermodynamic near-stability).
- predicted bandgap in [MOGA_TARGET_BANDGAP_MIN, MOGA_TARGET_BANDGAP_MAX].

Uses NSGA-II from *pymoo*.

Public API
----------
    run_moga(model, feature_names, ref_df) -> pymoo Result
    extract_pareto(result, feature_names) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from config import (
    ALL_DESCRIPTORS,
    TARGET_COLUMN,
    MOGA_POP_SIZE,
    MOGA_N_GENERATIONS,
    MOGA_CROSSOVER_PROB,
    MOGA_MUTATION_PROB,
    MOGA_TARGET_BANDGAP_MIN,
    MOGA_TARGET_BANDGAP_MAX,
    MOGA_EHULL_THRESHOLD,
    RANDOM_STATE,
)
from feature_engineering import engineer_features, get_feature_columns


# ──────────────────────────────────────────────────────────────────────
# Problem definition
# ──────────────────────────────────────────────────────────────────────

class BandGapProblem(Problem):
    """
    NSGA-II problem for band gap optimisation.

    Decision variables = raw descriptor values (ALL_DESCRIPTORS).
    We engineer features internally before calling the ML model.
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        descriptor_bounds: dict[str, tuple[float, float]],
        target_bg: float | None = None,
    ):
        self.model = model
        self.feature_names = feature_names
        self.descriptor_names = list(descriptor_bounds.keys())
        self.target_bg = target_bg or (
            (MOGA_TARGET_BANDGAP_MIN + MOGA_TARGET_BANDGAP_MAX) / 2
        )

        n_var = len(self.descriptor_names)
        xl = np.array([descriptor_bounds[d][0] for d in self.descriptor_names])
        xu = np.array([descriptor_bounds[d][1] for d in self.descriptor_names])

        # 2 objectives, 1 inequality constraint (ehull ≤ threshold)
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=1,
            xl=xl,
            xu=xu,
        )

    def _to_dataframe(self, x: np.ndarray) -> pd.DataFrame:
        """Convert decision-variable matrix to a DataFrame with descriptor names."""
        return pd.DataFrame(x, columns=self.descriptor_names)

    def _evaluate(self, x, out, *args, **kwargs):
        df = self._to_dataframe(x)
        df_eng = engineer_features(df, verbose=False)

        # Ensure columns align with the model's training features
        for col in self.feature_names:
            if col not in df_eng.columns:
                df_eng[col] = 0.0
        X_pred = np.nan_to_num(df_eng[self.feature_names].values, nan=0.0, posinf=1e5, neginf=-1e5)

        bg_pred = self.model.predict(X_pred)

        # Objective 1: |band gap − target|
        f1 = np.abs(bg_pred - self.target_bg)

        # Objective 2: ehull (lower = more stable)
        ehull_idx = self.descriptor_names.index("ehull")
        f2 = x[:, ehull_idx]

        out["F"] = np.column_stack([f1, f2])

        # Constraint: ehull ≤ threshold  →  g = ehull − threshold ≤ 0
        out["G"] = x[:, ehull_idx] - MOGA_EHULL_THRESHOLD


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def _descriptor_bounds(ref_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Extract (min, max) bounds from the reference data for each descriptor."""
    bounds = {}
    for col in ALL_DESCRIPTORS:
        if col in ref_df.columns:
            lo = float(ref_df[col].quantile(0.01))
            hi = float(ref_df[col].quantile(0.99))
            # Add small margin
            margin = (hi - lo) * 0.05
            bounds[col] = (lo - margin, hi + margin)
    return bounds


def run_moga(
    model,
    feature_names: list[str],
    ref_df: pd.DataFrame,
    target_bg: float | None = None,
    verbose: bool = True,
) -> object:
    """
    Run the multi-objective genetic algorithm.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    feature_names : list of feature column names the model was trained on
    ref_df : reference DataFrame (used to derive decision-variable bounds)
    target_bg : float, optional — desired band gap centre (eV)

    Returns
    -------
    pymoo.core.result.Result
    """
    bounds = _descriptor_bounds(ref_df)

    problem = BandGapProblem(
        model=model,
        feature_names=feature_names,
        descriptor_bounds=bounds,
        target_bg=target_bg,
    )

    algorithm = NSGA2(
        pop_size=MOGA_POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=MOGA_CROSSOVER_PROB, eta=15),
        mutation=PM(prob=MOGA_MUTATION_PROB, eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", MOGA_N_GENERATIONS)

    print(f"[MOGA] Running NSGA-II  (pop={MOGA_POP_SIZE}, "
          f"gen={MOGA_N_GENERATIONS}, target_bg={problem.target_bg:.2f} eV) …")

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=RANDOM_STATE,
        verbose=verbose,
    )

    print(f"[MOGA] Done. Pareto front size: {len(res.F)}")
    return res


def extract_pareto(
    result,
    ref_df: pd.DataFrame,
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Build a readable DataFrame of the Pareto-optimal solutions.

    Columns include all descriptors, predicted band gap, and objectives.
    """
    bounds = _descriptor_bounds(ref_df)
    descriptor_names = list(bounds.keys())

    df_pareto = pd.DataFrame(result.X, columns=descriptor_names)
    df_eng = engineer_features(df_pareto)

    for col in feature_names:
        if col not in df_eng.columns:
            df_eng[col] = 0.0

    X_pred_pareto = np.nan_to_num(df_eng[feature_names].values, nan=0.0, posinf=1e5, neginf=-1e5)
    df_pareto["predicted_bandgap"] = model.predict(X_pred_pareto)
    df_pareto["obj_bg_deviation"] = result.F[:, 0]
    df_pareto["obj_ehull"] = result.F[:, 1]

    df_pareto = df_pareto.sort_values("predicted_bandgap")
    return df_pareto


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Run via main_pipeline.py for the full workflow.")
