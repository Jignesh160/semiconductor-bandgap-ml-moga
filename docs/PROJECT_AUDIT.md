# Project audit and verified run

Audit started 2026-10-22 (Asia/Calcutta); full pipeline completed later the same day.

## Initial state and changes

The initial checkout was clean on branch `arjunkunnath-mcs-validation`, but its `data/reference_semiconductors.csv` was a surrogate dataset: all 26,840 IDs were `SURR-*`, with one formula value, `SurrogateCompound`. It was not authentic JARVIS input. The initial pipeline generated MCS before splitting and then re-split the combined real/synthetic data, so the augmented test set differed from the real-only test set and included synthetic rows. The baseline pipeline was not run against the surrogate cache.

Changes made before the completed run:

- The loader rejects surrogate/missing IDs, automatically tries to refresh an invalid cache, supports `JARVIS_DFT_3D_PATH`, and downloads into a project-local temporary directory rather than deleting files from the user-level JARVIS cache.
- The sampler runs after a fixed ID-based split, is fit on real training records only, and both model regimes use the same real test partition.
- MCS now uses physical positive/nonnegative lower bounds only. Fixed upper caps were removed because empirical inverse-CDF sampling already stays within the training marginals and the old caps truncated observed tails.
- Undefined/non-finite engineered values are handled safely; in particular the zero-density hydrogen record no longer creates an enormous mass/density ratio.
- NSGA-II outputs are constrained to feasible non-dominated rows and labelled as hypothetical descriptor-space candidates without material IDs.
- UQ intervals now use out-of-bag training residuals for calibration; bootstrap prediction spread alone had severe undercoverage.

## Data and provenance

The successful run retrieved JARVIS-DFT `dft_3d` version-11 Figshare file `64391379`, archive `jdft_3d-9-24-2025.json.zip` ([official Figshare record](https://doi.org/10.6084/m9.figshare.6815699)). It read 93,902 records, extracted descriptors, and retained 26,840 positive-gap records (67,062 skipped). The current cache has 26,840 unique non-surrogate JARVIS IDs, 17,818 distinct formulas, 0 missing cells, 0 duplicate full rows, and target range 0.001–18.179 eV (mean 2.02367 eV). Target column is `optb88vdw_bandgap`; the code treats it as eV.

MCS produced 30,000 rows from the 21,472 real training rows. Synthetic band-gap labels are generated estimates from the fitted training joint distribution, not independent DFT calculations. The sampler's maximum absolute Pearson-correlation difference was 0.183155; matching marginal statistics does not establish dependency fidelity.

## Completed experiment

Random seed 42 split the records into 21,472 train and 5,368 test records. Both model regimes were tested on the same untouched real test IDs.

| Training data | Best model | MAE (eV) | RMSE (eV) | R² | Test records |
| --- | --- | ---: | ---: | ---: | ---: |
| Real only | Random Forest | 0.560019 | 0.818547 | 0.794483 | 5,368 |
| Real + MCS | Random Forest | 0.607950 | 0.873358 | 0.766038 | 5,368 |

MCS augmentation worsened these Random Forest metrics on this split. SHAP was computed for the fitted real-only Random Forest on 100 held-out samples. OOB residual calibration used 236,796 training residuals; nominal 90% intervals covered 89.92% of the held-out set with a mean width of 2.805 eV. This is empirical coverage on one split, not a guarantee for new populations.

NSGA-II returned four feasible Pareto rows. They are continuous descriptor vectors with no verified material ID, formula, or structure; report them as hypothetical candidates only. The RAG component remains a hard-coded, uncited demonstration corpus, and its source-grounded quality is not evaluated.

## Commands and outcomes

- `python -u main_pipeline.py` — completed successfully using the verified local cache. The first attempt exposed the zero-density numerical issue and extremely slow SHAP at 500 examples; the corrected run completed with SHAP capped at 100.
- `python -m unittest discover -s tests -v` — 9 tests passed.
- `python -m py_compile ...` on changed Python files — passed.
- `git diff --check` — passed.
- Network fetch first failed in the sandbox due DNS restrictions. With authorized network access, Figshare download succeeded (48,447,610 bytes). Before the temporary-directory fix, a refresh also failed trying to delete a protected bad ZIP in the user cache; that external cache is no longer modified.

## Remaining limits

The split is random by material ID and is not composition- or structure-grouped. MCS dependency fidelity is imperfect. The synthetic labels are not ground truth. SHAP associations are not causal. Optimizer points are not verified materials. RAG quality and citations remain unverified. See [docs/EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) and [docs/RAG_EVALUATION.md](RAG_EVALUATION.md).
