# Experiment protocol

## Current protocol in code

The intended target is `optb88vdw_bandgap`, interpreted as eV, and the primary data source is JARVIS-DFT `dft_3d`. `data_loader.py` now rejects missing or `SURR-` IDs and no longer silently generates synthetic rows into the real-reference cache. The checked-in cache fails this provenance check, so no real-material result is currently reproducible from this checkout.

Once verified JARVIS records are supplied:

1. `split_real_records` splits records with unique `jid` values using `TEST_SIZE` and `RANDOM_STATE` from `config.py`.
2. Gaussian-copula MCS is fit only to `ref_train`. It rank-transforms the available numeric descriptors and target, estimates a Gaussian correlation matrix, samples correlated normals, maps each feature through its training empirical marginal, and applies configured bounds. A generated target is a synthetic label inferred from the training joint distribution, not a new DFT calculation or independent ground truth.
3. The real-only models use the real training partition. The augmented models use those same real training rows plus MCS rows. Both are scored against the same untouched real test rows.
4. Feature construction is deterministic algebra over each row. Undefined/non-finite derived values are mapped to zero; notably a zero-density row no longer creates an enormous mass/density ratio. Scaling for models that need it is contained in model pipelines and fit during estimator training. Any future learned preprocessing must be fit on training data only.
5. Report MAE/RMSE in eV, R² (dimensionless), and real/synthetic/train/test sample counts. Interpret augmentation impact as an empirical comparison, including degradation or no material change. MCS applies non-negative/positive lower bounds only; because the empirical inverse marginals already cap samples to observed training values, fixed upper limits are unnecessary and can truncate the distribution.

Current split is random by material ID. It does not group by chemical system, composition, structure, or near-duplicate family; a random-split score may be optimistic for novel compositions. That grouped generalization analysis remains future work.

## Reproduction status

Install dependencies with `python -m pip install -r requirements.txt`, obtain verified JARVIS records, then run `python main_pipeline.py`. At audit time the environment lacks `xgboost`, `shap`, `pymoo`, `jarvis-tools`, and `pytest`, and the bundled cache is surrogate. Therefore no ML, SHAP, MOGA or real-material performance result has been reproduced. `python -m unittest discover -s tests -v` runs dependency-light integrity and sampler checks.

The existing `FINAL_REPORT.md` and tracked result tables are historical artifacts, not current verified outputs. Do not interpret their reported metrics as evidence for JARVIS generalization. RAG is kept as a separately runnable component so its failure or unverified demo corpus does not prevent ML evaluation.
