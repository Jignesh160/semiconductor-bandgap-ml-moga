# Project audit (baseline before implementation)

Audit date: 2026-10-22 (Asia/Calcutta)

## Scope and repository state

- Inspected `README.md`, `requirements.txt`, `.gitignore`, the requested source modules (`main_pipeline.py`, `monte_carlo.py`, `models.py`, `moga.py`, `explainability.py`, `feature_engineering.py`), `data_loader.py`, `ner_rag_pipeline.py`, `FINAL_REPORT.md`, tracked outputs and Git history.
- Git was clean on branch `arjunkunnath-mcs-validation`; latest commits include the dashboard and reported benchmark additions. No repository `AGENTS.md` or test directory was present.
- The initial findings were written before implementation; the remainder of this file includes verification results recorded after the incremental changes.

## Actual data and provenance

The checked-in `data/reference_semiconductors.csv` has 26,840 rows and 21 columns. The target is `optb88vdw_bandgap` (the code treats values as eV and filters positive gaps when extracting JARVIS entries). Descriptor columns comprise `formation_energy_peratom`, `ehull`, `density`, atomic-mass/number/radius/electronegativity summaries, lattice lengths `a/b/c`, `volume`, `n_atoms`, and `n_elements`; `jid` and `formula` are metadata.

The checked-in file is **not a set of real JARVIS records**: all 26,840 IDs start `SURR-`; there is one formula value (`SurrogateCompound`). A diagnostic over the full CSV found 0 missing cells, 0 duplicate rows, 0 duplicate IDs, and `optb88vdw_bandgap` range 0.001–8.2382 eV (zero nonpositive targets). These numeric properties describe the surrogate CSV only. The generator in `data_loader.py` samples a multivariate normal from hard-coded approximate means, standard deviations, and correlations, then assigns those synthetic IDs/formulas. It does not download or verify the records represented by the file. `data/synthetic_semiconductors.csv` has 30,000 sampled rows and only the 19 numeric input/target columns, without source ID, record type, or generator metadata. The loader now also accepts a manually downloaded archive through `JARVIS_DFT_3D_PATH` when Figshare access returns 403.

At baseline, `load_reference_data()` accepted any existing CSV cache without provenance validation and, when download failed, silently built surrogate data in the reference cache path. This was corrected after the initial audit: the loader now rejects surrogate/missing IDs and fails without writing a substitute. A `JARVIS_DFT_3D_PATH` override now reads a downloaded JSON ZIP or JSON file locally, which supports networks that deny Figshare with HTTP 403. The override has only been tested with a controlled fixture, not a full real JARVIS archive.

## Architecture currently present

1. At baseline, `data_loader.py` fetched/extracted JARVIS records or generated the above surrogate fallback. The fallback is no longer used by `load_reference_data()`; verified IDs are required before caching.
2. `monte_carlo.py` rank-transforms all available numeric descriptors plus the target, fits a Gaussian copula, samples correlated normal vectors, maps each marginal through an empirical inverse CDF, and rejects draws outside a few configured bounds.
3. `feature_engineering.py` adds 24 deterministic features (and fills all remaining NaNs with zero); it does not learn imputation parameters.
4. `models.py` evaluates Linear Regression, Random Forest, Gradient Boosting, XGBoost, SVR, MLP and a VotingRegressor. Standard scaling is inside pipelines for LR, SVR and MLP; others are unscaled. There is no general training-only imputation pipeline.
5. `explainability.py` computes SHAP on the selected fitted estimator and a passed test matrix, and has bootstrap uncertainty helpers.
6. `moga.py` searches continuous descriptor vectors with NSGA-II. These are not tied to material IDs or structures and must be called hypothetical descriptor-space vectors. Its second objective is raw `ehull`; the 1st-percentile-derived bounds can admit negative `ehull`.
7. `ner_rag_pipeline.py` contains eight hard-coded sample passages with no identifiable bibliographic sources, retrieves by TF-IDF, and extracts values with regex. Its 32-field score is evaluated against annotations for those same passages, not an independent scientific evidence set. It is a synthetic demonstration, not evidence-grounded literature RAG.
8. `main_pipeline.py` sequences data loading, sampling, ML, SHAP, UQ and MOGA and writes directly into tracked `data/`, `results/`, and `figures/` paths. The RAG demo remains independently runnable and is no longer invoked as a prerequisite for ML.

## Baseline evaluation protocol and existing outputs

At baseline, `main_pipeline.py` generated synthetic rows before splitting and independently re-split the combined dataset, so its augmented test set differed from the real-only test set and included synthetic rows. This was corrected after the audit: it splits verified real IDs first, fits MCS on training rows, and reuses the untouched real test set for both training regimes.

The repository contains prior CSV/JSON results, including `results/pipeline_summary.json` reporting 26,840 reference rows, 30,000 synthetic rows, 42 features, `VotingEnsemble` R² 0.6887 on “reference”, and 0.6756 on “combined”. These are **reported, not reproduced**. They cannot be interpreted as real-material scores because the checked-in reference table is surrogate; the reference and combined test protocols differ; README, `FINAL_REPORT.md`, and saved result files also disagree on models, scores, augmentation effects, and Pareto counts. The audit did not run the full baseline because it would overwrite tracked results/figures and the required XGBoost, SHAP and pymoo modules are unavailable in the current Python environment.

## Reproduction environment

Commands run include repository/source/data inspection, Python import/version checks, `python -m unittest discover -s tests -v`, `python main_pipeline.py`, loader checks, syntax compilation and `git diff --check`. In the original audit environment Python was 3.12.10 with NumPy, pandas, SciPy, scikit-learn and matplotlib; `xgboost`, `shap`, `pymoo`, `pytest` and `jarvis` were missing. The pipeline stopped at import with `ModuleNotFoundError: No module named 'xgboost'`. Loader validation rejected the surrogate cache. The current seven unittest checks pass, including local archive parsing with a controlled fixture. The user also reported HTTP 403 from both Figshare download URLs and a `jarvis-tools` invalid-ZIP error; the new manual archive route is not yet verified against an authentic archive. No baseline or corrected ML metrics were reproduced.

## Findings and risks (priority order)

1. **Critical — provenance:** surrogate rows are cached and consumed as reference records; their fake IDs/formula make that explicit.
2. **Critical — evaluation leakage:** MCS is fitted before the real split and the augmented data is re-split, changing both training and test populations.
3. **High — reported results:** saved outputs and final report conflict, include unsupported scientific/material claims, and are not tied to verified real records.
4. **High — optimizer interpretation:** outputs are continuous feature vectors, not materials; negative energy-above-hull values appear in saved solutions and the result extraction hard-codes a 1.50 eV target.
5. **High — RAG provenance/evaluation:** the sample corpus has no citations/URLs/DOIs and its self-annotation score is not a factual RAG evaluation.
6. **Medium — model reproducibility:** ML evaluation cannot currently run in this environment due to missing dependencies; MLP has internal early stopping, but there is no documented validation split or tuning protocol.
7. **Medium — descriptor integrity:** correlated, independently sampled descriptors are not validated for duplicates, structural/chemical plausibility, or units; feature engineering may mask invalid/missing inputs by filling zeros.

## Recommended next steps

1. Prevent a surrogate table or unprovenanced cache from being accepted as real JARVIS data; retain it only as explicitly labelled synthetic benchmark input.
2. Split the records first, fit MCS on training rows only, and compare real-only and augmented models on the identical untouched real test rows. Mark synthetic outputs and settings.
3. Add focused tests for split disjointness, sampler reproducibility/training-only input, metric units, optimizer feasibility/non-dominance and RAG source handling.
4. Correct optimizer constraints/provenance labels; rank real records when a verified data source is available.
5. Replace or clearly demote hard-coded RAG demo passages; define a source-backed evaluation set before claiming RAG quality.
6. Rebuild results and report claims only after obtaining authenticated JARVIS data and installing the declared dependencies. Preserve the checked-in outputs until a verified replacement is available.

## Blockers

No real JARVIS records are present in the checkout. The cached dataset is explicitly surrogate, network-backed JARVIS acquisition is not assumed in this audit, and this environment lacks four required project/test dependencies. Real-material benchmark results, SHAP, MOGA, and source-grounded RAG quality therefore remain unverified.
