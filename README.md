# Semiconductor band-gap ML and optimization

This project contains an ML/MCS workflow, SHAP helpers, a continuous NSGA-II descriptor search, and a small TF-IDF/regex literature demo. The intended target is `optb88vdw_bandgap` in eV from JARVIS-DFT `dft_3d` records. The checked-in reference CSV is a **surrogate statistical dataset**, not real JARVIS data; the loader now rejects it for real-material evaluation. Existing metrics and figures are historical and unverified. See [the audit](docs/PROJECT_AUDIT.md) before interpreting any prior result.

## Components

- `data_loader.py`: retrieves JARVIS records, derives descriptors, and validates material IDs. It fails clearly if data cannot be retrieved and does not substitute surrogate samples.
- `monte_carlo.py`: seeded Gaussian-copula sampling from the provided training rows. Its generated target values are synthetic estimates, not independent calculated labels.
- `feature_engineering.py`, `models.py`: deterministic descriptor transforms and regression models.
- `explainability.py`: SHAP and bootstrap uncertainty utilities.
- `moga.py`: hypothetical descriptor-space search. Results are not material identities or verified structures.
- `ner_rag_pipeline.py`: unverified hard-coded demonstration passages; not source-grounded scientific RAG.
- `main_pipeline.py`: fixed real split followed by training-only MCS; real-only and augmented models share one real test set.

## Setup and commands

Python 3.10+ is intended. Install declared dependencies:

```bash
python -m pip install -r requirements.txt
```

Run integrity and sampler checks:

```bash
python -m unittest discover -s tests -v
```

Run the full pipeline after arranging access to JARVIS `dft_3d` data:

```bash
python main_pipeline.py
```

The full run requires JARVIS data access and the packages in `requirements.txt`. It writes CSV/JSON under `data/` and `results/`, and figures under `figures/`. The end-to-end run overwrites generated outputs at those paths. Current tracked files at those paths include historical outputs; preserve or back them up before a full run.

Refresh the verified data cache directly with:

```powershell
python -c "from data_loader import load_reference_data; load_reference_data(force_reload=True)"
```

If Figshare returns HTTP 403 on your network, download the `jdft_3d` JSON ZIP from the [official JARVIS-DFT Figshare record](https://doi.org/10.6084/m9.figshare.6815699), then set the local archive path and refresh in PowerShell:

```powershell
$env:JARVIS_DFT_3D_PATH = "C:\path\to\jdft_3d.zip"
python -c "from data_loader import load_reference_data; load_reference_data(force_reload=True)"
```

The loader reads the JSON inside the archive and validates JARVIS IDs before writing the cache. It leaves the current cache untouched if the local archive is missing or invalid.

## Data and experiment protocol

The expected input target is `optb88vdw_bandgap`, interpreted in eV by the project. Material records need unique, real `jid` identifiers and source provenance. `data/reference_semiconductors.csv` currently contains `SURR-*` IDs and `SurrogateCompound` formulas; it is deliberately rejected. No authentic JARVIS dataset is bundled.

The corrected workflow splits verified records with `RANDOM_STATE=42` and `TEST_SIZE=0.2`, fits the Gaussian copula on real training rows, and evaluates real-only and real+MCS training on the same held-out real test IDs. The sampler uses a rank-based Gaussian copula plus empirical inverse marginals and configured rejection bounds. Random record splits do not prevent composition/structure-family leakage; grouped generalization remains to be assessed. Detailed protocol is in [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Scientific limitations

- Synthetic labels reproduce dependencies in training labels; they are not independent ground truth and synthetic-score fidelity does not establish real-world predictive performance.
- SHAP attribution is association with model predictions, not causation.
- Continuous optimizer outputs are hypothetical descriptor vectors. They have no material ID, formula, or structure unless independently matched and verified.
- The current RAG passages have no verified bibliographic provenance. Its reported self-annotation accuracy is not evidence of source-grounded answers. See [docs/RAG_EVALUATION.md](docs/RAG_EVALUATION.md).
- `FINAL_REPORT.md` intentionally reports no quantitative findings until a verified real-data run is available.
