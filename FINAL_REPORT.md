# Verified pipeline run

## Data and method

The pipeline retrieved JARVIS-DFT `dft_3d` version-11 file `64391379` from the official [Figshare record](https://doi.org/10.6084/m9.figshare.6815699). It read 93,902 records, extracted descriptors, and retained 26,840 positive-gap records. The resulting cache contains 26,840 unique JARVIS IDs, no missing cells or duplicate rows, and target range 0.001–18.179 eV (mean 2.02367 eV).

With seed 42, 21,472 real records were used for training and 5,368 for testing. A Gaussian copula fit only to the real training rows generated 30,000 synthetic rows. Real-only and real+MCS models were evaluated on the same untouched real test records. MCS target labels are generated estimates, not independent DFT results.

## Model comparison

| Training data | Best model | Test MAE (eV) | Test RMSE (eV) | Test R² | Test count |
| --- | --- | ---: | ---: | ---: | ---: |
| Real only | Random Forest | 0.560019 | 0.818547 | 0.794483 | 5,368 |
| Real + MCS | Random Forest | 0.607950 | 0.873358 | 0.766038 | 5,368 |

On this split, MCS augmentation worsened the Random Forest results. The sampler's maximum absolute Pearson-correlation difference was 0.183155, so it did not fully preserve dependency patterns.

## Explainability, uncertainty and optimization

- SHAP was computed for the fitted real-only Random Forest on 100 held-out examples. Attributions describe model behavior and do not establish causation.
- Prediction intervals calibrated from 236,796 out-of-bag training residuals achieved 89.92% coverage for nominal 90% intervals on the held-out set; mean interval width was 2.805 eV. This is one split’s empirical coverage, not a guarantee for new populations.
- NSGA-II returned four feasible, non-dominated continuous descriptor candidates, with predicted band gaps from 1.4886 to 1.5000 eV. They have no verified material IDs, formulas, or structures and are hypothetical outputs, not material recommendations.

## Reproduction and limits

```powershell
python main_pipeline.py
python -m unittest discover -s tests -v
```

The completed pipeline produced tables and JSON under `results/` and plots under `figures/`; all nine tests passed. The RAG passages remain an uncited demonstration corpus and are not evaluated as scientific evidence. The random split is not composition/structure grouped. Predictions and optimization outputs are not experimental validation.
