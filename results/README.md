# Results

All artifacts produced by `code/Hue_Visitors_Linear_DL_Ensemble.ipynb` and reported in the paper.

## tables/
| File | Contents |
|---|---|
| `metrics_aggregate.csv` | MAE, RMSE, sMAPE, WAPE, MASE, R² per model × horizon, aggregated over the four sites (validation and test; `model_only` and `operational_closure` scenarios) |
| `metrics_by_destination_horizon.csv` | The same metrics broken out per destination and horizon |
| `predictions_test.csv` | Per-day actual vs. predicted values on the test period, all models |
| `predictions_validation.csv` | Per-day actual vs. predicted values on the validation period |
| `diebold_mariano_tests.csv` | Diebold–Mariano statistics vs. Seasonal Naive-7, with Holm-adjusted p-values |
| `ensemble_weights.csv` | Validation-fitted component weights per site–horizon |
| `optuna_summary.csv` | Best validation RMSE, trial, and epoch per deep model |
| `ridge_validation_search.csv` | ARX-Ridge lookback × alpha validation grid |
| `cross_destination_correlation.csv` | Pairwise correlation of the log visitor series |
| `zero_day_summary.csv` | Count and dates of observed zero-demand days per site |
| `split_summary.csv`, `data_audit.csv` | Chronological split sizes; row counts and SHA-256 per file |

## figures/
EDA plots (`eda_daily_visitors.png`, `eda_cross_destination_correlation.png`), the deep-model loss curves, the ARX-Ridge regularization curve, the test-set RMSE heatmap, and one-day test forecasts per destination.

## models/
Trained checkpoints: LSTM, GRU, 1D-CNN, and Transformer for seeds 2026/2027/2028 (`.keras`), plus the fitted `ARX_Ridge.joblib`.

## optuna/
Full Optuna trial logs for each deep model.

## logs/
`run_manifest.json` (splits, seeds, selected hyperparameters, data checksums), `environment_versions.json` (library versions), `best_hyperparameters.json`, and per-model training histories.

`Hue_4_Destinations_Forecasting_Results.xlsx` collects the main tables in one workbook.
