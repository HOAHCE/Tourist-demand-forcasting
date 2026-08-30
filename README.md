# Global GRU-Based Multi-Horizon Visitor Forecasting for Digital Heritage Management in Hue, Vietnam

Reproducibility repository for the DEFI 2026 paper of the same title. It contains the daily visitor data for four major heritage sites in Hue, the Colab/Jupyter notebook that runs the full experiment, and the complete set of result tables, figures, trained models, and run logs.

This study formulates daily visitor forecasting for the Imperial City (Đại Nội) and the Khai Dinh, Minh Mang, and Tu Duc royal tombs as a **global, multivariate, direct** problem at **1-, 7-, and 14-day** horizons. Seven approaches are compared under a chronological, leakage-controlled protocol: Seasonal Naive-7, ARX-Ridge, LSTM, GRU, 1D-CNN, Transformer, and a validation-weighted ensemble. On the independent test set, **GRU gives the lowest RMSE at every horizon** and in 11 of 12 site–horizon combinations, cutting aggregate RMSE by 46% versus the seasonal benchmark.

## Headline results (independent test set, aggregated over the four sites)

| Model | RMSE H=1 | RMSE H=7 | RMSE H=14 | Mean RMSE | Mean sMAPE (%) | Mean R² |
|---|--:|--:|--:|--:|--:|--:|
| Seasonal Naive-7 | 1160.22 | 1162.01 | 1233.94 | 1185.39 | 23.64 | 0.656 |
| ARX-Ridge | 665.02 | 952.76 | 994.90 | 870.89 | 22.72 | 0.809 |
| LSTM | 749.44 | 825.08 | 865.72 | 813.41 | 19.14 | 0.837 |
| **GRU** | **492.27** | **703.03** | **711.30** | **635.54** | **17.07** | **0.899** |
| 1D-CNN | 1096.46 | 1159.54 | 909.32 | 1055.10 | 27.55 | 0.724 |
| Transformer | 898.49 | 1041.88 | 1041.71 | 994.03 | 25.17 | 0.757 |
| Ensemble | 517.24 | 779.45 | 802.96 | 699.88 | 18.46 | 0.876 |

Relative to Seasonal Naive-7, GRU reduces RMSE by 57.57%, 39.50%, and 42.36% at the three horizons. Full per-destination results are in `results/tables/metrics_by_destination_horizon.csv`; the source of truth for every number reported in the paper is `results/tables/` and `results/Hue_4_Destinations_Forecasting_Results.xlsx`.

## Repository structure

```
.
├── data/                 Cleaned daily visitor series (model inputs)
├── code/                 The experiment notebook
├── results/              All outputs used in the paper
│   ├── tables/           Metrics, predictions, DM tests, ensemble weights
│   ├── figures/          EDA and result figures (PNG)
│   ├── models/           Trained model checkpoints (per seed)
│   ├── optuna/           Hyperparameter-search trial logs
│   ├── logs/             Run manifest, environment, training histories
│   └── Hue_4_Destinations_Forecasting_Results.xlsx
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Data

`data/` holds one cleaned CSV per destination, each covering **1,393 consecutive days** from 2022-11-01 to 2026-08-24 (the aligned modeling window used by the experiment starts 2022-11-10, giving 1,384 days per site). Each file has the columns:

| Column | Meaning |
|---|---|
| `date` | Calendar day (YYYY-MM-DD) |
| `destination_id`, `destination` | Site identifier and name |
| `domestic_visitors`, `international_visitors` | Daily counts by visitor type |
| `visitors` | Total daily visitors (`domestic + international`, the forecast target) |
| `log1p_visitors` | `log(1 + visitors)`, the modeling transform |
| `split` | `train` / `validation` / `test` assignment |
| `is_zero_day` | True on observed zero-demand days (kept as valid observations) |
| `data_quality_flag` | Provenance/quality marker |

The chronological split is **966 train / 208 validation / 210 test** days (no shuffling), spanning 2022-11-10 → 2025-07-02 (train), 2025-07-03 → 2026-01-26 (validation), and 2026-01-27 → 2026-08-24 (test). `data/hue_four_destinations_split_summary.csv` and `hue_four_destinations_share_summary.csv` summarize coverage and each site's share of total visitors. File integrity is recorded via SHA-256 in `results/tables/data_audit.csv` and `results/logs/run_manifest.json`.

The counts were compiled from public pages of the Hue Monuments Conservation Centre. Please credit that source if you reuse the data.

## Method in brief

The twelve prediction targets (four sites × three horizons) are produced directly, not recursively. Historical visitor counts are transformed with `log1p`, standardized on training data only, and fed to each model together with a separate branch of 17 calendar features (weekday, month, day-of-year sine/cosine encodings; weekend, Vietnamese public-holiday, and confirmed-closure indicators; a yearly trend; and weekday dummies) that are known at forecast issuance. A sample is assigned to a split only when all of its target dates fall inside that split, so no window crosses a boundary.

Hyperparameters were chosen with Optuna (Tree-structured Parzen Estimator, 15 trials per deep model) by minimizing aggregate validation RMSE on the original visitor scale. The validation-weighted ensemble combines, per site–horizon, the three models with the smallest validation RMSE using non-negative weights that sum to one and are frozen before any test prediction. Deep models use Adam, gradient clipping at 1.0, Huber loss, and early stopping; final models are refit on train+validation for a fixed number of epochs and averaged over seeds 2026, 2027, 2028. Selected configurations are in `results/logs/best_hyperparameters.json` and `run_manifest.json`.

Evaluation uses MAE, RMSE, sMAPE, WAPE, MASE, and R² (RMSE is the primary ranking metric; MAPE is avoided because of valid zeros). Diebold–Mariano tests with HAC variance and Holm correction compare each model against Seasonal Naive-7 (`results/tables/diebold_mariano_tests.csv`). Every selection decision was made before the test period was accessed.

## Reproducing the experiment

The experiment was developed in Google Colab with a GPU runtime.

1. Install dependencies (Python 3.13): `pip install -r requirements.txt`.
2. Open `code/Hue_Visitors_Linear_DL_Ensemble.ipynb` (locally with Jupyter, or in Colab).
3. Point the notebook's data directory at this repository's `data/` folder, then run all cells. The notebook rebuilds the tables, figures, predictions, and trained models that are mirrored under `results/`.

Because deep-model training is stochastic, exact figures can differ slightly across hardware and library builds; the versions used for the paper are pinned in `results/logs/environment_versions.json`. The pre-computed artifacts in `results/` let you inspect and re-plot the paper's numbers without retraining.

## Citation

If you use this repository, please cite the paper (see `CITATION.cff`):

> Tran Thai Hoa, "Global GRU-Based Multi-Horizon Visitor Forecasting for Digital Heritage Management in Hue, Vietnam," in *Proc. Int. Conf. on Digital Economy and Fintech Innovation (DEFI 2026)*, 2026.

## License

Code is released under the MIT License (`LICENSE`). The visitor data are redistributed for research reproducibility with credit to the Hue Monuments Conservation Centre.
