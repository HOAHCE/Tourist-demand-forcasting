# Camera-ready revision (DEFI 2026, November 2026)

This folder holds the code and outputs of the analyses added in response to the two reviewers. The original
experiment (`code/Hue_Visitors_Linear_DL_Ensemble.ipynb`, outputs in `results/`) is unchanged; every script here
reads the same `data/` files and reuses the notebook's features, scaling, sample construction and two-stage protocol
(`code/hue_common.py`).

## Reviewer comments → scripts → outputs

| Comment | What was added | Script | Output folder |
|---|---|---|---|
| R1 #1 — training, validation and test RMSE by fitting stage, loss curves, per-seed variability | In-sample RMSE of the published Stage-2 checkpoints on training+validation, Stage-1 validation RMSE, Stage-2 test RMSE, ratios to Seasonal Naive-7, per-seed test RMSE; Fig. 4 loss curves | `code/generalization_diagnostics.py`, `code/site_diagnostics.py` | `results/generalization/`, `results/site_diagnostics/fig4_loss_curves.png` |
| R1 #1–#2, R2 #3 — rolling-origin evaluation | Expanding-window evaluation over seven three-month blocks (Dec 2024–Aug 2026); 91-day inner validation per block; Table I hyperparameters fixed; three seeds | `code/rolling_origin.py`, `code/summarize_rolling.py`, `code/pairwise_tests.py` | `results/rolling/` |
| R1 #3, R2 #2 — direct comparisons with the strongest alternatives, uncertainty under temporal dependence, individual DM results | Diebold–Mariano tests of GRU against every model (HLN correction, Bartlett HAC with maximum lag h−1, Holm within family), moving-block bootstrap intervals for RMSE ratios, 90% model confidence sets; site-level comparisons with Seasonal Naive-7 re-run with the HLN correction | `code/pairwise_tests.py` | `results/pairwise_original_split/` |
| R2 #1 — models trained separately for each site | Joint (proposed), pooled (shared network, own-site inputs + site indicator) and local (one model per site) GRU and Ridge variants under the same protocol | `code/joint_local_ablation.py`, `code/compare_joint_local.py` | `results/joint_local/` |
| R2 #4 — low R² at Tu Duc Tomb | Bias/variance decomposition of GRU errors by site and horizon, monthly mean errors, Tu Duc share of four-site visits by season and year, cross-site log correlations by split | `code/site_diagnostics.py` | `results/site_diagnostics/` |

## Key results

- **Pairwise tests (original split, aggregate losses).** GRU's RMSE was 5–58% lower than every alternative. After Holm
  correction GRU beat the 1D-CNN at all horizons, the Transformer at 1 and 7 days and Seasonal Naive-7 at 1 day; it was
  not significantly better than LSTM, ARX-Ridge or the ensemble. 90% MCS: {GRU, ensemble} at H=1, {GRU, LSTM, ensemble}
  at H=7, {GRU} at H=14. Site level: 16 of 72 GRU comparisons significant, none against LSTM or the ensemble
  (`results/pairwise_original_split/`).
- **Generalization.** Mean RMSE ratio of GRU to Seasonal Naive-7: 0.50 in the Stage-2 fitting data, 0.70 in Stage-1
  validation, 0.54 in test. Single GRU networks: test RMSE 782 ± 85 (mean over horizons) versus 636 for the three-seed
  average (`results/generalization/`).
- **Rolling origin (7 blocks, 619 origins).** GRU won 4 blocks (F2, F4, F6, F7), LSTM 2 (F3, F5), ARX-Ridge 1 (F1). Pooled
  RMSE at H=1/7/14: GRU 609/735/762, LSTM 622/751/785, ensemble 626/755/779; differences among GRU, LSTM, ARX-Ridge and
  the ensemble not significant (`results/rolling/`).
- **Joint vs local.** Joint GRU 565/724/728 versus local GRU 724/941/972 (22–25% lower RMSE) and pooled GRU 543/762/794
  (not significantly different). For Ridge, joint ARX-Ridge had 6–11% higher RMSE than local and pooled Ridge
  (`results/joint_local/`).
- **Tu Duc.** Bias accounts for 48% of the one-day MSE; Tu Duc's June–August share of four-site visits fell from 16.9%
  (2025) to 14.4% (2026) (`results/site_diagnostics/`).

## Reproducing

Python 3.13 with the packages in `../requirements.txt` (the revision also uses `arch` and `joblib`). From the
repository root:

```bash
# pairwise tests on the published test predictions (no training)
python revision/code/pairwise_tests.py results/tables/predictions_test.csv revision/results/pairwise_original_split GRU
# generalization diagnostics from the published checkpoints (no training)
python revision/code/generalization_diagnostics.py . revision/results/generalization
python revision/code/site_diagnostics.py . revision/results/site_diagnostics
# rolling-origin evaluation (trains all models in 7 blocks; about 1 hour on 2 CPU cores)
python revision/code/rolling_origin.py data revision/results/rolling
python revision/code/summarize_rolling.py revision/results/rolling revision/results/rolling
python revision/code/pairwise_tests.py revision/results/rolling/rolling_predictions_long.csv revision/results/rolling/pairwise GRU
# joint vs pooled vs local (about 30 minutes on 2 CPU cores)
for v in ridge_all joint_gru local_gru pooled_gru; do python revision/code/joint_local_ablation.py data revision/results/joint_local $v; done
python revision/code/compare_joint_local.py revision/results/joint_local results/tables/predictions_test.csv
```

The rolling-origin and joint-versus-local experiments were run on CPU (TensorFlow 2.21). Neural-network training is
not bit-wise reproducible across hardware and library versions, so the joint GRU in these experiments differs from the
GPU run in `results/` (one-day aggregate RMSE 565 versus 492); comparisons are made only within each experiment.
The published checkpoints in `results/models/` reproduce the paper's test RMSE when their three seeds are averaged.
