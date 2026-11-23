"""Summarize rolling-origin predictions: RMSE by fold, model and horizon; ranks; clean-block summary.
Usage: python summarize_rolling.py <rolling_dir> <out_dir>
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd

D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(D / 'rolling_pred_F*.csv')))], ignore_index=True)
P['se'] = (P.actual - P.prediction) ** 2
main = P[~P.model.str.contains('_seed')]
main.to_csv(OUT / 'rolling_predictions_long.csv', index=False)

byh = main.groupby(['fold', 'model', 'horizon']).se.mean().pipe(np.sqrt).rename('RMSE').reset_index()
byh.to_csv(OUT / 'rolling_rmse_by_fold_model_horizon.csv', index=False)
mean = byh.groupby(['fold', 'model']).RMSE.mean().unstack('fold')
rank = mean.rank(axis=0)
summary = mean.copy()
summary['mean_rank'] = rank.mean(axis=1)
summary['wins'] = (rank == 1).sum(axis=1)
summary['pooled_RMSE_all_folds'] = main.groupby(['model', 'horizon']).se.mean().pipe(np.sqrt).groupby('model').mean()
clean = main[main.fold.isin(['F6', 'F7'])]
summary['pooled_RMSE_F6_F7'] = clean.groupby(['model', 'horizon']).se.mean().pipe(np.sqrt).groupby('model').mean()
summary = summary.sort_values('mean_rank')
summary.to_csv(OUT / 'rolling_summary.csv')
pooled_h = main.groupby(['model', 'horizon']).se.mean().pipe(np.sqrt).unstack()
pooled_h.to_csv(OUT / 'rolling_pooled_rmse_by_horizon.csv')

sw = P[P.model.str.contains('_seed')].copy(); sw['base'] = sw.model.str.replace(r'_seed\d+', '', regex=True)
ps = sw.groupby(['fold', 'base', 'model', 'horizon']).se.mean().pipe(np.sqrt).groupby(['fold', 'base', 'model']).mean()
ps.groupby(['fold', 'base']).agg(['mean', 'std']).to_csv(OUT / 'rolling_per_seed_rmse.csv')
pd.set_option('display.width', 250)
print(summary.round(1).to_string())
print(pooled_h.round(1).to_string())
