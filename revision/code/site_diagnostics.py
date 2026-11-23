"""Site-level error diagnostics for Reviewer 1 #4 (low R2 at Tu Duc) and the revision loss-curve figure.

Outputs: bias decomposition of GRU test errors by site and horizon, monthly mean error, Tu Duc share of four-site
visitors by season and year, cross-site log correlations by split, and Fig. 4 (Stage-1 loss curves).
Usage: python site_diagnostics.py <repo_dir> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_csv(REPO / 'results/tables/predictions_test.csv', encoding='utf-8-sig')
P = P[P.scenario.eq('operational_closure') & P.model.eq('GRU')].copy()
P['err'] = P.prediction - P.actual
rows = []
for (s, h), d in P.groupby(['destination', 'horizon']):
    mse = (d.err ** 2).mean(); b = d.err.mean(); var = d.actual.var(ddof=0)
    rows.append(dict(destination=s, horizon=h, mean_actual=d.actual.mean(), cv_actual=d.actual.std(ddof=0) / d.actual.mean(),
                     rmse=np.sqrt(mse), mean_error=b, bias_share_of_mse=b ** 2 / mse, r2=1 - mse / var))
pd.DataFrame(rows).to_csv(OUT / 'gru_bias_decomposition.csv', index=False)
P['month'] = pd.to_datetime(P.target_date).dt.to_period('M').astype(str)
P[P.horizon.eq(1)].pivot_table(index='month', columns='destination', values='err').to_csv(OUT / 'gru_monthly_mean_error_h1.csv')

frames = []
for f in sorted((REPO / 'data').glob('hue_*_daily_visitors_clean.csv')):
    d = pd.read_csv(f, encoding='utf-8-sig', parse_dates=['date']); frames.append(d)
D = pd.concat(frames); D = D[D.date >= '2022-11-10']
W = D.pivot_table(index='date', columns='destination', values='visitors')
td = [c for c in W.columns if 'Tự Đức' in c][0]
M = W.resample('MS').mean(); share = M[td] / M.sum(axis=1)
sh = pd.DataFrame({'share': share, 'year': share.index.year, 'month': share.index.month})
season = {6: 'Jun-Aug', 7: 'Jun-Aug', 8: 'Jun-Aug', 2: 'Feb-Apr', 3: 'Feb-Apr', 4: 'Feb-Apr'}
sh['season'] = sh.month.map(season)
sh.dropna().groupby(['season', 'year']).share.mean().unstack().to_csv(OUT / 'tu_duc_share_by_season.csv')
sp = D.drop_duplicates('date').set_index('date').split
corr = {s: np.log1p(W[sp.reindex(W.index).eq(s)]).corr()[td].drop(td) for s in ['train', 'validation', 'test']}
pd.DataFrame(corr).to_csv(OUT / 'tu_duc_log_correlation_by_split.csv')

# Fig. 4: Stage-1 training and validation loss, GRU and LSTM (one IEEE column wide)
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.45), sharey=True)
for ax, name in zip(axes, ['GRU', 'LSTM']):
    h = pd.read_csv(REPO / f'results/logs/{name}_training_history.csv', encoding='utf-8-sig')
    ep = h['epoch'] + 1; best = int(h['val_loss'].idxmin()) + 1
    ax.plot(ep, h['loss'], color='#1f4e79', lw=1.0, label='Training')
    ax.plot(ep, h['val_loss'], color='#c55a11', lw=1.0, label='Validation')
    ax.axvline(best, color='0.4', lw=0.7, ls='--')
    ax.text(best + 1, 0.66, f'best = {best}', fontsize=6, color='0.3')
    ax.set_title(name, fontsize=7.5, pad=2); ax.set_xlabel('Epoch', labelpad=1)
    ax.set_ylim(0, 0.72); ax.grid(alpha=0.25, lw=0.4)
    for sp_ in ['top', 'right']:
        ax.spines[sp_].set_visible(False)
axes[0].set_ylabel('Huber loss', labelpad=1)
axes[0].legend(frameon=False, fontsize=6, loc='center right')
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(OUT / 'fig4_loss_curves.png', dpi=600)
print('done')
