"""Pairwise forecast-accuracy tests (Reviewer 1 #2, Reviewer 2 #3) on any long-format prediction table.

Diebold-Mariano with squared-error loss, Bartlett HAC variance with maximum lag h-1, Harvey-Leybourne-Newbold
correction and Student-t reference; Holm adjustment within pre-defined families; moving-block bootstrap 95%
intervals for RMSE ratios; 90% model confidence set (stationary bootstrap).
Usage: python pairwise_tests.py <predictions.csv> <out_dir> [focal=GRU] [scenario=operational_closure]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests
from arch.bootstrap import MCS

SRC, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FOCAL = sys.argv[3] if len(sys.argv) > 3 else 'GRU'
SCEN = sys.argv[4] if len(sys.argv) > 4 else 'operational_closure'
BLOCK, B_CI, B_MCS = 14, 4000, 5000

P = pd.read_csv(SRC, encoding='utf-8-sig')
if 'scenario' in P:
    P = P[P.scenario.eq(SCEN)]
P = P[~P.model.astype(str).str.contains('_seed')].copy()
P['origin_date'] = P.origin_date.astype(str).str[:10]
P['se'] = (P.actual - P.prediction) ** 2
models = sorted(P.model.unique()); others = [m for m in models if m != FOCAL]
HS = sorted(P.horizon.unique())


def dm_hln(d, h):
    d = np.asarray(d, float); n = len(d); c = d - d.mean(); v = np.mean(c ** 2)
    for k in range(1, h):
        v += 2 * (1 - k / h) * np.mean(c[k:] * c[:-k])
    s = d.mean() / np.sqrt(max(v, 1e-12) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), n - 1)))


def mbb_ratio(a, b, seed=0):
    rng = np.random.default_rng(seed); n = len(a); nb = int(np.ceil(n / BLOCK))
    idx = (rng.integers(0, n - BLOCK + 1, size=(B_CI, nb))[:, :, None] + np.arange(BLOCK)).reshape(B_CI, -1)[:, :n]
    return np.percentile(np.sqrt(a[idx].mean(1) / b[idx].mean(1)), [2.5, 97.5])


W = P.pivot_table(index=['origin_date', 'horizon', 'destination'], columns='model', values='se').reset_index()
rows = []
for h in HS:
    g = W[W.horizon.eq(h)]
    agg = g.groupby('origin_date')[models].mean().sort_index()  # mean squared error over sites per origin
    for c in others:
        s, p = dm_hln(agg[FOCAL] - agg[c], h); lo, hi = mbb_ratio(agg[FOCAL].values, agg[c].values)
        rows.append(dict(level='aggregate', destination='All', horizon=h, model=FOCAL, comparator=c, n=len(agg),
                         rmse_model=np.sqrt(agg[FOCAL].mean()), rmse_comparator=np.sqrt(agg[c].mean()),
                         rmse_ratio=np.sqrt(agg[FOCAL].mean() / agg[c].mean()), ci_low=lo, ci_high=hi, DM=s, p_value=p))
    for st in sorted(g.destination.unique()):
        gs = g[g.destination.eq(st)].sort_values('origin_date')
        for c in others:
            s, p = dm_hln(gs[FOCAL] - gs[c], h)
            rows.append(dict(level='site', destination=st, horizon=h, model=FOCAL, comparator=c, n=len(gs),
                             rmse_model=np.sqrt(gs[FOCAL].mean()), rmse_comparator=np.sqrt(gs[c].mean()),
                             rmse_ratio=np.sqrt(gs[FOCAL].mean() / gs[c].mean()), DM=s, p_value=p))
R = pd.DataFrame(rows)
for lvl in ['aggregate', 'site']:
    m = R.level.eq(lvl); R.loc[m, 'p_holm'] = multipletests(R.loc[m, 'p_value'], method='holm')[1]
R.to_csv(OUT / f'pairwise_dm_{FOCAL}.csv', index=False)

# every model vs Seasonal Naive-7 at site level (re-run of the original family with the HLN correction)
if 'SeasonalNaive7' in models:
    sn = []
    for h in HS:
        g = W[W.horizon.eq(h)]
        for st in sorted(g.destination.unique()):
            gs = g[g.destination.eq(st)].sort_values('origin_date')
            for c in [m for m in models if m != 'SeasonalNaive7']:
                s, p = dm_hln(gs[c] - gs['SeasonalNaive7'], h)
                sn.append(dict(destination=st, horizon=h, model=c, benchmark='SeasonalNaive7', DM=s, p_value=p))
    SN = pd.DataFrame(sn); SN['p_holm'] = multipletests(SN.p_value, method='holm')[1]
    SN.to_csv(OUT / 'dm_vs_seasonal_naive_hln.csv', index=False)
    print('vs Seasonal Naive-7 (site level, HLN, Holm over', len(SN), '):', int((SN.p_holm < 0.05).sum()), 'significant')
    print(SN[SN.p_holm < 0.05].round(4).to_string(index=False))

mcs_rows = []
for h in HS:
    L = P[P.horizon.eq(h)].pivot_table(index='origin_date', columns='model', values='se', aggfunc='mean').sort_index()
    mc = MCS(L, size=0.10, reps=B_MCS, block_size=BLOCK, method='R', bootstrap='stationary', seed=1); mc.compute()
    for mname, pv in mc.pvalues['Pvalue'].items():
        mcs_rows.append(dict(horizon=h, model=mname, mcs_pvalue=float(pv), in_mcs90=mname in list(mc.included)))
M = pd.DataFrame(mcs_rows); M.to_csv(OUT / 'model_confidence_set.csv', index=False)
pd.set_option('display.width', 220)
print(R[R.level.eq('aggregate')][['horizon', 'comparator', 'rmse_ratio', 'ci_low', 'ci_high', 'DM', 'p_value', 'p_holm']].round(3).to_string(index=False))
print(M[M.in_mcs90].groupby('horizon').model.apply(list).to_string())
