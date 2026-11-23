"""Compare joint, pooled and local variants (Reviewer 1 #1). Usage: python compare_joint_local.py <joint_local_dir> <published predictions_test.csv>"""
import sys, numpy as np, pandas as pd
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests
from arch.bootstrap import MCS

D = sys.argv[1]; PUB = sys.argv[2]
fr = [pd.read_csv(f'{D}/pred_{v}.csv') for v in ['joint_gru', 'local_gru', 'pooled_gru', 'ridge_all']]
P = pd.concat(fr, ignore_index=True)
pub = pd.read_csv(PUB, encoding='utf-8-sig'); pub = pub[pub.scenario.eq('operational_closure') & pub.model.eq('GRU')]
pub = pub.assign(model='Joint-GRU (published, GPU)', origin_date=pub.origin_date.str[:10], target_date=pub.target_date.str[:10])
P = pd.concat([P, pub[P.columns]], ignore_index=True)
P['se'] = (P.actual - P.prediction) ** 2; P['err'] = P.prediction - P.actual
main = [m for m in P.model.unique() if 'seed' not in m]
rm = lambda d: np.sqrt(d.se.mean())

print('=== Aggregate test RMSE (seed-averaged)')
A = P[P.model.isin(main)].groupby(['model', 'horizon']).apply(rm, include_groups=False).unstack().round(1)
A['mean'] = A.mean(1).round(1); print(A.to_string())

print('\n=== Per-seed aggregate RMSE, mean ± sd over seeds')
sw = P[P.model.str.contains('seed')].copy(); sw['base'] = sw.model.str.replace(r'_seed\d+', '', regex=True)
R = sw.groupby(['base', 'model', 'horizon']).apply(rm, include_groups=False).reset_index(name='rmse')
print(R.groupby(['base', 'horizon']).rmse.agg(['mean', 'std', 'min', 'max']).round(1).to_string())
R[R.horizon.eq(1)].groupby('base').rmse.agg(['mean', 'std', 'min', 'max']).reset_index().to_csv(f'{D}/per_seed_summary.csv', index=False)

print('\n=== Site x horizon RMSE, GRU variants and Ridge variants')
gm = ['Joint-GRU', 'Local-GRU', 'Pooled-GRU', 'Joint-Ridge', 'Local-Ridge', 'Pooled-Ridge']
Sx = P[P.model.isin(gm)].groupby(['destination', 'horizon', 'model']).apply(rm, include_groups=False).unstack().round(0)
print(Sx[gm].to_string())
print('\n=== Mean error (bias) by site, H=1')
print(P[P.model.isin(gm) & P.horizon.eq(1)].pivot_table(index='destination', columns='model', values='err').round(0)[gm].to_string())

def dm(d, h):
    d = np.asarray(d, float); n = len(d); c = d - d.mean(); v = np.mean(c ** 2)
    for k in range(1, h):
        v += 2 * (1 - k / h) * np.mean(c[k:] * c[:-k])
    s = d.mean() / np.sqrt(max(v, 1e-12) / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return s, 2 * (1 - tdist.cdf(abs(s), n - 1))

def mbb(a, b, block=14, B=4000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); nb = int(np.ceil(n / block))
    idx = (rng.integers(0, n - block + 1, size=(B, nb))[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    return np.percentile(np.sqrt(a[idx].mean(1) / b[idx].mean(1)), [2.5, 97.5])

W = P[P.model.isin(gm)].pivot_table(index=['origin_date', 'horizon', 'destination'], columns='model', values='se').reset_index()
rows = []
for base, alts in [('Joint-GRU', ['Local-GRU', 'Pooled-GRU']), ('Joint-Ridge', ['Local-Ridge', 'Pooled-Ridge'])]:
    for h in [1, 7, 14]:
        g = W[W.horizon.eq(h)]; agg = g.groupby('origin_date')[[base] + alts].mean().sort_index()
        for a in alts:
            s, p = dm(agg[base] - agg[a], h); lo, hi = mbb(agg[base].values, agg[a].values)
            rows.append(dict(level='aggregate', site='All', h=h, comp=f'{base} vs {a}', ratio=np.sqrt(agg[base].mean() / agg[a].mean()), lo=lo, hi=hi, DM=s, p=p))
            for st in sorted(g.destination.unique()):
                gs = g[g.destination.eq(st)].sort_values('origin_date'); s2, p2 = dm(gs[base] - gs[a], h)
                rows.append(dict(level='site', site=st, h=h, comp=f'{base} vs {a}', ratio=np.sqrt(gs[base].mean() / gs[a].mean()), DM=s2, p=p2))
Rr = pd.DataFrame(rows)
for lvl in ['aggregate', 'site']:
    m = Rr.level.eq(lvl); Rr.loc[m, 'p_holm'] = multipletests(Rr.loc[m, 'p'], method='holm')[1]
print('\n=== DM (HLN, lag h-1), ratio <1 favours the joint model')
print(Rr[Rr.level.eq('aggregate')].round(3).to_string(index=False))
print(Rr[Rr.level.eq('site') & Rr.comp.str.contains('GRU')][['site', 'h', 'comp', 'ratio', 'p', 'p_holm']].round(3).to_string(index=False))
Rr.to_csv(f'{D}/global_local_dm.csv', index=False)

print('\n=== MCS 90% among six variants')
for h in [1, 7, 14]:
    L = P[P.model.isin(gm) & P.horizon.eq(h)].pivot_table(index='origin_date', columns='model', values='se', aggfunc='mean').sort_index()
    mc = MCS(L, size=0.10, reps=5000, block_size=14, method='R', bootstrap='stationary', seed=1); mc.compute()
    print(h, list(mc.included), mc.pvalues['Pvalue'].round(3).to_dict())
A.to_csv(f'{D}/global_local_aggregate_rmse.csv'); Sx.to_csv(f'{D}/global_local_site_rmse.csv')
