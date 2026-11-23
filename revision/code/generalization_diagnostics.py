"""Generalization diagnostics for Reviewer 2 #1, using the published (GPU) checkpoints and predictions.

Reports, per model and horizon: Stage-2 in-sample RMSE on training+validation (saved checkpoints),
Stage-1 validation RMSE (published validation predictions), Stage-2 test RMSE (published test predictions),
ratios to Seasonal Naive-7 in each period, and per-seed test RMSE.
Usage: python generalization_diagnostics.py <repo_dir> <out_dir>
"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import hue_common as hc
import joblib
from tensorflow import keras

REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
data = hc.Data(REPO / 'data')
prep = hc.Prepared(data, fit_mask=data.split == 'train')


def split_origins(L):
    out = {'train': [], 'validation': [], 'test': []}
    for o in range(L, data.T - hc.MAXH + 1):
        sp = set(data.split[[o + h - 1 for h in hc.HORIZONS]].tolist())
        if len(sp) == 1:
            out[sp.pop()].append(o)
    return {k: np.array(v) for k, v in out.items()}


def rmse_h(pred, y):
    return [float(np.sqrt(np.mean((pred[:, i] - y[:, i]) ** 2))) for i in range(hc.NH)]


rows, seed_rows = [], []
for name in ['SeasonalNaive7', 'ARX-Ridge'] + hc.DL_MODELS:
    L = hc.PARAMS.get(name, {'lookback': 112})['lookback']
    orig = split_origins(L)
    tv = np.concatenate([orig['train'], orig['validation']])
    b_tv, b_te = prep.bundle(tv, L), prep.bundle(orig['test'], L)
    if name == 'SeasonalNaive7':
        fit = rmse_h(hc.seasonal_naive(data, tv), b_tv['y_raw'])
        seeds = [rmse_h(hc.seasonal_naive(data, orig['test']), b_te['y_raw'])]
    elif name == 'ARX-Ridge':
        m = joblib.load(REPO / 'results/models/ARX_Ridge.joblib')
        fit = rmse_h(prep.inverse(m.predict(hc.flat(b_tv))), b_tv['y_raw'])
        seeds = [rmse_h(prep.inverse(m.predict(hc.flat(b_te))), b_te['y_raw'])]
    else:
        fits, seeds = [], []
        for sd in hc.FINAL_SEEDS:
            m = keras.models.load_model(REPO / f'results/models/{name.replace("-", "_")}_seed{sd}.keras', safe_mode=False, compile=False)
            fits.append(prep.inverse(m.predict([b_tv['X'], b_tv['F']], verbose=0)))
            seeds.append(rmse_h(prep.inverse(m.predict([b_te['X'], b_te['F']], verbose=0)), b_te['y_raw']))
        fit = rmse_h(np.mean(fits, axis=0), b_tv['y_raw'])
    for k, s in enumerate(seeds):
        for hi, h in enumerate(hc.HORIZONS):
            seed_rows.append(dict(model=name, seed=hc.FINAL_SEEDS[k] if len(seeds) > 1 else None, horizon=h, test_RMSE=s[hi]))
    for hi, h in enumerate(hc.HORIZONS):
        rows.append(dict(model=name, horizon=h, insample_trainval_RMSE=fit[hi], n_trainval=len(tv)))
    print(name, 'in-sample', np.round(fit, 1), 'per-seed test', np.round(seeds, 1).tolist(), flush=True)

G = pd.DataFrame(rows)
pub = lambda f: pd.read_csv(REPO / f'results/tables/{f}', encoding='utf-8-sig')
m = pub('metrics_aggregate.csv'); m = m[m.scenario.eq('operational_closure')]
val = m[m.split.eq('validation')].set_index(['model', 'horizon']).RMSE.rename('stage1_validation_RMSE')
test = m[m.split.eq('test')].set_index(['model', 'horizon']).RMSE.rename('stage2_test_RMSE')
G = G.join(val, on=['model', 'horizon']).join(test, on=['model', 'horizon'])
sn = G[G.model.eq('SeasonalNaive7')].set_index('horizon')
for c in ['insample_trainval_RMSE', 'stage1_validation_RMSE', 'stage2_test_RMSE']:
    G[c.replace('_RMSE', '_ratio_to_SN7')] = G[c] / G.horizon.map(sn[c])
S = pd.DataFrame(seed_rows)
ps = S[S.seed.notna()].groupby(['model', 'horizon']).test_RMSE.agg(['mean', 'std', 'min', 'max']).add_prefix('per_seed_test_RMSE_')
G = G.join(ps, on=['model', 'horizon'])
G.to_csv(OUT / 'generalization_by_stage.csv', index=False)
S.to_csv(OUT / 'per_seed_test_rmse.csv', index=False)
pd.set_option('display.width', 250)
print(G.round(3).to_string(index=False))
