"""Expanding-window rolling-origin evaluation (revision, Reviewer 1 #3 and Reviewer 2 #1-#2).

Seven three-month evaluation blocks from Dec 2024 to Aug 2026. For each block, the 91 days before it
form an inner validation window (early stopping, Ridge alpha, ensemble weights); scalers are estimated
on the rows before that window; models are refitted on training + inner validation with three seeds.
Hyperparameters stay fixed at the Table I configurations.
Usage: python rolling_origin.py <data_dir> <out_dir> [fold ids, e.g. F7,F1]
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import hue_common as hc

DATA_DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
FOLDS = {'F1': ('2024-12-01', '2025-02-28'), 'F2': ('2025-03-01', '2025-05-31'), 'F3': ('2025-06-01', '2025-08-31'),
         'F4': ('2025-09-01', '2025-11-30'), 'F5': ('2025-12-01', '2026-02-28'), 'F6': ('2026-03-01', '2026-05-31'),
         'F7': ('2026-06-01', '2026-08-24')}
INNER_VAL_DAYS = 91
order = sys.argv[3].split(',') if len(sys.argv) > 3 else list(FOLDS)
data = hc.Data(DATA_DIR)
logf = open(OUT / 'rolling_origin.log', 'a')


def log(m):
    print(m, flush=True); logf.write(m + '\n'); logf.flush()


for fold in order:
    b0d, b1d = FOLDS[fold]
    b0, b1 = data.idx(b0d), data.idx(b1d)
    vs = b0 - INNER_VAL_DAYS
    prep = hc.Prepared(data, fit_mask=np.arange(data.T) < vs)
    test_o = np.array([o for o in range(b0, b1 + 1) if o + hc.MAXH - 1 <= data.T - 1])
    val_o = np.array([o for o in range(vs, b0) if o + hc.MAXH - 1 <= b0 - 1])
    log(f'== {fold}: test {b0d}..{b1d} ({len(test_o)} origins), inner validation from {data.dates[vs].date()} '
        f'({len(val_o)} origins), fit rows < {data.dates[vs].date()}')
    t_fold = time.time()
    val_pred, test_pred, meta = {}, {}, {'fold': fold}
    sn = hc.seasonal_naive(data, test_o)
    frames = [hc.to_long(data, sn, test_o, 'SeasonalNaive7', fold=fold)]
    # ARX-Ridge
    L = hc.PARAMS['ARX-Ridge']['lookback']
    tr_o = np.array([o for o in range(L, vs) if o + hc.MAXH - 1 <= vs - 1])
    rb = {k: prep.bundle(v, L) for k, v in [('train', tr_o), ('validation', val_o), ('test', test_o)]}
    val_pred['ARX-Ridge'], test_pred['ARX-Ridge'], meta['ridge_alpha'] = hc.fit_ridge(prep, rb['train'], rb['validation'], rb['test'])
    y_val = rb['validation']['y_raw']
    # deep models
    for name in hc.DL_MODELS:
        p = hc.PARAMS[name]; L = p['lookback']
        tr_o = np.array([o for o in range(L, vs) if o + hc.MAXH - 1 <= vs - 1])
        b = {k: prep.bundle(v, L) for k, v in [('train', tr_o), ('validation', val_o), ('test', test_o)]}
        s1, s2, best, _ = hc.two_stage(name, p, b['train'], b['validation'], {'test': b['test']}, log=log)
        val_pred[name] = prep.inverse(s1['validation'])
        seedwise = [prep.inverse(x) for x in s2['test']]
        test_pred[name] = np.mean(seedwise, axis=0)
        for k, sd in enumerate(hc.FINAL_SEEDS):
            frames.append(hc.to_long(data, seedwise[k], test_o, f'{name}_seed{sd}', fold=fold))
        meta[f'{name}_best_epoch'] = best
    ens, wrows = hc.ensemble(val_pred, test_pred, y_val)
    test_pred['Ensemble'] = ens
    for name, pr in test_pred.items():
        frames.append(hc.to_long(data, pr, test_o, name, fold=fold))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT / f'rolling_pred_{fold}.csv', index=False)
    pd.DataFrame(wrows).assign(fold=fold).to_csv(OUT / f'rolling_ensemble_weights_{fold}.csv', index=False)
    json.dump(meta, open(OUT / f'rolling_meta_{fold}.json', 'w'), indent=1)
    agg = out[~out.model.str.contains('seed')].groupby(['model', 'horizon']).apply(
        lambda d: np.sqrt(np.mean((d.actual - d.prediction) ** 2)), include_groups=False).unstack()
    log(agg.round(1).to_string())
    log(f'== {fold} done in {time.time() - t_fold:.0f}s')
