"""Global (joint) vs local vs pooled ablation for the Hue four-site paper.

Re-implements the notebook's preprocessing and two-stage protocol so that the
only thing that changes between variants is how the four sites are combined.
Usage: python joint_local_ablation.py <data_dir> <out_dir> <variant> [seeds]
variant in {joint_gru, local_gru, pooled_gru, ridge_all}
"""
import os, sys, json, time, math, random
from pathlib import Path
import numpy as np, pandas as pd

DATA_DIR, OUT_DIR, VARIANT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SEEDS = [int(s) for s in sys.argv[4].split(',')] if len(sys.argv) > 4 else [2026, 2027, 2028]
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import holidays

SEED = 2026
HORIZONS = [1, 7, 14]; MAXH = 14; H = len(HORIZONS)
PATIENCE = 18; MAX_EPOCHS_STAGE1 = 200
COMMON_START = pd.Timestamp('2022-11-10')
CLOSURES = {pd.Timestamp(d) for d in ['2025-10-28', '2025-10-29', '2025-10-30']}
FILES = [('Dai_Noi', 'Đại Nội', 'hue_dai_noi_daily_visitors_clean.csv'),
         ('Khai_Dinh', 'Lăng Khải Định', 'hue_lang_khai_dinh_daily_visitors_clean.csv'),
         ('Minh_Mang', 'Lăng Minh Mạng', 'hue_lang_minh_mang_daily_visitors_clean.csv'),
         ('Tu_Duc', 'Lăng Tự Đức', 'hue_lang_tu_duc_daily_visitors_clean.csv')]
KEYS = [f[0] for f in FILES]; NAMES = [f[1] for f in FILES]; S = len(FILES)
GRU_P = dict(lookback=112, units=32, n_layers=1, dense_units=96, dropout=0.2642995828650003,
             learning_rate=0.0011882822203134762, batch_size=16)


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)


# ---------------- data (identical to notebook cells 2 and 4) ----------------
frames = []
for key, _, fn in FILES:
    d = pd.read_csv(DATA_DIR / fn, encoding='utf-8-sig')
    d['date'] = pd.to_datetime(d['date'])
    d = d.sort_values('date').drop_duplicates('date', keep=False)
    d = d[d['date'] >= COMMON_START]
    d['split'] = d['split'].astype(str).str.strip().str.lower()
    d = d[['date', 'split', 'domestic_visitors', 'international_visitors', 'visitors']].rename(columns={
        'domestic_visitors': f'{key}__domestic', 'international_visitors': f'{key}__international',
        'visitors': f'{key}__visitors'})
    frames.append(d)
wide = frames[0]
for i, p in enumerate(frames[1:], 1):
    wide = wide.merge(p, on='date', suffixes=('', f'_s{i}'))
    assert (wide['split'] == wide[f'split_s{i}']).all()
    wide = wide.drop(columns=f'split_s{i}')
wide = wide.sort_values('date').reset_index(drop=True)
dates = pd.DatetimeIndex(wide['date'])
vn_h = holidays.Vietnam(years=range(dates.min().year, dates.max().year + 1))


def calendar(idx):
    dow, month, doy = idx.dayofweek.to_numpy(), idx.month.to_numpy(), idx.dayofyear.to_numpy()
    cols = [np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), np.sin(2*np.pi*(month-1)/12), np.cos(2*np.pi*(month-1)/12),
            np.sin(2*np.pi*(doy-1)/365.25), np.cos(2*np.pi*(doy-1)/365.25), (dow >= 5).astype(float),
            np.array([float(t.date() in vn_h) for t in idx]), np.array([float(t.normalize() in CLOSURES) for t in idx]),
            (idx - COMMON_START).days.to_numpy()/365.25]
    cols += [(dow == k).astype(float) for k in range(7)]
    return np.column_stack(cols)


cal = calendar(dates)
split = wide['split'].to_numpy(); trmask = split == 'train'
T = len(wide)
hist_log = {k: np.column_stack([np.log1p(wide[f'{k}__{s}'].to_numpy(float)) for s in ['visitors', 'domestic', 'international']]) for k in KEYS}
y_raw_all = np.column_stack([wide[f'{k}__visitors'].to_numpy(float) for k in KEYS])
fut_scaler = StandardScaler().fit(cal[trmask]); fut_scaled = fut_scaler.transform(cal).astype('float32')


def origins(lookback):
    out = {'train': [], 'validation': [], 'test': []}
    for o in range(lookback, T - MAXH + 1):
        ti = [o + h - 1 for h in HORIZONS]
        sp = set(split[ti].tolist())
        if len(sp) == 1:
            out[sp.pop()].append(o)
    return {k: np.array(v) for k, v in out.items()}


def make_xy(site_ids, lookback, onehot=False):
    """Sequence/future/target arrays for the given sites (site-local scaling when one site, shared for joint)."""
    hist = np.concatenate([hist_log[KEYS[s]] for s in site_ids], axis=1)
    seq = np.concatenate([hist, cal], axis=1)
    sc = StandardScaler().fit(seq[trmask]); seq_s = sc.transform(seq).astype('float32')
    if onehot:
        oh = np.zeros((T, S), dtype='float32'); oh[:, site_ids[0]] = 1.0
        seq_s = np.concatenate([seq_s, oh], axis=1)
    tlog = np.log1p(y_raw_all[:, site_ids]); tsc = StandardScaler().fit(tlog[trmask])
    t_s = tsc.transform(tlog).astype('float32')
    return seq_s, t_s, tsc


def bundle(seq_s, t_s, lookback, orig):
    b = {}
    for sp, os_ in orig.items():
        ti = os_[:, None] + np.array(HORIZONS)[None, :] - 1
        b[sp] = dict(X=np.stack([seq_s[o - lookback:o] for o in os_]), F=fut_scaled[ti],
                     y=t_s[ti].reshape(len(os_), -1), o=os_)
    return b


def inv(pred, tsc, n_sites):
    n = pred.shape[0]
    lg = tsc.inverse_transform(pred.reshape(n * H, n_sites))
    return np.clip(np.expm1(lg).reshape(n, H, n_sites), 0, None)


def build_gru(seq_shape, fut_shape, n_out, p):
    si = keras.Input(shape=seq_shape); fi = keras.Input(shape=fut_shape)
    x = layers.GRU(p['units'])(si); x = layers.Dropout(p['dropout'])(x)
    f = layers.Flatten()(fi); f = layers.Dense(p['dense_units'], activation='relu')(f)
    m = layers.Concatenate()([x, f]); m = layers.Dense(p['dense_units'], activation='relu')(m)
    m = layers.Dropout(p['dropout'])(m); out = layers.Dense(n_out)(m)
    model = keras.Model([si, fi], out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=p['learning_rate'], clipnorm=1.0),
                  loss=keras.losses.Huber(delta=1.0))
    return model


def two_stage(tr, va, te, n_out, p, log):
    """Stage 1: train->validation with early stopping; Stage 2: refit train+val for best epoch, seed average."""
    tf.keras.backend.clear_session(); set_seeds(SEED)
    m = build_gru(tr['X'].shape[1:], tr['F'].shape[1:], n_out, p)
    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, min_delta=1e-5),
          keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=max(3, PATIENCE // 2), factor=0.5, min_lr=1e-6)]
    t0 = time.time()
    h = m.fit([tr['X'], tr['F']], tr['y'], validation_data=([va['X'], va['F']], va['y']), epochs=MAX_EPOCHS_STAGE1,
              batch_size=p['batch_size'], shuffle=False, verbose=0, callbacks=cb)
    best = int(np.argmin(h.history['val_loss']) + 1)
    val_pred = m.predict([va['X'], va['F']], verbose=0)
    log(f'  stage1 epochs={len(h.history["loss"])} best={best} ({time.time()-t0:.0f}s)')
    Xtv = np.concatenate([tr['X'], va['X']]); Ftv = np.concatenate([tr['F'], va['F']]); ytv = np.concatenate([tr['y'], va['y']])
    preds = []
    for sd in SEEDS:
        tf.keras.backend.clear_session(); set_seeds(sd); t0 = time.time()
        fm = build_gru(tr['X'].shape[1:], tr['F'].shape[1:], n_out, p)
        fm.fit([Xtv, Ftv], ytv, epochs=max(1, best), batch_size=p['batch_size'], shuffle=False, verbose=0)
        preds.append(fm.predict([te['X'], te['F']], verbose=0)); log(f'  seed {sd} ({time.time()-t0:.0f}s)')
    return val_pred, np.stack(preds), best, pd.DataFrame(h.history)


def to_long(pred, orig_test, model, site_names):
    """pred: (n, H, n_sites) in visitors."""
    rows = []
    for i, o in enumerate(orig_test):
        for hi, h in enumerate(HORIZONS):
            ti = o + h - 1
            for j, nm in enumerate(site_names):
                s = NAMES.index(nm)
                rows.append(dict(model=model, origin_date=dates[o].date().isoformat(), target_date=dates[ti].date().isoformat(),
                                 horizon=h, destination=nm, actual=y_raw_all[ti, s], prediction=float(pred[i, hi, j])))
    return pd.DataFrame(rows)


logf = open(OUT_DIR / f'{VARIANT}.log', 'a')
def log(msg):
    print(msg, flush=True); logf.write(msg + '\n'); logf.flush()

log(f'== {VARIANT} | TF {tf.__version__} | seeds {SEEDS}')
p = GRU_P; L = p['lookback']; orig = origins(L)
log(f'origins: ' + str({k: len(v) for k, v in orig.items()}))
frames_out, meta = [], {}

if VARIANT == 'joint_gru':
    seq_s, t_s, tsc = make_xy(list(range(S)), L)
    b = bundle(seq_s, t_s, L, orig)
    vp, sp_, best, hist = two_stage(b['train'], b['validation'], b['test'], H * S, p, log)
    seedwise = [inv(x, tsc, S) for x in sp_]
    frames_out.append(to_long(np.mean(seedwise, 0), orig['test'], 'Joint-GRU', NAMES))
    for k, sd in enumerate(SEEDS):
        frames_out.append(to_long(seedwise[k], orig['test'], f'Joint-GRU_seed{sd}', NAMES))
    meta['joint'] = dict(best_epoch=best); hist.to_csv(OUT_DIR / 'joint_gru_history.csv', index=False)

elif VARIANT == 'local_gru':
    for s in range(S):
        log(f'site {NAMES[s]}')
        seq_s, t_s, tsc = make_xy([s], L)
        b = bundle(seq_s, t_s, L, orig)
        vp, sp_, best, hist = two_stage(b['train'], b['validation'], b['test'], H, p, log)
        seedwise = [inv(x, tsc, 1) for x in sp_]
        frames_out.append(to_long(np.mean(seedwise, 0), orig['test'], 'Local-GRU', [NAMES[s]]))
        for k, sd in enumerate(SEEDS):
            frames_out.append(to_long(seedwise[k], orig['test'], f'Local-GRU_seed{sd}', [NAMES[s]]))
        meta[NAMES[s]] = dict(best_epoch=best)

elif VARIANT == 'pooled_gru':
    parts = {sp: dict(X=[], F=[], y=[], o=[]) for sp in orig}; scalers = []
    for s in range(S):
        seq_s, t_s, tsc = make_xy([s], L, onehot=True); scalers.append(tsc)
        b = bundle(seq_s, t_s, L, orig)
        for sp in orig:
            for k in ['X', 'F', 'y', 'o']:
                parts[sp][k].append(b[sp][k])
    st = {sp: {k: np.concatenate(v[k]) for k in ['X', 'F', 'y']} for sp, v in parts.items()}
    nte = len(orig['test'])
    vp, sp_, best, hist = two_stage(st['train'], st['validation'], st['test'], H, p, log)
    for k_seed in [None] + list(range(len(SEEDS))):
        raw = sp_.mean(0) if k_seed is None else sp_[k_seed]
        name = 'Pooled-GRU' if k_seed is None else f'Pooled-GRU_seed{SEEDS[k_seed]}'
        for s in range(S):
            seg = raw[s * nte:(s + 1) * nte]
            frames_out.append(to_long(inv(seg, scalers[s], 1), orig['test'], name, [NAMES[s]]))
    meta['pooled'] = dict(best_epoch=best)

elif VARIANT == 'ridge_all':
    alphas = np.logspace(-3, 5, 17); Lr = 28; orr = origins(Lr)
    def flat(bb): return np.concatenate([bb['X'].reshape(len(bb['X']), -1), bb['F'].reshape(len(bb['F']), -1)], 1)
    def fit_ridge(site_ids, onehot=False, pooled=False):
        if pooled:
            per = []
            for s in range(S):
                seq_s, t_s, tsc = make_xy([s], Lr, onehot=True); per.append((bundle(seq_s, t_s, Lr, orr), tsc))
            def cat(sp): return np.concatenate([flat(b_[sp]) for b_, _ in per]), np.concatenate([b_[sp]['y'] for b_, _ in per])
            Xtr, ytr = cat('train'); Xva, yva = cat('validation'); Xte, _ = cat('test')
            def inv_all(pred, n):
                return np.concatenate([inv(pred[k*n:(k+1)*n], per[k][1], 1) for k in range(S)], axis=2)
            nva, nte_ = len(orr['validation']), len(orr['test'])
            yva_raw = np.concatenate([y_raw_all[orr['validation'][:, None] + np.array(HORIZONS) - 1][:, :, [s]] for s in range(S)], 2)
            best = min(alphas, key=lambda a: np.sqrt(np.mean((inv_all(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva), nva) - yva_raw) ** 2)))
            fm = Ridge(alpha=best).fit(np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva]))
            return inv_all(fm.predict(Xte), nte_), best
        seq_s, t_s, tsc = make_xy(site_ids, Lr)
        b = bundle(seq_s, t_s, Lr, orr); n = len(site_ids)
        yva_raw = y_raw_all[orr['validation'][:, None] + np.array(HORIZONS) - 1][:, :, site_ids]
        best = min(alphas, key=lambda a: np.sqrt(np.mean((inv(Ridge(alpha=a).fit(flat(b['train']), b['train']['y']).predict(flat(b['validation'])), tsc, n) - yva_raw) ** 2)))
        fm = Ridge(alpha=best).fit(np.concatenate([flat(b['train']), flat(b['validation'])]), np.concatenate([b['train']['y'], b['validation']['y']]))
        return inv(fm.predict(flat(b['test'])), tsc, n), best
    pj, aj = fit_ridge(list(range(S))); frames_out.append(to_long(pj, orr['test'], 'Joint-Ridge', NAMES)); meta['Joint-Ridge'] = float(aj)
    for s in range(S):
        pl, al = fit_ridge([s]); frames_out.append(to_long(pl, orr['test'], 'Local-Ridge', [NAMES[s]])); meta[f'Local-Ridge {NAMES[s]}'] = float(al)
    pp, ap = fit_ridge(None, pooled=True); frames_out.append(to_long(pp, orr['test'], 'Pooled-Ridge', NAMES)); meta['Pooled-Ridge'] = float(ap)
    log(json.dumps(meta, ensure_ascii=False))

out = pd.concat(frames_out, ignore_index=True)
out.to_csv(OUT_DIR / f'pred_{VARIANT}.csv', index=False)
json.dump(meta, open(OUT_DIR / f'meta_{VARIANT}.json', 'w'), ensure_ascii=False, indent=1)
for m, g in out[~out.model.str.contains('seed')].groupby('model'):
    r = g.groupby('horizon').apply(lambda d: np.sqrt(np.mean((d.actual - d.prediction) ** 2)), include_groups=False)
    log(f'{m}: aggregate test RMSE ' + ', '.join(f'H{h}={v:.1f}' for h, v in r.items()))
log('done')
