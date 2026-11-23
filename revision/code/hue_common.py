"""Shared data preparation, models and training protocol for the DEFI 2026 revision experiments.

Mirrors code/Hue_Visitors_Linear_DL_Ensemble.ipynb (cells 2, 4, 5, 6, 8, 10, 11) so that the
revision experiments use the same features, scaling, sample construction and two-stage protocol.
"""
import os, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import holidays

SEED = 2026
FINAL_SEEDS = [2026, 2027, 2028]
HORIZONS = [1, 7, 14]
MAXH = max(HORIZONS)
NH = len(HORIZONS)
PATIENCE = 18
MAX_EPOCHS_STAGE1 = 200
TOP_K_ENSEMBLE = 3
RIDGE_ALPHAS = np.logspace(-3, 5, 17)
COMMON_START = pd.Timestamp('2022-11-10')
CLOSURES = {pd.Timestamp(d) for d in ['2025-10-28', '2025-10-29', '2025-10-30']}
FILES = [('Dai_Noi', 'Đại Nội', 'hue_dai_noi_daily_visitors_clean.csv'),
         ('Khai_Dinh', 'Lăng Khải Định', 'hue_lang_khai_dinh_daily_visitors_clean.csv'),
         ('Minh_Mang', 'Lăng Minh Mạng', 'hue_lang_minh_mang_daily_visitors_clean.csv'),
         ('Tu_Duc', 'Lăng Tự Đức', 'hue_lang_tu_duc_daily_visitors_clean.csv')]
KEYS = [f[0] for f in FILES]
NAMES = [f[1] for f in FILES]
NS = len(FILES)
DL_MODELS = ['LSTM', 'GRU', '1D-CNN', 'Transformer']
# Validation-selected configurations reported in Table I (results/logs/best_hyperparameters.json).
PARAMS = {
    'ARX-Ridge': dict(lookback=28),
    'LSTM': dict(lookback=84, dense_units=64, dropout=0.3377266601306773, learning_rate=0.00047969666213237426,
                 batch_size=16, units=64, n_layers=1),
    'GRU': dict(lookback=112, dense_units=96, dropout=0.2642995828650003, learning_rate=0.0011882822203134762,
                batch_size=16, units=32, n_layers=1),
    '1D-CNN': dict(lookback=84, dense_units=96, dropout=0.12834714045503087, learning_rate=0.00280987361408003,
                   batch_size=16, filters=32, kernel_size=5, n_blocks=3),
    'Transformer': dict(lookback=28, dense_units=32, dropout=0.29435810875639434, learning_rate=0.0019493355483246903,
                        batch_size=16, d_model=32, num_heads=4, n_layers=1, ff_dim=64),
}


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


class Data:
    """Aligned four-site daily panel with calendar features."""

    def __init__(self, data_dir):
        frames = []
        for key, _, fn in FILES:
            d = pd.read_csv(Path(data_dir) / fn, encoding='utf-8-sig')
            d['date'] = pd.to_datetime(d['date'])
            d = d.sort_values('date').drop_duplicates('date', keep=False)
            d = d[d['date'] >= COMMON_START]
            d['split'] = d['split'].astype(str).str.strip().str.lower()
            d = d[['date', 'split', 'domestic_visitors', 'international_visitors', 'visitors']].rename(columns={
                'domestic_visitors': f'{key}__domestic', 'international_visitors': f'{key}__international',
                'visitors': f'{key}__visitors'})
            frames.append(d)
        wide = frames[0]
        for i, part in enumerate(frames[1:], 1):
            wide = wide.merge(part, on='date', suffixes=('', f'_s{i}'))
            assert (wide['split'] == wide[f'split_s{i}']).all()
            wide = wide.drop(columns=f'split_s{i}')
        self.wide = wide.sort_values('date').reset_index(drop=True)
        self.dates = pd.DatetimeIndex(self.wide['date'])
        self.T = len(self.wide)
        self.split = self.wide['split'].to_numpy()
        vn = holidays.Vietnam(years=range(self.dates.min().year, self.dates.max().year + 1))
        idx = self.dates
        dow, month, doy = idx.dayofweek.to_numpy(), idx.month.to_numpy(), idx.dayofyear.to_numpy()
        cols = [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
                np.sin(2 * np.pi * (doy - 1) / 365.25), np.cos(2 * np.pi * (doy - 1) / 365.25),
                (dow >= 5).astype(float), np.array([float(t.date() in vn) for t in idx]),
                np.array([float(t.normalize() in CLOSURES) for t in idx]),
                (idx - COMMON_START).days.to_numpy() / 365.25]
        cols += [(dow == k).astype(float) for k in range(7)]
        self.cal = np.column_stack(cols)
        self.hist = np.column_stack([np.log1p(self.wide[f'{k}__{s}'].to_numpy(float))
                                     for k in KEYS for s in ['visitors', 'domestic', 'international']])
        self.y_raw = np.column_stack([self.wide[f'{k}__visitors'].to_numpy(float) for k in KEYS])

    def idx(self, date):
        return int(np.searchsorted(self.dates.values, np.datetime64(pd.Timestamp(date))))


class Prepared:
    """Scaled arrays for one fitting window: scalers are estimated on rows where fit_mask is True."""

    def __init__(self, data, fit_mask):
        self.d = data
        seq = np.concatenate([data.hist, data.cal], axis=1)
        self.seq = StandardScaler().fit(seq[fit_mask]).transform(seq).astype('float32')
        self.fut = StandardScaler().fit(data.cal[fit_mask]).transform(data.cal).astype('float32')
        tlog = np.log1p(data.y_raw)
        self.tsc = StandardScaler().fit(tlog[fit_mask])
        self.tgt = self.tsc.transform(tlog).astype('float32')

    def bundle(self, origins, lookback):
        origins = np.asarray(origins, dtype=int)
        ti = origins[:, None] + np.array(HORIZONS)[None, :] - 1
        return dict(X=np.stack([self.seq[o - lookback:o] for o in origins]).astype('float32'),
                    F=self.fut[ti], y=self.tgt[ti].reshape(len(origins), -1), o=origins,
                    y_raw=self.d.y_raw[ti])

    def inverse(self, pred):
        pred = np.asarray(pred); n = pred.shape[0]
        lg = self.tsc.inverse_transform(pred.reshape(n * NH, NS))
        return np.clip(np.expm1(lg).reshape(n, NH, NS), 0, None)


def seasonal_naive(data, origins, season=7):
    out = np.empty((len(origins), NH, NS))
    for i, o in enumerate(origins):
        for hi, h in enumerate(HORIZONS):
            t = o + h - 1
            src = t - season * math.ceil(h / season)
            assert src <= o - 1
            out[i, hi] = data.y_raw[src]
    return out


def build_deep_model(name, seq_shape, fut_shape, p, n_out=NH * NS):
    si = keras.Input(shape=seq_shape, name='past_sequence')
    fi = keras.Input(shape=fut_shape, name='known_future')
    dr = float(p['dropout'])
    if name in ['LSTM', 'GRU']:
        cls = layers.LSTM if name == 'LSTM' else layers.GRU
        x = si
        for li in range(int(p['n_layers'])):
            x = cls(int(p['units']), return_sequences=li < int(p['n_layers']) - 1)(x)
            x = layers.Dropout(dr)(x)
    elif name == '1D-CNN':
        f = int(p['filters'])
        x = layers.Dense(f)(si)
        for dil in [1, 2, 4][:int(p['n_blocks'])]:
            res = x
            y = layers.Conv1D(f, int(p['kernel_size']), padding='causal', dilation_rate=dil, activation='relu')(x)
            y = layers.BatchNormalization()(y)
            y = layers.Dropout(dr)(y)
            y = layers.Conv1D(f, 1, padding='same')(y)
            x = layers.Activation('relu')(layers.Add()([res, y]))
        x = layers.GlobalAveragePooling1D()(x)
    elif name == 'Transformer':
        dm = int(p['d_model'])
        x = layers.Dense(dm)(si)
        pos = layers.Embedding(input_dim=seq_shape[0], output_dim=dm)(tf.range(start=0, limit=seq_shape[0], delta=1))
        x = x + pos
        for _ in range(int(p['n_layers'])):
            a = layers.MultiHeadAttention(num_heads=int(p['num_heads']), key_dim=dm // int(p['num_heads']), dropout=dr)(x, x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dr)(a))
            ff = layers.Dense(int(p['ff_dim']), activation='gelu')(x)
            ff = layers.Dense(dm)(ff)
            x = layers.LayerNormalization(epsilon=1e-6)(x + layers.Dropout(dr)(ff))
        x = layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(name)
    fb = layers.Dense(int(p['dense_units']), activation='relu')(layers.Flatten()(fi))
    m = layers.Dense(int(p['dense_units']), activation='relu')(layers.Concatenate()([x, fb]))
    out = layers.Dense(n_out, name='forecast')(layers.Dropout(dr)(m))
    model = keras.Model([si, fi], out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(p['learning_rate']), clipnorm=1.0),
                  loss=keras.losses.Huber(delta=1.0))
    return model


def two_stage(name, p, tr, va, targets, log=print, seeds=FINAL_SEEDS):
    """Stage 1: fit on tr with early stopping on va (seed 2026) -> best epoch and validation predictions.
    Stage 2: refit on tr+va for the best epoch with each seed; returns predictions for every bundle in targets."""
    tf.keras.backend.clear_session(); set_seeds(SEED)
    m = build_deep_model(name, tr['X'].shape[1:], tr['F'].shape[1:], p)
    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, min_delta=1e-5),
          keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=max(3, PATIENCE // 2), factor=0.5, min_lr=1e-6)]
    t0 = time.time()
    h = m.fit([tr['X'], tr['F']], tr['y'], validation_data=([va['X'], va['F']], va['y']),
              epochs=MAX_EPOCHS_STAGE1, batch_size=int(p['batch_size']), shuffle=False, verbose=0, callbacks=cb)
    best = int(np.argmin(h.history['val_loss']) + 1)
    stage1 = {'train': m.predict([tr['X'], tr['F']], verbose=0), 'validation': m.predict([va['X'], va['F']], verbose=0)}
    log(f'    {name}: stage 1 {len(h.history["loss"])} epochs, best {best} ({time.time() - t0:.0f}s)')
    X = np.concatenate([tr['X'], va['X']]); F = np.concatenate([tr['F'], va['F']]); y = np.concatenate([tr['y'], va['y']])
    out = {k: [] for k in targets}
    for sd in seeds:
        tf.keras.backend.clear_session(); set_seeds(sd)
        fm = build_deep_model(name, tr['X'].shape[1:], tr['F'].shape[1:], p)
        fm.fit([X, F], y, epochs=max(1, best), batch_size=int(p['batch_size']), shuffle=False, verbose=0)
        for k, b in targets.items():
            out[k].append(fm.predict([b['X'], b['F']], verbose=0))
    return stage1, {k: np.stack(v) for k, v in out.items()}, best, pd.DataFrame(h.history)


def flat(b):
    return np.concatenate([b['X'].reshape(len(b['X']), -1), b['F'].reshape(len(b['F']), -1)], axis=1)


def fit_ridge(prep, tr, va, te):
    """Select alpha on validation (original visitor scale), refit on train+validation."""
    def score(a):
        pr = prep.inverse(Ridge(alpha=a).fit(flat(tr), tr['y']).predict(flat(va)))
        return np.sqrt(np.mean((pr - va['y_raw']) ** 2))
    best = min(RIDGE_ALPHAS, key=score)
    s1 = Ridge(alpha=best).fit(flat(tr), tr['y'])
    fm = Ridge(alpha=best).fit(np.concatenate([flat(tr), flat(va)]), np.concatenate([tr['y'], va['y']]))
    return prep.inverse(s1.predict(flat(va))), prep.inverse(fm.predict(flat(te))), float(best)


def simplex_weights(P, y):
    n = P.shape[1]
    res = minimize(lambda w: np.mean((P @ w - y) ** 2), x0=np.repeat(1 / n, n), method='SLSQP',
                   bounds=[(0.0, 1.0)] * n, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    if not res.success or not np.all(np.isfinite(res.x)):
        r = np.sqrt(np.mean((P - y[:, None]) ** 2, axis=0)); inv = 1 / np.maximum(r, 1e-8)
        return inv / inv.sum()
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def ensemble(val_preds, test_preds, y_val):
    """Top-3 by validation RMSE per site-horizon, non-negative weights summing to one (notebook cell 11)."""
    cands = list(val_preds)
    ens_test = np.zeros_like(next(iter(test_preds.values())))
    rows = []
    for hi, h in enumerate(HORIZONS):
        for s in range(NS):
            rank = sorted(cands, key=lambda m: np.sqrt(np.mean((val_preds[m][:, hi, s] - y_val[:, hi, s]) ** 2)))[:TOP_K_ENSEMBLE]
            w = simplex_weights(np.column_stack([val_preds[m][:, hi, s] for m in rank]), y_val[:, hi, s])
            ens_test[:, hi, s] = np.column_stack([test_preds[m][:, hi, s] for m in rank]) @ w
            rows += [dict(destination=NAMES[s], horizon=h, component_model=m, weight=float(x)) for m, x in zip(rank, w)]
    return ens_test, rows


def to_long(data, pred, origins, model, **extra):
    rows = []
    for i, o in enumerate(origins):
        for hi, h in enumerate(HORIZONS):
            t = o + h - 1
            for s in range(NS):
                rows.append(dict(model=model, origin_date=data.dates[o].date().isoformat(),
                                 target_date=data.dates[t].date().isoformat(), horizon=h, destination=NAMES[s],
                                 actual=float(data.y_raw[t, s]), prediction=float(pred[i, hi, s]), **extra))
    return pd.DataFrame(rows)
