"""Export a trained PI-NGB-TSTR model (official ngboost NGBRegressor) to the app's portable format.

Run once where the model was trained (e.g. Colab, with scikit-learn and ngboost installed). The training
pipeline is reproduced exactly: physics descriptors, training-median / mode filling, one-hot encoding
of the categorical inputs with the first category dropped, numerical inputs passed through unscaled."""
import json
import numpy as np
import pandas as pd
from .features import prepare_inputs


def _training_frames(xlsx, target, drop_cols=()):
    tr, te, va = [pd.read_excel(xlsx, s) for s in ('Training', 'Testing', 'Validation')]
    tr, te, va = [d.drop(columns=['Original_sheet'], errors='ignore').dropna(subset=[target]) for d in (tr, te, va)]
    tr, te, va = prepare_inputs(tr), prepare_inputs(te), prepare_inputs(va)
    fill = {}
    for c in tr.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(tr[c]):
            fill[c] = float(tr[c].median())
        else:
            mv = tr[c].mode(); fill[c] = str(mv.iloc[0]) if len(mv) else None
    for d in (tr, te, va):
        for c, v in fill.items():
            if c in d and v is not None:
                d[c] = d[c].fillna(v)
        d.drop(columns=[c for c in drop_cols if c in d], inplace=True)
    return tr, te, va, fill


def _spec_from_training(Xtr):
    """Replicates ColumnTransformer([('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cats)], remainder='passthrough')."""
    cat_cols = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    cats = {c: sorted(Xtr[c].astype(str).unique().tolist()) for c in cat_cols}
    names = [f'cat__{c}_{v}' for c in cat_cols for v in cats[c][1:]] + [f'remainder__{c}' for c in num_cols]
    return dict(x_cols=list(Xtr.columns), cat_cols=cat_cols, num_cols=num_cols, categories=cats,
                drop_idx={c: 0 for c in cat_cols}, feature_names=names)


def _trees(model, n_features):
    L, R, F, TH, V, roots, tparam, tmult = [], [], [], [], [], [], [], []
    off, depth = 0, 0
    for stage, (models, s, cidx) in enumerate(zip(model.base_models, model.scalings, model.col_idxs)):
        cidx = np.asarray(cidx) if cidx is not None and len(np.atleast_1d(cidx)) else np.arange(n_features)
        for p, est in enumerate(models):
            t = est.tree_; n = t.node_count; depth = max(depth, int(t.max_depth))
            left, right = t.children_left.copy(), t.children_right.copy()
            leaf = left < 0; own = np.arange(n)
            left[leaf] = own[leaf]; right[leaf] = own[leaf]
            feat = np.where(leaf, 0, cidx[np.maximum(t.feature, 0)])
            thr = np.where(leaf, np.inf, t.threshold)
            L.append(left + off); R.append(right + off); F.append(feat); TH.append(thr); V.append(t.value[:, 0, 0])
            roots.append(off); tparam.append(p); tmult.append(model.learning_rate * float(s)); off += n
    return dict(left=np.concatenate(L).astype(np.int32), right=np.concatenate(R).astype(np.int32),
                feat=np.concatenate(F).astype(np.int32), thr=np.concatenate(TH).astype(np.float64),
                val=np.concatenate(V).astype(np.float64), roots=np.array(roots, np.int32),
                tree_param=np.array(tparam, np.int32), tree_mult=np.array(tmult, np.float64)), depth


def export(model, xlsx, target, out_path, drop_cols=(), source='', n_bg=50, seed=0, dist=None):
    """Write models/<target>.npz. `model` is the fitted NGBRegressor; `xlsx` the TSTR data file it was trained on.
    Returns the held-out metrics (to compare with the paper) and the maximum difference between the exported
    model and model.predict on all data (should be ~1e-12)."""
    from .engine import NGBEnsemble
    tr, te, va, fill = _training_frames(xlsx, target, drop_cols)
    Xtr = tr.drop(columns=[target])
    spec = _spec_from_training(Xtr)
    if dist is None:   # 'normal' or 'lognormal'; pass it explicitly when known (official ngboost wraps its Dist classes)
        names = ' '.join(str(c) for c in getattr(getattr(model, 'Dist', None), '__mro__', [getattr(model, 'Dist', '')]))
        dist = 'lognormal' if ('LogNormal' in names or getattr(getattr(model, 'D', None), 'kind', '') == 'lognormal') else 'normal'
    trees, depth = _trees(model, len(spec['feature_names']))
    rng = np.random.default_rng(seed)
    spec.update(target=target, dist=dist, fill=fill, init_params=[float(v) for v in np.ravel(model.init_params)],
                max_depth=depth, source=source, dropped_inputs=list(drop_cols))
    tmp = dict(spec=json.dumps(dict(spec, calibration=dict(mode='absolute', scores=[0.0], sigma=1.0))), bg=np.zeros((1, 1)), **trees)
    np.savez_compressed(out_path, **tmp)
    eng = NGBEnsemble(out_path)
    mats = {k: eng.design_matrix(d) for k, d in (('Training', tr), ('Testing', te), ('Validation', va))}
    # exactness check against the original model on the same processed matrices
    diff = max(float(np.max(np.abs(eng.mean(M) - np.asarray(model.predict(M))))) for M in mats.values())
    ys = {k: d[target].values.astype(float) for k, d in (('Training', tr), ('Testing', te), ('Validation', va))}
    preds = {k: eng.mean(M) for k, M in mats.items()}
    y_ho = np.r_[ys['Testing'], ys['Validation']]; p_ho = np.r_[preds['Testing'], preds['Validation']]
    mode = 'relative' if target == 'RD' else 'absolute'
    scores = np.abs(y_ho - p_ho) / (np.abs(p_ho) if mode == 'relative' else 1.0)
    cal = dict(mode=mode, scores=sorted(float(v) for v in scores), sigma=float(np.sqrt(np.mean((y_ho - p_ho) ** 2))))
    metrics = {}
    for k in ('Training', 'Testing', 'Validation'):
        y, p = ys[k][:132] if k == 'Training' else ys[k], preds[k][:132] if k == 'Training' else preds[k]
        metrics[k] = dict(R2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                          MAPE=float(100 * np.mean(np.abs(y - p) / np.abs(y))), n=int(len(y)))
    bg = mats['Training'][rng.choice(132, size=min(n_bg, 132), replace=False)]
    spec.update(calibration=cal, metrics=metrics)
    np.savez_compressed(out_path, spec=json.dumps(spec), bg=bg, **trees)
    return metrics, diff
