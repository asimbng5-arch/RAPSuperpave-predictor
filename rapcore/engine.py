"""Prediction engine for the exported PI-NGB-TSTR models.

Each model file (models/<TARGET>.npz) holds the NGBoost tree ensemble as plain arrays, the one-hot
encoding of the categorical inputs, the training fill values, a background sample for SHAP and the
split-conformal calibration scores. Only NumPy is needed to evaluate it, so the app does not depend on
the scikit-learn or ngboost versions used for training."""
import json, math, os
import numpy as np
import pandas as pd
from .features import prepare_inputs, check_inputs, TARGETS

TSR_LIMIT = 0.80


class NGBEnsemble:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.spec = json.loads(str(z['spec']))
        for k in ('left', 'right', 'feat', 'thr', 'val', 'roots', 'tree_param', 'tree_mult', 'bg'):
            setattr(self, k, z[k])
        s = self.spec
        self.target, self.dist = s['target'], s['dist']
        self.init = np.array(s['init_params'], float)
        self.W = np.zeros((len(self.roots), 2)); self.W[np.arange(len(self.roots)), self.tree_param] = self.tree_mult
        self.groups = self._groups()

    # ---------------------------------------------------------------- preprocessing (= notebook ColumnTransformer)
    def design_matrix(self, prepared):
        s = self.spec; X = prepared.copy()
        for c in s['x_cols']:
            if c not in X:
                X[c] = np.nan
            X[c] = X[c].where(X[c].notna(), s['fill'].get(c, np.nan))
        blocks = []
        for c in s['cat_cols']:
            cats, drop = s['categories'][c], s['drop_idx'][c]
            v = X[c].astype(str).values
            for k, cat in enumerate(cats):
                if drop is not None and k == drop:
                    continue
                blocks.append((v == cat).astype(float))
        for c in s['num_cols']:
            blocks.append(pd.to_numeric(X[c], errors='coerce').values.astype(float))
        return np.column_stack(blocks)

    def _groups(self):
        """Processed columns belonging to each original input (one SHAP player per input)."""
        s, g, j = self.spec, [], 0
        for c in s['cat_cols']:
            n = len(s['categories'][c]) - (s['drop_idx'][c] is not None)
            g.append((c, np.arange(j, j + n))); j += n
        for c in s['num_cols']:
            g.append((c, np.array([j]))); j += 1
        return g

    # ---------------------------------------------------------------- NGBoost: parameters and mean
    def params(self, M, chunk=1500):
        M = np.asarray(M, float).astype(np.float32).astype(np.float64)   # scikit-learn trees split on float32 inputs
        out = np.empty((len(M), 2))
        T = len(self.roots); depth = int(self.spec['max_depth'])
        for a in range(0, len(M), chunk):
            X = M[a:a + chunk]
            idx = np.broadcast_to(self.roots, (len(X), T)).copy()
            for _ in range(depth):
                xv = np.take_along_axis(X, self.feat[idx], axis=1)
                idx = np.where(xv <= self.thr[idx], self.left[idx], self.right[idx])
            out[a:a + chunk] = self.init - self.val[idx] @ self.W
        return out

    def mean(self, M):
        P = self.params(M); loc, scale = P[:, 0], np.exp(P[:, 1])
        return np.exp(loc + scale ** 2 / 2) if self.dist == 'lognormal' else loc

    # ---------------------------------------------------------------- split-conformal interval
    def interval(self, mu, level=0.90):
        cal = self.spec['calibration']; sc = np.sort(np.array(cal['scores'], float)); n = len(sc)
        k = min(n, int(math.ceil((n + 1) * level))); q = sc[k - 1]
        if cal['mode'] == 'relative':
            return np.maximum(mu * (1 - q), 0.0), mu * (1 + q)
        return mu - q, mu + q

    # ---------------------------------------------------------------- SHAP (interventional, antithetic permutations)
    def shap(self, m_row, n_perm=10, seed=0):
        rng = np.random.default_rng(seed); BG = self.bg; P = len(self.groups)
        perms = []
        for _ in range(max(1, n_perm // 2)):
            p = rng.permutation(P); perms += [p, p[::-1]]
        rows = []
        for perm in perms:
            cur = BG.copy(); rows.append(cur.copy())
            for j in perm:
                cur[:, self.groups[j][1]] = m_row[self.groups[j][1]]; rows.append(cur.copy())
        out = self.mean(np.vstack(rows)).reshape(len(perms), P + 1, len(BG)).mean(axis=2)
        phi = np.zeros(P)
        for k, perm in enumerate(perms):
            phi[perm] += np.diff(out[k])
        phi /= len(perms)
        base = out[:, 0].mean()
        return pd.Series(phi, index=[g[0] for g in self.groups]), base


class Predictor:
    """Loads the three PI-NGB-TSTR models and the app metadata."""
    def __init__(self, model_dir):
        self.models = {t: NGBEnsemble(os.path.join(model_dir, f'{t}.npz')) for t in TARGETS}
        self.meta = json.load(open(os.path.join(model_dir, 'meta.json')))

    def predict(self, raw, level=0.90):
        prep = prepare_inputs(raw)
        res = pd.DataFrame(index=raw.index)
        for t, m in self.models.items():
            mu = m.mean(m.design_matrix(prep)); lo, hi = m.interval(mu, level)
            res[f'{t}_pred'], res[f'{t}_lo'], res[f'{t}_hi'] = mu, lo, hi
            if t == 'TSR':
                sig = m.spec['calibration']['sigma']
                z = (TSR_LIMIT - mu) / sig
                p = 1 - 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
                res['TSR_P_pass'] = p
                res['TSR_decision'] = np.where(p >= 0.90, 'Pass', np.where(p <= 0.10, 'Fail', 'Uncertain – test in laboratory'))
        return res, prep

    def checks(self, raw):
        return [check_inputs(r, self.meta['domain']) for r in raw.to_dict('records')]

    def explain(self, prep_row_df, target, n_perm=10):
        m = self.models[target]
        return m.shap(m.design_matrix(prep_row_df)[0], n_perm=n_perm)
