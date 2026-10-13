"""Mixture inputs, physics-informed descriptors and input checks.

The descriptor formulas are copied from the PI-NGB-TSTR training notebooks so that the app computes
exactly the features the models were trained on."""
import numpy as np
import pandas as pd

TARGETS = ['ITSD', 'TSR', 'RD']
SIEVES = ['19 mm', '12.5 mm', '9.5 mm', '4.75 mm', '2.36 mm', '0.6 mm', '0.15 mm', '0.075 mm']
SIEVE_SIZES = np.array([19.0, 12.5, 9.5, 4.75, 2.36, 0.6, 0.15, 0.075])
PHYSICS = ['Gradation_n', 'Gradation_RMSE', 'Gradation_R2', 'AV_VMA_Ratio', 'VFA_Calc', 'VFA_Residual',
           'Volumetric_Consistency', 'Volumetric_Consistency_Error', 'AV_VMA_Violation']

# Inputs as in Table 2 of the paper: column, symbol, description, unit, category, kind ('num' or 'cat')
CATEGORIES = ['Aggregate gradation', 'Aggregate characteristics', 'Recyclable materials and binder properties',
              'Mixture volumetrics', 'Experimental protocols']
INPUTS = [
    *[(s, s, f'Passing the {s} sieve', '%', 'Aggregate gradation', 'num') for s in SIEVES],
    ('NMAS', 'NMAS', 'Nominal maximum aggregate size', 'mm', 'Aggregate characteristics', 'num'),
    ('LAA', 'LAA', 'Los Angeles abrasion loss', '%', 'Aggregate characteristics', 'num'),
    ('AT', 'AT', 'Aggregate type', '–', 'Aggregate characteristics', 'cat'),
    ('RAP', 'RAP', 'Reclaimed asphalt pavement content', '%', 'Recyclable materials and binder properties', 'num'),
    ('RAPPG', 'RAPPG', 'RAP binder performance grade', '–', 'Recyclable materials and binder properties', 'cat'),
    ('RT', 'RJ', 'Rejuvenator', '–', 'Recyclable materials and binder properties', 'cat'),
    ('VAPG', 'VAPG', 'Virgin asphalt performance grade', '–', 'Recyclable materials and binder properties', 'cat'),
    ('OBC', 'OBC', 'Optimum binder content', '%', 'Mixture volumetrics', 'num'),
    ('Gmm', 'Gmm', 'Theoretical maximum specific gravity', '–', 'Mixture volumetrics', 'num'),
    ('AV', 'AV', 'Air voids', '%', 'Mixture volumetrics', 'num'),
    ('VMA', 'VMA', 'Voids in mineral aggregate', '%', 'Mixture volumetrics', 'num'),
    ('VFA', 'VFA', 'Voids filled with asphalt', '%', 'Mixture volumetrics', 'num'),
    ('Ndes', 'Ndes', 'Number of design gyrations', '–', 'Experimental protocols', 'num'),
    ('RCT', 'RCT', 'Rutting cycle type', '–', 'Experimental protocols', 'cat'),
]
OUTPUTS = [('ITSD', 'Dry indirect tensile strength', 'kPa'), ('TSR', 'Tensile strength ratio', '–'), ('RD', 'Rut depth', 'mm')]
INPUT_NAMES = [i[0] for i in INPUTS]
CATEGORICAL = [i[0] for i in INPUTS if i[5] == 'cat']
NUMERIC = [i[0] for i in INPUTS if i[5] == 'num']
SYMBOL = {i[0]: i[1] for i in INPUTS}
DESC = {i[0]: i[2] for i in INPUTS}
UNITS = {i[0]: i[3] for i in INPUTS}
CATEGORY = {i[0]: i[4] for i in INPUTS}
LABELS = {c: (DESC[c] if c in SIEVES else f'{SYMBOL[c]} – {DESC[c]}') for c in INPUT_NAMES}
ALIASES = {'RJ': 'RT'}          # paper symbol -> database column


def form_label(c):
    """'Symbol – description (unit)' as in Table 2."""
    return f'{SYMBOL[c]} – {DESC[c]} ({UNITS[c]})'


def gradation_features(data):
    """Power-law (maximum-density) fit of the gradation: exponent n, fit RMSE and fit R² (notebook code)."""
    data = data.copy(); n_values, rmse_values, r2_values = [], [], []
    for _, row in data.iterrows():
        g = pd.to_numeric(row[SIEVES], errors='coerce').values.astype(float)
        valid = np.isfinite(g) & (g > 0) & (g <= 100)
        if np.sum(valid) >= 2:
            x = np.log(SIEVE_SIZES[valid] / 19.0); y = np.log(g[valid] / 100.0)
            den = np.sum(x * x)
            if den > 1e-12:
                n = np.clip(np.sum(x * y) / den, 0.05, 2.0)
                pred = 100.0 * (SIEVE_SIZES[valid] / 19.0) ** n; obs = g[valid]
                rmse = np.sqrt(np.mean((obs - pred) ** 2))
                ss_res = np.sum((obs - pred) ** 2); ss_tot = np.sum((obs - np.mean(obs)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            else:
                n, rmse, r2 = 0.45, np.nan, np.nan
        else:
            n, rmse, r2 = 0.45, np.nan, np.nan
        n_values.append(n); rmse_values.append(rmse); r2_values.append(r2)
    data['Gradation_n'] = n_values; data['Gradation_RMSE'] = rmse_values; data['Gradation_R2'] = r2_values
    return data


def volumetric_features(data):
    """Superpave volumetric relationships AV–VMA–VFA (notebook code)."""
    data = data.copy()
    av, vma, vfa = (pd.to_numeric(data[c], errors='coerce') for c in ('AV', 'VMA', 'VFA'))
    safe_vma = np.maximum(vma.values, 1e-8)
    data['AV_VMA_Ratio'] = av.values / safe_vma
    vfa_calc = 100.0 * (vma.values - av.values) / safe_vma
    data['VFA_Calc'] = vfa_calc
    data['VFA_Residual'] = vfa.values - vfa_calc
    data['Volumetric_Consistency'] = vfa.values / 100.0 + av.values / safe_vma
    data['Volumetric_Consistency_Error'] = np.abs(data['Volumetric_Consistency'] - 1.0)
    data['AV_VMA_Violation'] = (av.values > vma.values).astype(int)
    return data


def prepare_inputs(df):
    """Raw mixture table -> table with the nine physics descriptors added (same order as in training)."""
    df = df.copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return volumetric_features(gradation_features(df))


def physics_table(df_prepared, i=0):
    """Descriptor values of one mixture, with plain-language names, for display."""
    names = {'Gradation_n': 'Gradation exponent n (power-law fit)', 'Gradation_RMSE': 'Gradation fit RMSE (% passing)',
             'Gradation_R2': 'Gradation fit R²', 'AV_VMA_Ratio': 'AV/VMA ratio', 'VFA_Calc': 'Calculated VFA (%)',
             'VFA_Residual': 'VFA residual (entered − calculated, %)', 'Volumetric_Consistency': 'Volumetric consistency (ideal = 1)',
             'Volumetric_Consistency_Error': 'Volumetric consistency error', 'AV_VMA_Violation': 'AV > VMA flag'}
    r = df_prepared.iloc[i]
    return pd.DataFrame({'Descriptor': [names[p] for p in PHYSICS], 'Value': [r[p] for p in PHYSICS]})


def check_inputs(row, domain):
    """Engineering checks of one mixture. Returns a list of (level, message); level is 'error' or 'warning'.
    `domain` holds the ranges and categories of the 189 experimental mixtures used to build the models."""
    msgs = []
    g = np.array([row.get(s, np.nan) for s in SIEVES], float)
    missing = [LABELS[c] for c in INPUT_NAMES if (pd.isna(row.get(c, np.nan)) or row.get(c, '') == '')]
    if missing:
        msgs.append(('warning', 'Missing inputs were filled with the training median or most frequent value: ' + ', '.join(missing) + '.'))
    if np.any((g < 0) | (g > 100)):
        msgs.append(('error', 'Percent passing must lie between 0 and 100 for every sieve.'))
    gg = g[np.isfinite(g)]
    if len(gg) > 1 and np.any(np.diff(gg) > 1e-9):
        msgs.append(('error', 'Gradation must not increase from coarser to finer sieves (percent passing must decrease or stay equal).'))
    av, vma, vfa = (row.get(c, np.nan) for c in ('AV', 'VMA', 'VFA'))
    if np.isfinite(av) and np.isfinite(vma):
        if av >= vma:
            msgs.append(('error', 'Air voids must be smaller than VMA (AV < VMA).'))
        elif np.isfinite(vfa):
            vc = 100 * (vma - av) / vma
            if abs(vfa - vc) > 3:
                msgs.append(('warning', f'VFA ({vfa:.1f}%) is inconsistent with AV and VMA; the Superpave identity gives {vc:.1f}%.'))
    out = []
    for c in NUMERIC:
        v = row.get(c, np.nan)
        lo, hi = domain['ranges'][c]
        if np.isfinite(v) and (v < lo - 1e-9 or v > hi + 1e-9):
            u = '' if UNITS[c] == '–' else f' {UNITS[c]}'
            out.append(f'{LABELS[c]} = {v:g}{u} (database range {lo:g}–{hi:g}{u})')
    for c in CATEGORICAL:
        v = row.get(c, None)
        if isinstance(v, str) and v not in domain['categories'][c]:
            out.append(f'{LABELS[c]} = {v} (not in the database)')
    if out:
        msgs.append(('warning', 'Outside the experimental database, so the prediction is an extrapolation and its interval may be '
                                'too narrow: ' + '; '.join(out) + '.'))
    return msgs
