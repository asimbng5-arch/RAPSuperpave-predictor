"""Train the three PI-NGB-TSTR models exactly as in the PI-NGB-TSTR notebooks and export them for the app.

Run by the GitHub Actions workflow (.github/workflows/build.yml) with scikit-learn 1.6.1 and ngboost 0.5.11.
Writes models/ITSD.npz, models/TSR.npz, models/RD.npz and a Markdown report of the held-out accuracy."""
import os, sys, traceback


def _annotate(level, msg):
    """Report to the GitHub Actions run page (readable without the raw log)."""
    print(f"::{level}::" + str(msg).replace('%', '%25').replace('\r', '').replace('\n', '%0A'), flush=True)


def _excepthook(tp, val, tb):
    _annotate('error', 'train_export.py failed:\n' + ''.join(traceback.format_exception(tp, val, tb))[-6000:])
    sys.__excepthook__(tp, val, tb)


sys.excepthook = _excepthook
import numpy as np
import sklearn, ngboost, pandas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import r2_score
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
from ngboost.scores import MLE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rapcore.export import _training_frames, export
from rapcore.engine import NGBEnsemble

FILES = {'ITSD': '2. TSTR_ITSD_RAPSuperpave.xlsx', 'TSR': '4. TSTR_TSR_RAPSuperpave_resplit.xlsx',
         'RD': '6. TSTR_RD_RAPSuperpave_resplit.xlsx'}
SETTINGS = {
    'ITSD': dict(base=DecisionTreeRegressor(max_depth=6, min_samples_leaf=13, random_state=42), Dist=Normal,
                 n_estimators=1500, learning_rate=0.025, minibatch_frac=1.0, col_sample=1.0, random_state=42, drop=[], weight=None),
    'TSR':  dict(base=ExtraTreeRegressor(max_depth=15, min_samples_leaf=4, max_features=0.7, random_state=7), Dist=Normal,
                 n_estimators=80, learning_rate=0.05, minibatch_frac=0.9, col_sample=1.0, random_state=63,
                 drop=['AT', 'RAP', 'RCT', 'Ndes'], weight=(132, 3.0)),
    'RD':   dict(base=DecisionTreeRegressor(max_depth=6, min_samples_leaf=8, random_state=42), Dist=LogNormal,
                 n_estimators=400, learning_rate=0.03, minibatch_frac=0.3, col_sample=0.85, random_state=42, drop=[], weight=None)}
PAPER = {'ITSD': (0.8327, 0.8166), 'TSR': (0.8270, 0.8012), 'RD': (0.8575, 0.8289)}

_annotate('notice', f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pandas.__version__}, scikit-learn {sklearn.__version__}, ngboost {ngboost.__version__}')
report = [f'## PI-NGB-TSTR models (ngboost {ngboost.__version__}, scikit-learn {sklearn.__version__})', '',
          '| Property | Testing R² | Paper | Validation R² | Paper | Preprocessing identical | Export max diff |',
          '| --- | --- | --- | --- | --- | --- | --- |']
ok = True
src = f'official ngboost {ngboost.__version__}, scikit-learn {sklearn.__version__} (PI-NGB-TSTR notebooks, trained on GitHub Actions)'
for t, s in SETTINGS.items():
    xlsx = os.path.join(ROOT, 'data', FILES[t])
    tr, te, va, _ = _training_frames(xlsx, t, s['drop'])
    Xtr, Xte, Xva = (d.drop(columns=[t]) for d in (tr, te, va))
    cats = Xtr.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    pre = ColumnTransformer([('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cats)],
                            remainder='passthrough')
    A, B, C = pre.fit_transform(Xtr), pre.transform(Xte), pre.transform(Xva)
    model = NGBRegressor(Base=s['base'], Dist=s['Dist'], Score=MLE, natural_gradient=True, n_estimators=s['n_estimators'],
                         learning_rate=s['learning_rate'], minibatch_frac=s['minibatch_frac'], col_sample=s['col_sample'],
                         random_state=s['random_state'], verbose=False)
    if s['weight']:
        w = np.ones(len(A)); w[:s['weight'][0]] = s['weight'][1]
        model.fit(A, tr[t].values, sample_weight=w)
    else:
        model.fit(A, tr[t].values)
    out = os.path.join(ROOT, 'models', f'{t}.npz')
    metrics, diff = export(model, xlsx, t, out, drop_cols=s['drop'], source=src,
                           dist='lognormal' if s['Dist'] is LogNormal else 'normal')
    eng = NGBEnsemble(out)
    same = max(float(np.abs(eng.design_matrix(d) - M).max()) for d, M in zip((tr, te, va), (A, B, C))) == 0.0
    ok &= same and diff < 1e-8
    r2 = (metrics['Testing']['R2'], metrics['Validation']['R2'])
    line = (f'{t}: test R2 {r2[0]:.4f} (paper {PAPER[t][0]}), val R2 {r2[1]:.4f} (paper {PAPER[t][1]}), '
            f'preprocessing identical {same}, export diff {diff:.1e}')
    print(line, flush=True); _annotate('notice', line)
    report.append(f'| {t} | {r2[0]:.4f} | {PAPER[t][0]} | {r2[1]:.4f} | {PAPER[t][1]} | {same} | {diff:.1e} |')
open(os.path.join(ROOT, 'training_report.md'), 'w').write('\n'.join(report) + '\n')
if os.environ.get('GITHUB_STEP_SUMMARY'):
    open(os.environ['GITHUB_STEP_SUMMARY'], 'a').write('\n'.join(report) + '\n')
if not ok:
    _annotate('error', 'Exported models do not reproduce the trained models exactly (see notices above).')
sys.exit(0 if ok else 1)
