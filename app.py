"""RAP Superpave Performance Predictor (PI-NGB-TSTR).

Streamlit app: enter a RAP Superpave mix design and obtain the predicted dry indirect tensile strength (ITSD),
tensile strength ratio (TSR) and rut depth (RD) with split-conformal prediction intervals, the probability of
meeting TSR >= 0.80, engineering input checks and a SHAP explanation of each prediction.
Run locally with:  streamlit run app.py
"""
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rapcore import Predictor, prepare_inputs, physics_table
from rapcore.features import (INPUTS, INPUT_NAMES, NUMERIC, CATEGORICAL, LABELS, UNITS, SIEVES, SIEVE_SIZES, TARGETS,
                               CATEGORIES, CATEGORY, SYMBOL, DESC, OUTPUTS, ALIASES, form_label)

HERE = os.path.dirname(os.path.abspath(__file__))
IN_BROWSER = sys.platform == 'emscripten'      # running in the browser (stlite / Pyodide)
N_PERM = 6 if IN_BROWSER else 10                # SHAP permutations (10 as in the paper; 6 in the browser for speed)
BLUE, ORANGE, GREEN, RED, GREY, INK = '#2a78d6', '#eb6834', '#1a9e6e', '#d64545', '#a3a29d', '#1f1f1e'
TNAME = {'ITSD': 'Dry indirect tensile strength', 'TSR': 'Tensile strength ratio', 'RD': 'Rut depth'}
TUNIT = {'ITSD': 'kPa', 'TSR': '', 'RD': 'mm'}
TFMT = {'ITSD': '{:,.0f}', 'TSR': '{:.3f}', 'RD': '{:.2f}'}
TROLE = {'ITSD': 'Tensile strength and cohesion – resistance to cracking throughout service',
         'TSR': 'Moisture-damage resistance – durability over the service life (criterion TSR ≥ 0.80)',
         'RD': 'Permanent-deformation resistance – early-life rutting under traffic'}

st.set_page_config(page_title='RAP Superpave Performance Predictor', page_icon='🛣️', layout='wide')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': INK, 'xtick.color': '#52514e', 'ytick.color': '#52514e'})


@st.cache_resource
def load_predictor():
    return Predictor(os.path.join(HERE, 'models'))


P = load_predictor()
META = P.meta
DOMAIN = META['domain']
PRESETS = META.get('presets', {})
APP = META.get('app', {})
RCT_DEF = {k: v for k, v in APP.get('rct_definitions', {}).items() if v}


def fmt(t, v):
    return TFMT[t].format(v)


# ------------------------------------------------------------------------------------------------- session state
def _default_inputs():
    first = next(iter(PRESETS.values())) if PRESETS else {}
    d = {}
    for c in INPUT_NAMES:
        if c in first:
            d[c] = first[c]
        elif c in NUMERIC:
            d[c] = float(np.mean(DOMAIN['ranges'][c]))
        else:
            d[c] = DOMAIN['categories'][c][0]
    return d


for c, v in _default_inputs().items():
    st.session_state.setdefault(f'in_{c}', v)
st.session_state.setdefault('result', None)
st.session_state.setdefault('shap', {})


def load_preset():
    rec = PRESETS.get(st.session_state.get('preset'))
    if rec:
        for c in INPUT_NAMES:
            st.session_state[f'in_{c}'] = rec[c]
        st.session_state['result'] = None
        st.session_state['shap'] = {}


def set_consistent_vfa():
    av, vma = st.session_state['in_AV'], st.session_state['in_VMA']
    if vma > av > 0:
        st.session_state['in_VFA'] = round(100 * (vma - av) / vma, 2)


def current_inputs():
    return pd.DataFrame([{c: st.session_state[f'in_{c}'] for c in INPUT_NAMES}])


# ------------------------------------------------------------------------------------------------- figures
def gradation_figure(row, prep_row):
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    g = np.array([row[s] for s in SIEVES], float)
    x = SIEVE_SIZES ** 0.45
    ax.plot(x, g, 'o-', color=BLUE, lw=1.6, ms=4, label='Entered gradation')
    n = prep_row['Gradation_n']
    xs = np.linspace(SIEVE_SIZES.min(), 19, 100)
    ax.plot(xs ** 0.45, 100 * (xs / 19) ** n, '--', color=GREY, lw=1.2, label=f'Power-law fit (n = {n:.3f})')
    ax.set_xticks(x); ax.set_xticklabels([f'{s:g}' for s in SIEVE_SIZES], fontsize=7)
    lab = ax.get_xticklabels(); lab[-1].set_ha('right'); lab[-2].set_ha('left')     # 0.15 and 0.075 mm sit close together
    ax.set_xlabel('Sieve size (mm), 0.45-power scale'); ax.set_ylabel('Passing (%)')
    ax.set_ylim(0, 105); ax.grid(alpha=0.3, lw=0.5); ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    return fig


def interval_figure(t, mu, lo, hi):
    """Prediction and interval against the distribution of measured values in the database."""
    vals = np.array(META.get('measured', {}).get(t, []), float)
    fig, ax = plt.subplots(figsize=(3.6, 1.35))
    if len(vals):
        ax.hist(vals, bins=24, color='#e4e3df', edgecolor='white')
    top = ax.get_ylim()[1]
    ax.axvspan(lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.axvline(mu, color=BLUE, lw=2)
    if t == 'TSR':
        ax.axvline(0.80, color=RED, lw=1, ls='--'); ax.text(0.80, top * 0.95, ' 0.80', color=RED, fontsize=7, va='top')
    ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'{t}' + (f' ({TUNIT[t]})' if TUNIT[t] else '') + ' – grey: measured values in the database', fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    return fig


def shap_figure(phi, base, pred, t, top=10):
    s = phi.reindex(phi.abs().sort_values(ascending=False).index)[:top][::-1]
    rest = phi.sum() - s.sum()
    labels = [LABELS.get(k, PHYS_LABEL.get(k, k)) for k in s.index]
    fig, ax = plt.subplots(figsize=(6.2, 0.33 * len(s) + 1.0))
    ax.barh(range(len(s)), s.values, color=[RED if v > 0 else BLUE for v in s.values], height=0.65)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='#52514e', lw=0.8)
    u = TUNIT[t] if TUNIT[t] else '–'
    ax.set_xlabel(f'Contribution to predicted {t} ({u})')
    ax.set_title(f'Average prediction {fmt(t, base)}  →  this mixture {fmt(t, pred)}   (other inputs: {rest:+.3g})',
                 fontsize=8, loc='left')
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    fig.tight_layout()
    return fig


PHYS_LABEL = {'Gradation_n': 'Gradation exponent n*', 'Gradation_RMSE': 'Gradation fit RMSE*', 'Gradation_R2': 'Gradation fit R²*',
              'AV_VMA_Ratio': 'AV/VMA ratio*', 'VFA_Calc': 'Calculated VFA*', 'VFA_Residual': 'VFA residual*',
              'Volumetric_Consistency': 'Volumetric consistency*', 'Volumetric_Consistency_Error': 'Volumetric consistency error*',
              'AV_VMA_Violation': 'AV > VMA flag*'}


def fig_to_streamlit(fig):
    st.pyplot(fig)
    plt.close(fig)


def render_results(R, raw, level):
    if not R['raw'].equals(raw) or R['level'] != level:
        st.info('Inputs or interval level changed since the last prediction – press **Predict** to update.')
    errors = [m for l, m in R['checks'] if l == 'error']
    warns = [m for l, m in R['checks'] if l == 'warning']
    for m in errors:
        st.error(m)
    if errors:
        return
    for m in warns:
        st.warning(m)
    res = R['res'].iloc[0]
    lv = int(round(R['level'] * 100))
    st.markdown('#### RAP Superpave performance (predicted)')
    cols = st.columns(3, gap='large')
    for col, t in zip(cols, TARGETS):
        with col:
            with st.container(border=True):
                u = f' {TUNIT[t]}' if TUNIT[t] else ''
                st.markdown(f'**{t} – {TNAME[t]} ({TUNIT[t] or "–"})**')
                st.caption(TROLE[t])
                st.metric(f'Predicted {t}', fmt(t, res[f"{t}_pred"]) + u)
                st.markdown(f'{lv}% interval: **{fmt(t, res[f"{t}_lo"])} – {fmt(t, res[f"{t}_hi"])}**{u}')
                if t == 'TSR':
                    p, dec = res['TSR_P_pass'], res['TSR_decision']
                    color = GREEN if dec == 'Pass' else (RED if dec == 'Fail' else ORANGE)
                    st.markdown(f'P(TSR ≥ 0.80) = **{p:.2f}** &nbsp; <span style="background:{color};color:white;'
                                f'padding:2px 8px;border-radius:4px;font-size:0.85rem">{dec}</span>', unsafe_allow_html=True)
                fig_to_streamlit(interval_figure(t, res[f'{t}_pred'], res[f'{t}_lo'], res[f'{t}_hi']))

    with st.expander('Physics-informed descriptors computed from the inputs'):
        st.dataframe(physics_table(R['prep']).style.format({'Value': '{:.4g}'}), hide_index=True)
        st.caption('These nine descriptors encode the power-law gradation fit and the Superpave AV–VMA–VFA relationships; '
                   'they are computed automatically and used by the model together with the entered variables.')

    st.markdown('#### Explain the prediction')
    e1, e2, _ = st.columns([1.3, 1, 2])
    tsel = e1.selectbox('Property', TARGETS, format_func=lambda t: f'{TNAME[t]} ({t})', key='shap_target')
    e2.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    if IN_BROWSER:
        st.caption('In the browser version the explanation takes about 10–60 seconds (longest for ITSD); the page is busy meanwhile.')
    if e2.button('Explain'):
        with st.spinner('Computing SHAP contributions...'):
            st.session_state['shap'][tsel] = P.explain(R['prep'], tsel, n_perm=N_PERM)
    if tsel in st.session_state['shap']:
        phi, base = st.session_state['shap'][tsel]
        fig_to_streamlit(shap_figure(phi, base, res[f'{tsel}_pred'], tsel))
        st.caption('Red bars raise and blue bars lower the prediction relative to the average prediction over 50 experimental '
                   'mixtures; * marks physics-informed descriptors. Contributions are associations learned by the model, '
                   'not causal effects.')

    out = pd.concat([R['raw'], R['res']], axis=1)
    st.download_button('Download this prediction (CSV)', out.to_csv(index=False).encode(), file_name='prediction.csv', mime='text/csv')


def render_batch(up, level):
    try:
        df = pd.read_csv(up) if up.name.lower().endswith('.csv') else pd.read_excel(up, sheet_name=0)
    except Exception as e:
        st.error(f'Could not read the file: {e}')
        return
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=ALIASES)
    miss = [SYMBOL[c] for c in INPUT_NAMES if c not in df.columns]
    if miss:
        st.error('Missing columns: ' + ', '.join(miss))
        return
    df = df[INPUT_NAMES].dropna(how='all').reset_index(drop=True)
    with st.spinner(f'Predicting {len(df)} mixtures...'):
        res, _ = P.predict(df, level)
        chk = P.checks(df)
    res['Errors'] = [' | '.join(m for l, m in c if l == 'error') for c in chk]
    res['Warnings'] = [' | '.join(m for l, m in c if l == 'warning') for c in chk]
    bad = res['Errors'] != ''
    res.loc[bad, [c for c in res.columns if c.endswith(('_pred', '_lo', '_hi')) or c.startswith('TSR_P')]] = np.nan
    res.loc[bad, 'TSR_decision'] = 'Not predicted (input error)'
    n_err, n_warn = int((res['Errors'] != '').sum()), int((res['Warnings'] != '').sum())
    st.success(f'{len(df)} mixtures predicted at the {int(round(level * 100))}% interval level. '
               f'{n_err} with input errors (not predicted), {n_warn} with warnings.')
    view = pd.concat([df[['RAP', 'OBC', 'AV', 'VMA']], res], axis=1)
    st.dataframe(view.style.format(precision=3), hide_index=False)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        pd.concat([df, res], axis=1).to_excel(w, sheet_name='Predictions', index=False)
    st.download_button('Download predictions (Excel)', out.getvalue(), file_name='RAP_Superpave_predictions.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


WIDGET = {**{s_: (0.0, 100.0, 0.5, '%.1f') for s_ in SIEVES},
          'LAA': (0.0, 80.0, 0.5, '%.1f'), 'RAP': (0.0, 100.0, 1.0, '%.1f'), 'OBC': (0.0, 15.0, 0.05, '%.2f'),
          'Gmm': (1.5, 3.5, 0.001, '%.3f'), 'AV': (0.0, 20.0, 0.1, '%.2f'), 'VMA': (0.0, 40.0, 0.1, '%.2f'),
          'VFA': (0.0, 100.0, 0.5, '%.2f'), 'Ndes': (10.0, 250.0, 5.0, '%.0f')}
NOTES = {'VFA': ' Superpave identity: VFA = 100 (VMA − AV)/VMA.', 'RCT': ' Number of load cycles at which the rut depth is reported.',
         'NMAS': ' Superpave sizes 9.5, 12.5 and 19 mm are in the database.'}


def cat_label(c, v):
    return f'{v} – {RCT_DEF[v]}' if c == 'RCT' and v in RCT_DEF else v


def range_text(c):
    if c in CATEGORICAL:
        return 'Database: ' + ', '.join(cat_label(c, v) for v in DOMAIN['categories'][c])
    lo, hi = DOMAIN['ranges'][c]; f = WIDGET.get(c, (0, 0, 0, '%g'))[3]
    u = '' if UNITS[c] == '–' else f' {UNITS[c]}'
    return f'Range in database: {f % lo} – {f % hi}{u}'


def field(c, where):
    """One input widget: 'symbol – description (unit)', tooltip with description and range, range shown underneath."""
    unused = c in P.models['TSR'].spec.get('dropped_inputs', [])
    help_ = f'{DESC[c]}.{NOTES.get(c, "")} {range_text(c)}.' + (' Not used by the TSR model.' if unused else '')
    key = f'in_{c}'
    if c == 'NMAS':
        opts = [9.5, 12.5, 19.0]
        if st.session_state[key] not in opts:
            st.session_state[key] = min(opts, key=lambda v: abs(v - st.session_state[key]))
        where.selectbox(form_label(c), opts, key=key, help=help_)
    elif c in CATEGORICAL:
        opts = DOMAIN['categories'][c]
        if st.session_state[key] not in opts:
            st.session_state[key] = opts[0]
        where.selectbox(form_label(c), opts, key=key, help=help_, format_func=lambda v: cat_label(c, v))
    else:
        lo, hi, step, f = WIDGET[c]
        where.number_input(form_label(c), min_value=lo, max_value=hi, step=step, format=f, key=key, help=help_)
    if c not in CATEGORICAL:
        where.caption(range_text(c))


def variables_table():
    """Table 2 of the paper with the ranges of the experimental database."""
    rows = [{'Category': CATEGORY[c], 'Variable': SYMBOL[c], 'Description': DESC[c], 'Role – type':
             'Input – ' + ('categorical' if c in CATEGORICAL else 'continuous'), 'Unit': UNITS[c], 'Range / values in database':
             range_text(c).split(': ', 1)[1]} for c in INPUT_NAMES]
    for t, d, u in OUTPUTS:
        v = np.array(META.get('measured', {}).get(t, [np.nan]), float); f = {'ITSD': '%.0f', 'TSR': '%.2f', 'RD': '%.2f'}[t]
        rows.append({'Category': 'RAP Superpave performance', 'Variable': t, 'Description': d, 'Role – type': 'Output – continuous',
                     'Unit': u, 'Range / values in database': f'{f % np.nanmin(v)} – {f % np.nanmax(v)}' + ('' if u == '–' else f' {u}')})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### {APP.get('title', 'RAP Superpave Performance Predictor')}")
    st.caption(APP.get('model', 'PI-NGB-TSTR'))
    level = st.select_slider('Prediction interval', options=[0.80, 0.90, 0.95], value=0.90,
                             format_func=lambda v: f'{int(round(v * 100))}%',
                             help='Split-conformal interval calibrated on 57 experimental mixtures that were not used for training. '
                                  'The paper reports the 90% interval.')
    st.divider()
    st.markdown('**How to use**\n1. Load an example or enter your mix design.\n2. Press **Predict**.\n'
                '3. Read the interval and the TSR decision; check any warnings.\n4. Use **Explain** to see what drives each prediction.')
    st.divider()
    src = P.models['ITSD'].spec.get('source', '')
    if src.startswith('interim'):
        st.warning('Interim models are loaded (re-implemented NGBoost). Replace them with the official PI-NGB-TSTR export '
                   'before publication – see README.', icon='⚠️')
    st.caption('Predictions are laboratory performance indicators for design screening. They do not replace the laboratory '
               'tests required by the specification, especially for mixtures flagged as uncertain or outside the database.')

st.title(APP.get('title', 'RAP Superpave Performance Predictor'))
st.markdown(APP.get('subtitle', ''))

tab_single, tab_batch, tab_info = st.tabs(['Single mixture', 'Batch prediction', 'Model information'])

# ================================================================================================= single mixture
with tab_single:
    if PRESETS:
        c1, c2, _ = st.columns([2.2, 1, 2])
        c1.selectbox('Example mixtures from the database', list(PRESETS), key='preset')
        c2.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
        c2.button('Load example', on_click=load_preset)

    colA, colB, colC = st.columns([1.25, 1, 1], gap='large')
    with colA:
        with st.container(border=True):
            st.markdown('**Aggregate gradation**')
            g1, g2 = st.columns(2)
            for k, s_ in enumerate(SIEVES):
                field(s_, g1 if k < 4 else g2)
            gfig_slot = st.empty()
    with colB:
        for cat in ('Aggregate characteristics', 'Recyclable materials and binder properties'):
            with st.container(border=True):
                st.markdown(f'**{cat}**')
                for c in INPUT_NAMES:
                    if CATEGORY[c] == cat:
                        field(c, st)
    with colC:
        with st.container(border=True):
            st.markdown('**Mixture volumetrics**')
            for c in ('OBC', 'Gmm', 'AV', 'VMA', 'VFA'):
                field(c, st)
            av, vma = st.session_state['in_AV'], st.session_state['in_VMA']
            if vma > av > 0:
                st.caption(f'Consistent VFA = 100 (VMA − AV)/VMA = **{100 * (vma - av) / vma:.2f}%**')
                st.button('Use consistent VFA', on_click=set_consistent_vfa)
        with st.container(border=True):
            st.markdown('**Experimental protocols**')
            for c in ('Ndes', 'RCT'):
                field(c, st)

    raw = current_inputs()
    prep_now = prepare_inputs(raw)
    with gfig_slot.container():
        fig_to_streamlit(gradation_figure(raw.iloc[0], prep_now.iloc[0]))

    st.divider()
    if st.button('Predict', type='primary'):
        res, prep = P.predict(raw, level)
        st.session_state['result'] = dict(raw=raw, res=res, prep=prep, checks=P.checks(raw)[0], level=level)
        st.session_state['shap'] = {}

    if st.session_state['result'] is not None:
        render_results(st.session_state['result'], raw, level)

# ================================================================================================= batch
with tab_batch:
    st.markdown('Predict many mixtures at once. Download the template, fill one row per mixture and upload it '
                '(Excel or CSV). Column names must match the template (the *Column guide* sheet lists each variable with its '
                'description, unit and database range); empty cells are filled with training values and flagged.')
    tmpl = pd.DataFrame(list(PRESETS.values()))[INPUT_NAMES] if PRESETS else pd.DataFrame(columns=INPUT_NAMES)
    tmpl = tmpl.rename(columns={c: SYMBOL[c] for c in INPUT_NAMES})          # paper symbols (RJ for rejuvenator)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        tmpl.to_excel(w, sheet_name='Mixtures', index=False)
        variables_table().to_excel(w, sheet_name='Column guide', index=False)
    st.download_button('Download input template (Excel)', buf.getvalue(), file_name='RAP_Superpave_batch_template.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    up = st.file_uploader('Upload mixtures', type=['xlsx', 'csv'])
    if up is not None:
        render_batch(up, level)

# ================================================================================================= model information
with tab_info:
    st.markdown(f"#### Model\n{APP.get('model', '')}. Each property is predicted by a separate natural-gradient boosting "
                "model (NGBoost) that learns a full predictive distribution – Normal for ITSD and TSR, LogNormal for rut depth, "
                "which guarantees positive rut depths. The models were trained on 132 experimental mixtures plus 924 synthetic "
                "mixtures generated by a physics-informed tabular variational autoencoder (PI-TVAE), and use nine "
                "physics-informed descriptors of gradation and volumetrics in addition to the mixture variables.")
    st.markdown("#### Prediction intervals and TSR decision\nIntervals are split-conformal intervals calibrated on the 57 "
                "experimental mixtures held out from training (testing and validation subsets): absolute for ITSD and TSR, "
                "proportional to the prediction for rut depth. The probability of meeting TSR ≥ 0.80 uses a Gaussian predictive "
                "distribution with the held-out error spread. A mixture is classified as **Pass** if P ≥ 0.90, **Fail** if "
                "P ≤ 0.10 and **Uncertain – test in laboratory** otherwise.")
    rows = []
    for t in TARGETS:
        s = P.models[t].spec
        for k in ('Testing', 'Validation'):
            m = s['metrics'][k]
            rows.append({'Property': t, 'Subset': f"{k} (n = {m['n']})", 'R²': m['R2'], 'RMSE': m['RMSE'], 'MAPE (%)': m['MAPE']})
    st.markdown('#### Input and output variables')
    st.dataframe(variables_table(), hide_index=True)
    st.markdown('#### Accuracy of the loaded models on held-out experimental mixtures')
    st.dataframe(pd.DataFrame(rows).style.format({'R²': '{:.3f}', 'RMSE': '{:.3f}', 'MAPE (%)': '{:.1f}'}), hide_index=True)
    st.caption('Model source: ' + P.models['ITSD'].spec.get('source', ''))
    if P.models['TSR'].spec.get('dropped_inputs'):
        st.caption('The TSR model does not use: ' + ', '.join(LABELS[c] for c in P.models['TSR'].spec['dropped_inputs']) + '.')
    st.markdown("#### Applicability\nThe models were developed from mixtures compiled from 24 published studies "
                f"({DOMAIN.get('n_mixtures', '')} distinct experimental mix designs). Inputs outside the database ranges or "
                "categories are flagged; predictions for such mixtures are extrapolations. ITSD, TSR and rut depth are laboratory "
                "indicators of life-cycle performance; they are not field service-life predictions.")
    if APP.get('paper'):
        st.markdown(f"#### Reference\n*{APP['paper']}*" + (f"  \n{APP['authors']}" if APP.get('authors') else '') +
                    (f"  \n{APP['affiliation']}" if APP.get('affiliation') else '') +
                    (f"  \n{APP['citation']}" if APP.get('citation') else ''))
