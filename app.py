'''Turkey Creek (USGS 02250030): 10 NSGA-II calibrations of the UMN HLM CRB3 runoff model on pairs of
composite performance metrics (CPMs), compared on 2017. Data: data/ (made by export_data.py on the cluster;
the app only reads these files).'''
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative
from plotly.subplots import make_subplots

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
st.set_page_config(page_title='Turkey Creek CPM calibrations', layout='wide')

CPMS = ['AllEq', 'NSE_Cor', 'KGE_Perc1090', '6Percentile', '2Mom6Perc']
COMPONENTS = ['NSE', 'mean_ratio', 'std_ratio', 'P5', 'P10', 'P25', 'P75', 'P90', 'P95', 'pearson', 'spearman']
PARAMS = ['crb3_vs', 'crb3_vg', 'crb3_vh', 'crb3_ssmax', 'crb3_sfc', 'crb3_lambda']
WARMUP_END = pd.Timestamp('2017-01-01', tz='UTC')

# How each sortable metric is ranked: 'max' = higher is better, 'one' = closest to 1 is better
SORTABLE = {**{c: 'max' for c in CPMS}, 'NSE': 'max', 'KGE': 'max', 'r': 'max', 'log_NSE': 'max', 'spearman': 'max',
            'alpha': 'one', **{'c_' + c: 'one' for c in ['std_ratio', 'P5', 'P10', 'P25', 'P75', 'P90', 'P95']}}
LABELS = {'alpha': 'alpha (std ratio)', 'log_NSE': 'log NSE', **{'c_' + c: 'component ' + c for c in COMPONENTS}}


@st.cache_data
def load():
    m = pd.read_csv(os.path.join(DATA, 'metrics.csv')).fillna({'cpm_a': '', 'cpm_b': ''})
    q = pd.read_csv(os.path.join(DATA, 'q_hourly.csv.gz'), parse_dates=['time_utc']).set_index('time_utc')
    s = pd.read_csv(os.path.join(DATA, 'storage_hourly.csv.gz'), parse_dates=['time_utc'])
    p = pd.read_csv(os.path.join(DATA, 'precip_daily.csv'), parse_dates=['date'])
    e = pd.read_csv(os.path.join(DATA, 'evaluations.csv.gz'))
    r = pd.read_csv(os.path.join(DATA, 'param_ranges.csv'))
    w = pd.read_csv(os.path.join(DATA, 'cpm_weights.csv')).set_index('cpm')
    return m, q, s, p, e, r, w


metrics, qh, storage, precip, evals, ranges, weights = load()
RUNS = list(metrics['run'])
PAIRS = [r for r in RUNS if r != 'default']
COLOR = {r: qualitative.D3[i % 10] for i, r in enumerate(PAIRS)}
COLOR['default'] = '#7f7f7f'
COLOR['observed'] = '#000000'
DASH = {'default': 'dash'}


def pair_label(row):
    return '' if row['run'] == 'default' else '%s + %s' % (row['cpm_a'], row['cpm_b'])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('About')
    st.markdown(
        'Turkey Creek at Palm Bay, FL (USGS **02250030**), UMN HLM coupled runoff + routing, **CRB3** '
        'runoff model, 91 hillslopes (outlet link 509412).\n\n'
        'Each of the 10 runs is an **NSGA-II** calibration (population 40 × 15 generations, 6 parameters) '
        'on **two** of five composite performance metrics (CPMs). Shown for each: the point of its final '
        'Pareto front with the **highest sum of its two CPMs**.\n\n'
        '- Run 2016-10-01 to 2017-12-31; Oct–Dec 2016 is warm-up; metrics on the **8661 hourly values of 2017**.\n'
        '- Discharge compared as **Q / mean(Q)**: the model outlet drains 12.33 of 271.95 km², so volumes are not '
        'comparable (mean_ratio is always 1).\n'
        '- CPM = 1 − √Σ((wᵢ / max w)(xᵢ − 1))², each component xᵢ = 1 when perfect (NSE capped to 0–1; '
        'mean, std and percentile ratios sim/obs; Pearson; Spearman).')
    st.subheader('CPM weights')
    st.dataframe(weights.astype(float), width='stretch')
    st.warning('Groundwater (G) starts every run at ~310 mm, a hot start made with the default parameters. '
               'The calibrated parameters drain it through the warm-up and into mid-2017, so part of the '
               'Jan–Jun 2017 baseflow comes from that initial storage.', icon='⚠️')

st.title('Turkey Creek: CRB3 calibrations on pairs of CPMs')

# ── Ranking ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 1, 3])
with c1:
    key = st.selectbox('Sort by', list(SORTABLE), index=0, format_func=lambda k: LABELS.get(k, k))
with c2:
    reverse = st.toggle('Reverse order', value=False)
with c3:
    st.caption('CPMs, NSE, KGE, r, Spearman: higher is better. Ratios (alpha, components): closest to 1 is better '
               '(sorted by |x − 1|). ★ marks the two CPMs a run was calibrated on.')

score = metrics[key] if SORTABLE[key] == 'max' else -(metrics[key] - 1).abs()
order = score.sort_values(ascending=reverse, kind='mergesort').index
ranked = metrics.loc[order].reset_index(drop=True)
table = pd.DataFrame({'rank': np.arange(1, len(ranked) + 1), 'run': ranked['run'],
                      'calibrated on': ranked.apply(pair_label, axis=1)})
table[LABELS.get(key, key) + ' (sort)'] = ranked[key]
for c in CPMS:
    table[c] = ranked[c]
    table[c + ' ★'] = np.where((ranked['cpm_a'] == c) | (ranked['cpm_b'] == c), '★', '')
for c in ['NSE', 'KGE', 'r', 'spearman', 'log_NSE']:
    table[c] = ranked[c]
cols = ['rank', 'run', 'calibrated on', LABELS.get(key, key) + ' (sort)']
for c in CPMS:
    cols += [c, c + ' ★']
cols += ['NSE', 'KGE', 'r', 'spearman', 'log_NSE']
fmt = {c: st.column_config.NumberColumn(c, format='%.3f') for c in cols if table[c].dtype.kind == 'f'}
fmt.update({c + ' ★': st.column_config.TextColumn('', width='small') for c in CPMS})
st.dataframe(table[cols], hide_index=True, width='stretch', column_config=fmt)

top = [r for r in ranked['run'] if r != 'default'][:3]
sel = st.multiselect('Runs to plot', RUNS, default=top + ['default'],
                     help='Defaults to the top 3 of the current ranking plus the default parameters.')

# ── Views ────────────────────────────────────────────────────────────────────
tabs = st.tabs(['Hydrograph 2017', 'Flow-duration curve', 'Storages S and G', 'Pareto & parameters', 'All metrics'])

with tabs[0]:
    a, b = st.columns([1, 3])
    with a:
        logy = st.toggle('Log scale', value=True, key='hlog')
    with b:
        d0, d1 = qh.index.min().date(), qh.index.max().date()
        rng = st.slider('Dates', min_value=d0, max_value=d1, value=(d0, d1), format='YYYY-MM-DD')
    w = qh[(qh.index.date >= rng[0]) & (qh.index.date <= rng[1])]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=w.index, y=w['observed'], name='observed', line=dict(color='black', width=1.4)))
    for r in sel:
        fig.add_trace(go.Scattergl(x=w.index, y=w[r], name=r, line=dict(color=COLOR[r], width=1.1, dash=DASH.get(r))))
    fig.update_layout(height=520, yaxis_title='Q / mean(Q)', hovermode='x unified', legend=dict(orientation='h', y=-0.15),
                      margin=dict(t=30))
    if logy:
        fig.update_yaxes(type='log', range=[-2, None])   # a few observed hours in Apr-Jun are ~0 or negative
    st.plotly_chart(fig, width='stretch')

with tabs[1]:
    low = st.toggle('Zoom on low flows (60–100 % exceedance)', value=False)
    fig = go.Figure()
    for r in ['observed'] + sel:
        z = np.sort(qh[r].dropna().to_numpy())[::-1]
        ex = 100 * np.arange(1, len(z) + 1) / (len(z) + 1)
        fig.add_trace(go.Scattergl(x=ex, y=z, name=r, line=dict(color=COLOR[r], width=2.2 if r == 'observed' else 1.3,
                                                              dash=DASH.get(r))))
    for p in (5, 10, 25, 75, 90, 95):
        fig.add_vline(x=100 - p, line=dict(color='lightgrey', width=1), annotation_text='P%d' % p,
                      annotation_position='top')
    fig.update_yaxes(type='log', range=[-2, None], title='Q / mean(Q)')
    fig.update_xaxes(title='exceedance probability [%]', range=[60, 100] if low else [0, 100])
    fig.update_layout(height=520, legend=dict(orientation='h', y=-0.15), margin=dict(t=40))
    st.plotly_chart(fig, width='stretch')

with tabs[2]:
    st.caption('Area-weighted basin averages over the 91 hillslopes of the CRB3 states: soil water S (state 0) and '
               'groundwater G (state 2), hourly. Shaded: warm-up (Oct–Dec 2016). Top: grid-mean AORC precipitation.')
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.2, 0.4, 0.4], vertical_spacing=0.04)
    fig.add_trace(go.Bar(x=precip['date'], y=precip['precip_mm'], name='P [mm/day]', marker_color='#1f77b4',
                         showlegend=False), row=1, col=1)
    for r in sel:
        d = storage[storage['run'] == r]
        for row, v in ((2, 'S_mm'), (3, 'G_mm')):
            fig.add_trace(go.Scattergl(x=d['time_utc'], y=d[v], name=r, legendgroup=r, showlegend=row == 2,
                                       line=dict(color=COLOR[r], width=1.2, dash=DASH.get(r))), row=row, col=1)
    for row in (1, 2, 3):
        fig.add_vrect(x0=storage['time_utc'].min(), x1=WARMUP_END, fillcolor='lightgrey', opacity=0.35, line_width=0,
                      row=row, col=1)
    fig.update_yaxes(title_text='P [mm/day]', autorange='reversed', row=1, col=1)
    fig.update_yaxes(title_text='S (soil) [mm]', row=2, col=1)
    fig.update_yaxes(title_text='G (groundwater) [mm]', row=3, col=1)
    fig.update_layout(height=780, hovermode='x unified', legend=dict(orientation='h', y=-0.08), margin=dict(t=30))
    st.plotly_chart(fig, width='stretch')
    y = storage[storage['time_utc'] >= WARMUP_END]
    summ = y.groupby('run').agg(S_mean=('S_mm', 'mean'), S_min=('S_mm', 'min'), S_max=('S_mm', 'max'),
                                G_mean=('G_mm', 'mean'), G_min=('G_mm', 'min'), G_max=('G_mm', 'max'))
    first, last = y.groupby('run').first(), y.groupby('run').last()
    summ['dS_2017'] = last['S_mm'] - first['S_mm']
    summ['dG_2017'] = last['G_mm'] - first['G_mm']
    st.dataframe(summ.loc[[r for r in sel if r in summ.index]].round(1), width='stretch')

with tabs[3]:
    a, b = st.columns(2)
    with a:
        pr = st.selectbox('Calibration', PAIRS, index=PAIRS.index(sel[0]) if sel and sel[0] in PAIRS else 0)
        row = metrics.set_index('run').loc[pr]
        e = evals[evals['run'] == pr]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=e['cpm_a_value'], y=e['cpm_b_value'], mode='markers', name='all evaluations',
                                 marker=dict(color='lightgrey', size=5), text=e['eval_id'],
                                 hovertemplate='eval %{text}<br>%{x:.3f}, %{y:.3f}<extra></extra>'))
        f = e[e['pareto'] == 1]
        fig.add_trace(go.Scatter(x=f['cpm_a_value'], y=f['cpm_b_value'], mode='markers', name='final Pareto front',
                                 marker=dict(color=COLOR[pr], size=9), text=f['eval_id'],
                                 hovertemplate='eval %{text}<br>%{x:.3f}, %{y:.3f}<extra></extra>'))
        bb = e[e['is_best'] == 1]
        fig.add_trace(go.Scatter(x=bb['cpm_a_value'], y=bb['cpm_b_value'], mode='markers', name='shown (max sum)',
                                 marker=dict(color='black', size=18, symbol='star')))
        lo = np.nanpercentile(np.r_[e['cpm_a_value'], e['cpm_b_value']], 20)
        fig.update_xaxes(title=row['cpm_a'], range=[lo - 0.05, 1])
        fig.update_yaxes(title=row['cpm_b'], range=[lo - 0.05, 1])
        fig.update_layout(height=480, legend=dict(orientation='h', y=-0.2), margin=dict(t=30),
                          title='%s: %d evaluations (zoomed on the best 80 %%)' % (pr, len(e)))
        st.plotly_chart(fig, width='stretch')
    with b:
        fig = go.Figure()
        rr = ranges.set_index('parameter')
        for r in sel:
            vals = []
            for k in PARAMS:
                v, lo_, hi_ = metrics.set_index('run').loc[r, k], rr.loc[k, 'min'], rr.loc[k, 'max']
                vals.append((np.log10(v) - np.log10(lo_)) / (np.log10(hi_) - np.log10(lo_)) if rr.loc[k, 'log']
                            else (v - lo_) / (hi_ - lo_))
            fig.add_trace(go.Scatter(x=[k + (' (log)' if rr.loc[k, 'log'] else '') for k in PARAMS], y=vals, name=r,
                                     mode='lines+markers', line=dict(color=COLOR[r], dash=DASH.get(r)),
                                     customdata=[metrics.set_index('run').loc[r, k] for k in PARAMS],
                                     hovertemplate='%{x}: %{customdata:.4g}<extra>' + r + '</extra>'))
        fig.update_yaxes(title='position in calibration range (0 = min, 1 = max)', range=[-0.05, 1.05])
        fig.update_layout(height=480, legend=dict(orientation='h', y=-0.2), margin=dict(t=30),
                          title='Parameters of the selected runs')
        st.plotly_chart(fig, width='stretch')
    st.dataframe(metrics.set_index('run').loc[sel, PARAMS], width='stretch',
                 column_config={k: st.column_config.NumberColumn(k, format='%.4g') for k in PARAMS})

with tabs[4]:
    st.caption('Every metric of every run on 2017 (Q / mean(Q)). c_* are the CPM components (1 = perfect). '
               'volume_ratio = raw mean(sim) / mean(obs), not calibrated (drainage-area mismatch).')
    st.dataframe(metrics, hide_index=True, width='stretch')
    st.download_button('Download metrics.csv', metrics.to_csv(index=False).encode(), 'turkeycreek_cpm_metrics.csv',
                       'text/csv')
    if sel:
        st.download_button('Download hourly Q / mean(Q) of the selected runs',
                           qh[['observed'] + sel].to_csv().encode(), 'turkeycreek_q_selected.csv', 'text/csv')
