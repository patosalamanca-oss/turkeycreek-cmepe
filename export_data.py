#!/usr/bin/env python3
'''Exports the Turkey Creek CMePe calibration results to small files in data/ for the Streamlit app.
Run once on the cluster (it is the only part that needs the model outputs), from anywhere:

    ~/.local/share/umn_hlm_calib_env/bin/python streamlit_cmepe/export_data.py

Reads ../calibration/cmepe/<pair>/ (calibration_summary.json, evaluations.csv, best/outputs/) and
../outputs_2016/ (default parameters), with the metric and CPM code of calib_tools/umn_hlm_calibrate.py
(the same numbers as compare_cmepe.ipynb).'''
import glob
import json
import os
import sys
from datetime import datetime, timezone

import netCDF4
import numpy as np
import pandas as pd
import yaml

APP = os.path.dirname(os.path.abspath(__file__))
TC = os.path.dirname(APP)                                    # HLM_tests/TurkeyCreek
sys.path.insert(0, os.path.join(TC, '..', '..', 'calib_tools'))
import umn_hlm_calibrate as uc                               # noqa: E402

LINK = 509412
START = datetime(2016, 10, 1, tzinfo=timezone.utc)
EVAL0, EVAL1 = pd.Timestamp('2017-01-01', tz='UTC'), pd.Timestamp('2018-01-01', tz='UTC')
OUT = os.path.join(APP, 'data')
CPM_FILES = sorted(glob.glob(os.path.join(TC, '..', '..', 'calib_tools', 'CMePe', '*.json')))
CPMS = [uc.load_cpm(f, lambda p: p) for f in CPM_FILES]
PAIRS = [l.strip() for l in open(os.path.join(TC, 'calib_cmepe', 'pairs.txt')) if l.strip()]
AREAS = pd.read_csv(os.path.join(TC, 'TurkeyCreek_runoff.csv')).set_index('stream')['area_sqkm']
PARAMS = ['crb3_vs', 'crb3_vg', 'crb3_vh', 'crb3_ssmax', 'crb3_sfc', 'crb3_lambda']


def read_q(out_dir):
    minutes, q = uc.read_gauge_series(out_dir, LINK, START)
    t = pd.to_datetime(START) + pd.to_timedelta(minutes, unit='min')
    return pd.Series(q, index=t).resample('1h', label='left', closed='left').mean()


def read_states(out_dir):
    parts = []
    for f in sorted(glob.glob(os.path.join(out_dir, 'dense_*.nc'))):
        with netCDF4.Dataset(f) as d:
            ids = np.asarray(d['system'][:]).astype(int)
            w = AREAS.reindex(ids).to_numpy()
            x = np.asarray(d['outputs'][:], dtype=float)                 # (system, time, variable)
            avg = np.tensordot(w / w.sum(), x, axes=(0, 0))
            t = pd.to_datetime(uc.minutes_origin(d['time'].units.replace('T', ' ').rstrip('Z'))) \
                + pd.to_timedelta(np.asarray(d['time'][:]), unit='min')
        parts.append(pd.DataFrame({'S_mm': avg[:, 0] * 1000, 'G_mm': avg[:, 2] * 1000}, index=t))
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep='last')]


def main():
    os.makedirs(OUT, exist_ok=True)
    obs = pd.read_csv(os.path.join(TC, 'observations', 'usgs_02250030_2017_hourly_cms.csv'), parse_dates=['time_utc'])
    obs = obs.set_index('time_utc')['discharge_cms'].astype(float)
    obs = obs[(obs.index >= EVAL0) & (obs.index < EVAL1)].dropna()

    default_params = pd.read_csv(os.path.join(TC, 'TurkeyCreek_runoff.csv'))[PARAMS].mean().to_dict()
    runs = {'default': {'dir': os.path.join(TC, 'outputs_2016'), 'pair': ('', ''), 'params': default_params}}
    for p in PAIRS:
        d = os.path.join(TC, 'calibration', 'cmepe', p)
        summ = json.load(open(os.path.join(d, 'calibration_summary.json')))
        runs[p] = {'dir': os.path.join(d, 'best', 'outputs'), 'pair': tuple(summ['metrics']),
                   'params': summ['best']['params'], 'best_id': summ['best']['id'], 'cal': d}

    zcols, mrows, srows = {'observed': obs / obs.mean()}, [], []
    for name, r in runs.items():
        sim = read_q(r['dir']).reindex(obs.index)
        ok = sim.notna().to_numpy()
        s, o = sim.to_numpy()[ok], obs.to_numpy()[ok]
        m = uc.compute_metrics(s / s.mean(), o / o.mean(), obs.index.year.to_numpy()[ok], CPMS)
        row = {'run': name, 'cpm_a': r['pair'][0].replace('CPM_', ''), 'cpm_b': r['pair'][1].replace('CPM_', '')}
        row.update({c['label'].replace('CPM_', ''): m[c['label']] for c in CPMS})
        row.update({k: m[k] for k in ('NSE', 'KGE', 'r', 'alpha', 'log_NSE')})
        row['spearman'] = m['cpm_spearman']
        row.update({'c_' + c: m[uc.CPM_PREFIX + c] for c in uc.CPM_COMPONENTS})
        row['volume_ratio'] = s.mean() / o.mean()
        row['n_hours'] = int(ok.sum())
        row.update({k: r['params'][k] for k in PARAMS})
        mrows.append(row)
        zcols[name] = sim / sim.mean()
        st = read_states(r['dir'])
        st = st[(st.index >= pd.Timestamp(START)) & (st.index <= EVAL1)]
        srows.append(st.assign(run=name))
        print('%-30s AllEq %.3f  NSE %.3f  hours %d  storage rows %d' % (name, row['AllEq'], row['NSE'], row['n_hours'], len(st)))

    metrics = pd.DataFrame(mrows)
    metrics.to_csv(os.path.join(OUT, 'metrics.csv'), index=False, float_format='%.6g')

    q = pd.DataFrame(zcols)
    q.index.name = 'time_utc'
    q.to_csv(os.path.join(OUT, 'q_hourly.csv.gz'), float_format='%.5g')

    st = pd.concat(srows)
    st.index.name = 'time_utc'
    st[['run', 'S_mm', 'G_mm']].to_csv(os.path.join(OUT, 'storage_hourly.csv.gz'), float_format='%.4g')

    pr = []
    for f in sorted(glob.glob(os.path.join(TC, 'pr_hourly_AORC_*.nc'))):
        with netCDF4.Dataset(f) as d:
            t = pd.to_datetime(np.asarray(d['time'][:]), unit='s', utc=True)      # seconds since 1970-01-01
            pr.append(pd.Series(np.asarray(d['pr'][:]).mean(axis=(1, 2)), index=t))
    pr = pd.concat(pr).sort_index()
    pr = pr[(pr.index >= pd.Timestamp(START)) & (pr.index < EVAL1)].resample('1D').sum()
    pr.rename('precip_mm').rename_axis('date').to_csv(os.path.join(OUT, 'precip_daily.csv'), float_format='%.3f')

    ev_rows = []
    for p in PAIRS:
        r = runs[p]
        a, b = r['pair']
        ev = pd.read_csv(os.path.join(r['cal'], 'evaluations.csv'))
        ev = ev[ev.status == 'ok']
        out = pd.DataFrame({'run': p, 'eval_id': ev['id'], 'stage': ev['stage'], 'cpm_a_value': ev[a], 'cpm_b_value': ev[b],
                            'pareto': ev['pareto'], 'is_best': (ev['id'] == r['best_id']).astype(int)})
        for k in PARAMS:
            out[k] = ev[k]
        ev_rows.append(out)
    pd.concat(ev_rows).to_csv(os.path.join(OUT, 'evaluations.csv.gz'), index=False, float_format='%.6g')

    spec = yaml.safe_load(open(os.path.join(TC, 'calib_cmepe', PAIRS[0] + '.yaml')))['parameters']
    pd.DataFrame([{'parameter': k, 'min': v['min'], 'max': v['max'], 'log': bool(v.get('log', False))}
                  for k, v in spec.items()]).to_csv(os.path.join(OUT, 'param_ranges.csv'), index=False)

    w = pd.DataFrame({c['label'].replace('CPM_', ''): c['weights'] for c in CPMS}).T.reindex(columns=uc.CPM_COMPONENTS).fillna(0)
    w.rename_axis('cpm').to_csv(os.path.join(OUT, 'cpm_weights.csv'))

    for f in sorted(os.listdir(OUT)):
        print('  data/%-24s %8.1f kB' % (f, os.path.getsize(os.path.join(OUT, f)) / 1024))


if __name__ == '__main__':
    main()
