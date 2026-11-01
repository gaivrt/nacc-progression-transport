"""e4_prep.py -- landmark tables for E4 sensitivity variants (row-level output only under NACC_DERIVED).

imp : impaired-not-MCI (NACCUDSD=2) grouped with MCI -> task A event {2,3,4}; task B risk set {2,3}, event {4}
mid : event time placed at the midpoint between the last event-free visit and the diagnosis visit
Also writes the visit-feature cache with etiologic diagnoses (variant 'etio').
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_v2 as BV

OUT = BV.OUT
CUT = pd.Timestamp('2014-12-31')


def write(res_key, el, dev):
    ev = BV.one_per_participant(el)
    val = el[el.lm_date >= CUT + pd.Timedelta(days=1)]
    vev = BV.one_per_participant(val)
    for k, x in [('eligible', el), ('eval', ev), ('dev', dev), ('val_eval', vev)]:
        x.to_parquet(os.path.join(OUT, f'task{res_key}_landmarks_{k}_v2.parquet'), index=False)
    return dict(variant=res_key, eligible=len(el), participants=el.NACCID.nunique(), eval=len(ev),
                dev=len(dev), val_eval=len(vev), eval_events=int((ev.status == 1).sum()),
                eval_deaths=int((ev.status == 2).sum()))


def midpoint(x, visits):
    """replace event time by midpoint between last visit before the event visit and the event visit."""
    x = x.copy()
    ev = x.status.values == 1
    prev = np.full(len(x), np.datetime64('NaT'), dtype='datetime64[ns]')
    for i in np.where(ev)[0]:
        dts = visits[x.NACCID.iat[i]]
        j = np.searchsorted(dts, np.datetime64(x.next_ev.iat[i]), side='left') - 1
        prev[i] = dts[max(j, 0)]
    pv = pd.to_datetime(prev)
    mid = pv + (pd.to_datetime(x.next_ev) - pv) / 2
    tmid = (mid - x.lm_date).dt.days / 365.25
    x['time_orig'] = x['time']
    x.loc[ev, 'time'] = np.maximum(tmid[ev].values, 1 / 365.25)
    return x


def main():
    d = BV.recode(BV.load())
    deaths = BV.death_table(d)
    rows = []
    BV.TASKS['Aimp'] = dict(landmark={1}, event={2, 3, 4})
    BV.TASKS['Bimp'] = dict(landmark={2, 3}, event={4})
    for t in ['Aimp', 'Bimp']:
        el = BV.eligible(BV.build_task(d, deaths, t))
        dev = BV.eligible(BV.build_task(d, deaths, t, cutoff=CUT))
        rows.append(write(t, el, dev))
    visits = {k: g.values.astype('datetime64[ns]') for k, g in d.sort_values('date').groupby('NACCID')['date']}
    for t in ['A', 'B']:
        rd = lambda k: pd.read_parquet(os.path.join(OUT, f'task{t}_landmarks_{k}_v2.parquet'))
        for k in ['eligible', 'eval', 'dev', 'val_eval']:
            x = midpoint(rd(k), visits)
            x.to_parquet(os.path.join(OUT, f'task{t}mid_landmarks_{k}_v2.parquet'), index=False)
            if k == 'eval':
                e = x.status == 1
                rows.append(dict(variant=f'{t}mid', eval=len(x), eval_events=int(e.sum()),
                                 median_shift_y=float((x.time_orig - x.time)[e].median())))
    if os.environ.get('E4_SKIP_ETIO') != '1':
        import run_fold as RF
        RF.load_visits(etio=True)
    pd.DataFrame(rows).to_csv(sys.argv[1] if len(sys.argv) > 1 else 'e4_prep_counts.csv', index=False)


if __name__ == '__main__':
    main()
