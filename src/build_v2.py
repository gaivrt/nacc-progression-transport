"""
build_v2.py -- E0 v2 landmark dataset construction (revised eligibility: later visit OR death within 24-month window) (NACC progression manuscript).

Builds landmark tables for
  Task A: landmark NACCUDSD==1 (normal cognition) -> event: later visit NACCUDSD in {3,4}
  Task B: landmark NACCUDSD==3 (MCI)               -> event: later visit NACCUDSD == 4
Competing event: death before the event, counted ONLY if death date <= last event-free
visit + W months (W = 24 primary; 12, 36 sensitivity; 'inf' = any recorded death, extra
sensitivity). Otherwise censored at last event-free visit.

Death date: NACCYOD/NACCMOD, day 15. NACCMOD==99 (month unknown, year known) -> 1 July.
If imputed death date < last visit date (day-15 imputation artefact), death date is set to
the last visit date. Times in years = days / 365.25.

Writes row-level parquet files ONLY to <dataset>/derived/ (never to the artifact store).
Usage: build(df) -> dict of DataFrames; or run as script.
"""
import numpy as np
import pandas as pd

import os
DATA = os.environ.get("NACC_DATA", "./data/investigator_ftldlbd_nacc75.parquet")
OUT = os.environ.get("NACC_DERIVED", "./data/derived")
PRIMARY_W = 24
COLS = ['NACCID', 'NACCADC', 'NACCVNUM', 'VISITDATE', 'PACKET', 'FORMVER', 'NACCUDSD',
        'NACCAGE', 'NACCSEX', 'EDUC', 'NACCNE4S', 'NACCNIHR', 'NACCHISP', 'CDRSUM',
        'NACCDIED', 'NACCMOD', 'NACCYOD']
WINDOWS = {'w24': 24, 'w12': 12, 'w36': 36, 'winf': None}
TASKS = {'A': dict(landmark={1}, event={3, 4}), 'B': dict(landmark={3}, event={4})}
DPM = 365.25 / 12  # days per month


def load(path=DATA):
    d = pd.read_parquet(path, columns=COLS)
    d['date'] = pd.to_datetime(d['VISITDATE'], errors='raise')
    d = d.sort_values(['NACCID', 'date']).reset_index(drop=True)
    return d


def recode(d):
    d = d.copy()
    d['sex'] = d['NACCSEX'].map({1: 'Male', 2: 'Female'})            # 8/9 -> NaN
    d['female'] = np.where(d['sex'].isna(), np.nan, (d['NACCSEX'] == 2).astype(float))
    d['educ'] = d['EDUC'].where(d['EDUC'].between(0, 36)).astype(float)  # 99 unknown
    d['apoe4'] = d['NACCNE4S'].where(d['NACCNE4S'].isin([0, 1, 2])).astype(float)  # 9 missing
    d['cdrsb'] = d['CDRSUM'].where(d['CDRSUM'].between(0, 18)).astype(float)       # 99 not assessed
    race = np.select(
        [d['NACCHISP'] == 1,
         (d['NACCHISP'] == 0) & (d['NACCNIHR'] == 1),
         (d['NACCHISP'] == 0) & (d['NACCNIHR'] == 2)],
        ['Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black'], default='Other/unknown')
    d['race_eth'] = race
    d['age'] = d['NACCAGE'].astype(float)
    d['age_group'] = pd.cut(d['age'], [-np.inf, 70, 80, np.inf], right=False,
                            labels=['<70', '70-79', '>=80'])
    return d


def death_table(d):
    """One row per participant: last visit date, death flag, (adjusted) death date."""
    p = d.groupby('NACCID').agg(last_visit=('date', 'max'), died=('NACCDIED', 'max'),
                                yod=('NACCYOD', 'max'), mod=('NACCMOD', 'max'))
    dead = p['died'] == 1
    yr_ok = dead & p['yod'].between(1900, 2100)
    mon_ok = p['mod'].between(1, 12)
    month = np.where(mon_ok, p['mod'], 7)
    day = np.where(mon_ok, 15, 1)
    dod = pd.to_datetime(pd.DataFrame({'year': np.where(yr_ok, p['yod'], 2000),
                                       'month': month, 'day': day}))
    dod.index = p.index
    p['dod_raw'] = dod.where(yr_ok)
    p['dod_month_imputed'] = yr_ok & ~mon_ok
    p['death_date_missing'] = dead & ~yr_ok
    p['dod_before_last'] = p['dod_raw'] < p['last_visit']
    p['dod'] = p[['dod_raw', 'last_visit']].max(axis=1).where(yr_ok)
    return p


def _outcomes(lm, cutoff=None):
    """lm has: lm_date, next_ev, last_after (last visit date <= cutoff), died, dod."""
    out = {}
    dod = lm['dod']
    if cutoff is not None:
        dod = dod.where(dod <= cutoff)
    has_ev = lm['next_ev'].notna()
    for key, w in WINDOWS.items():
        if w is None:
            in_win = dod.notna()
        else:
            in_win = dod.notna() & (dod <= lm['last_after'] + pd.to_timedelta(w * DPM, unit='D'))
        status = np.where(has_ev, 1, np.where(in_win, 2, 0))
        end = np.where(has_ev, lm['next_ev'], np.where(in_win, dod, lm['last_after']))
        tdays = (pd.to_datetime(end) - lm['lm_date']).dt.days.astype(float)
        tdays = np.where((status == 2) & (tdays <= 0), 1.0, tdays)  # death on landmark day
        sfx = '' if key == 'w24' else '_' + key
        out['status' + sfx] = status.astype(int)
        out['time' + sfx] = tdays / 365.25
    return pd.DataFrame(out, index=lm.index)


def build_task(d, deaths, task, cutoff=None):
    """All candidate landmark visits for a task (before age/follow-up filters), with flags."""
    spec = TASKS[task]
    v = d if cutoff is None else d[d['date'] <= cutoff]
    v = v.sort_values(['NACCID', 'date']).copy()
    ev_date = v['date'].where(v['NACCUDSD'].isin(spec['event']))
    # first event visit strictly after each row
    v['next_ev'] = ev_date.groupby(v['NACCID']).shift(-1)
    v['next_ev'] = v.groupby('NACCID')['next_ev'].bfill()
    v['last_after'] = v.groupby('NACCID')['date'].transform('max')
    v['n_later'] = v.groupby('NACCID').cumcount(ascending=False)
    lm = v[v['NACCUDSD'].isin(spec['landmark'])].copy()
    lm = lm.rename(columns={'date': 'lm_date'})
    lm = lm.join(deaths[['died', 'dod', 'death_date_missing']], on='NACCID')
    # last event-free visit: if an event occurs, censoring date irrelevant; otherwise last visit
    lm['death_after_lm'] = lm['dod'].notna() & (lm['dod'] > lm['lm_date'])
    if cutoff is not None:
        lm['death_after_lm'] &= lm['dod'] <= cutoff
    lm['age50'] = lm['age'] >= 50
    lm['age60'] = lm['age'] >= 60
    win = pd.to_timedelta(PRIMARY_W * DPM, unit='D')
    lm['death_in_win_lm'] = lm['dod'].notna() & (lm['dod'] >= lm['lm_date']) & (lm['dod'] <= lm['lm_date'] + win)
    if cutoff is not None:
        lm['death_in_win_lm'] &= lm['dod'] <= cutoff
    lm['has_fu_v1'] = (lm['n_later'] > 0) | lm['death_after_lm']   # E0 v1 rule (any later death)
    lm['has_fu'] = (lm['n_later'] > 0) | lm['death_in_win_lm']     # v2 rule (Methods 2.1/2.2)
    lm = pd.concat([lm, _outcomes(lm, cutoff)], axis=1)
    lm['task'] = task
    keep = ['task', 'NACCID', 'NACCADC', 'NACCVNUM', 'lm_date', 'age', 'age_group', 'sex', 'female',
            'educ', 'race_eth', 'apoe4', 'cdrsb', 'FORMVER', 'PACKET', 'NACCUDSD',
            'n_later', 'last_after', 'next_ev', 'died', 'dod', 'death_date_missing',
            'death_after_lm', 'death_in_win_lm', 'has_fu_v1', 'age50', 'age60', 'has_fu',
            'time', 'status', 'time_w12', 'status_w12', 'time_w36', 'status_w36',
            'time_winf', 'status_winf']
    return lm[keep].reset_index(drop=True)


def eligible(lm):
    return lm[lm['age50'] & lm['has_fu']].copy()


def one_per_participant(lm, seed=20261019):
    rng = np.random.default_rng(seed)
    x = lm.assign(_u=rng.random(len(lm)))
    idx = x.groupby('NACCID')['_u'].idxmin()
    return lm.loc[idx.sort_values().values].reset_index(drop=True)


def build(d=None, write=True, cut=pd.Timestamp('2014-12-31')):
    if d is None:
        d = load()
    d = recode(d)
    deaths = death_table(d)
    res = {'deaths': deaths}
    for t in TASKS:
        cand = build_task(d, deaths, t)
        el = eligible(cand)
        res[f'{t}_candidates'] = cand
        res[f'{t}_eligible'] = el
        res[f'{t}_eval'] = one_per_participant(el)
        dev = eligible(build_task(d, deaths, t, cutoff=cut))
        val = el[el['lm_date'] >= cut + pd.Timedelta(days=1)].copy()
        res[f'{t}_dev'] = dev
        res[f'{t}_val'] = val
        res[f'{t}_val_eval'] = one_per_participant(val)
    if write:
        os.makedirs(OUT, exist_ok=True)
        for t in TASKS:
            for k in ['eligible', 'eval', 'dev', 'val_eval']:
                res[f'{t}_{k}'].to_parquet(os.path.join(OUT, f'task{t}_landmarks_{k}_v2.parquet'), index=False)
    return d, res


if __name__ == '__main__':
    build()
