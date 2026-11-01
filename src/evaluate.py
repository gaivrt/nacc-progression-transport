"""evaluate.py -- aggregate performance tables from held-out predictions (one task x scheme per call).

python evaluate.py --runs RUN_DIR --task B --scheme loco --out OUT_DIR [--B 1000] [--Bc 1000] [--Bsub 200]
Outputs (aggregate only): metrics_<task>_<scheme>.csv, center_<task>.csv, pooled_<task>.csv,
calib_deciles_*.csv, calib_curves_*.csv, decision_*.csv, subgroups_*.csv, center_calib_*.csv.
"""
import argparse
import glob
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as MT

DC_THR = {1.5: np.round(np.arange(0.02, 0.301, 0.01), 2), 3.0: np.round(np.arange(0.10, 0.401, 0.01), 2),
          5.0: np.round(np.arange(0.10, 0.601, 0.01), 2)}


def subgroups(df):
    g = {'Overall': np.ones(len(df), bool),
         'Female': (df.female == 1).values, 'Male': (df.female == 0).values,
         'Non-Hispanic White': (df.race_eth == 'Non-Hispanic White').values,
         'Non-Hispanic Black': (df.race_eth == 'Non-Hispanic Black').values,
         'Hispanic': (df.race_eth == 'Hispanic').values,
         'Age <70': (df.age < 70).values, 'Age 70-79': ((df.age >= 70) & (df.age < 80)).values,
         'Age >=80': (df.age >= 80).values,
         'APOE e4 carrier': (df.apoe4 >= 1).values, 'APOE e4 non-carrier': (df.apoe4 == 0).values}
    return g


def rows_from(out, base):
    r = []
    for k, (e, lo, hi, se, nb) in out.items():
        r.append(dict(base, metric=k, est=e, lo=lo, hi=hi, n_boot=nb))
    return r


def n_ev(df, h):
    return int(((df.status == 1) & (df.time <= h)).sum())


def recal_crossfit(df, col, h, seed=7):
    """Intercept-only (logit-shift) recalibration with 2-fold cross-fitting inside a held-out centre."""
    from scipy.optimize import brentq
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(df)) % 2
    p = np.clip(df[col].values, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    out = p.copy()
    for k in (0, 1):
        oth = half != k
        S = MT.Sample(df.time.values[oth], df.status.values[oth], p[oth], h)
        import torch
        aj = float(S.weights(torch.ones(1, S.n, device=MT.DEV, dtype=torch.float64))['aj'][0])
        aj = min(max(aj, 1e-4), 1 - 1e-4)
        f = lambda d: np.mean(1 / (1 + np.exp(-(lp[oth] + d)))) - aj
        try:
            d = brentq(f, -10, 10)
        except ValueError:
            d = 0.0
        out[~oth] = 1 / (1 + np.exp(-(lp[~oth] + d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--task', required=True)
    ap.add_argument('--scheme', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--Bc', type=int, default=1000)
    ap.add_argument('--Bsub', type=int, default=200)
    ap.add_argument('--Bcenter', type=int, default=200)
    ap.add_argument('--models', default='all')
    ap.add_argument('--what', default='main,calib,dc,alllm,sub,center,recal')
    ap.add_argument('--tag_extra', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(sum([glob.glob(os.path.join(r, f'pred_{a.task}_{a.scheme}_*.parquet')) for r in a.runs], []))
    P = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tag = f'{a.task}_{a.scheme}'
    sfx = ('' if a.models == 'all' else '_' + a.models.replace(',', '+')) + a.tag_extra
    W_ = set(a.what.split(','))
    H = sorted({float(c.split('_')[1]) for c in P.columns if c.startswith('F1_')})
    models = list(dict.fromkeys(P.model)) if a.models == 'all' else a.models.split(',')
    M, CD, CC, DC, SG, CEN, CENCAL = [], [], [], [], [], [], []
    t0 = time.time()
    for mname in models:
        X = P[P.model == mname]
        E = X[X.is_eval]
        for h in H:
            col = f'F1_{h:g}'
            base = dict(task=a.task, scheme=a.scheme, model=mname, horizon=h)
            if 'main' not in W_ and 'calib' not in W_ and 'dc' not in W_:
                S = None
            elif a.scheme != 'loco':
                out, S, _ = MT.evaluate(E.time.values, E.status.values, E[col].values, h, B=a.B, c_boot=a.Bc)
                M += rows_from(out, dict(base, sample='eval_one_per_participant', n=len(E), events=n_ev(E, h)))
            else:
                out, S, _ = MT.evaluate(E.time.values, E.status.values, E[col].values, h, B=a.Bsub, c_boot=0,
                                        do_c=True)
                M += rows_from(out, dict(base, sample='loco_concatenated_heldout', n=len(E), events=n_ev(E, h)))
            if 'calib' in W_ and S is not None:
                dec, cur = MT.calibration_tables(S)
                CD += [dict(base, **r) for r in dec]
                CC += [dict(base, **r) for r in cur]
            if 'dc' in W_ and S is not None:
                DC += [dict(base, **r) for r in MT.decision_curve(S, DC_THR.get(h, DC_THR[3.0]))]
            if 'alllm' in W_:
                o2, _, _ = MT.evaluate(X.time.values, X.status.values, X[col].values, h, B=0, do_c=False)
                M += rows_from(o2, dict(base, sample='all_eligible_landmarks', n=len(X), events=n_ev(X, h)))
            for gname, gm in (subgroups(E).items() if 'sub' in W_ else []):
                if gname == 'Overall':
                    continue
                Eg = E[gm]
                if n_ev(Eg, h) < 10 or len(Eg) < 50:
                    continue
                og, _, _ = MT.evaluate(Eg.time.values, Eg.status.values, Eg[col].values, h, B=a.Bsub,
                                       do_c=False)
                SG += rows_from(og, dict(base, subgroup=gname, n=len(Eg), events=n_ev(Eg, h)))
            if a.scheme == 'loco' and 'center' in W_:
                for cen, Ec in E.groupby('NACCADC'):
                    ne = n_ev(Ec, h)
                    if ne < 1 or ne == len(Ec):
                        continue
                    for recal in ((False, True) if 'recal' in W_ else (False,)):
                        pc = recal_crossfit(Ec, col, h) if recal else Ec[col].values
                        oc, Sc, reps = MT.evaluate(Ec.time.values, Ec.status.values, pc, h, B=a.Bcenter,
                                                   c_boot=a.Bcenter)
                        for k, (e, lo, hi, se, nb) in oc.items():
                            CEN.append(dict(base, center=int(cen), recalibrated=recal, n=len(Ec), events=ne,
                                            metric=k, est=e, se_t=MT.se_transformed(reps[k], MT.TRANSFORM.get(k))))
                        if ne >= 100 and not recal:
                            dc_, cu_ = MT.calibration_tables(Sc)
                            CENCAL += [dict(base, center=int(cen), center_n=len(Ec), center_events=ne, kind='decile', **r) for r in dc_]
                            CENCAL += [dict(base, center=int(cen), center_n=len(Ec), center_events=ne, kind='curve', **r) for r in cu_]
        print(f'{tag} {mname} done {time.time() - t0:.0f}s', flush=True)
    if M:
        pd.DataFrame(M).to_csv(os.path.join(a.out, f'metrics_{tag}{sfx}.csv'), index=False)
    if CD:
        pd.DataFrame(CD).to_csv(os.path.join(a.out, f'calib_deciles_{tag}{sfx}.csv'), index=False)
    if CC:
        pd.DataFrame(CC).to_csv(os.path.join(a.out, f'calib_curves_{tag}{sfx}.csv'), index=False)
    if DC:
        pd.DataFrame(DC).to_csv(os.path.join(a.out, f'decision_{tag}{sfx}.csv'), index=False)
    if SG:
        pd.DataFrame(SG).to_csv(os.path.join(a.out, f'subgroups_{tag}{sfx}.csv'), index=False)
    if a.scheme == 'loco' and CEN:
        C = pd.DataFrame(CEN)
        C.to_csv(os.path.join(a.out, f'center_{a.task}{sfx}.csv'), index=False)
        pd.DataFrame(CENCAL).to_csv(os.path.join(a.out, f'center_calib_{a.task}{sfx}.csv'), index=False)
        pool = []
        for (mname, h, recal, met), g in C.groupby(['model', 'horizon', 'recalibrated', 'metric']):
            g = g[np.isfinite(g.est) & np.isfinite(g.se_t) & (g.se_t > 1e-3)]
            if MT.TRANSFORM.get(met) == 'logit':
                g = g[(g.est > 0) & (g.est < 1)]
            if met in ('oe', 'ici'):
                g = g[g.est > 0]
            r = MT.reml_pool(g.est.values, g.se_t.values, MT.TRANSFORM.get(met))
            pool.append(dict(task=a.task, scheme='loco', model=mname, horizon=h, recalibrated=recal, metric=met,
                             centers=int(len(g)), **r))
        pd.DataFrame(pool).to_csv(os.path.join(a.out, f'pooled_{a.task}{sfx}.csv'), index=False)
    print(f'{tag} finished {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
