"""e4_idm.py -- Weibull illness-death model with interval-censored onset (Methods 2.8 sensitivity).

States: 0 event-free (alive), 1 event (MCI/dementia for task A; dementia for task B), 2 dead.
Transitions 0->1, 0->2, 1->2, each Weibull in time since the landmark (Markov).  Onset of the event is
interval-censored between the last event-free visit (L) and the diagnosis visit (R); a death without a
diagnosis may have been preceded by an unobserved onset after L (Joly-Commenges likelihood). All recorded
deaths are used (no window), because the model itself accounts for possible undiagnosed onset before death.
Output: cumulative incidence of the event by 3 and 5 y, F01(t) = int_0^t S0(u) h01(u) du, overall and by age
group, with percentile bootstrap CIs (participants resampled), next to the Aalen-Johansen estimates.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_v2 as BV

GX, GW = np.polynomial.legendre.leggauss(24)
EPS = 1e-6


def prepare(ev, visits):
    t_last_free = np.where(ev.status.values == 1, np.nan, (ev.last_after - ev.lm_date).dt.days.values / 365.25)
    L = np.zeros(len(ev))
    for i in range(len(ev)):
        if ev.status.iat[i] == 1:
            dts = visits[ev.NACCID.iat[i]]
            j = np.searchsorted(dts, np.datetime64(ev.next_ev.iat[i]), side='left') - 1
            L[i] = (dts[max(j, 0)] - np.datetime64(ev.lm_date.iat[i])) / np.timedelta64(1, 'D') / 365.25
        else:
            L[i] = t_last_free[i]
    R = np.where(ev.status.values == 1, ev.time.values, np.nan)
    dead = (ev.died.values == 1) & ev.dod.notna().values & (ev.status.values != 1)
    D = np.where(dead, (ev.dod - ev.lm_date).dt.days.values / 365.25, np.nan)
    D = np.where(dead, np.maximum(D, L + 1 / 365.25), np.nan)
    kind = np.where(ev.status.values == 1, 1, np.where(dead, 2, 0))
    return dict(L=np.maximum(L, 0.0), R=R, D=D, kind=kind, age=ev.age.values)


def wb(theta):
    a = np.exp(theta[0::2]); b = np.exp(theta[1::2])          # shapes, scales for 01, 02, 12
    A = lambda k, t: (np.maximum(t, EPS) / b[k]) ** a[k]
    h = lambda k, t: a[k] / b[k] * (np.maximum(t, EPS) / b[k]) ** (a[k] - 1)
    return A, h


def nll(theta, X, w):
    A, h = wb(theta)
    L, R, D, kind = X['L'], X['R'], X['D'], X['kind']
    ll = np.zeros(len(L))
    c = kind == 0
    ll[c] = -(A(0, L[c]) + A(1, L[c]))
    e = kind == 1
    if e.any():
        lo, hi = L[e], R[e]
        u = (hi - lo)[:, None] / 2 * GX[None, :] + (hi + lo)[:, None] / 2
        f = np.exp(-(A(0, u) + A(1, u))) * h(0, u) * np.exp(-(A(2, hi[:, None]) - A(2, u)))
        ll[e] = np.log(np.maximum((f * GW).sum(1) * (hi - lo) / 2, 1e-300))
    dd = kind == 2
    if dd.any():
        lo, hi = L[dd], D[dd]
        u = (hi - lo)[:, None] / 2 * GX[None, :] + (hi + lo)[:, None] / 2
        f = np.exp(-(A(0, u) + A(1, u))) * h(0, u) * np.exp(-(A(2, hi[:, None]) - A(2, u))) * h(2, hi[:, None])
        integ = (f * GW).sum(1) * (hi - lo) / 2
        direct = np.exp(-(A(0, hi) + A(1, hi))) * h(1, hi)
        ll[dd] = np.log(np.maximum(direct + integ, 1e-300))
    return -(w * ll).sum() / w.sum()


def fit(X, w, th0=None):
    th0 = np.array([0.0, 1.5, 0.0, 2.5, 0.0, 1.0]) if th0 is None else th0
    r = minimize(nll, th0, args=(X, w), method='L-BFGS-B', options=dict(maxiter=500))
    return r.x, r


def cif01(theta, t):
    A, h = wb(theta)
    u = t / 2 * GX + t / 2
    return float((np.exp(-(A(0, u) + A(1, u))) * h(0, u) * GW).sum() * t / 2)


def _boot(args):
    X, seed, th = args
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(X['L']), np.full(len(X['L']), 1 / len(X['L']))).astype(float)
    tb, _ = fit(X, w, th)
    return [cif01(tb, 3.0), cif01(tb, 5.0)]


def main(out, B=100, procs=6):
    d = BV.recode(BV.load())
    visits = {k: g.values.astype('datetime64[ns]') for k, g in d.sort_values('date').groupby('NACCID')['date']}
    rows = []
    for t in ['A', 'B']:
        ev = pd.read_parquet(os.path.join(BV.OUT, f'task{t}_landmarks_eval_v2.parquet'))
        Xall = prepare(ev, visits)
        for g, m in [('Overall', np.ones(len(ev), bool)), ('<70', ev.age.values < 70),
                     ('70-79', (ev.age.values >= 70) & (ev.age.values < 80)), ('>=80', ev.age.values >= 80)]:
            X = {k: v[m] for k, v in Xall.items()}
            w = np.ones(m.sum())
            th, r = fit(X, w)
            jobs = [(X, 20261019 + b, th) for b in range(B)]
            if procs > 1:
                with Pool(procs) as p:
                    bs = np.array(p.map(_boot, jobs))
            else:
                bs = np.array([_boot(j) for j in jobs])
            for j, hz in enumerate([3.0, 5.0]):
                rows.append(dict(task=t, age_group=g, n=int(m.sum()), horizon_y=hz,
                                 n_events=int((X['kind'] == 1).sum()), n_deaths_no_dx=int((X['kind'] == 2).sum()),
                                 idm_cif_event=cif01(th, hz), idm_lo=float(np.percentile(bs[:, j], 2.5)),
                                 idm_hi=float(np.percentile(bs[:, j], 97.5)), n_boot=B, converged=bool(r.success),
                                 nll=float(r.fun), shapes=';'.join(f'{x:.3f}' for x in np.exp(th[0::2])),
                                 scales=';'.join(f'{x:.3f}' for x in np.exp(th[1::2]))))
            print(t, g, rows[-2]['idm_cif_event'], rows[-1]['idm_cif_event'], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'idm_results.csv', B=int(os.environ.get('IDM_B', '100')))
