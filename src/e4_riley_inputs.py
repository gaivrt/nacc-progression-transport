"""e4_riley_inputs.py -- aggregate inputs for Riley et al. (BMJ 2020) sample-size criteria.

Writes: number of candidate predictor parameters (after missing indicators), development sizes for each
LOCO fold / temporal / random fold, outcome proportions, event rates, and anticipated Cox-Snell R2 from
out-of-fold predictions of the first full run (random scheme).
"""
import glob
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FT
import models as MD
import run_fold as RF


def r2cs_binary(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); x = np.log(p / (1 - p)); n = len(y)
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e8, max_iter=1000).fit(x[:, None], y)
    q = np.clip(m.predict_proba(x[:, None])[:, 1], 1e-12, 1 - 1e-12)
    ll = np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
    phi = y.mean(); ll0 = n * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi))
    lr = 2 * (ll - ll0)
    return 1 - np.exp(-lr / n), 1 - np.exp(2 * ll0 / n), lr, n


def r2cs_cox(t, s, lp):
    from lifelines import CoxPHFitter
    df = pd.DataFrame(dict(T=np.maximum(np.minimum(t, 5.0), 1e-3), E=((s == 1) & (t <= 5.0)).astype(int), lp=lp))
    cph = CoxPHFitter().fit(df, 'T', 'E')
    lr = cph.log_likelihood_ratio_test().test_statistic
    n = len(df)
    return 1 - np.exp(-lr / n), lr, n


def main(out, runs):
    meta, Fd, blocks = RF.load_visits()
    rows = []
    add = lambda **k: rows.append(k)
    for task in ['A', 'B']:
        el, ev, dev, vev = RF.load_sets(task, 'primary')
        el = FT.attach_landmarks(el, meta[['NACCID', 'NACCVNUM', 'vidx', 'pos']])
        X = Fd.values.astype(np.float32)[el.vidx.values]
        pr = MD.Prep().fit(X)
        add(task=task, item='candidate_features_raw', value=Fd.shape[1])
        add(task=task, item='parameters_full_after_indicators', value=len(pr.keep) + len(pr.ind))
        add(task=task, item='features_kept', value=len(pr.keep))
        add(task=task, item='missing_indicators', value=len(pr.ind))
        par = [Fd.columns.get_loc(c) for c in FT.PARSIMONIOUS]
        pp = MD.Prep().fit(X[:, par])
        add(task=task, item='parameters_parsimonious_after_indicators', value=len(pp.keep) + len(pp.ind))
        # outcome information in the one-landmark-per-participant evaluation sample
        y3 = MD.horizon_labels(ev.time.values, ev.status.values, 3.0)
        add(task=task, item='eval_n', value=len(ev))
        add(task=task, item='eval_events_by3y', value=int(((ev.status == 1) & (ev.time <= 3)).sum()))
        add(task=task, item='eval_known_status_3y', value=int(np.isfinite(y3).sum()))
        add(task=task, item='eval_prop_event_3y_known', value=float(np.nanmean(y3)))
        tt = np.minimum(ev.time.values, 5.0)
        add(task=task, item='eval_events_by5y', value=int(((ev.status == 1) & (ev.time <= 5)).sum()))
        add(task=task, item='eval_persontime_5y', value=float(tt.sum()))
        add(task=task, item='eval_mean_followup_5y', value=float(tt.mean()))
        # development sizes
        cens = np.sort(el.NACCADC.unique())
        for c in cens:
            trp = ev[ev.NACCADC != c]
            add(task=task, item='loco_train', center=int(c), n_participants=int(el[el.NACCADC != c].NACCID.nunique()),
                n_landmarks=int((el.NACCADC != c).sum()),
                events3y_one_per_participant=int(((trp.status == 1) & (trp.time <= 3)).sum()),
                events_landmarks_3y=int(((el.status == 1) & (el.time <= 3) & (el.NACCADC != c)).sum()))
        dev1 = __import__('build_v2').one_per_participant(dev)
        add(task=task, item='temporal_dev', n_participants=int(dev.NACCID.nunique()), n_landmarks=len(dev),
            events3y_one_per_participant=int(((dev1.status == 1) & (dev1.time <= 3)).sum()),
            events_landmarks_3y=int(((dev.status == 1) & (dev.time <= 3)).sum()))
        # anticipated R2_CS from out-of-fold predictions (random scheme, first full run)
        P = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(runs, f'pred_{task}_random_*.parquet'))])
        P = P[P.is_eval]
        for mname in ['logistic', 'cox', 'gbt', 'crseq', 'finegray_w']:
            x = P[P.model == mname]
            if x.empty:
                continue
            y = MD.horizon_labels(x.time.values, x.status.values, 3.0); k = np.isfinite(y)
            r2, mx, lr, n = r2cs_binary(y[k], x['F1_3'].values[k])
            add(task=task, item='r2cs_binary_3y', model=mname, value=r2, max_r2cs=mx, lr=lr, n=n)
            F3 = np.clip(x['F1_3'].values, 1e-6, 1 - 1e-6)
            r2s, lrs, ns = r2cs_cox(x.time.values, x.status.values, np.log(-np.log(1 - F3)))
            add(task=task, item='r2cs_survival_5y_trunc', model=mname, value=r2s, lr=lrs, n=ns)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
