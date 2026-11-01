"""e5_analysis.py -- analyses on saved row-level predictions (no retraining), E2 full_v1 primary run, 3-y horizon.

(1) Local sample size for intercept recalibration: in each held-out LOCO center, split evaluation participants
    into two halves (cross-fitted); draw calibration samples of n participants from one half, estimate the logit
    shift delta so that mean predicted risk equals the Aalen-Johansen CIF of the calibration sample, evaluate O:E
    and ICI on the other half.  n in {0, 25, 50, 100, 200, 400, half}, plus an oracle shift fitted on the
    evaluation half itself (floor of what an intercept can do).
(2) Per-center bootstrap (B=200) of all metrics for all 15 models; REML pooling of calibration slope and log O:E
    (between-center heterogeneity tau).
(3) Paired differences with shared bootstrap weights: random and temporal schemes (participant bootstrap,
    B=1000); LOCO (per-center paired differences, REML pooled with prediction interval).
"""
import sys, os, glob, time, argparse
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

ROOT = os.environ.get('NACC_RUNS', './runs/full_v1')  # directory with pred_*.parquet from run_fold.py
TAU = 3.0
PCOL = 'F1_3'
MODELS = ['logistic', 'gbt', 'cox', 'rsf', 'finegray', 'rnn', 'ddh', 'crseq', 'gbt_w', 'cox_w', 'finegray_w',
          'crseq_unweighted', 'crseq_deathcens', 'crseq_lmonly', 'crseq_nomask']
MAIN = MODELS[:11]
PAIRS = [('crseq', m) for m in MODELS if m != 'crseq'] + [
    ('finegray_w', 'finegray'), ('finegray_w', 'cox_w'), ('finegray_w', 'gbt_w'), ('cox_w', 'cox'),
    ('gbt_w', 'gbt'), ('finegray', 'cox'), ('finegray_w', 'cox')]
METS = ['auc', 'brier', 'ipa', 'oe', 'slope', 'ici', 'obs_cif', 'mean_pred']


def load_wide(task, scheme):
    fs = sorted(glob.glob(f'{ROOT}/pred_{task}_{scheme}_*.parquet'))
    cols = ['NACCID', 'NACCVNUM', 'NACCADC', 'is_eval', 'time', 'status', 'fold', 'model', PCOL]
    D = pd.concat([pd.read_parquet(f, columns=cols) for f in fs], ignore_index=True)
    D = D[D.is_eval]
    W = D.pivot_table(index=['NACCID', 'NACCVNUM', 'fold', 'NACCADC', 'time', 'status'], columns='model',
                      values=PCOL, aggfunc='first').reset_index()
    miss = [m for m in MODELS if m not in W.columns]
    assert not miss, miss
    assert W[MODELS].notna().all().all(), (task, scheme, W[MODELS].isna().sum().to_dict())
    return W


# ------------------------------------------------------------------ (1) recalibration sample size (CPU workers)
def _init_worker():
    M.DEV = torch.device('cpu')
    torch.set_num_threads(1)


def _aj(t, s):
    S = M.Sample(t, s, np.full(len(t), 0.5), TAU)
    W1 = torch.ones(1, S.n, dtype=torch.float64)
    return float(S.weights(W1)['aj'][0])


def _eval(t, s, p):
    S = M.Sample(t, s, p, TAU)
    W1 = torch.ones(1, S.n, dtype=torch.float64)
    wd = S.weights(W1)
    oe = float(S.oe(W1, wd['aj'])[0])
    ici = float(S.calib(W1, wd['om'])[0][0])
    return oe, ici


def _shift(p, target):
    lp = np.log(p / (1 - p))
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if np.mean(1 / (1 + np.exp(-(lp + mid)))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def recal_center(args):
    task, center, sub, sizes, R, seed = args
    rng = np.random.default_rng(seed)
    n = len(sub)
    idx = rng.permutation(n)
    halves = [idx[: n // 2], idx[n // 2:]]
    t_all = sub['time'].to_numpy(float); s_all = sub['status'].to_numpy(int)
    rows = []
    for h in (0, 1):
        pool, ev = halves[h], halves[1 - h]
        te, se_ = t_all[ev], s_all[ev]
        if ((se_ == 1) & (te <= TAU)).sum() < 5:
            continue
        aj_ev = _aj(te, se_)
        draws = []                       # (size label, rep, indices into pool)
        draws.append(('0', 0, None))
        for k in sizes:
            if k <= len(pool):
                for r in range(R):
                    draws.append((str(k), r, rng.choice(pool, k, replace=False)))
        draws.append(('half', 0, pool))
        draws.append(('oracle', 0, ev))
        for lab, r, ci in draws:
            if ci is None:
                aj_c, ncal, evc = np.nan, 0, 0
            else:
                aj_c = _aj(t_all[ci], s_all[ci]); ncal = len(ci)
                evc = int(((s_all[ci] == 1) & (t_all[ci] <= TAU)).sum())
            for m in MAIN:
                p_all = np.clip(sub[m].to_numpy(float), 1e-6, 1 - 1e-6)
                pe = p_all[ev]
                if ci is None:
                    d = 0.0
                else:
                    d = _shift(p_all[ci], float(np.clip(aj_c, 0.002, 0.998)))
                ps = 1 / (1 + np.exp(-(np.log(pe / (1 - pe)) + d)))
                oe, ici = _eval(te, se_, ps)
                rows.append(dict(task=task, center=center, model=m, half=h, size=lab, rep=r, n_cal=ncal,
                                 events_cal=evc, aj_cal=aj_c, n_eval=len(ev), aj_eval=aj_ev, delta=d, oe=oe,
                                 ici=ici))
    return rows


# ------------------------------------------------------------------ (2)+(3) bootstrap metrics on GPU
def boot_all(sub, B, seed):
    t = sub['time'].to_numpy(float); s = sub['status'].to_numpy(int)
    W = M.boot_weights(len(t), B, seed)
    W1 = torch.ones(1, len(t), device=M.DEV, dtype=torch.float64)
    est, reps = {}, {}
    for m in MODELS:
        try:
            S = M.Sample(t, s, sub[m].to_numpy(float), TAU)
            e = S.all_metrics(W1, do_c=False)
            rr = {k: [] for k in METS}
            for b0 in range(0, B, 50):
                try:
                    r = S.all_metrics(W[b0:b0 + 50], do_c=False)
                except Exception:
                    continue
                for k in METS:
                    rr[k].append(r[k])
            est[m] = {k: float(e[k][0]) for k in METS}
            reps[m] = {k: (np.concatenate(rr[k]) if rr[k] else np.array([np.nan])) for k in METS}
        except Exception as ex:
            print('skip', m, len(t), repr(ex)[:80], flush=True)
            est[m] = {k: np.nan for k in METS}
            reps[m] = {k: np.array([np.nan]) for k in METS}
    return est, reps


def diff_rows(est, reps, extra):
    rows = []
    for a, b in PAIRS:
        for k in ['auc', 'ipa', 'ici', 'oe', 'slope', 'brier']:
            if k == 'oe':
                d = np.log(est[a][k]) - np.log(est[b][k]); dr = np.log(reps[a][k]) - np.log(reps[b][k])
            else:
                d = est[a][k] - est[b][k]; dr = reps[a][k] - reps[b][k]
            dr = dr[np.isfinite(dr)]
            if len(dr) < 10 or not np.isfinite(d):
                rows.append(dict(**extra, model_a=a, model_b=b, metric=k, diff=np.nan, lo=np.nan, hi=np.nan,
                                 se=np.nan, p_two=np.nan, B=len(dr)))
                continue
            rows.append(dict(**extra, model_a=a, model_b=b, metric=k, diff=d, lo=np.percentile(dr, 2.5),
                             hi=np.percentile(dr, 97.5), se=np.std(dr), p_two=2 * min((dr <= 0).mean(), (dr >= 0).mean()),
                             B=len(dr)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./runs/e5')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--skip_recal', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tasks = ['B'] if a.smoke else ['A', 'B']
    sizes = [25, 50, 100, 200, 400]
    R = 2 if a.smoke else 40
    Bp = 20 if a.smoke else 1000
    Bc = 10 if a.smoke else 200
    print('DEV', M.DEV, flush=True)

    loco = {tk: load_wide(tk, 'loco') for tk in tasks}
    print('loaded loco', {k: v.shape for k, v in loco.items()}, round(time.time() - t0), flush=True)

    # (1) recalibration curve
    jobs = []
    for tk in tasks:
        cents = sorted(loco[tk].NACCADC.unique())
        if a.smoke:
            cents = cents[:2]
        for i, cen in enumerate(cents):
            sub = loco[tk][loco[tk].NACCADC == cen].reset_index(drop=True)
            jobs.append((tk, int(cen), sub, sizes, R, 20261021 + i))
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    if a.skip_recal:
        jobs = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'), initializer=_init_worker) as ex:
        for rr in ex.map(recal_center, jobs, chunksize=1):
            rows.extend(rr)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'recal_curve_rows.csv'), index=False)
    print('recal done', len(rows), round(time.time() - t0), flush=True)

    # (2)+(3) LOCO per-center bootstrap and paired differences
    crow, drow = [], []
    for tk in tasks:
        cents = sorted(loco[tk].NACCADC.unique())
        if a.smoke:
            cents = cents[:2]
        for i, cen in enumerate(cents):
            sub = loco[tk][loco[tk].NACCADC == cen].reset_index(drop=True)
            nev = int(((sub.status == 1) & (sub.time <= TAU)).sum())
            if nev < 5 or len(sub) < 30:
                continue
            est, reps = boot_all(sub, Bc, 7000 + i)
            for m in MODELS:
                for k in METS:
                    v = reps[m][k]; v = v[np.isfinite(v)]
                    vt = M._tf(v, M.TRANSFORM.get(k))
                    crow.append(dict(task=tk, center=int(cen), n=len(sub), events=nev, model=m, metric=k,
                                     est=est[m][k], se_t=float(np.std(vt)) if len(v) else np.nan))
            drow += diff_rows(est, reps, dict(task=tk, scheme='loco_center', center=int(cen), n=len(sub), events=nev))
        print('loco boot', tk, round(time.time() - t0), flush=True)
    C = pd.DataFrame(crow); C.to_csv(os.path.join(a.out, 'center_boot.csv'), index=False)
    DL = pd.DataFrame(drow); DL.to_csv(os.path.join(a.out, 'paired_loco_center.csv'), index=False)

    pooled = []
    for (tk, m, k), g in C.groupby(['task', 'model', 'metric']):
        if k not in ('slope', 'oe', 'auc', 'ici', 'ipa'):
            continue
        g = g[(g.events >= 5) & np.isfinite(g.est) & np.isfinite(g.se_t) & (g.se_t > 1e-6)]
        if k in ('oe', 'ici'):
            g = g[g.est > 0]
        if k == 'auc':
            g = g[(g.est > 0) & (g.est < 1)]
        try:
            r = M.reml_pool(g.est.values, g.se_t.values, M.TRANSFORM.get(k))
        except Exception as ex:
            r = dict(err=repr(ex)[:80])
        pooled.append(dict(task=tk, model=m, metric=k, centers_used=len(g), **r))
    pd.DataFrame(pooled).to_csv(os.path.join(a.out, 'center_pooled.csv'), index=False)

    pd_rows = []
    for (tk, ma, mb, k), g in DL.groupby(['task', 'model_a', 'model_b', 'metric']):
        g = g[(g.events >= 5) & np.isfinite(g['diff']) & np.isfinite(g.se) & (g.se > 1e-6)]
        try:
            r = M.reml_pool(g['diff'].values, g.se.values, None)
        except Exception as ex:
            r = dict(err=repr(ex)[:80])
        pd_rows.append(dict(task=tk, model_a=ma, model_b=mb, metric=k, centers_used=len(g),
                            frac_centers_a_better=float((g['diff'] > 0).mean()), **r))
    pd.DataFrame(pd_rows).to_csv(os.path.join(a.out, 'paired_loco_pooled.csv'), index=False)
    print('loco pooled', round(time.time() - t0), flush=True)

    # (3) random and temporal paired bootstrap
    prow, sanity = [], []
    for tk in tasks:
        for sch in ['random', 'temporal']:
            Wd = load_wide(tk, sch)
            est, reps = boot_all(Wd, Bp, 20261019)
            for m in MODELS:
                sanity.append(dict(task=tk, scheme=sch, model=m, n=len(Wd), **est[m]))
            prow += diff_rows(est, reps, dict(task=tk, scheme=sch))
            print('paired', tk, sch, len(Wd), round(time.time() - t0), flush=True)
    pd.DataFrame(prow).to_csv(os.path.join(a.out, 'paired_random_temporal.csv'), index=False)
    pd.DataFrame(sanity).to_csv(os.path.join(a.out, 'sanity_point_estimates.csv'), index=False)
    print('ALL DONE', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
