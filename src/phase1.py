"""phase1.py -- E0 v2 / E1 aggregate tables (revised eligibility). Run: python phase1.py OUTDIR"""
import numpy as np
import pandas as pd
import os, sys, json, time
from build_v2 import one_per_participant, build, load, DPM

SEED = 20261019
GRID = np.round(np.arange(0, 6.0001, 0.02), 4)
HZ = [3.0, 5.0]
TASKLAB = {'A': 'Task A (NC -> MCI/dementia)', 'B': 'Task B (MCI -> dementia)'}


# ---------------------------------------------------------------- estimators
class CR:
    """Weighted Aalen-Johansen / KM on a fixed sample; fast bootstrap via multinomial weights."""
    def __init__(self, t, s):
        t = np.asarray(t, float); s = np.asarray(s, int)
        self.u, self.inv = np.unique(t, return_inverse=True)
        self.s = s; self.n = len(t)

    def curves(self, w=None, grid=GRID):
        if w is None:
            w = np.ones(self.n)
        nu = len(self.u)
        d1 = np.bincount(self.inv, w * (self.s == 1), nu)
        d2 = np.bincount(self.inv, w * (self.s == 2), nu)
        tot = np.bincount(self.inv, w, nu)
        nrisk = np.cumsum(tot[::-1])[::-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            h1 = np.where(nrisk > 0, d1 / nrisk, 0.0)
            h2 = np.where(nrisk > 0, d2 / nrisk, 0.0)
        S = np.cumprod(1 - h1 - h2)
        Sprev = np.concatenate([[1.0], S[:-1]])
        cif1 = np.cumsum(Sprev * h1)
        cif2 = np.cumsum(Sprev * h2)
        km1 = 1 - np.cumprod(1 - h1)            # deaths treated as censored
        idx = np.searchsorted(self.u, grid, side='right') - 1
        def at(a):
            return np.where(idx >= 0, a[np.clip(idx, 0, None)], 0.0)
        return {'aj': at(cif1), 'km': at(km1), 'death': at(cif2)}


def boot(t, s, B=1000, seed=SEED, grid=GRID):
    cr = CR(t, s); rng = np.random.default_rng(seed)
    est = cr.curves(grid=grid)
    reps = {k: np.empty((B, len(grid))) for k in est}
    for b in range(B):
        w = rng.multinomial(cr.n, np.full(cr.n, 1.0 / cr.n)).astype(float)
        c = cr.curves(w, grid)
        for k in c:
            reps[k][b] = c[k]
    lo = {k: np.percentile(reps[k], 2.5, axis=0) for k in reps}
    hi = {k: np.percentile(reps[k], 97.5, axis=0) for k in reps}
    return est, lo, hi


def km_median_iqr(t, e):
    """KM quantiles (25/50/75%) of time to 'e'==1."""
    t = np.asarray(t, float); e = np.asarray(e, int)
    u, inv = np.unique(t, return_inverse=True)
    d = np.bincount(inv, e, len(u)); tot = np.bincount(inv, minlength=len(u))
    nr = np.cumsum(tot[::-1])[::-1]
    S = np.cumprod(1 - d / nr)
    q = []
    for p in [0.25, 0.5, 0.75]:
        k = np.where(S <= 1 - p)[0]
        q.append(u[k[0]] if len(k) else np.nan)
    return q


# ---------------------------------------------------------------- formatting
def mean_sd(x):
    x = pd.Series(x).dropna(); return f"{x.mean():.1f} ({x.std():.1f})"


def n_pct(k, n, dp=1):
    return f"{int(k):,} ({100 * k / n:.{dp}f})"


def med_iqr(x, dp=1):
    x = pd.Series(x).dropna(); q = x.quantile([.25, .5, .75]).values
    return f"{q[1]:.{dp}f} ({q[0]:.{dp}f}-{q[2]:.{dp}f})"

# ---------------------------------------------------------------- E0 tables
def flow_table(d0, res):
    rows = []
    def add(task, step, n_part, n_lm, note=''):
        rows.append(dict(task=task, step=step, participants=n_part, landmarks=n_lm, note=note))
    for t in 'AB':
        lab = {'A': 'NACCUDSD==1 (normal cognition)', 'B': 'NACCUDSD==3 (MCI)'}[t]
        add(t, '0 All UDS visits in file', d0.NACCID.nunique(), len(d0), 'visits = landmarks column')
        c = res[f'{t}_candidates']
        add(t, f'1 Visits with {lab}', c.NACCID.nunique(), len(c))
        a = c[c.age50]
        add(t, '1x Excluded: age <50 at landmark', np.nan, int((~c.age50).sum()))
        add(t, '2 Age >=50 at landmark', a.NACCID.nunique(), len(a))
        nofu = a[~a.has_fu]
        add(t, '2x Excluded: no later visit and no death within 24 months of the landmark', np.nan, len(nofu),
            f'{nofu.NACCID.nunique()} participants touched; {(nofu.lm_date >= pd.Timestamp("2025-01-01")).sum()} of these landmarks dated 2025+')
        x1 = nofu[nofu.death_after_lm]
        add(t, '2x1  of which: no later visit, death >24 months after landmark (eligible under E0 v1 rule)', x1.NACCID.nunique(), len(x1))
        x2 = nofu[~nofu.death_after_lm]
        add(t, '2x2  of which: no later visit and no recorded death after landmark', x2.NACCID.nunique(), len(x2))
        el = res[f'{t}_eligible']
        add(t, '3 Eligible landmarks (>=1 later visit or death within 24 months)', el.NACCID.nunique(), len(el))
        add(t, '3a  of which: with >=1 later visit', el[el.n_later > 0].NACCID.nunique(), int((el.n_later > 0).sum()))
        add(t, '3b  of which: death within 24 months but no later visit', el[el.n_later == 0].NACCID.nunique(), int((el.n_later == 0).sum()))
        e60 = el[el.age60]
        add(t, '3s Sensitivity: eligible with age >=60', e60.NACCID.nunique(), len(e60))
        ev = res[f'{t}_eval']
        add(t, '4 Evaluation sample: one random landmark per participant (seed 20261019)', len(ev), len(ev))
        add(t, '4s Evaluation sample restricted to age >=60', int(ev.age60.sum()), int(ev.age60.sum()))
        for k, lab2 in [(0, 'censored'), (1, 'event'), (2, 'competing death (24-mo window)')]:
            add(t, f'5 Eval outcome: {lab2} (any time)', int((ev.status == k).sum()), int((ev.status == k).sum()))
        add(t, '5z Eval: censored at time 0 (should be 0 under v2 rule)', int(((ev.time <= 0) & (ev.status == 0)).sum()), int(((ev.time <= 0) & (ev.status == 0)).sum()))
    return pd.DataFrame(rows)


def table1(res):
    out = {}
    for t in 'AB':
        el = res[f'{t}_eligible']; ev = res[f'{t}_eval']; n = len(ev)
        r = {}
        r['Participants, n'] = f'{n:,}'
        r['Eligible landmarks (all), n'] = f'{len(el):,}'
        r['Landmarks per participant, median (IQR)'] = med_iqr(el.groupby('NACCID').size(), 0)
        r['Age at landmark, mean (SD), y'] = mean_sd(ev.age)
        for g in ['<70', '70-79', '>=80']:
            r[f'  Age {g}, n (%)'] = n_pct((ev.age_group == g).sum(), n)
        r['Female, n (%)'] = n_pct((ev.female == 1).sum(), n)
        r['  Sex unknown, n'] = f'{int(ev.female.isna().sum())}'
        r['Education, mean (SD), y'] = mean_sd(ev.educ)
        r['  Education missing, n'] = f'{int(ev.educ.isna().sum())}'
        for g in ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'Other/unknown']:
            r[f'Race/ethnicity: {g}, n (%)'] = n_pct((ev.race_eth == g).sum(), n)
        gen = ev.apoe4.notna()
        r['APOE e4 carrier, n (% of genotyped)'] = n_pct((ev.apoe4 >= 1).sum(), gen.sum())
        r['  APOE genotyped, n (%)'] = n_pct(gen.sum(), n)
        r['CDR-SB, median (IQR)'] = med_iqr(ev.cdrsb)
        r['  CDR-SB missing, n'] = f'{int(ev.cdrsb.isna().sum())}'
        for v, lab in [(1.0, 'v1'), (2.0, 'v2'), (3.0, 'v3.0'), (3.2, 'v3.2'), (4.0, 'v4')]:
            r[f'UDS version at landmark: {lab}, n (%)'] = n_pct((ev.FORMVER == v).sum(), n)
        r['Telephone packet at landmark (T/IT), n (%)'] = n_pct(ev.PACKET.isin(['T', 'IT']).sum(), n)
        q = km_median_iqr(ev.time, (ev.status == 0).astype(int))
        r['Follow-up after landmark, reverse-KM median (IQR), y'] = f'{q[1]:.1f} ({q[0]:.1f}-{q[2]:.1f})'
        r['Observed follow-up (time to event/death/censoring), median (IQR), y'] = med_iqr(ev.time)
        r['Events within 3 y, n'] = f'{int(((ev.status == 1) & (ev.time <= 3)).sum()):,}'
        r['Deaths before event within 3 y, n'] = f'{int(((ev.status == 2) & (ev.time <= 3)).sum()):,}'
        r['Events within 5 y, n'] = f'{int(((ev.status == 1) & (ev.time <= 5)).sum()):,}'
        r['Deaths before event within 5 y, n'] = f'{int(((ev.status == 2) & (ev.time <= 5)).sum()):,}'
        out[TASKLAB[t]] = r
    return pd.DataFrame(out).rename_axis('characteristic').reset_index()


def centers_table(res):
    rows = []; summ = []
    for t in 'AB':
        ev = res[f'{t}_eval']
        g = ev.groupby('NACCADC')
        tab = pd.DataFrame({
            'participants': g.size(),
            'events_3y': g.apply(lambda x: int(((x.status == 1) & (x.time <= 3)).sum())),
            'deaths_3y': g.apply(lambda x: int(((x.status == 2) & (x.time <= 3)).sum())),
            'events_5y': g.apply(lambda x: int(((x.status == 1) & (x.time <= 5)).sum())),
            'deaths_5y': g.apply(lambda x: int(((x.status == 2) & (x.time <= 5)).sum())),
            'landmarks_all_eligible': res[f'{t}_eligible'].groupby('NACCADC').size(),
        }).fillna(0).astype(int).reset_index()
        tab.insert(0, 'task', t)
        rows.append(tab)
        for h in ['3y', '5y']:
            e = tab[f'events_{h}']
            summ.append(dict(task=t, horizon=h, n_centers=len(tab), centers_ge100=int((e >= 100).sum()),
                             centers_ge50=int((e >= 50).sum()), centers_lt20=int((e < 20).sum()),
                             min_events=int(e.min()), median_events=float(e.median()), max_events=int(e.max())))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summ)


def temporal_v4(d0, res, freeze):
    rows = []
    def cnt(x, t, part, sample):
        r = dict(task=t, part=part, sample=sample, participants=x.NACCID.nunique(), landmarks=len(x),
                 events_any=int((x.status == 1).sum()), deaths_any=int((x.status == 2).sum()))
        for h in [1, 2, 3, 5]:
            r[f'events_{h}y'] = int(((x.status == 1) & (x.time <= h)).sum())
            r[f'deaths_{h}y'] = int(((x.status == 2) & (x.time <= h)).sum())
        r['max_followup_y'] = round(float(x.time.max()), 2) if len(x) else np.nan
        r['median_followup_obs_y'] = round(float(x.time.median()), 2) if len(x) else np.nan
        return r
    v4first = d0.loc[d0.FORMVER == 4, 'date'].min()
    for t in 'AB':
        dev, val = res[f'{t}_dev'], res[f'{t}_val']
        rows.append(cnt(dev, t, 'development (landmark <=2014-12-31, admin-censored 2014-12-31)', 'all eligible landmarks'))
        rows.append(cnt(one_per_participant(dev), t, 'development (landmark <=2014-12-31, admin-censored 2014-12-31)', 'one landmark per participant'))
        rows.append(cnt(val, t, 'validation (landmark >=2015-01-01)', 'all eligible landmarks'))
        rows.append(cnt(res[f'{t}_val_eval'], t, 'validation (landmark >=2015-01-01)', 'one landmark per participant'))
        ov = len(set(dev.NACCID) & set(val.NACCID))
        rows[-1]['note'] = f'{ov} participants contribute landmarks to both periods'
        c = res[f'{t}_candidates']; el = res[f'{t}_eligible']
        c4 = c[(c.FORMVER == 4) & c.age50]
        r = cnt(el[el.FORMVER == 4], t, 'UDS v4 landmarks (FORMVER==4)', 'all eligible landmarks')
        pot = (freeze - c4.lm_date).dt.days / 365.25
        r['note'] = (f'v4 first visit {v4first.date()}; age>=50 v4 candidate landmarks {len(c4)} '
                     f'({c4.NACCID.nunique()} participants), {int((~c4.has_fu).sum())} without follow-up yet; '
                     f'potential follow-up to freeze {freeze.date()}: max {pot.max():.2f} y, '
                     f'n with >=1 y potential {int((pot >= 1).sum())}; eligible with observed follow-up >=1 y '
                     f'{int((el[el.FORMVER == 4].time >= 1).sum())}')
        rows.append(r)
        rows.append(cnt(one_per_participant(el[el.FORMVER == 4]), t, 'UDS v4 landmarks (FORMVER==4)', 'one landmark per participant'))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- E1
GROUPS = ['Overall', '<70', '70-79', '>=80']


def boot_full(t, s, B=1000, seed=SEED, grid=GRID):
    cr = CR(t, s); rng = np.random.default_rng(seed)
    est = cr.curves(grid=grid); est['diff'] = est['km'] - est['aj']
    reps = {k: np.empty((B, len(grid))) for k in est}
    for b in range(B):
        w = rng.multinomial(cr.n, np.full(cr.n, 1.0 / cr.n)).astype(float)
        c = cr.curves(w, grid); c['diff'] = c['km'] - c['aj']
        for k in c:
            reps[k][b] = c[k]
    lo = {k: np.percentile(reps[k], 2.5, axis=0) for k in reps}
    hi = {k: np.percentile(reps[k], 97.5, axis=0) for k in reps}
    return est, lo, hi


def run_e1(res, B=1000):
    curves = {}; rows = []
    for t in 'AB':
        ev = res[f'{t}_eval']
        for g in GROUPS:
            x = ev if g == 'Overall' else ev[ev.age_group == g]
            est, lo, hi = boot_full(x.time.values, x.status.values, B=B)
            curves[(t, g)] = (est, lo, hi)
            for h in HZ:
                i = int(np.where(np.isclose(GRID, h))[0][0])
                r = dict(task=t, window='24 mo (primary)', age_group=g, n=len(x), horizon_y=h,
                         n_events_by_h=int(((x.status == 1) & (x.time <= h)).sum()),
                         n_deaths_by_h=int(((x.status == 2) & (x.time <= h)).sum()))
                for k, nm in [('aj', 'AJ_cif_event'), ('km', 'oneminusKM_event'), ('death', 'AJ_cif_death'), ('diff', 'KM_minus_AJ')]:
                    r[nm] = est[k][i]; r[nm + '_lo'] = lo[k][i]; r[nm + '_hi'] = hi[k][i]
                r['relative_overestimation_pct'] = 100 * (est['km'][i] / est['aj'][i] - 1)
                rows.append(r)
            for wk, wl in [('w12', '12 mo'), ('w36', '36 mo'), ('winf', 'any recorded death (no window)')]:
                tt, ss = x['time_' + wk].values, x['status_' + wk].values
                c = CR(tt, ss).curves(grid=np.array(HZ))
                for j, h in enumerate(HZ):
                    rows.append(dict(task=t, window=wl, age_group=g, n=len(x), horizon_y=h,
                                     n_events_by_h=int(((ss == 1) & (tt <= h)).sum()),
                                     n_deaths_by_h=int(((ss == 2) & (tt <= h)).sum()),
                                     AJ_cif_event=c['aj'][j], oneminusKM_event=c['km'][j],
                                     AJ_cif_death=c['death'][j], KM_minus_AJ=c['km'][j] - c['aj'][j],
                                     relative_overestimation_pct=100 * (c['km'][j] / c['aj'][j] - 1)))
    return curves, pd.DataFrame(rows)



# ---------------------------------------------------------------- exploratory form-version scheme
def formversion_counts(d0, res):
    """Candidate definitions for the exploratory UDS v4 scheme. Development: landmarks and follow-up
    up to 2024-12-31 (admin-censored). Evaluation: each participant's LAST pre-2025 visit (the landmark)
    with landmark status, dated in a recency window, age>=50, with >=1 later visit or death within 24 mo;
    outcomes ascertained after 2025-01-01 (UDS v4 began 2025-01-07). Horizon measured from the landmark."""
    from build_v2 import build_task, eligible, death_table, recode
    cut = pd.Timestamp('2024-12-31')
    d = d0.sort_values(['NACCID', 'date'])
    fv_next = d.groupby('NACCID')['FORMVER'].shift(-1)
    d = d.assign(fv_next=fv_next)
    fv_at = d.set_index(['NACCID', 'date'])['FORMVER']
    fv_at = fv_at[~fv_at.index.duplicated()]
    deaths = res['deaths']
    rows = []
    for t in 'AB':
        el = res[f'{t}_eligible']
        dev = eligible(build_task(d0, deaths, t, cutoff=cut))
        rows.append(dict(task=t, definition='development: landmarks <=2024-12-31, follow-up admin-censored 2024-12-31',
                         window='-', horizon_y=np.nan, participants=dev.NACCID.nunique(), landmarks=len(dev),
                         events_by_h=int((dev.status == 1).sum()), deaths_by_h=int((dev.status == 2).sum())))
        pre = d[d.date <= cut]
        last_pre = pre.groupby('NACCID').tail(1)[['NACCID', 'date', 'fv_next']].rename(columns={'date': 'lm_date'})
        x = el.merge(last_pre, on=['NACCID', 'lm_date'], how='inner')
        # form version of the visit at which the event was ascertained
        evfv = [fv_at.get((i, e), np.nan) if pd.notna(e) else np.nan for i, e in zip(x.NACCID, x.next_ev)]
        x['ev_fv'] = evfv
        for wname, start in [('landmark 2023-01-01..2024-12-31', '2023-01-01'), ('landmark 2023-07-01..2024-12-31', '2023-07-01'),
                             ('landmark 2024-01-01..2024-12-31', '2024-01-01')]:
            for nv4 in [False, True]:
                y = x[x.lm_date >= pd.Timestamp(start)]
                if nv4:
                    y = y[(y.fv_next == 4) | (y.n_later == 0)]
                lab = ('last pre-2025 visit as landmark' + ('; next visit is UDS v4 (or death only)' if nv4 else ''))
                for h in [1.0, 1.5, 2.0]:
                    ev = (y.status == 1) & (y.time <= h)
                    rows.append(dict(task=t, definition=lab, window=wname, horizon_y=h, participants=y.NACCID.nunique(),
                                     landmarks=len(y), events_by_h=int(ev.sum()),
                                     events_by_h_at_v4_visit=int((ev & (y.ev_fv == 4)).sum()),
                                     deaths_by_h=int(((y.status == 2) & (y.time <= h)).sum()),
                                     censored_before_h=int(((y.status == 0) & (y.time < h)).sum()),
                                     next_visit_v4_pct=round(100 * float((y.fv_next == 4).mean()), 1) if len(y) else np.nan,
                                     max_followup_y=round(float(y.time.max()), 2) if len(y) else np.nan))
    return pd.DataFrame(rows)


def curves_long(curves):
    rows = []
    for (t, g), (est, lo, hi) in curves.items():
        m = GRID <= 6.0001
        for k in ['aj', 'km', 'death', 'diff']:
            rows.append(pd.DataFrame(dict(task=t, age_group=g, quantity=k, t=GRID[m], est=est[k][m], lo=lo[k][m], hi=hi[k][m])))
    return pd.concat(rows, ignore_index=True)


def main(out):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d0, res = build(write=True)
    flow_table(d0, res).to_csv(os.path.join(out, 'flow_v2.csv'), index=False)
    table1(res).to_csv(os.path.join(out, 'table1_v2.csv'), index=False)
    cen, summ = centers_table(res)
    cen.to_csv(os.path.join(out, 'tableS3_v2.csv'), index=False)
    summ.to_csv(os.path.join(out, 'tableS3_summary_v2.csv'), index=False)
    temporal_v4(d0, res, d0.date.max()).to_csv(os.path.join(out, 'temporal_counts_v2.csv'), index=False)
    formversion_counts(d0, res).to_csv(os.path.join(out, 'formversion_counts.csv'), index=False)
    curves, e1 = run_e1(res, B=int(os.environ.get('PHASE1_B', '1000')))
    e1.to_csv(os.path.join(out, 'e1_cif_summary_v2.csv'), index=False)
    curves_long(curves).to_csv(os.path.join(out, 'e1_curves_v2.csv'), index=False)
    # deaths: gap last visit -> death
    dt = res['deaths']; dd = dt[dt.died == 1]
    gap = (dd.dod - dd.last_visit).dt.days / DPM
    json.dump(dict(n_died=int(len(dd)), gap_median_months=float(gap.median()),
                   pct_within_12=float((gap <= 12).mean() * 100), pct_within_24=float((gap <= 24).mean() * 100),
                   pct_within_36=float((gap <= 36).mean() * 100), runtime_s=time.time() - t0,
                   freeze=str(d0.date.max().date()), v4_first=str(d0.loc[d0.FORMVER == 4, 'date'].min().date())),
              open(os.path.join(out, 'phase1_meta.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'out_phase1')
