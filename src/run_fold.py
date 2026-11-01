"""run_fold.py -- train all requested models for one (task, scheme, fold) and write held-out predictions.

python run_fold.py --task B --scheme loco --fold 3 --models all --out RUN_DIR [--max_train N (debug only)]
Schemes: random (5 participant folds), loco (46 held-out centres), temporal (develop <=2014-12-31,
validate landmarks >=2015-01-01), formver (exploratory: develop <=2024-12-31, evaluate last pre-2025 visit).
Predictions (row-level) are written under RUN_DIR (scratch); only aggregate tables leave HPC4.
"""
import argparse
import json
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as FT
import models as MD
import build_v2 as BV

DATA = BV.DATA
DER = BV.OUT
SEED = 20261019
CUT_TEMPORAL = pd.Timestamp('2014-12-31')
CUT_FORMVER = pd.Timestamp('2024-12-31')
FORMVER_START = pd.Timestamp('2023-07-01')
H_MAIN = [3.0, 5.0]
H_FORMVER = [1.5]
SUBCOLS = ['female', 'race_eth', 'age_group', 'apoe4', 'age']


def load_visits(etio=False):
    cache = os.path.join(DER, 'visit_features_etio_v2.pkl' if etio else 'visit_features_v2.pkl')
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    cols = FT.RAW_COLS + (FT.ETIO if etio else [])
    raw = pd.read_parquet(DATA, columns=sorted(set(cols)))
    meta, Fd, blocks = FT.build_visit_features(raw, etio=etio)
    obj = (meta, Fd, blocks)
    tmp = cache + f'.tmp{os.getpid()}'
    pd.to_pickle(obj, tmp)
    os.replace(tmp, cache)
    return obj


VARIANTS = ['primary', 'w12', 'w36', 'winf', 'imp', 'mid', 'notel', 'etio']


def load_sets(task, variant):
    """eligible, eval, dev, val_eval landmark tables for a sensitivity variant (Methods 2.8)."""
    key = task + ({'imp': 'imp', 'mid': 'mid'}.get(variant, ''))
    rd = lambda k: pd.read_parquet(os.path.join(DER, f'task{key}_landmarks_{k}_v2.parquet'))
    el, ev, dev, vev = rd('eligible'), rd('eval'), rd('dev'), rd('val_eval')
    if variant in ('w12', 'w36', 'winf'):
        for x in (el, ev, dev, vev):
            x['time'], x['status'] = x['time_' + variant].values, x['status_' + variant].values
    if variant == 'notel':
        tel = lambda x: x[~x.PACKET.isin(['T', 'IT'])].copy()
        el, dev = tel(el), tel(dev)
        ev = BV.one_per_participant(el)
        vev = BV.one_per_participant(el[el.lm_date >= CUT_TEMPORAL + pd.Timedelta(days=1)])
    return el, ev, dev, vev


def formver_sets(task):
    d = BV.recode(BV.load())
    deaths = BV.death_table(d)
    dev = BV.eligible(BV.build_task(d, deaths, task, cutoff=CUT_FORMVER))
    el = BV.eligible(BV.build_task(d, deaths, task))
    last_pre = d[d.date <= CUT_FORMVER].groupby('NACCID').tail(1)[['NACCID', 'NACCVNUM']]
    ev = el.merge(last_pre, on=['NACCID', 'NACCVNUM'], how='inner')
    ev = ev[ev.lm_date >= FORMVER_START].copy()
    return dev, ev


class FoldData:
    def __init__(self, task, scheme, fold, max_train=None, variant='primary'):
        self.task, self.scheme, self.fold, self.variant = task, scheme, fold, variant
        meta, Fd, blocks = load_visits(etio=(variant == 'etio'))
        self.meta = meta
        self.fnames = list(Fd.columns)
        self.blocks = blocks
        self.F = Fd.values.astype(np.float32)
        codes, uniq = pd.factorize(meta.NACCID)
        meta['pid'] = codes
        self.pstart = meta.groupby('pid')['vidx'].min().values
        self.plen = meta.groupby('pid').size().values
        el, ev, dev_, vev_ = load_sets(task, variant)
        cut = None
        if scheme == 'random':
            parts = np.sort(el.NACCID.unique())
            rng = np.random.default_rng(SEED)
            fid = pd.Series(rng.permutation(len(parts)) % 5, index=parts)
            infold = el.NACCID.map(fid).values == fold
            tr, te, evkeys = el[~infold], el[infold], ev
            self.H = H_MAIN
        elif scheme == 'loco':
            cens = np.sort(el.NACCADC.unique())
            c = cens[fold]
            self.center = int(c)
            tr, te, evkeys = el[el.NACCADC != c], el[el.NACCADC == c], ev
            self.H = H_MAIN
        elif scheme == 'temporal':
            tr = dev_
            te = el[el.lm_date >= CUT_TEMPORAL + pd.Timedelta(days=1)]
            evkeys = vev_
            cut = CUT_TEMPORAL
            self.H = H_MAIN
        elif scheme == 'formver':
            tr, te = formver_sets(task)
            evkeys = te
            cut = CUT_FORMVER
            self.H = H_FORMVER
        else:
            raise ValueError(scheme)
        if max_train:
            keep = np.random.default_rng(0).permutation(tr.NACCID.unique())[:max_train]
            tr = tr[tr.NACCID.isin(keep)]
        ekey = set(zip(evkeys.NACCID, evkeys.NACCVNUM))
        te = te.copy()
        te['is_eval'] = [k in ekey for k in zip(te.NACCID, te.NACCVNUM)]
        self.tr = FT.attach_landmarks(tr.reset_index(drop=True), meta[['NACCID', 'NACCVNUM', 'vidx', 'pos']])
        self.te = FT.attach_landmarks(te.reset_index(drop=True), meta[['NACCID', 'NACCVNUM', 'vidx', 'pos']])
        self.tr_pid = meta.pid.values[self.tr.vidx.values]
        self.te_pid = meta.pid.values[self.te.vidx.values]
        self.tr_vidx, self.te_vidx = self.tr.vidx.values, self.te.vidx.values
        self.lm_pos_tr, self.lm_pos_te = self.tr.pos.values, self.te.pos.values
        if cut is not None:
            self.plen_cut = meta.assign(ok=meta.date <= cut).groupby('pid')['ok'].sum().values.astype(int)
            self.plen_cut = np.maximum(self.plen_cut, 1)
        else:
            self.plen_cut = self.plen
        self.cut = cut
        self.par_idx = [self.fnames.index(f) for f in FT.PARSIMONIOUS]
        self._gpu = False

    def lm_features(self, df, parsimonious=False):
        X = self.F[df.vidx.values]
        return X[:, self.par_idx] if parsimonious else X

    def prepare_gpu(self):
        if self._gpu:
            return
        dev = MD.DEV
        rows = np.concatenate([np.arange(s, s + l) for s, l in
                               zip(self.pstart[np.unique(self.tr_pid)], self.plen_cut[np.unique(self.tr_pid)])])
        Xt = self.F[rows]
        mu = np.nanmean(Xt, 0); sd = np.nanstd(Xt, 0)
        keep = np.where(np.isfinite(mu) & (sd > 0))[0]
        Z = (self.F[:, keep] - mu[keep]) / sd[keep]
        M = ~np.isnan(Z)
        Z = np.clip(np.nan_to_num(Z), -6, 6)
        self.Xg = torch.tensor(Z, dtype=torch.float32, device=dev)
        self.Mg = torch.tensor(M, dtype=torch.float32, device=dev)
        self.verg = torch.tensor(self.meta.ver.values, device=dev)
        self.t1g = torch.tensor(self.meta.t_since_first.values, dtype=torch.float32, device=dev)
        self.gapg = torch.tensor(self.meta.gap.values, dtype=torch.float32, device=dev)
        self.p = len(keep)
        kn = [self.fnames[i] for i in keep]
        bl = {}
        for j, f in enumerate(kn):
            bl.setdefault(self.blocks[f], []).append(j)
        probs = {'np12': 0.25, 'np3': 0.25, 'npsh': 0.25, 'np4': 0.25, 'faq': 0.1, 'npi': 0.1, 'gds': 0.1,
                 'hist': 0.1, 'med': 0.1, 'vitals': 0.1, 'sensory': 0.1, 'b9': 0.1}
        self.block_drop = [(torch.tensor(v, device=dev), probs[b]) for b, v in bl.items() if b in probs]
        self.np_cols = torch.tensor(sum([bl.get(b, []) for b in ['np12', 'np3', 'npsh', 'np4']], []), device=dev)
        self._gpu = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--scheme', required=True)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--models', default='all')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_train', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', default='primary', choices=VARIANTS)
    ap.add_argument('--merge', action='store_true', help='re-run only --models and merge into existing outputs')
    a = ap.parse_args()
    models = MD.ALL_MODELS if a.models == 'all' else a.models.split(',')
    os.makedirs(a.out, exist_ok=True)
    tag = f'{a.task}_{a.scheme}_{a.fold:02d}'
    if (not a.merge) and os.path.exists(os.path.join(a.out, f'runtime_{tag}.csv')) and os.environ.get('SKIP_DONE', '1') == '1':
        print(f'{tag} already done; skipping', flush=True)
        return
    t0 = time.time()
    fd = FoldData(a.task, a.scheme, a.fold, a.max_train, a.variant)
    t_data = time.time() - t0
    base = fd.te[['NACCID', 'NACCVNUM', 'NACCADC', 'lm_date', 'is_eval', 'time', 'status'] + SUBCOLS].copy()
    base['task'], base['scheme'], base['fold'] = a.task, a.scheme, a.fold
    base['seed'], base['variant'] = a.seed, a.variant
    rt, preds = [], []
    for mname in models:
        t1 = time.time()
        info = dict(task=a.task, scheme=a.scheme, fold=a.fold, model=mname, seed=a.seed, variant=a.variant, n_train=len(fd.tr),
                    n_train_participants=int(fd.tr.NACCID.nunique()), n_test=len(fd.te),
                    n_test_eval=int(fd.te.is_eval.sum()), data_s=round(t_data, 1),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
        try:
            if len(fd.te) == 0:
                raise RuntimeError('empty test set')
            m = MD.make_model(mname, a.seed).fit(fd)
            t2 = time.time()
            out = m.predict(fd)
            df = base.copy(); df['model'] = mname
            for i, h in enumerate(fd.H):
                df[f'F1_{h:g}'] = out['F1'][:, i]
                df[f'F2_{h:g}'] = out['F2'][:, i] if out['F2'] is not None else np.nan
            preds.append(df)
            info.update(ok=True, fit_s=round(t2 - t1, 1), predict_s=round(time.time() - t2, 1),
                        epochs=getattr(m, 'epochs', None), lam=getattr(m, 'lam', None))
            del m
        except Exception as e:
            info.update(ok=False, error=f'{type(e).__name__}: {e}', tb=traceback.format_exc()[-1500:])
        info['total_s'] = round(time.time() - t1, 1)
        rt.append(info)
        print(json.dumps({k: v for k, v in info.items() if k != 'tb'}), flush=True)
        if not info['ok']:
            print(info.get('tb', ''), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pp, rp = os.path.join(a.out, f'pred_{tag}.parquet'), os.path.join(a.out, f'runtime_{tag}.csv')
    RT = pd.DataFrame(rt)
    if a.merge:
        if os.path.exists(pp):
            old = pd.read_parquet(pp)
            preds = [old[~old.model.isin(models)]] + preds
        if os.path.exists(rp):
            o = pd.read_csv(rp)
            RT = pd.concat([o[~o.model.isin(models)], RT], ignore_index=True)
    if preds:
        pd.concat(preds).to_parquet(pp, index=False)
    RT.to_csv(rp, index=False)


if __name__ == '__main__':
    main()
