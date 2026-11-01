"""models.py -- comparator families (Methods 2.4) and the competing-risk sequence model (Methods 2.5).

Every model exposes fit(fd) and predict(fd) -> dict(F1=[n_test, len(H)], F2=[n_test, len(H)] or None)
where fd is a FoldData (see run_fold.py) and H the evaluation horizons in years.
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

TMAX = 5.0          # follow-up used for model development is truncated at the longest horizon
WIDTH = 0.25        # discrete-time interval width (years) for the sequence models -> K = 20
K = int(round(TMAX / WIDTH))
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trunc(time_, status):
    t = np.minimum(time_, TMAX)
    s = np.where(time_ > TMAX, 0, status)
    return t, s


def horizon_labels(time_, status, h):
    """1 if event of interest by h; 0 if known event-free at h (followed >= h, or died first); nan if unknown."""
    y = np.where((status == 1) & (time_ <= h), 1.0, 0.0)
    known = (time_ >= h) | ((status > 0) & (time_ <= h))
    return np.where(known, y, np.nan)


# ------------------------------------------------------------------ tabular preprocessing
class Prep:
    """Median imputation + missing indicators + standardisation, fitted on training rows only."""
    def __init__(self, indicators=True, scale=True):
        self.indicators, self.scale = indicators, scale

    def fit(self, X):
        ok = ~np.all(np.isnan(X), 0)
        sd = np.nanstd(np.where(ok, X, 0), 0)
        self.keep = np.where(ok & (sd > 0))[0]
        Xk = X[:, self.keep]
        self.med = np.nanmedian(Xk, 0)
        mf = np.isnan(Xk).mean(0)
        self.ind = np.where((mf > 0.001) & (mf < 0.999))[0] if self.indicators else np.array([], int)
        Z = self._raw(Xk)
        self.mu = Z.mean(0) if self.scale else np.zeros(Z.shape[1])
        s = Z.std(0) if self.scale else np.ones(Z.shape[1])
        self.sd = np.where(s > 0, s, 1.0)
        return self

    def _raw(self, Xk):
        miss = np.isnan(Xk)
        Z = np.where(miss, self.med, Xk)
        if len(self.ind):
            Z = np.concatenate([Z, miss[:, self.ind].astype(np.float32)], 1)
        return Z.astype(np.float32)

    def transform(self, X):
        Z = self._raw(X[:, self.keep])
        return ((Z - self.mu) / self.sd).astype(np.float32)


def inner_split(fd, frac=0.15, seed=11):
    """Validation subset of TRAINING centers (Methods 2.5); returns boolean mask over fd.tr rows."""
    rng = np.random.default_rng(seed)
    cen = fd.tr['NACCADC'].values
    uc = np.unique(cen)
    order = rng.permutation(uc)
    sizes = fd.tr.groupby('NACCADC').size()
    tot, chosen = 0, []
    for cc in order:
        if tot >= frac * len(cen):
            break
        chosen.append(cc); tot += sizes[cc]
    return np.isin(cen, chosen)


# ------------------------------------------------------------------ 1. logistic horizon classifier
class LogisticHorizon:
    name = 'logistic'

    def fit(self, fd):
        from sklearn.linear_model import LogisticRegression
        X = fd.lm_features(fd.tr, parsimonious=True)
        self.prep = Prep().fit(X)
        Z = self.prep.transform(X)
        va = inner_split(fd)
        self.m = {}
        for h in fd.H:
            y = horizon_labels(fd.tr.time.values, fd.tr.status.values, h)
            k = ~np.isnan(y)
            best = None
            for C in [0.01, 0.1, 1.0, 10.0]:
                m = LogisticRegression(C=C, max_iter=2000).fit(Z[k & ~va], y[k & ~va])
                p = np.clip(m.predict_proba(Z[k & va])[:, 1], 1e-6, 1 - 1e-6)
                ll = -np.mean(y[k & va] * np.log(p) + (1 - y[k & va]) * np.log(1 - p))
                if best is None or ll < best[0]:
                    best = (ll, C)
            self.m[h] = LogisticRegression(C=best[1], max_iter=2000).fit(Z[k], y[k])
        return self

    def predict(self, fd):
        Z = self.prep.transform(fd.lm_features(fd.te, parsimonious=True))
        return dict(F1=np.stack([self.m[h].predict_proba(Z)[:, 1] for h in fd.H], 1), F2=None)


# ------------------------------------------------------------------ 2. gradient-boosted trees
class GBTHorizon:
    name = 'gbt'
    weighted = False
    seed = 0
    GRID = [dict(max_leaf_nodes=15, min_samples_leaf=100), dict(max_leaf_nodes=31, min_samples_leaf=50),
            dict(max_leaf_nodes=63, min_samples_leaf=200)]

    @staticmethod
    def _usable(X):
        """columns with >= 2 distinct non-missing values in these rows (HistGB binning requirement)."""
        ok = []
        for j in range(X.shape[1]):
            v = X[:, j]
            v = v[~np.isnan(v)]
            if len(v) and v.min() < v.max():
                ok.append(j)
        return np.array(ok, int)

    def fit(self, fd):
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        X = fd.lm_features(fd.tr)
        va = inner_split(fd)
        self.m, self.cols = {}, {}
        kw = dict(learning_rate=0.05, max_iter=2000, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=30, l2_regularization=1.0, random_state=self.seed)
        W = participant_weights(fd.tr) if self.weighted else np.ones(len(fd.tr), np.float32)
        for h in fd.H:
            y = horizon_labels(fd.tr.time.values, fd.tr.status.values, h)
            k = ~np.isnan(y)
            ci = self._usable(X[k & ~va])
            best = None
            for g in self.GRID:
                m = H(**kw, **g).fit(X[k & ~va][:, ci], y[k & ~va], sample_weight=W[k & ~va])
                p = np.clip(m.predict_proba(X[k & va][:, ci])[:, 1], 1e-6, 1 - 1e-6)
                yy, ww = y[k & va], W[k & va]
                ll = -np.sum(ww * (yy * np.log(p) + (1 - yy) * np.log(1 - p))) / ww.sum()
                if best is None or ll < best[0]:
                    best = (ll, g)
            cf = self._usable(X[k])
            self.cols[h] = cf
            self.m[h] = H(**kw, **best[1]).fit(X[k][:, cf], y[k], sample_weight=W[k])
        return self

    def predict(self, fd):
        X = fd.lm_features(fd.te)
        return dict(F1=np.stack([self.m[h].predict_proba(X[:, self.cols[h]])[:, 1] for h in fd.H], 1), F2=None)


class GBTHorizonW(GBTHorizon):
    name = 'gbt_w'
    weighted = True


# ------------------------------------------------------------------ 3/5. Cox and Fine-Gray (ridge, torch)
def _km_censor(t, s):
    """KM of the censoring distribution; returns function G(t-) (left limit)."""
    u, inv = np.unique(t, return_inverse=True)
    c = np.bincount(inv, (s == 0).astype(float), len(u))
    n = np.bincount(inv, minlength=len(u)).astype(float)
    nr = np.cumsum(n[::-1])[::-1]
    G = np.cumprod(1 - c / nr)
    Gprev = np.concatenate([[1.0], G[:-1]])

    def Gm(x):
        i = np.searchsorted(u, x, side='left')   # number of unique times < x
        return np.where(i > 0, G[np.clip(i - 1, 0, None)], 1.0)
    return Gm, u, Gprev


class _PH:
    """Weighted (sub)distribution proportional hazards with Breslow ties; events = cause `cause`.
    fine_gray=True keeps competing-event subjects in the risk set with IPCW weights G(t-)/G(T_i-)."""

    def __init__(self, cause=1, fine_gray=False):
        self.cause, self.fg = cause, fine_gray

    def _setup(self, t, s, w=None):
        ut, inv = np.unique(t, return_inverse=True)
        w = np.ones(len(t), np.float32) if w is None else np.asarray(w, np.float32)
        d = dict(inv=torch.tensor(inv, device=DEV), ut=ut, nU=len(ut),
                 w=torch.tensor(w, device=DEV),
                 ev=torch.tensor((s == self.cause).astype(np.float32), device=DEV))
        if self.fg:
            Gm, _, _ = _km_censor(t, s)
            comp = (s != self.cause) & (s > 0)
            d['comp'] = torch.tensor(comp.astype(np.float32), device=DEV)
            d['Gti'] = torch.tensor(np.maximum(Gm(t), 1e-3).astype(np.float32), device=DEV)
            d['Gu'] = torch.tensor(np.maximum(Gm(ut), 1e-3).astype(np.float32), device=DEV)
        return d

    def _risk_sums(self, eta, d):
        e = torch.exp(eta) * d['w']
        nU = d['nU']
        tot = torch.zeros(nU, device=DEV).index_add_(0, d['inv'], e)
        at_risk = torch.flip(torch.cumsum(torch.flip(tot, [0]), 0), [0])     # T_j >= u
        if self.fg:
            ce = torch.zeros(nU, device=DEV).index_add_(0, d['inv'], e * d['comp'] / d['Gti'])
            cum = torch.cumsum(ce, 0) - ce                                     # competing with T_j < u
            at_risk = at_risk + d['Gu'] * cum
        return at_risk

    def _nll(self, beta, Z, d):
        eta = Z @ beta
        R = self._risk_sums(eta, d)
        dk = torch.zeros(d['nU'], device=DEV).index_add_(0, d['inv'], d['ev'] * d['w'])
        ll = (d['w'] * d['ev'] * eta).sum() - (dk * torch.log(R + 1e-30)).sum()
        return -ll / d['w'].sum()

    def fit(self, Z, t, s, lam, w=None):
        Zt = torch.tensor(Z, device=DEV)
        d = self._setup(t, s, w)
        beta = torch.zeros(Z.shape[1], device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([beta], lr=1, max_iter=200, history_size=20, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-7, tolerance_change=1e-10)

        def closure():
            opt.zero_grad()
            loss = self._nll(beta, Zt, d) + 0.5 * lam * (beta ** 2).sum()
            loss.backward()
            return loss
        for _ in range(3):
            opt.step(closure)
        self.beta = beta.detach()
        with torch.no_grad():
            R = self._risk_sums(Zt @ self.beta, d)
            dk = torch.zeros(d['nU'], device=DEV).index_add_(0, d['inv'], d['ev'] * d['w'])
            self.H0 = np.cumsum((dk / R).cpu().numpy())
        self.ut = d['ut']
        return self

    def score(self, Z, t, s, w=None):
        with torch.no_grad():
            return float(self._nll(self.beta, torch.tensor(Z, device=DEV), self._setup(t, s, w)))

    def cif(self, Z, H):
        lp = (torch.tensor(Z, device=DEV) @ self.beta).cpu().numpy()
        idx = np.searchsorted(self.ut, H, side='right') - 1
        H0 = np.where(idx >= 0, self.H0[np.clip(idx, 0, None)], 0.0)
        return 1 - np.exp(-np.exp(lp)[:, None] * H0[None, :])


class CoxCensorDeath:
    name = 'cox'
    fine_gray = False
    weighted = False
    LAMS = [1e-4, 1e-3, 1e-2, 1e-1]

    def fit(self, fd):
        X = fd.lm_features(fd.tr)
        self.prep = Prep().fit(X)
        Z = self.prep.transform(X)
        t, s = trunc(fd.tr.time.values, fd.tr.status.values)
        if not self.fine_gray:
            s = np.where(s == 2, 0, s)          # death treated as censoring
        w = participant_weights(fd.tr) if self.weighted else None
        wv = (lambda m_: None if w is None else w[m_])
        va = inner_split(fd)
        best = None
        for lam in self.LAMS:
            m = _PH(1, self.fine_gray).fit(Z[~va], t[~va], s[~va], lam, wv(~va))
            sc = m.score(Z[va], t[va], s[va], wv(va))
            if best is None or sc < best[0]:
                best = (sc, lam)
        self.lam = best[1]
        self.m = _PH(1, self.fine_gray).fit(Z, t, s, self.lam, w)
        return self

    def predict(self, fd):
        Z = self.prep.transform(fd.lm_features(fd.te))
        return dict(F1=self.m.cif(Z, np.array(fd.H)), F2=None)


def participant_weights(df):
    """1 / number of training landmarks of the participant (every participant contributes equally)."""
    return (1.0 / df.groupby('NACCID')['NACCID'].transform('size').values).astype(np.float32)


class FineGray(CoxCensorDeath):
    name = 'finegray'
    fine_gray = True


class CoxW(CoxCensorDeath):
    name = 'cox_w'
    weighted = True


class FineGrayW(CoxCensorDeath):
    name = 'finegray_w'
    fine_gray = True
    weighted = True


# ------------------------------------------------------------------ 4. random survival forest
class RSFCensorDeath:
    name = 'rsf'

    def __init__(self, n_estimators=200, min_samples_leaf=50, n_jobs=6):
        self.kw = dict(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features='sqrt',
                       n_jobs=n_jobs, random_state=0, low_memory=False)

    def fit(self, fd):
        from sksurv.ensemble import RandomSurvivalForest
        X = fd.lm_features(fd.tr)
        self.prep = Prep(scale=False).fit(X)
        Z = self.prep.transform(X)
        t, s = trunc(fd.tr.time.values, fd.tr.status.values)
        t = np.ceil(t * 12 - 1e-9) / 12          # monthly time grid
        t = np.maximum(t, 1 / 12)
        y = np.array(list(zip(s == 1, t)), dtype=[('e', bool), ('t', float)])
        self.m = RandomSurvivalForest(**self.kw).fit(Z, y)
        return self

    def predict(self, fd):
        Z = self.prep.transform(fd.lm_features(fd.te))
        out = []
        for i in range(0, len(Z), 4000):
            sf = self.m.predict_survival_function(Z[i:i + 4000], return_array=True)
            ut = self.m.unique_times_
            idx = np.searchsorted(ut, np.array(fd.H), side='right') - 1
            out.append(1 - np.where(idx >= 0, sf[:, np.clip(idx, 0, None)], 1.0))
        return dict(F1=np.concatenate(out), F2=None)


# ------------------------------------------------------------------ sequence models
class TimeEnc(nn.Module):
    def __init__(self, d, nf=8):
        super().__init__()
        self.freq = nn.Parameter(torch.exp(torch.linspace(math.log(0.1), math.log(4.0), nf)), requires_grad=False)
        self.lin = nn.Linear(2 * nf + 1, d)

    def forward(self, t):
        a = t[..., None] * self.freq * 2 * math.pi
        return self.lin(torch.cat([torch.sin(a), torch.cos(a), torch.log1p(t.clamp(min=0))[..., None]], -1))


class CRSeqNet(nn.Module):
    def __init__(self, p, d=64, layers=2, heads=4, ncls=3, use_version=True):
        super().__init__()
        self.Wx = nn.Linear(p, d)
        self.Wm = nn.Linear(p, d, bias=False)
        self.ver = nn.Embedding(4, d) if use_version else None
        if self.ver is not None:
            nn.init.zeros_(self.ver.weight)   # unseen form versions contribute 0 (not a random vector)
        self.g1, self.g2 = TimeEnc(d), TimeEnc(d)
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, K * ncls))
        self.ncls = ncls

    def forward(self, x, m, ver, t1, gap, pad):
        e = self.Wx(x * m) + self.Wm(m) + self.g1(t1) + self.g2(gap)
        if self.ver is not None:
            e = e + self.ver(ver)
        L = x.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
        h = self.enc(e, mask=causal, src_key_padding_mask=pad)
        return self.norm(h)

    def out(self, h):
        return self.head(h).view(*h.shape[:-1], K, self.ncls)


class GRUNet(nn.Module):
    """Time-aware GRU (inputs: values, mask, gap, time since first visit, version one-hot)."""
    def __init__(self, p, d=64, nout=2):
        super().__init__()
        self.inp = nn.Linear(2 * p + 2 + 4, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, nout))

    def forward(self, x, m, ver, t1, gap, pad):
        z = torch.cat([x * m, m, gap[..., None], t1[..., None] / 10, Fnn.one_hot(ver, 4).float()], -1)
        h, _ = self.gru(torch.relu(self.inp(z)))
        return h


class DDHNet(nn.Module):
    """Dynamic-DeepHit (Lee et al.): RNN + temporal attention + cause-specific subnetworks, joint softmax."""
    def __init__(self, p, d=64, ncause=2):
        super().__init__()
        self.inp = nn.Linear(2 * p + 1, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.att = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, 1))
        self.cs = nn.ModuleList([nn.Sequential(nn.Linear(d + 2 * p, d), nn.ReLU(), nn.Dropout(0.1),
                                               nn.Linear(d, d), nn.ReLU(), nn.Linear(d, K)) for _ in range(ncause)])
        self.surv = nn.Parameter(torch.zeros(1))
        self.longi = nn.Linear(d, p)
        self.ncause = ncause

    def forward(self, x, m, ver, t1, gap, pad):
        z = torch.cat([x * m, m, gap[..., None]], -1)
        h, _ = self.gru(torch.relu(self.inp(z)))
        self._xm = torch.cat([x * m, m], -1)
        return h

    def out_at(self, h, b, pos):
        """attention over h[b, :pos+1] with query h[b,pos]; returns logits [N, 2K+1]."""
        H = h[b]                                   # [N, L, d]
        q = h[b, pos]                              # [N, d]
        L = H.shape[1]
        valid = torch.arange(L, device=h.device)[None, :] <= pos[:, None]
        sc = self.att(torch.cat([H, q[:, None].expand(-1, L, -1)], -1)).squeeze(-1)
        sc = sc.masked_fill(~valid, -1e9)
        a = torch.softmax(sc, 1)
        ctx = (a[..., None] * H).sum(1)
        inp = torch.cat([ctx, self._xm[b, pos]], -1)
        logits = torch.cat([f(inp) for f in self.cs] + [self.surv.expand(len(b))[:, None]], 1)
        return logits


def cr_loglik(logits, kappa, status, ncls):
    """Discrete-time softmax hazards. logits [N,K,ncls]; kappa in 1..K+1 (K+1 = event-free through K)."""
    lp = torch.log_softmax(logits, -1)
    cls0 = torch.cumsum(lp[..., 0], 1)                       # log S(k), k=1..K
    zero = torch.zeros_like(cls0[:, :1])
    logS_prev = torch.cat([zero, cls0], 1)                   # log S(k-1), k=1..K+1
    kk = (kappa - 1).clamp(max=K)
    base = logS_prev.gather(1, kk[:, None]).squeeze(1)
    ev = status > 0
    ki = (kappa - 1).clamp(max=K - 1)
    sel = lp.gather(1, ki[:, None, None].expand(-1, 1, ncls)).squeeze(1)    # [N, ncls]
    lev = sel.gather(1, status.clamp(max=ncls - 1)[:, None]).squeeze(1)
    return base + torch.where(ev, lev, torch.zeros_like(lev))


def cr_cif(logits, ncls):
    p = torch.softmax(logits, -1)
    S = torch.cumprod(p[..., 0], 1)
    Sprev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], 1)
    return [torch.cumsum(Sprev * p[..., e], 1) for e in range(1, ncls)]    # each [N, K]


def ddh_loss(logits, kappa, status, tval, alpha=1.0, sigma=0.1):
    N = logits.shape[0]
    p = torch.softmax(logits, 1)
    pe = [p[:, e * K:(e + 1) * K] for e in range(2)]
    Fe = [torch.cumsum(q, 1) for q in pe]
    ki = (kappa - 1).clamp(max=K - 1)
    ev = status > 0
    pev = torch.where(status == 1, pe[0].gather(1, ki[:, None]).squeeze(1), pe[1].gather(1, ki[:, None]).squeeze(1))
    Fall = Fe[0] + Fe[1]
    Fall_prev = torch.cat([torch.zeros_like(Fall[:, :1]), Fall], 1)
    cens = 1 - Fall_prev.gather(1, (kappa - 1).clamp(max=K)[:, None]).squeeze(1)
    l1 = torch.where(ev, torch.log(pev + 1e-8), torch.log(cens.clamp(min=1e-8)))
    # ranking loss (per cause): i has event e at kappa_i, j has T_j > T_i
    l2 = torch.zeros((), device=logits.device)
    for e in range(2):
        ii = torch.where(status == e + 1)[0]
        if len(ii) == 0:
            continue
        Fi = Fe[e][ii].gather(1, ki[ii][:, None])                     # [ni,1]
        Fj = Fe[e][:, ki[ii]].T                                         # [ni, N]
        acc = (tval[None, :] > tval[ii][:, None]).float()
        l2 = l2 + (acc * torch.exp(-(Fi - Fj) / sigma)).sum() / max(1.0, float(acc.sum()))
    return l1, alpha * l2


class SeqModel:
    """Shared trainer for sequence models. kind in {'crseq','rnn','ddh'}."""
    def __init__(self, kind='crseq', name=None, landmark_only=False, death_censored=False, block_mask=True,
                 weighted=None, max_epochs=60, patience=6, lr=1e-3, batch=256, seed=0, beta_long=0.5):
        self.kind = kind
        self.name = name or kind
        self.lmo, self.dc, self.bm = landmark_only, death_censored, block_mask
        self.weighted = (kind == 'crseq') if weighted is None else weighted
        self.max_epochs, self.patience, self.lr, self.batch, self.seed = max_epochs, patience, lr, batch, seed
        self.beta_long = beta_long

    # ---------------- data helpers
    def _labels(self, lm, fd):
        t, s = trunc(lm.time.values, lm.status.values)
        if self.dc:
            s = np.where(s == 2, 0, s)
        kappa = np.where(lm.time.values > TMAX, K + 1, np.clip(np.ceil(t / WIDTH - 1e-9), 1, K)).astype(int)
        kappa = np.where((s == 0) & (lm.time.values > TMAX), K + 1, kappa)
        lab = dict(kappa=kappa, status=s.astype(int), t=lm.time.values.astype(np.float32))
        for i, h in enumerate(fd.H):
            lab[f'y{i}'] = horizon_labels(lm.time.values, lm.status.values, h)
        n_lm = lm.groupby('NACCID')['NACCID'].transform('size').values
        lab['w'] = (1.0 / n_lm) if self.weighted else np.ones(len(lm))
        return lab

    def _batch(self, fd, pidx, vidx, cut, augment):
        """pidx: participant ids (index into fd.pstart); vidx: landmark visit rows (landmark-only ablation)."""
        if self.lmo:
            idx = np.asarray(vidx)[:, None]
            pad = np.zeros_like(idx, bool)
        else:
            starts = fd.pstart[pidx]
            lens = fd.plen_cut[pidx] if cut else fd.plen[pidx]
            L = int(lens.max())
            ar = np.arange(L)[None, :]
            pad = ar >= lens[:, None]
            idx = np.where(pad, starts[:, None], starts[:, None] + ar)
        it = torch.tensor(idx, device=DEV)
        X, M = fd.Xg[it], fd.Mg[it]
        ver, t1, gap = fd.verg[it], fd.t1g[it], fd.gapg[it]
        if self.lmo:
            t1, gap = torch.zeros_like(t1), torch.zeros_like(gap)
        if augment and self.bm and self.kind == 'crseq':
            M = self._block_mask(M)
        return X * M, M, ver, t1, gap, torch.tensor(pad, device=DEV)

    def _block_mask(self, M):
        """Form-block masking: drop whole version-specific test blocks (and other form blocks) per history."""
        B, L = M.shape[0], M.shape[1]
        M = M.clone()
        for bcols, prob in self.fdref.block_drop:
            drop = torch.rand(B, device=DEV) < prob
            if drop.any():
                sub = M[:, :, bcols]
                sub[drop] = 0
                M[:, :, bcols] = sub
        npc = self.fdref.np_cols            # whole battery missing at single visits (telephone-like)
        vd = torch.rand(B, L, device=DEV) < 0.1
        sub = M[:, :, npc]
        sub[vd] = 0
        M[:, :, npc] = sub
        return M

    def _net(self, p):
        if self.kind == 'crseq':
            return CRSeqNet(p, ncls=2 if self.dc else 3).to(DEV)
        if self.kind == 'rnn':
            return GRUNet(p, nout=len(self.H)).to(DEV)
        return DDHNet(p).to(DEV)

    def _forward_loss(self, net, fd, plist, lm_by_p, lab, train):
        rows = np.concatenate([lm_by_p[q] for q in plist])
        bmap = np.repeat(np.arange(len(plist)), [len(lm_by_p[q]) for q in plist])
        pos = fd.lm_pos_tr[rows]
        X, M, ver, t1, gap, pad = self._batch(fd, plist, fd.tr_vidx[rows], True, train)
        if self.lmo:
            bmap = np.arange(len(rows)); pos = np.zeros(len(rows), int)
        h = net(X, M, ver, t1, gap, pad)
        b = torch.tensor(bmap, device=DEV); ps = torch.tensor(pos, device=DEV)
        w = torch.tensor(lab['w'][rows], device=DEV, dtype=torch.float32)
        kap = torch.tensor(lab['kappa'][rows], device=DEV)
        st = torch.tensor(lab['status'][rows], device=DEV)
        if self.kind == 'crseq':
            logits = net.out(h[b, ps])
            ll = cr_loglik(logits, kap, st, net.ncls)
            return -(w * ll).sum() / w.sum()
        if self.kind == 'rnn':
            out = net.head(h[b, ps])
            loss = out.sum() * 0.0
            for i in range(len(self.H)):
                y = torch.tensor(lab[f'y{i}'][rows], device=DEV, dtype=torch.float32)
                k = ~torch.isnan(y)
                if k.any():
                    loss = loss + Fnn.binary_cross_entropy_with_logits(out[k, i], y[k])
            return loss
        logits = net.out_at(h, b, ps)
        l1, l2 = ddh_loss(logits, kap, st, torch.tensor(lab['t'][rows], device=DEV))
        loss = -(l1.mean()) + l2
        if self.beta_long > 0 and not self.lmo:
            # longitudinal loss: predict next visit's standardised values (where observed)
            nxt = ps + 1
            okn = nxt < (~pad).sum(1)[b]
            if okn.any():
                bb, nn_ = b[okn], nxt[okn]
                pred = net.longi(h[bb, ps[okn]])
                tgt, mk = X[bb, nn_], M[bb, nn_]
                loss = loss + self.beta_long * ((pred - tgt) ** 2 * mk).sum() / mk.sum().clamp(min=1)
        return loss

    def fit(self, fd):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.H = list(fd.H)
        self.fdref = fd
        fd.prepare_gpu()
        lab = self._labels(fd.tr, fd)
        va_rows = inner_split(fd)
        pid = fd.tr_pid
        lm_by_p = {}
        for r, q in enumerate(pid):
            lm_by_p.setdefault(q, []).append(r)
        lm_by_p = {q: np.array(v) for q, v in lm_by_p.items()}
        pv = np.unique(pid[va_rows]); pt = np.unique(pid[~va_rows])
        net = self._net(fd.p)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, bad, state = np.inf, 0, None
        rng = np.random.default_rng(self.seed)
        self.epochs = 0
        for ep in range(self.max_epochs):
            net.train()
            perm = rng.permutation(pt)
            for i in range(0, len(perm), self.batch):
                loss = self._forward_loss(net, fd, perm[i:i + self.batch], lm_by_p, lab, True)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                vl, nb = 0.0, 0
                for i in range(0, len(pv), 1024):
                    vl += float(self._forward_loss(net, fd, pv[i:i + 1024], lm_by_p, lab, False)); nb += 1
                vl /= max(nb, 1)
            self.epochs = ep + 1
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(state)
        self.net = net.eval()
        self.best_val = best
        return self

    @torch.no_grad()
    def predict(self, fd):
        te = fd.te
        pid = fd.te_pid
        Hb = [int(round(h / WIDTH)) for h in fd.H]
        F1 = np.zeros((len(te), len(fd.H))); F2 = np.zeros((len(te), len(fd.H)))
        up = np.unique(pid)
        rows_by = {}
        for r, q in enumerate(pid):
            rows_by.setdefault(q, []).append(r)
        for i in range(0, len(up), 512):
            plist = up[i:i + 512]
            rows = np.concatenate([rows_by[q] for q in plist])
            bmap = np.repeat(np.arange(len(plist)), [len(rows_by[q]) for q in plist])
            pos = fd.lm_pos_te[rows]
            X, M, ver, t1, gap, pad = self._batch(fd, plist, fd.te_vidx[rows], False, False)
            if self.lmo:
                bmap = np.arange(len(rows)); pos = np.zeros(len(rows), int)
            h = self.net(X, M, ver, t1, gap, pad)
            b = torch.tensor(bmap, device=DEV); ps = torch.tensor(pos, device=DEV)
            if self.kind == 'crseq':
                cifs = cr_cif(self.net.out(h[b, ps]), self.net.ncls)
                c1 = cifs[0].cpu().numpy()
                F1[rows] = c1[:, [k - 1 for k in Hb]]
                if len(cifs) > 1:
                    F2[rows] = cifs[1].cpu().numpy()[:, [k - 1 for k in Hb]]
            elif self.kind == 'rnn':
                F1[rows] = torch.sigmoid(self.net.head(h[b, ps])).cpu().numpy()
            else:
                p = torch.softmax(self.net.out_at(h, b, ps), 1)
                c1 = torch.cumsum(p[:, :K], 1).cpu().numpy(); c2 = torch.cumsum(p[:, K:2 * K], 1).cpu().numpy()
                F1[rows] = c1[:, [k - 1 for k in Hb]]; F2[rows] = c2[:, [k - 1 for k in Hb]]
        has2 = (self.kind == 'ddh') or (self.kind == 'crseq' and not self.dc)
        return dict(F1=F1, F2=F2 if has2 else None)


def make_model(name, seed=0):
    """seed: training seed for stochastic models (network init/order, GBT early-stopping split, RSF)."""
    def gbt(cls):
        g = cls(); g.seed = seed; return g
    def rsf():
        r = RSFCensorDeath(); r.kw['random_state'] = seed; return r
    return {
        'logistic': lambda: LogisticHorizon(),
        'gbt': lambda: gbt(GBTHorizon),
        'cox': lambda: CoxCensorDeath(),
        'rsf': rsf,
        'finegray': lambda: FineGray(),
        'rnn': lambda: SeqModel('rnn', 'rnn', seed=seed),
        'ddh': lambda: SeqModel('ddh', 'ddh', seed=seed),
        'crseq': lambda: SeqModel('crseq', 'crseq', seed=seed),
        'crseq_lmonly': lambda: SeqModel('crseq', 'crseq_lmonly', landmark_only=True, seed=seed),
        'crseq_deathcens': lambda: SeqModel('crseq', 'crseq_deathcens', death_censored=True, seed=seed),
        'crseq_nomask': lambda: SeqModel('crseq', 'crseq_nomask', block_mask=False, seed=seed),
        'crseq_unweighted': lambda: SeqModel('crseq', 'crseq_unweighted', weighted=False, seed=seed),
        'cox_w': lambda: CoxW(),
        'finegray_w': lambda: FineGrayW(),
        'gbt_w': lambda: gbt(GBTHorizonW),
    }[name]()


ALL_MODELS = ['logistic', 'gbt', 'cox', 'rsf', 'finegray', 'rnn', 'ddh', 'crseq', 'crseq_lmonly',
              'crseq_deathcens', 'crseq_nomask']
