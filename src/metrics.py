"""metrics.py -- competing-risk performance measures at horizon tau (Methods 2.7), batched over bootstrap
weight vectors on the GPU (torch).  Conventions follow van Geloven et al. (BMJ 2022) / riskRegression::Score:

- Censoring weights from the reverse Kaplan-Meier G (estimated in the evaluation sample; per bootstrap rep).
  omega_i = 1/G(T_i-) if T_i <= tau and status in {1,2}; 1/G(tau) if T_i > tau; 0 if censored before tau.
- AUC(tau), cause 1: cases = event of interest by tau; controls = event-free at tau OR competing death by tau
  (Blanche et al. 2013, definition 2), IPCW-weighted, ties count 1/2.
- Truncated C-index for competing risks (Wolbers et al.): case i (event 1 at T_i <= tau) vs j with T_j > T_i
  or a competing event at T_j <= T_i; IPCW weights 1/(G(T_i-)G(T_i)) and 1/(G(T_i-)G(T_j-)).
- Brier(tau) = mean omega_i (Y_i - p_i)^2 ; IPA = 1 - Brier/Brier_null with null = Aalen-Johansen CIF of the
  evaluation sample.  O:E = AJ CIF(tau) / mean predicted risk.
- Calibration slope: IPCW-weighted logistic regression of Y on logit(p) (known-status subjects).
- Smoothed calibration / ICI: IPCW-weighted least squares of Y on a restricted cubic spline of logit(p)
  (4 knots); ICI = mean |calibrated - predicted|.  (IPCW outcomes are a censoring-weighted analogue of the
  pseudo-observation approach.)
"""
import numpy as np
import torch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-6


class Sample:
    """Fixed evaluation sample (t, s, p) on DEV; methods take weights W [B, n] (multinomial bootstrap)."""
    def __init__(self, t, s, p, tau):
        t = np.asarray(t, float); s = np.asarray(s, int); p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        self.n = len(t); self.tau = tau
        u, inv = np.unique(t, return_inverse=True)
        self.u = u; self.nU = len(u)
        self.inv = torch.tensor(inv, device=DEV)
        self.iu_tau = int(np.searchsorted(u, tau, side='right') - 1)
        self.t = torch.tensor(t, device=DEV, dtype=torch.float64)
        self.s = torch.tensor(s, device=DEV)
        self.p = torch.tensor(p, device=DEV, dtype=torch.float64)
        self.lp = torch.log(self.p / (1 - self.p))
        self.Y = ((self.s == 1) & (self.t <= tau)).double()
        self.case = self.Y > 0
        self.comp = (self.s == 2) & (self.t <= tau)
        self.after = self.t > tau
        # ties in p for AUC
        pu, pinv = np.unique(p, return_inverse=True)
        self.pinv = torch.tensor(pinv, device=DEV); self.nP = len(pu)
        # spline basis (restricted cubic spline, 4 knots at 5/35/65/95 % of logit p)
        x = self.lp.cpu().numpy()
        self.knots = np.quantile(x, [0.05, 0.35, 0.65, 0.95])
        self.Xs = torch.tensor(rcs(x, self.knots), device=DEV)

    def _bin(self, W, mask):
        B = W.shape[0]
        return torch.zeros(B, self.nU, device=DEV, dtype=W.dtype).index_add_(1, self.inv, W * mask)

    def km_aj(self, W):
        tot = self._bin(W, torch.ones_like(self.t))
        R = torch.flip(torch.cumsum(torch.flip(tot, [1]), 1), [1]).clamp(min=1e-12)
        d1 = self._bin(W, (self.s == 1).double()); d2 = self._bin(W, (self.s == 2).double())
        c = self._bin(W, (self.s == 0).double())
        h1, h2 = d1 / R, d2 / R
        S = torch.cumprod(1 - h1 - h2, 1)
        Sprev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], 1)
        cif1 = torch.cumsum(Sprev * h1, 1)
        G = torch.cumprod(1 - c / R, 1)
        Gprev = torch.cat([torch.ones_like(G[:, :1]), G[:, :-1]], 1)
        return cif1, G, Gprev

    def weights(self, W):
        cif1, G, Gprev = self.km_aj(W)
        iu = self.iu_tau
        aj = cif1[:, iu] if iu >= 0 else torch.zeros(W.shape[0], device=DEV, dtype=W.dtype)
        Gtau = G[:, iu] if iu >= 0 else torch.ones_like(aj)
        Gm = Gprev[:, self.inv].clamp(min=0.02)          # G(T_i-)
        Gi = G[:, self.inv].clamp(min=0.02)              # G(T_i)
        om = torch.where(self.case | self.comp, 1 / Gm, torch.zeros_like(Gm))
        om = torch.where(self.after, (1 / Gtau.clamp(min=0.02))[:, None].expand_as(om), om)
        return dict(aj=aj, om=om, Gm=Gm, Gi=Gi)

    def auc(self, W, om):
        wc = W * om * self.case
        wk = W * om * (~self.case)
        B = W.shape[0]
        gc = torch.zeros(B, self.nP, device=DEV, dtype=W.dtype).index_add_(1, self.pinv, wc)
        gk = torch.zeros(B, self.nP, device=DEV, dtype=W.dtype).index_add_(1, self.pinv, wk)
        below = torch.cumsum(gk, 1) - gk
        num = (gc * (below + 0.5 * gk)).sum(1)
        return num / (gc.sum(1) * gk.sum(1)).clamp(min=1e-12)

    def brier(self, W, om, aj):
        n = W.sum(1)
        bs = (W * om * (self.Y - self.p) ** 2).sum(1) / n
        bs0 = (W * om * (self.Y - aj[:, None]) ** 2).sum(1) / n
        return bs, 1 - bs / bs0

    def oe(self, W, aj):
        return aj / ((W * self.p).sum(1) / W.sum(1))

    def slope(self, W, om, iters=25):
        w = W * om
        x = self.lp; y = self.Y
        B = W.shape[0]
        beta = torch.zeros(B, 2, device=DEV, dtype=W.dtype)
        X = torch.stack([torch.ones_like(x), x], 1)                     # [n,2]
        for _ in range(iters):
            eta = beta @ X.T
            mu = torch.sigmoid(eta)
            g = ((y - mu) * w) @ X                                       # [B,2]
            Hm = torch.einsum('bn,ni,nj->bij', w * mu * (1 - mu), X, X) + 1e-9 * torch.eye(2, device=DEV, dtype=W.dtype)
            beta = beta + torch.linalg.solve(Hm, g[..., None]).squeeze(-1)
        return beta[:, 1]

    def calib(self, W, om):
        w = W * om
        X = self.Xs
        XtWX = torch.einsum('bn,ni,nj->bij', w, X, X) + 1e-8 * torch.eye(X.shape[1], device=DEV, dtype=W.dtype)
        XtWy = (w * self.Y) @ X
        beta = torch.linalg.solve(XtWX, XtWy[..., None]).squeeze(-1)
        cal = (beta @ X.T).clamp(0, 1)
        ici = (W * (cal - self.p).abs()).sum(1) / W.sum(1)
        return ici, beta

    def cindex(self, W, Gm, Gi, chunk=512):
        """Truncated competing-risk C-index; loops over reps in W (keep B moderate)."""
        out = []
        ci = torch.where(self.case)[0]
        tj, sj, pj = self.t, self.s, self.p
        for b in range(W.shape[0]):
            w = W[b]; num = torch.zeros((), device=DEV, dtype=W.dtype); den = torch.zeros_like(num)
            for k in range(0, len(ci), chunk):
                ii = ci[k:k + chunk]
                ti = tj[ii][:, None]
                A1 = (tj[None, :] > ti)
                A2 = (sj[None, :] == 2) & (tj[None, :] <= ti)
                wij = (w[ii] / Gm[b, ii])[:, None] * w[None, :] * (A1 / Gi[b, ii][:, None] + A2 / Gm[b][None, :])
                conc = (pj[ii][:, None] > pj[None, :]).double() + 0.5 * (pj[ii][:, None] == pj[None, :]).double()
                num = num + (wij * conc).sum(); den = den + wij.sum()
            out.append(num / den.clamp(min=1e-12))
        return torch.stack(out)

    def aj_subset(self, W, mask):
        cif1, _, _ = self.km_aj(W * mask)
        return cif1[:, self.iu_tau] if self.iu_tau >= 0 else torch.zeros(W.shape[0], device=DEV, dtype=W.dtype)

    def topq(self, W, thr):
        return self.aj_subset(W, (self.p >= thr).double())

    def all_metrics(self, W, do_c=True, thr_q=None):
        wd = self.weights(W)
        om, aj = wd['om'], wd['aj']
        m = {}
        m['auc'] = self.auc(W, om)
        m['brier'], m['ipa'] = self.brier(W, om, aj)
        m['oe'] = self.oe(W, aj)
        m['slope'] = self.slope(W, om)
        m['ici'], _ = self.calib(W, om)
        m['obs_cif'] = aj
        m['mean_pred'] = (W * self.p).sum(1) / W.sum(1)
        if thr_q is not None:
            m['topq_obs'] = self.topq(W, thr_q)
        if do_c:
            m['cindex'] = self.cindex(W, wd['Gm'], wd['Gi'])
        return {k: v.detach().cpu().numpy() for k, v in m.items()}


def rcs(x, knots):
    k = knots
    def pos3(z):
        return np.maximum(z, 0) ** 3
    cols = [np.ones_like(x), x]
    den = (k[-1] - k[0]) ** 2
    for j in range(len(k) - 2):
        cols.append((pos3(x - k[j]) - pos3(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + pos3(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / den)
    return np.stack(cols, 1)


def boot_weights(n, B, seed, dtype=torch.float64):
    g = torch.Generator(device='cpu').manual_seed(seed)
    idx = torch.randint(0, n, (B, n), generator=g)
    W = torch.zeros(B, n, dtype=dtype).scatter_add_(1, idx, torch.ones(B, n, dtype=dtype))
    return W.to(DEV)


def evaluate(t, s, p, tau, B=1000, seed=20261019, do_c=True, c_boot=None, chunk=100):
    """Point estimates + percentile bootstrap CIs. Returns dict metric -> (est, lo, hi, se)."""
    S = Sample(t, s, p, tau)
    thr = float(np.quantile(np.asarray(p, float), 0.75))
    W1 = torch.ones(1, S.n, device=DEV, dtype=torch.float64)
    est = S.all_metrics(W1, do_c=do_c, thr_q=thr)
    reps = {k: [] for k in est}
    nb_c = B if c_boot is None else c_boot
    done = 0
    while done < B:
        b = min(chunk, B - done)
        W = boot_weights(S.n, b, seed + done)
        r = S.all_metrics(W, do_c=do_c and done < nb_c, thr_q=thr)
        for k in reps:
            if k in r:
                reps[k].append(r[k])
        done += b
    out = {}
    allreps = {}
    for k in est:
        v = np.concatenate(reps[k]) if reps[k] else np.array([np.nan])
        allreps[k] = v.copy()
        v = v[np.isfinite(v)]
        out[k] = (float(est[k][0]), float(np.percentile(v, 2.5)) if len(v) else np.nan,
                  float(np.percentile(v, 97.5)) if len(v) else np.nan, float(np.std(v)) if len(v) else np.nan,
                  len(v))
    return out, S, allreps


def calibration_tables(S, ngrid=50):
    """Deciles: mean predicted vs AJ observed; smooth curve on a grid of predicted risk."""
    p = S.p.cpu().numpy()
    W1 = torch.ones(1, S.n, device=DEV, dtype=torch.float64)
    q = np.quantile(p, np.linspace(0, 1, 11))
    dec = np.clip(np.searchsorted(q[1:-1], p, side='right'), 0, 9)
    rows = []
    for k in range(10):
        m = torch.tensor((dec == k).astype(float), device=DEV)
        if m.sum() == 0:
            continue
        rows.append(dict(decile=k + 1, n=int(m.sum()), mean_pred=float(p[dec == k].mean()),
                         obs_aj=float(S.aj_subset(W1, m)[0])))
    wd = S.weights(W1)
    _, beta = S.calib(W1, wd['om'])
    g = np.linspace(np.quantile(p, 0.01), np.quantile(p, 0.99), ngrid)
    lg = np.log(g / (1 - g))
    cal = np.clip(rcs(lg, S.knots) @ beta[0].cpu().numpy(), 0, 1)
    curve = [dict(pred=float(a), cal=float(b)) for a, b in zip(g, cal)]
    return rows, curve


def decision_curve(S, thresholds):
    W1 = torch.ones(1, S.n, device=DEV, dtype=torch.float64)
    aj_all = float(S.weights(W1)['aj'][0])
    p = S.p
    rows = []
    for pt in thresholds:
        hi = (p >= pt).double()
        pi = float(hi.mean())
        aj_hi = float(S.aj_subset(W1, hi)[0]) if pi > 0 else 0.0
        odds = pt / (1 - pt)
        nb = aj_hi * pi - (1 - aj_hi) * pi * odds
        rows.append(dict(threshold=pt, nb_model=nb, nb_all=aj_all - (1 - aj_all) * odds, nb_none=0.0,
                         frac_high_risk=pi))
    return rows


# --------------------------------------------------------------- random-effects meta-analysis
TRANSFORM = {'auc': 'logit', 'cindex': 'logit', 'oe': 'log', 'slope': None, 'ipa': None, 'brier': None,
             'ici': 'log', 'obs_cif': None, 'mean_pred': None, 'topq_obs': None}


def _tf(x, kind):
    x = np.asarray(x, float)
    if kind == 'logit':
        x = np.clip(x, 1e-4, 1 - 1e-4); return np.log(x / (1 - x))
    if kind == 'log':
        return np.log(np.clip(x, 1e-4, None))
    return x


def _itf(x, kind):
    if kind == 'logit':
        return 1 / (1 + np.exp(-x))
    if kind == 'log':
        return np.exp(x)
    return x


def reml_pool(y, se, kind=None):
    """REML random-effects pooling; Hartung-Knapp-Sidik-Jonkman CI; prediction interval with t_{k-2}."""
    from scipy import stats, optimize
    y = _tf(y, kind)
    v = np.asarray(se, float) ** 2
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    if k < 3:
        return dict(k=k, est=np.nan, ci_lo=np.nan, ci_hi=np.nan, pi_lo=np.nan, pi_hi=np.nan, tau2=np.nan)

    def negll(t2):
        w = 1 / (v + t2); mu = (w * y).sum() / w.sum()
        return 0.5 * (np.log(v + t2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())
    r = optimize.minimize_scalar(negll, bounds=(0, max(10 * np.var(y), 1e-6)), method='bounded')
    t2 = max(r.x, 0)
    w = 1 / (v + t2); mu = (w * y).sum() / w.sum()
    q = (w * (y - mu) ** 2).sum() / (k - 1)
    se_hk = np.sqrt(max(q, 1e-12) / w.sum())
    tq = stats.t.ppf(0.975, k - 1)
    se_mu = np.sqrt(1 / w.sum())
    tp = stats.t.ppf(0.975, k - 2)
    piw = tp * np.sqrt(t2 + se_mu ** 2)
    return dict(k=k, est=float(_itf(mu, kind)), ci_lo=float(_itf(mu - tq * se_hk, kind)),
                ci_hi=float(_itf(mu + tq * se_hk, kind)), pi_lo=float(_itf(mu - piw, kind)),
                pi_hi=float(_itf(mu + piw, kind)), tau2=float(t2))


def se_transformed(reps, kind):
    v = _tf(reps, kind)
    v = v[np.isfinite(v)]
    return float(np.std(v)) if len(v) > 5 else np.nan
