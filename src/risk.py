"""Uncertainty-aware decision, risk-certified selective prediction and evaluation (Section 7).

Decision for epoch n with candidate links l:
    l* = argmax_l p_l * R_l                          (expected goodput)
    s  = p_{l*} - gamma * sigma_{l*}                  (uncertainty-penalised confidence)
    accept  if s >= lambda  -> aggressive MCS on l*
    abstain otherwise       -> robust MCS on l*        (safe fallback)

Risk certification (Learn-then-Test): exact binomial p-values for H0: R_sel(gamma, lambda) > alpha on the
accepted calibration points, Bonferroni over the (gamma, lambda) grid, so that with probability >= 1 - delta
every certified pair -- and hence the selected one -- has selective risk <= alpha.
"""
import numpy as np
from scipy.stats import binom

GAMMAS = (0.0, 0.5, 1.0, 2.0)
LAMBDAS = np.round(np.linspace(0.99, 0.0, 100), 4)
P_TX, T_TX, R_ROB = 10.0, 20.0, 1.0                       # W, s, Mbit/s (robust mode)


def decide(P, S, rate, mask, gamma):
    g = np.where(mask > 0, P * rate, -1.0)
    l = np.argmax(g, axis=1)
    p = P[np.arange(len(l)), l]
    s = S[np.arange(len(l)), l]
    return l, p - gamma * s


def outcomes(D, l, PR=None):
    """Aggressive error on l*, and success of the diversity fallback: robust MCS duplicated on the two
    links with the highest predicted robust-mode success."""
    n = np.arange(len(l))
    has = D["mask"][n, l] > 0
    err = np.where(has, 1.0 - D["y"][n, l], 1.0)
    if PR is not None:
        Pm = np.where(D["mask"] > 0, PR, -1.0)
        o = np.argsort(-Pm, axis=1)
        f1, f2 = o[:, 0], o[:, 1] if Pm.shape[1] > 1 else o[:, 0]
        h1, h2 = D["mask"][n, f1] > 0, D["mask"][n, f2] > 0
        rob_ok = np.maximum(np.where(h1, D["y_rob"][n, f1], 0.0), np.where(h2, D["y_rob"][n, f2], 0.0))
        n_links = h1.astype(float) + h2.astype(float)
    else:
        rob_ok = np.where(has, D["y_rob"][n, l], 0.0)
        n_links = np.ones(len(l))
    return err, rob_ok, D["rate"][n, l], n_links


def binom_p(k, n, alpha):
    """Exact p-value for H0: selective risk > alpha, given k errors among n accepted calibration points.
    Conditionally on n, the accepted points are i.i.d. draws from the law given {s >= lambda}, so
    k ~ Binomial(n, R_sel); the p-value is P(Binomial(n, alpha) <= k)."""
    if n == 0:
        return 1.0
    return float(binom.cdf(k, n, alpha))


def ltt(score, err, alpha, delta, n_tests=1):
    """Learn-then-Test over the lambda grid with Bonferroni family-wise control.
    Returns the most lenient lambda whose null is rejected (largest coverage)."""
    level = delta / (n_tests * len(LAMBDAS))
    cert = [lam for lam in LAMBDAS
            if binom_p(int(err[score >= lam].sum()), int((score >= lam).sum()), alpha) <= level]
    return min(cert) if cert else None


def calibrate(P, S, D, idx, alpha, delta, gammas=GAMMAS):
    """Returns (gamma, lambda) maximising calibration coverage among certified pairs."""
    best = (-1, None, None)
    for gm in gammas:
        l, s = decide(P[idx], S[idx], D["rate"][idx], D["mask"][idx], gm)
        err = outcomes({k: D[k][idx] for k in ("mask", "y", "y_rob", "rate")}, l)[0]
        lam = ltt(s, err, alpha, delta, n_tests=len(gammas))
        if lam is None:
            continue
        cov = (s >= lam).mean()
        if cov > best[0]:
            best = (cov, gm, lam)
    if best[1] is None:
        return gammas[0], np.inf                              # nothing certified: abstain always
    return best[1], best[2]


def calibrate_groups(P, S, D, idx, groups, alpha, delta, gammas=GAMMAS):
    """Mondrian (group-conditional) certification: one (gamma, lambda) per sea-state group."""
    out = {}
    for g in np.unique(groups[idx]):
        gi = idx[groups[idx] == g]
        out[g] = calibrate(P, S, D, gi, alpha, delta, gammas)
    return out


def evaluate(P, S, D, idx, gamma, lam, lam_vec=None, PR=None):
    """Accept: aggressive MCS on l* (critical + bulk data).  Abstain: robust MCS duplicated on the two most
    probable links (critical data protected by diversity, bulk at the robust rate, twice the energy)."""
    l, s = decide(P[idx], S[idx], D["rate"][idx], D["mask"][idx], gamma)
    sub = {k: D[k][idx] for k in ("mask", "y", "y_rob", "rate")}
    err, rob_ok, rate, n_links = outcomes(sub, l, (PR if PR is not None else P)[idx])
    thr = lam if lam_vec is None else lam_vec
    acc = s >= thr
    bits = np.where(acc, (1 - err) * rate * T_TX, rob_ok * R_ROB * T_TX)          # Mbit per epoch
    miss = np.where(acc, err, 1 - rob_ok)                                         # critical deadline miss
    energy = np.where(acc, 1.0, n_links) * P_TX * T_TX                            # J per epoch
    cov = acc.mean()
    return dict(coverage=float(cov), sel_risk=float(err[acc].mean()) if acc.any() else 0.0,
                miss_rate=float(miss.mean()), goodput=float(bits.mean()),
                energy_per_mbit=float(energy.mean() / max(bits.mean(), 1e-9)),
                accept=acc, err=err, score=s, miss=miss)


def risk_coverage(score, err):
    o = np.argsort(-score)
    e = err[o]
    cov = np.arange(1, len(e) + 1) / len(e)
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return cov, risk, float(np.trapezoid(risk, cov))           # AURC


def ece(P, D, idx, bins=15):
    m = D["mask"][idx] > 0
    p, y = P[idx][m], D["y"][idx][m]
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        s = (p >= a) & (p < b if b < 1 else p <= b)
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)


def online_adapt(score, err, lam0, alpha, eta=0.01, lo=0.0, hi=1.0):
    """Online threshold tracking under drift: lam_{t+1} = lam_t + eta * 1{acc_t}(e_t - alpha)."""
    lam = lam0
    acc = np.zeros(len(score), bool)
    lam_hist = np.zeros(len(score))
    for t in range(len(score)):
        lam_hist[t] = lam
        acc[t] = score[t] >= lam
        if acc[t]:
            lam = float(np.clip(lam + eta * (err[t] - alpha), lo, hi))
    return acc, lam_hist
