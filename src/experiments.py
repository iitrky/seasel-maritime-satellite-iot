"""Full experimental campaign of the paper (Sections 8-10).

    python experiments.py            # ~40 min on 4 cores, writes ../results/*.json and ../results/cache/
"""
import json, os, pickle, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from testbed import simulate                                    # noqa: E402
from features import build, Scaler                              # noqa: E402
from model import train_one, Ensemble, forward                  # noqa: E402
import risk as R                                                # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier     # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.metrics import roc_auc_score                       # noqa: E402

OUT = os.path.join(HERE, "..", "results")
CACHE = os.path.join(OUT, "cache")
ALPHA, DELTA, NSPLIT = 0.05, 0.10, 100
E = 1440
SCEN = {
    "train_day": (dict(), 0),
    "test_day": (dict(), 1),
    "ood_storm": (dict(storm_prob=0.85, storm_hs=(4.0, 7.0), storm_rain=(40.0, 90.0)), 2),
    "ood_tropical": (dict(lat_band=(-15.0, 15.0), base_rain_cells=160), 3),
    "ood_constellation": (dict(sat_fail=0.30, sat_loss_db=2.0), 4),
}


# ----------------------------------------------------------------------------- data
def _sim(item):
    name, (cfg, seed) = item
    p = os.path.join(CACHE, name + ".pkl")
    if not os.path.exists(p):
        pickle.dump(simulate(dict(cfg, n_vessels=40, hours=24), seed=seed), open(p, "wb"))
    return name


def load(name, **fk):
    return build(pickle.load(open(os.path.join(CACHE, name + ".pkl"), "rb")), seed=SCEN[name][1], **fk)


def stack(sc, F, vessels):
    D = sc.transform(F, np.asarray(vessels))
    D["Y2"] = np.stack([D["y"], D["y_rob"]], -1)
    return D


def concat(*Ds):
    return {k: np.concatenate([d[k] for d in Ds]) for k in Ds[0]}


# ----------------------------------------------------------------------------- training helpers
_G = {}


def _train(args):
    idx, seed, mode, epochs = args
    return train_one(_G["D"], idx, seed, mode=mode, epochs=epochs)


def train_ens(D, idx_tr, idx_val, M=5, mode="pba", epochs=12, seed=0, tag=""):
    """Members are trained in parallel; the data reach the workers by fork (no pickling).
    Trained members are cached in results/cache so that a rerun does not retrain."""
    path = os.path.join(CACHE, f"ens_{mode}{tag}_M{M}_e{epochs}_s{seed}.pkl")
    if os.path.exists(path):
        members = pickle.load(open(path, "rb"))
    else:
        _G["D"] = D
        with Pool(min(M, 4)) as p:
            members = p.map(_train, [(idx_tr, seed * 100 + m, mode, epochs) for m in range(M)])
        pickle.dump(members, open(path, "wb"))
    ens = Ensemble(members, mode)
    ens.fit_temperature(D, idx_val)
    return ens


def flat_features(D):
    """Per-token design matrix for the tabular baselines (token features + context)."""
    N, K, ft = D["tok"].shape
    X = np.concatenate([D["tok"], np.repeat(D["ctx"][:, None, :], K, 1)], -1).reshape(N * K, -1)
    return X, D["mask"].reshape(-1) > 0


def tabular(model, Dtr, Dall):
    Xtr, mtr = flat_features(Dtr)
    model.fit(Xtr[mtr], Dtr["y"].reshape(-1)[mtr])
    rob = type(model)(**model.get_params()).fit(Xtr[mtr], Dtr["y_rob"].reshape(-1)[mtr])
    X, _ = flat_features(Dall)
    N, K = Dall["y"].shape
    return model.predict_proba(X)[:, 1].reshape(N, K), rob.predict_proba(X)[:, 1].reshape(N, K)


# ----------------------------------------------------------------------------- evaluation helpers
def splits(n_pool, rng):
    p = rng.permutation(n_pool)
    return p[: n_pool // 2], p[n_pool // 2:]


def certified_eval(P, S, PR, D, pool_idx, alpha=ALPHA, delta=DELTA, nsplit=NSPLIT, gammas=R.GAMMAS):
    rng = np.random.default_rng(123)
    rows = []
    for _ in range(nsplit):
        a, b = splits(len(pool_idx), rng)
        cal, te = pool_idx[a], pool_idx[b]
        g, lam = R.calibrate(P, S, D, cal, alpha, delta, gammas)
        r = R.evaluate(P, S, D, te, g, lam, PR=PR)
        rows.append([r["coverage"], r["sel_risk"], r["miss_rate"], r["goodput"], r["energy_per_mbit"], g, lam])
    A = np.array(rows, float)
    return summarize(A, alpha)


def tuned_eval(P, S, PR, D, val_idx, pool_idx, alpha=ALPHA, nsplit=NSPLIT):
    """Uncertified softmax-response baseline: threshold = smallest lambda whose validation risk <= alpha."""
    l, s = R.decide(P[val_idx], S[val_idx] * 0, D["rate"][val_idx], D["mask"][val_idx], 0.0)
    err = R.outcomes({k: D[k][val_idx] for k in ("mask", "y", "y_rob", "rate")}, l)[0]
    lam = 1.01
    for t in sorted(R.LAMBDAS):
        acc = s >= t
        if acc.any() and err[acc].mean() <= alpha:
            lam = t
            break
    rng = np.random.default_rng(123)
    rows = []
    for _ in range(nsplit):
        _, b = splits(len(pool_idx), rng)
        r = R.evaluate(P, S * 0, D, pool_idx[b], 0.0, lam, PR=PR)
        rows.append([r["coverage"], r["sel_risk"], r["miss_rate"], r["goodput"], r["energy_per_mbit"], 0.0, lam])
    return summarize(np.array(rows, float), alpha)


def fixed_eval(P, S, PR, D, pool_idx, lam, nsplit=NSPLIT):
    rng = np.random.default_rng(123)
    rows = []
    for _ in range(nsplit):
        _, b = splits(len(pool_idx), rng)
        r = R.evaluate(P, S, D, pool_idx[b], 0.0, lam, PR=PR)
        rows.append([r["coverage"], r["sel_risk"], r["miss_rate"], r["goodput"], r["energy_per_mbit"], 0.0, lam])
    return summarize(np.array(rows, float), ALPHA)


def summarize(A, alpha):
    keys = ["coverage", "sel_risk", "miss_rate", "goodput", "energy_per_mbit", "gamma", "lam"]
    out = {k: dict(mean=float(np.nanmean(A[:, i])), std=float(np.nanstd(A[:, i]))) for i, k in enumerate(keys)}
    out["violation"] = float(np.mean(A[:, 1] > alpha))
    return out


def scores(P, D, idx):
    m = D["mask"][idx] > 0
    l, s = R.decide(P[idx], P[idx] * 0, D["rate"][idx], D["mask"][idx], 0.0)
    err = R.outcomes({k: D[k][idx] for k in ("mask", "y", "y_rob", "rate")}, l)[0]
    cov, risk, aurc = R.risk_coverage(s, err)
    return dict(auroc=float(roc_auc_score(D["y"][idx][m], P[idx][m])), ece=R.ece(P, D, idx), aurc=aurc,
                rc_cov=cov[::50].tolist(), rc_risk=risk[::50].tolist())


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    with Pool(4) as pool:
        pool.map(_sim, list(SCEN.items()))
    print("simulations ready %.0fs" % (time.time() - t0), flush=True)
    if True:
        Ftr = load("train_day")
        Fte = load("test_day")
        tr, va, ho = np.arange(0, 24), np.arange(24, 28), np.arange(28, 40)
        sc = Scaler().fit(Ftr, tr)
        Dtr, Dva, Dho, Dte = stack(sc, Ftr, tr), stack(sc, Ftr, va), stack(sc, Ftr, ho), stack(sc, Fte, np.arange(40))
        D = concat(Dtr, Dva, Dho, Dte)
        ntr, nva = len(Dtr["y"]), len(Dva["y"])
        itr, iva = np.arange(ntr), np.arange(ntr, ntr + nva)
        ipool = np.arange(ntr + nva, len(D["y"]))                      # held-out vessels + independent day
        res = dict(meta=dict(n_train=int(ntr), n_val=int(nva), n_pool=int(len(ipool)), alpha=ALPHA, delta=DELTA,
                             nsplit=NSPLIT))
        # stats of the testbed
        m = D["mask"][ipool] > 0
        res["testbed"] = dict(p_aggr=float(D["y"][ipool][m].mean()), p_rob=float(D["y_rob"][ipool][m].mean()),
                              mean_candidates=float(m.sum(1).mean()),
                              ter_share=float((D["is_ter"][ipool] * D["mask"][ipool]).sum(1).clip(0, 1).mean()),
                              no_link=float(1 - (D["y_rob"][ipool] * D["mask"][ipool]).max(1).mean()))

        # ---------------- proposed and learned variants
        t1 = time.time()
        ens = train_ens(D, itr, iva, M=5, mode="pba")
        print("SeaSel ensemble %.0fs T=%.2f" % (time.time() - t1, ens.T), flush=True)
        P, S, PR = ens.predict(D)
        t_inf = time.time(); ens.predict(D, ipool[:2000]); t_inf = (time.time() - t_inf) / 2000 * 1e3
        res["complexity"] = dict(params=int(sum(v.size for v in ens.members[0].values())), members=5,
                                 ms_per_decision=float(t_inf), beta=[float(np.log1p(np.exp(p["beta"][0]))) for p in ens.members])
        single = Ensemble(ens.members[:1], "pba"); single.fit_temperature(D, iva)
        P1, S1, PR1 = single.predict(D)

        # ---------------- baselines
        Dtab = {k: D[k][:ntr] for k in ("tok", "ctx", "mask", "y", "y_rob")}
        Pg, PRg = tabular(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0), Dtab, D)
        Pl, PRl = tabular(LogisticRegression(max_iter=2000, C=1.0), Dtab, D)
        # physics heuristics: elevation and link-budget margin, mapped to [0, 1] scores
        el = np.where(D["mask"] > 0, D["el"], -90)
        Pel = np.where(D["is_ter"] > 0, 0.5, el / 90.0)
        raw = lambda c: D["tok"][..., c] * sc.ts[c] + sc.tm[c]
        margin = 10 * raw(11) - np.where(raw(8) > 0.5, 10 * raw(7), 0.0) - 10.0      # dB above the MCS threshold
        margin = np.where(D["is_ter"] > 0, 0.0, margin)
        Pbud = 1 / (1 + np.exp(-margin / 3.0))
        Z = np.zeros_like(P)

        methods = {}
        methods["Max-elevation (always aggressive)"] = fixed_eval(Pel, Z, Pel, D, ipool, -1.0)
        methods["Always robust (dual diversity)"] = fixed_eval(P, Z, PR, D, ipool, 9.0)
        methods["Link-budget margin, tuned"] = tuned_eval(Pbud, Z, Pbud, D, iva, ipool)
        methods["Logistic regression, tuned"] = tuned_eval(Pl, Z, PRl, D, iva, ipool)
        methods["Gradient boosting, tuned"] = tuned_eval(Pg, Z, PRg, D, iva, ipool)
        methods["Gradient boosting + LTT"] = certified_eval(Pg, Z, PRg, D, ipool, gammas=(0.0,))
        methods["PBA single model + LTT"] = certified_eval(P1, S1, PR1, D, ipool, gammas=(0.0,))
        methods["SeaSel (PBA ensemble + LTT)"] = certified_eval(P, S, PR, D, ipool)
        res["methods"] = methods
        res["discrimination"] = {k: scores(Pk, D, ipool) for k, Pk in
                                 [("Link-budget margin", Pbud), ("Logistic regression", Pl), ("Gradient boosting", Pg),
                                  ("PBA single model", P1), ("SeaSel ensemble", P)]}
        # reliability diagram data
        mm = D["mask"][ipool] > 0
        rel = {}
        for k, Pk in [("Gradient boosting", Pg), ("PBA single model", P1), ("SeaSel ensemble", P)]:
            p_, y_ = Pk[ipool][mm], D["y"][ipool][mm]
            b = np.clip((p_ * 10).astype(int), 0, 9)
            rel[k] = dict(conf=[float(p_[b == i].mean()) if (b == i).any() else None for i in range(10)],
                          acc=[float(y_[b == i].mean()) if (b == i).any() else None for i in range(10)],
                          n=[int((b == i).sum()) for i in range(10)])
        res["reliability"] = rel
        print("main comparison %.0fs" % (time.time() - t0), flush=True)

        # ---------------- alpha sweep (guarantee validity)
        sweep = {}
        for a in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
            sweep[a] = dict(seasel=certified_eval(P, S, PR, D, ipool, alpha=a, nsplit=60),
                            gbdt=certified_eval(Pg, Z, PRg, D, ipool, alpha=a, nsplit=60, gammas=(0.0,)),
                            tuned=tuned_eval(Pg, Z, PRg, D, iva, ipool, alpha=a, nsplit=60))
        res["alpha_sweep"] = sweep
        # operating curve: goodput vs miss for lambda sweep (gamma = 0)
        curve = {}
        for k, (Pk, Sk, PRk) in {"SeaSel": (P, S, PR), "Gradient boosting": (Pg, Z, PRg), "Link-budget": (Pbud, Z, Pbud)}.items():
            pts = []
            for lam in np.linspace(0.0, 1.0, 41):
                r = R.evaluate(Pk, Sk, D, ipool, 0.0, lam, PR=PRk)
                pts.append([r["miss_rate"], r["goodput"], r["coverage"]])
            curve[k] = pts
        res["pareto"] = curve
        print("sweeps %.0fs" % (time.time() - t0), flush=True)

        # ---------------- ablations (retrained)
        abl = {"Full SeaSel": methods["SeaSel (PBA ensemble + LTT)"]}
        abl_disc = {"Full SeaSel": res["discrimination"]["SeaSel ensemble"]}
        for name, mode in [("Without physics bias (plain attention)", "attn"), ("Mean pooling instead of attention", "pool"),
                           ("Without attention (per-link MLP)", "none")]:
            e2 = train_ens(D, itr, iva, M=5, mode=mode)
            Pa, Sa, PRa = e2.predict(D)
            abl[name] = certified_eval(Pa, Sa, PRa, D, ipool)
            abl_disc[name] = scores(Pa, D, ipool)
            print("ablation", name, "%.0fs" % (time.time() - t0), flush=True)
        for name, cols_tok, cols_ctx in [("Without motion and weather context", [], [0, 1, 2, 3, 4]),
                                         ("Without link-history features", [6, 7, 8], [])]:
            D2 = {k: v.copy() if k in ("tok", "ctx") else v for k, v in D.items()}
            for c in cols_tok:
                D2["tok"][..., c] = 0.0
            for c in cols_ctx:
                D2["ctx"][..., c] = 0.0
            e2 = train_ens(D2, itr, iva, M=5, mode="pba", tag="_" + name.split()[1])
            Pa, Sa, PRa = e2.predict(D2)
            abl[name] = certified_eval(Pa, Sa, PRa, D2, ipool)
            abl_disc[name] = scores(Pa, D2, ipool)
            print("ablation", name, "%.0fs" % (time.time() - t0), flush=True)
        abl["Without uncertainty penalty (gamma = 0)"] = certified_eval(P, S, PR, D, ipool, gammas=(0.0,))
        ens_noT = Ensemble(ens.members, "pba")
        Pn, Sn, PRn = ens_noT.predict(D)
        abl["Without temperature scaling"] = certified_eval(Pn, Sn, PRn, D, ipool)
        abl_disc["Without temperature scaling"] = scores(Pn, D, ipool)
        abl["Single member (M = 1)"] = methods["PBA single model + LTT"]
        abl_disc["Single member (M = 1)"] = res["discrimination"]["PBA single model"]
        res["ablation"] = abl
        res["ablation_disc"] = abl_disc

        # ---------------- robustness (sensor corruption at test time; calibration on clean vs corrupted)
        rob = {}
        for name, fk in [("IMU noise 1 deg", dict(imu_noise=1.0)), ("IMU noise 3 deg", dict(imu_noise=3.0)),
                         ("Rain sensor noise x3", dict(rain_noise=0.9)), ("30% context missing", dict(drop_ctx=0.3)),
                         ("50% link history missing", dict(drop_hist=0.5))]:
            Fc = load("test_day", **fk)
            Dc = stack(sc, Fc, np.arange(40))
            Pc, Sc, PRc = ens.predict(Dc)
            # clean calibration (threshold from the clean pool), corrupted test
            g, lam = R.calibrate(P, S, D, ipool, ALPHA, DELTA)
            r_clean = R.evaluate(Pc, Sc, Dc, np.arange(len(Dc["y"])), g, lam, PR=PRc)
            ic = np.arange(len(Dc["y"]))
            r_re = certified_eval(Pc, Sc, PRc, Dc, ic, nsplit=40)
            rob[name] = dict(clean_cal=dict(coverage=r_clean["coverage"], sel_risk=r_clean["sel_risk"],
                                            miss_rate=r_clean["miss_rate"], goodput=r_clean["goodput"]),
                             recal=r_re, auroc=scores(Pc, Dc, ic)["auroc"])
        res["robustness"] = rob
        print("robustness %.0fs" % (time.time() - t0), flush=True)

        # ---------------- OOD: pooled vs sea-state-conditional (Mondrian) vs online adaptive vs oracle recalibration
        groups_edges = np.quantile(D["roll_rms"][ipool], [1 / 3, 2 / 3])
        grp = lambda DD: np.digitize(DD["roll_rms"], groups_edges)
        g0, lam0 = R.calibrate(P, S, D, ipool, ALPHA, DELTA)
        gm = R.calibrate_groups(P, S, D, ipool, grp(D), ALPHA, DELTA)
        ood = {}
        traces = {}
        for name in ("ood_storm", "ood_tropical", "ood_constellation"):
            Fo = load(name)
            Do = stack(sc, Fo, np.arange(40))
            Po, So, PRo = ens.predict(Do)
            io = np.arange(len(Do["y"]))
            r_naive = R.evaluate(Po, So, Do, io, g0, lam0, PR=PRo)
            # Mondrian: per-group gamma/lambda chosen by each sample's sea-state group
            go = grp(Do)
            acc_any = np.zeros(len(io), bool)
            lam_vec = np.zeros(len(io)); gam_vec = np.zeros(len(io))
            for gi, (gg, ll) in gm.items():
                lam_vec[go == gi], gam_vec[go == gi] = ll, gg
            l, s0 = R.decide(Po, So * 0, Do["rate"], Do["mask"], 0.0)
            s_g = s0 - gam_vec * So[np.arange(len(l)), l]
            sub = {k: Do[k] for k in ("mask", "y", "y_rob", "rate")}
            err, rob_ok, rate, nl = R.outcomes(sub, l, PRo)
            acc = s_g >= lam_vec
            r_mond = dict(coverage=float(acc.mean()), sel_risk=float(err[acc].mean()) if acc.any() else 0.0,
                          miss_rate=float(np.where(acc, err, 1 - rob_ok).mean()),
                          goodput=float(np.where(acc, (1 - err) * rate * R.T_TX, rob_ok * R.R_ROB * R.T_TX).mean()))
            # online adaptation, vessels processed as time-ordered streams (interleaved by epoch)
            order = np.argsort(np.tile(np.arange(E), 40), kind="stable")
            l0, s_on = R.decide(Po[order], So[order], Do["rate"][order], Do["mask"][order], g0)
            e_on = R.outcomes({k: Do[k][order] for k in ("mask", "y", "y_rob", "rate")}, l0, PRo[order])
            acc_on, lam_hist = R.online_adapt(s_on, e_on[0], lam0, ALPHA, eta=0.02)
            r_onl = dict(coverage=float(acc_on.mean()), sel_risk=float(e_on[0][acc_on].mean()),
                         miss_rate=float(np.where(acc_on, e_on[0], 1 - e_on[1]).mean()),
                         goodput=float(np.where(acc_on, (1 - e_on[0]) * e_on[2] * R.T_TX, e_on[1] * R.R_ROB * R.T_TX).mean()))
            r_or = certified_eval(Po, So, PRo, Do, io, nsplit=40)
            r_all = R.evaluate(Po, So, Do, io, 0.0, -1.0, PR=PRo)
            ood[name] = dict(naive={k: r_naive[k] for k in ("coverage", "sel_risk", "miss_rate", "goodput")},
                             mondrian=r_mond, online=r_onl, oracle_recal=r_or,
                             accept_all={k: r_all[k] for k in ("coverage", "sel_risk", "miss_rate", "goodput")},
                             auroc=scores(Po, Do, io)["auroc"])
            # running selective risk for the figure (window of 1000 accepted decisions)
            w = 2000
            run_naive = np.convolve(((s_on >= lam0) * e_on[0]).astype(float), np.ones(w), "valid") / \
                np.maximum(np.convolve((s_on >= lam0).astype(float), np.ones(w), "valid"), 1)
            run_onl = np.convolve((acc_on * e_on[0]).astype(float), np.ones(w), "valid") / \
                np.maximum(np.convolve(acc_on.astype(float), np.ones(w), "valid"), 1)
            traces[name] = dict(naive=run_naive[::100].tolist(), online=run_onl[::100].tolist(), lam=lam_hist[::100].tolist())
        res["ood"] = ood
        # ---------------- cross-link transfer: models trained and tested with 50% of link histories missing
        print("cross-link transfer", flush=True)
        rng_h = np.random.default_rng(77)
        Dh = {k: v.copy() if k == "tok" else v for k, v in D.items()}
        drop = (rng_h.random(Dh["tok"].shape[:2]) < 0.5) & (D["is_ter"] < 0.5)
        Dh["tok"][..., 6][drop] = 0.0
        Dh["tok"][..., 7][drop] = 0.0
        Dh["tok"][..., 8][drop] = (0.0 - sc.tm[8]) / sc.ts[8]
        transfer = {}
        for name, mode in [("Physics-biased attention", "pba"), ("Plain attention", "attn"), ("Per-link MLP", "none")]:
            e3 = train_ens(Dh, itr, iva, M=5, mode=mode, tag="_hd")
            Ph, Sh, PRh = e3.predict(Dh)
            sc_h = scores(Ph, Dh, ipool)
            # where history is missing, which tokens benefit: newly visible links
            mm = (Dh["mask"][ipool] > 0) & drop[ipool]
            auroc_missing = float(roc_auc_score(Dh["y"][ipool][mm], Ph[ipool][mm]))
            transfer[name] = dict(auroc=sc_h["auroc"], auroc_missing=auroc_missing, ece=sc_h["ece"], aurc=sc_h["aurc"],
                                  cert=certified_eval(Ph, Sh, PRh, Dh, ipool))
            print("  ", name, round(sc_h["auroc"], 4), round(auroc_missing, 4), flush=True)
        res["transfer"] = transfer
        res["ood_traces"] = traces
        res["mondrian_groups"] = {str(k): dict(gamma=float(v[0]), lam=float(v[1])) for k, v in gm.items()}
        res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
