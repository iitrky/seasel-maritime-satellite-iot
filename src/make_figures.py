"""Figures, LaTeX tables and numeric macros for the manuscript, generated from results/results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "..", "results", "results.json")))
FIG = os.path.join(HERE, "..", "figures")
GEN = os.path.join(HERE, "..", "manuscript", "generated")
os.makedirs(GEN, exist_ok=True)

C = {"SeaSel": "#2a78d6", "GBDT": "#eb6834", "Link": "#1baf7a", "Tuned": "#e34948", "Grey": "#52514e", "Violet": "#6250d6"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.7,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
M = RES["methods"]
mean = lambda d, k: d[k]["mean"] if isinstance(d[k], dict) else d[k]
std = lambda d, k: d[k]["std"] if isinstance(d[k], dict) else 0.0


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ Fig 4: risk-coverage curves
fig, ax = plt.subplots(figsize=(4.6, 3.3), constrained_layout=True)
for k, col, ls in [("Link-budget margin", C["Link"], "--"), ("Logistic regression", C["Grey"], ":"),
                   ("Gradient boosting", C["GBDT"], "-"), ("PBA single model", C["Violet"], "-."),
                   ("SeaSel ensemble", C["SeaSel"], "-")]:
    d = RES["discrimination"][k]
    ax.plot(d["rc_cov"], d["rc_risk"], ls, color=col, label=f"{k} (AURC {d['aurc']:.3f})")
ax.axhline(RES["meta"]["alpha"], color="#0b0b0b", lw=0.9, ls=":")
ax.text(0.02, RES["meta"]["alpha"] + 0.004, r"target $\alpha$", fontsize=7)
ax.set_xlabel("coverage (fraction of epochs accepted)"); ax.set_ylabel("selective risk (aggressive failures)")
ax.set_ylim(0, 0.2); ax.legend(fontsize=6.6, loc="upper left")
save(fig, "fig_risk_coverage")

# ------------------------------------------------------------------ Fig 5: alpha sweep (validity + coverage)
A = sorted(RES["alpha_sweep"], key=float)
al = np.array([float(a) for a in A])
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), constrained_layout=True)
for key, lab, col, mk in [("seasel", "SeaSel (certified)", C["SeaSel"], "o"), ("gbdt", "GBDT + LTT (certified)", C["GBDT"], "s"),
                          ("tuned", "GBDT, tuned threshold", C["Tuned"], "^")]:
    r = [RES["alpha_sweep"][a][key] for a in A]
    ax[0].errorbar(al, [x["sel_risk"]["mean"] for x in r], yerr=[x["sel_risk"]["std"] for x in r], color=col, marker=mk,
                   ms=4, capsize=2, label=lab)
    ax[1].plot(al, [x["coverage"]["mean"] for x in r], color=col, marker=mk, ms=4, label=lab)
ax[0].plot(al, al, color="#0b0b0b", lw=0.9, ls=":", label=r"risk = $\alpha$")
ax[0].set_xlabel(r"target risk $\alpha$"); ax[0].set_ylabel("realised selective risk"); ax[0].legend(fontsize=7)
ax[1].set_xlabel(r"target risk $\alpha$"); ax[1].set_ylabel("coverage")
ax[0].set_title("(a)", loc="left"); ax[1].set_title("(b)", loc="left")
save(fig, "fig_alpha_sweep")

# ------------------------------------------------------------------ Fig 6: reliability diagram
fig, ax = plt.subplots(figsize=(3.6, 3.3), constrained_layout=True)
ax.plot([0, 1], [0, 1], color="#0b0b0b", lw=0.8, ls=":")
for k, col, mk in [("Gradient boosting", C["GBDT"], "s"), ("PBA single model", C["Violet"], "^"), ("SeaSel ensemble", C["SeaSel"], "o")]:
    d = RES["reliability"][k]
    pts = [(c, a) for c, a, n in zip(d["conf"], d["acc"], d["n"]) if c is not None and n > 30]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=4, color=col,
            label=f"{k} (ECE {RES['discrimination'][k]['ece']:.3f})")
ax.set_xlabel("predicted success probability"); ax.set_ylabel("observed success frequency"); ax.legend(fontsize=6.6)
save(fig, "fig_reliability")

# ------------------------------------------------------------------ Fig 7: operating curves goodput vs miss
fig, ax = plt.subplots(figsize=(4.6, 3.3), constrained_layout=True)
for k, col in [("Link-budget", C["Link"]), ("Gradient boosting", C["GBDT"]), ("SeaSel", C["SeaSel"])]:
    pts = np.array(RES["pareto"][k])
    o = np.argsort(pts[:, 0])
    ax.plot(100 * pts[o, 0], pts[o, 1], color=col, label=k)
s = M["SeaSel (PBA ensemble + LTT)"]
ax.scatter([100 * s["miss_rate"]["mean"]], [s["goodput"]["mean"]], s=60, color=C["SeaSel"], edgecolor="white", zorder=4,
           label=r"SeaSel certified point ($\alpha$ = 0.05)")
for k, mk in [("Max-elevation (always aggressive)", "X"), ("Always robust (dual diversity)", "D")]:
    ax.scatter([100 * M[k]["miss_rate"]["mean"]], [M[k]["goodput"]["mean"]], s=50, marker=mk, color=C["Grey"], zorder=4, label=k)
ax.set_xlabel("critical-message miss rate (%)"); ax.set_ylabel("goodput (Mbit per epoch)")
ax.legend(fontsize=6.4, loc="lower right")
save(fig, "fig_pareto")

# ------------------------------------------------------------------ Fig 8: OOD bars + online trace
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True)
names = [("ood_storm", "Severe storms"), ("ood_tropical", "Tropical rain"), ("ood_constellation", "30% satellite loss")]
w = 0.2
for i, (key, lab, col) in enumerate([("naive", "in-distribution certificate", C["Tuned"]), ("mondrian", "sea-state conditional", C["GBDT"]),
                                      ("online", "online adaptive", C["SeaSel"]), ("oracle_recal", "recalibrated (reference)", C["Grey"])]):
    vals = [RES["ood"][n][key]["sel_risk"] if not isinstance(RES["ood"][n][key]["sel_risk"], dict)
            else RES["ood"][n][key]["sel_risk"]["mean"] for n, _ in names]
    ax[0].bar(np.arange(3) + (i - 1.5) * w, vals, w, color=col, label=lab)
ax[0].axhline(RES["meta"]["alpha"], color="#0b0b0b", lw=0.9, ls=":")
ax[0].set_xticks(np.arange(3)); ax[0].set_xticklabels([n for _, n in names])
ax[0].set_ylabel("selective risk"); ax[0].legend(fontsize=6.6); ax[0].set_title("(a)", loc="left")
tr = RES["ood_traces"]["ood_storm"]
x = np.arange(len(tr["naive"])) * 100 / 40 / 60
ax[1].plot(x, tr["naive"], color=C["Tuned"], label="fixed certificate")
ax[1].plot(x, tr["online"], color=C["SeaSel"], label="online adaptive")
ax[1].axhline(RES["meta"]["alpha"], color="#0b0b0b", lw=0.9, ls=":")
ax[1].set_xlabel("time (h), severe-storm day"); ax[1].set_ylabel("running selective risk")
ax[1].legend(fontsize=7); ax[1].set_title("(b)", loc="left")
save(fig, "fig_ood")


# ------------------------------------------------------------------ tables
def pm(d, k, f="{:.3f}", s=1.0):
    return (f + r" $\pm$ " + f).format(mean(d, k) * s, std(d, k) * s)


rows = ["Max-elevation (always aggressive)", "Always robust (dual diversity)", "Link-budget margin, tuned",
        "Logistic regression, tuned", "Gradient boosting, tuned", "Gradient boosting + LTT", "PBA single model + LTT",
        "SeaSel (PBA ensemble + LTT)"]
L = [r"\begin{table}[t]\centering\small",
     r"\caption{\textbf{Main results on the held-out pool} (target $\alpha=0.05$, $\delta=0.1$; mean $\pm$ s.d. over 100 random calibration/test splits). Viol.: fraction of splits with selective risk above $\alpha$.}",
     r"\label{tab:main}", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lccccccc@{}}\toprule",
     r"Method & Certified & Coverage & Selective risk & Miss rate & Goodput & Energy & Viol.\\",
     r" & & & & (\%) & (Mbit/epoch) & (J/Mbit) & \\\midrule"]
for r_ in rows:
    d = M[r_]
    cert = "yes" if ("LTT" in r_) else "no"
    name = r"\textbf{" + r_ + "}" if r_.startswith("SeaSel") else r_
    L.append(f"{name} & {cert} & {pm(d, 'coverage')} & {pm(d, 'sel_risk')} & {pm(d, 'miss_rate', '{:.2f}', 100)} & "
             f"{pm(d, 'goodput', '{:.1f}')} & {pm(d, 'energy_per_mbit', '{:.2f}')} & {d['violation']:.2f}\\\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(os.path.join(GEN, "tab_main.tex"), "w").write("\n".join(L))

L = [r"\begin{table}[t]\centering\small", r"\caption{\textbf{Discrimination and calibration of the link-success predictors} on the held-out pool.}",
     r"\label{tab:disc}", r"\begin{tabular}{@{}lccc@{}}\toprule", r"Predictor & AUROC $\uparrow$ & ECE $\downarrow$ & AURC $\downarrow$\\\midrule"]
for k, v in RES["discrimination"].items():
    L.append(f"{k} & {v['auroc']:.3f} & {v['ece']:.3f} & {v['aurc']:.4f}\\\\")
L += [r"\bottomrule\end{tabular}\end{table}"]
open(os.path.join(GEN, "tab_disc.tex"), "w").write("\n".join(L))

L = [r"\begin{table}[t]\centering\small", r"\caption{\textbf{Ablation study} (retrained variants; $\alpha=0.05$, $\delta=0.1$, 100 splits).}",
     r"\label{tab:ablation}", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lcccccc@{}}\toprule",
     r"Variant & AUROC & ECE & Coverage & Selective risk & Goodput (Mbit) & Viol.\\\midrule"]
for k, d in RES["ablation"].items():
    disc = RES["ablation_disc"].get(k, RES["ablation_disc"]["Full SeaSel"])
    name = r"\textbf{" + k + "}" if k == "Full SeaSel" else k
    L.append(f"{name} & {disc['auroc']:.3f} & {disc['ece']:.3f} & {pm(d, 'coverage')} & {pm(d, 'sel_risk')} & "
             f"{pm(d, 'goodput', '{:.1f}')} & {d['violation']:.2f}\\\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(os.path.join(GEN, "tab_ablation.tex"), "w").write("\n".join(L))

L = [r"\begin{table}[t]\centering\small", r"\caption{\textbf{Robustness to sensor corruption at test time.} Fixed certificate from clean calibration versus recalibration on corrupted data ($\alpha=0.05$).}",
     r"\label{tab:robust}", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lccccc@{}}\toprule",
     r" & & \multicolumn{2}{c}{Clean certificate} & \multicolumn{2}{c}{Recalibrated}\\",
     r"Corruption & AUROC & Coverage & Selective risk & Coverage & Selective risk\\\midrule"]
for k, d in RES["robustness"].items():
    L.append(f"{k.replace('%', chr(92) + '%')} & {d['auroc']:.3f} & {d['clean_cal']['coverage']:.3f} & {d['clean_cal']['sel_risk']:.3f} & "
             f"{d['recal']['coverage']['mean']:.3f} & {d['recal']['sel_risk']['mean']:.3f}\\\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(os.path.join(GEN, "tab_robust.tex"), "w").write("\n".join(L))

L = [r"\begin{table}[t]\centering\small", r"\caption{\textbf{Out-of-distribution operation} (target $\alpha=0.05$). Sel.: selective risk; Cov.: coverage; Miss: critical-message miss rate.}",
     r"\label{tab:ood}", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lcccccccccc@{}}\toprule",
     r" & & \multicolumn{2}{c}{Always aggressive} & \multicolumn{2}{c}{In-distribution certificate} & \multicolumn{2}{c}{Sea-state conditional} & \multicolumn{2}{c}{Online adaptive}\\",
     r"Shift & AUROC & Sel. & Miss & Sel. & Cov. & Sel. & Cov. & Sel. & Cov.\\\midrule"]
for n, lab in names:
    o = RES["ood"][n]
    L.append(f"{lab.replace('%', chr(92) + '%')} & {o['auroc']:.3f} & {o['accept_all']['sel_risk']:.3f} & {100 * o['accept_all']['miss_rate']:.1f}\\% & "
             f"{o['naive']['sel_risk']:.3f} & {o['naive']['coverage']:.3f} & {o['mondrian']['sel_risk']:.3f} & {o['mondrian']['coverage']:.3f} & "
             f"{o['online']['sel_risk']:.3f} & {o['online']['coverage']:.3f}\\\\")
L += [r"\bottomrule\end{tabular}}\end{table}"]
open(os.path.join(GEN, "tab_ood.tex"), "w").write("\n".join(L))

if "transfer" in RES:
    L = [r"\begin{table}[t]\centering\small",
         r"\caption{\textbf{Cross-link transfer when half of the link histories are missing} (models retrained and evaluated with 50\% of satellite tokens lacking beacon history; $\alpha=0.05$).}",
         r"\label{tab:transfer}", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lcccccc@{}}\toprule",
         r"Predictor & AUROC (all) & AUROC (no history) & ECE & AURC & Coverage & Selective risk\\\midrule"]
    for k, d in RES["transfer"].items():
        L.append(f"{k} & {d['auroc']:.3f} & {d['auroc_missing']:.3f} & {d['ece']:.3f} & {d['aurc']:.4f} & "
                 f"{pm(d['cert'], 'coverage')} & {pm(d['cert'], 'sel_risk')}\\\\")
    L += [r"\bottomrule\end{tabular}}\end{table}"]
    open(os.path.join(GEN, "tab_transfer.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ numeric macros
S = M["SeaSel (PBA ensemble + LTT)"]
AA = M["Max-elevation (always aggressive)"]
AR = M["Always robust (dual diversity)"]
G = M["Gradient boosting + LTT"]
GT = M["Gradient boosting, tuned"]
D = RES["discrimination"]
TB = RES["testbed"]
num = dict(
    NTrain=f"{RES['meta']['n_train']:,}", NPool=f"{RES['meta']['n_pool']:,}",
    PAggr=f"{100 * TB['p_aggr']:.1f}", PRob=f"{100 * TB['p_rob']:.1f}", NCand=f"{TB['mean_candidates']:.1f}",
    NoLink=f"{100 * TB['no_link']:.1f}",
    SCov=f"{100 * mean(S, 'coverage'):.1f}", SRisk=f"{100 * mean(S, 'sel_risk'):.2f}", SMiss=f"{100 * mean(S, 'miss_rate'):.2f}",
    SGood=f"{mean(S, 'goodput'):.1f}", SEnergy=f"{mean(S, 'energy_per_mbit'):.2f}", SViol=f"{S['violation']:.2f}",
    AAMiss=f"{100 * mean(AA, 'miss_rate'):.1f}", AAGood=f"{mean(AA, 'goodput'):.1f}", AAEnergy=f"{mean(AA, 'energy_per_mbit'):.2f}",
    ARMiss=f"{100 * mean(AR, 'miss_rate'):.2f}", ARGood=f"{mean(AR, 'goodput'):.1f}", AREnergy=f"{mean(AR, 'energy_per_mbit'):.1f}",
    GCov=f"{100 * mean(G, 'coverage'):.1f}", GRisk=f"{100 * mean(G, 'sel_risk'):.2f}", GTViol=f"{GT['violation']:.2f}",
    GTRisk=f"{100 * mean(GT, 'sel_risk'):.2f}",
    MissRed=f"{100 * (1 - mean(S, 'miss_rate') / mean(AA, 'miss_rate')):.0f}",
    GoodRatio=f"{mean(S, 'goodput') / mean(AR, 'goodput'):.1f}",
    GoodKeep=f"{100 * mean(S, 'goodput') / mean(AA, 'goodput'):.0f}",
    CovGain=f"{100 * (mean(S, 'coverage') - mean(G, 'coverage')):.1f}",
    SAuroc=f"{D['SeaSel ensemble']['auroc']:.3f}", SEce=f"{D['SeaSel ensemble']['ece']:.3f}", SAurc=f"{D['SeaSel ensemble']['aurc']:.4f}",
    GAuroc=f"{D['Gradient boosting']['auroc']:.3f}", GEce=f"{D['Gradient boosting']['ece']:.3f}", GAurc=f"{D['Gradient boosting']['aurc']:.4f}",
    LAuroc=f"{D['Link-budget margin']['auroc']:.3f}",
    Params=f"{RES['complexity']['params']:,}", MsDec=f"{RES['complexity']['ms_per_decision']:.2f}",
    BetaMean=f"{np.mean(RES['complexity']['beta']):.2f}",
    Runtime=f"{RES['runtime_s'] / 60:.0f}",
)
for n, short in [("ood_storm", "Storm"), ("ood_tropical", "Trop"), ("ood_constellation", "Const")]:
    o = RES["ood"][n]
    num[f"{short}Naive"] = f"{100 * o['naive']['sel_risk']:.1f}"
    num[f"{short}Online"] = f"{100 * o['online']['sel_risk']:.1f}"
    num[f"{short}Mond"] = f"{100 * o['mondrian']['sel_risk']:.1f}"
    num[f"{short}OnlineCov"] = f"{100 * o['online']['coverage']:.1f}"
    num[f"{short}AAMiss"] = f"{100 * o['accept_all']['miss_rate']:.1f}"
    num[f"{short}Auroc"] = f"{o['auroc']:.3f}"
ab = RES["ablation"]; abd = RES["ablation_disc"]
for k, short in [("Without physics bias (plain attention)", "NoBias"), ("Mean pooling instead of attention", "Pool"),
                 ("Without attention (per-link MLP)", "NoAttn"), ("Without motion and weather context", "NoCtx"),
                 ("Without link-history features", "NoHist"), ("Without uncertainty penalty (gamma = 0)", "NoGamma"),
                 ("Single member (M = 1)", "Single"), ("Without temperature scaling", "NoTemp")]:
    num[f"Ab{short}Cov"] = f"{100 * mean(ab[k], 'coverage'):.1f}"
    num[f"Ab{short}Auroc"] = f"{abd.get(k, abd['Full SeaSel'])['auroc']:.3f}"
if "transfer" in RES:
    for k, short in [("Physics-biased attention", "PBA"), ("Plain attention", "ATT"), ("Per-link MLP", "MLP")]:
        t = RES["transfer"][k]
        num[f"Tr{short}Auroc"] = f"{t['auroc']:.3f}"
        num[f"Tr{short}Miss"] = f"{t['auroc_missing']:.3f}"
        num[f"Tr{short}Cov"] = f"{100 * t['cert']['coverage']['mean']:.1f}"
        num[f"Tr{short}Aurc"] = f"{t['aurc']:.4f}"
open(os.path.join(GEN, "numbers.tex"), "w").write("\n".join(f"\\newcommand{{\\R{k}}}{{{v}}}" for k, v in num.items()) + "\n")
print(json.dumps(num, indent=1))
