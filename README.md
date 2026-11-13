# SeaSel: certified selective link prediction for maritime IoT over LEO satellites

Code, sandbox testbed, results and manuscript for

> **Certified selective link prediction for maritime Internet of Things over low Earth orbit satellites**
> A. Bamila Virgin Louis, Department of Computer Science and Engineering, St. Xavier's Catholic College of Engineering, Nagercoil, India
> bamilavirgin@sxcce.edu.in · ORCID [0000-0003-0297-4004](https://orcid.org/0000-0003-0297-4004)

## What SeaSel does

A ship's IoT gateway must choose every minute among visible low Earth orbit (LEO) satellites and, near ports, a coastal 5G link. Satellites set, rain cells pass, masts block the sky, and the ship rolls with the sea. SeaSel:

1. predicts the success of every candidate link with a **physics-biased attention ensemble**. Attention logits are penalised by the angular separation of links, so geometrically close links share evidence;
2. **certifies** an acceptance threshold by Learn-then-Test with exact binomial p-values. With probability ≥ 1 − δ, the failure rate of accepted aggressive transmissions stays ≤ α;
3. **abstains** to a robust dual-link mode otherwise, which turns the statistical guarantee into a bound on the critical-message miss rate;
4. handles **distribution shift** with a sea-state-conditional certificate and an online threshold update.

## Key results (sandbox testbed; α = 0.05, δ = 0.1, 100 calibration/test splits)

| Policy | Coverage | Selective risk | Critical miss rate | Goodput (Mbit/epoch) | Splits violating α |
|---|---|---|---|---|---|
| Max-elevation (always aggressive) | 100% | 13.3% | 13.3% | 348.9 | 100% |
| Always robust (dual diversity) | 0% | – | 1.2% | 19.8 | – |
| Gradient boosting, tuned threshold | 93.6% | 6.2% | 6.8% | 353.8 | 100% |
| Gradient boosting + certificate | 89.8% | 4.6% | 5.2% | 346.0 | 1% |
| **SeaSel** | **90.6%** | **4.6%** | **5.2%** | **349.0** | **1%** |

* 61% fewer critical-message misses than always-aggressive operation, at the same goodput; 17.7× the goodput of always-robust operation.
* Uncertified tuned thresholds exceed the target in essentially every split; every certified variant stays within δ.
* Under shift (severe storms, tropical rain, 30% satellite loss) the in-distribution certificate rises to 5.7%, 5.8% and 10.1% risk. The online rule restores 4.9%, 4.9% and 5.0%.
* Honest note: in distribution, the attention architecture gives modest gains over strong tabular models (AUROC 0.907 vs 0.901), but it is the best calibrated (ECE 0.005 vs 0.020) and it transfers evidence best when link history is missing.

## Repository layout

```
src/
  testbed.py         sandbox testbed: 2-shell Walker constellation, vessels, weather, ship motion, link budget
  features.py        preprocessing, IMU segmentation, FFT motion features, tokenisation, scaling
  model.py           physics-biased attention, multi-task heads, ensemble, temperature scaling (autograd)
  risk.py            decision rule, dual-link fallback, binomial p-values, Learn-then-Test, Mondrian, online rule
  experiments.py     full campaign: baselines, 100-split evaluation, alpha sweep, ablations, robustness, OOD, transfer
  make_figures.py    figures, LaTeX tables and numeric macros for the manuscript
tests/test_seasel.py 7 checks of the paper's claims (geometry, p-value validity, guarantee, equivariance, ...)
scripts/             quick_demo.py, reproduce.sh, build_diagrams.sh
results/             results.json (all numbers in the paper), run_log.txt
figures/             result figures; figures/diagrams/ = 3 system-model views + Algorithm 1 (TikZ sources)
manuscript/          LaTeX sources, Crossref-verified references (30, all with DOI), PDF and Word manuscript
docs/IMPLEMENTATION.md  equation-to-code map
```

## Quick start

```bash
pip install -r requirements.txt
python tests/test_seasel.py          # 7 checks, about 1 minute
python scripts/quick_demo.py         # small end-to-end run, about 3 minutes
```

## Reproduce the paper

```bash
sh scripts/reproduce.sh              # about 15-45 minutes on 4 CPU cores
```

All seeds are fixed. `results/cache/` (created on first run) stores simulated days and trained ensembles, so reruns skip completed work. Building the manuscript needs `pdflatex` and, for the Word version, `pandoc` and `python-docx`.

## Citation

See `CITATION.cff`. Code is released under the MIT licence.
