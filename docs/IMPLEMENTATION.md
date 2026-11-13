# Implementation details

This document maps the manuscript to the code.

## 1. Modules

| File | Content |
|---|---|
| `src/testbed.py` | Sandbox testbed: two Walker shells, Earth rotation, look angles, vessel routes, weather (storms, rain cells, sea state), IMU synthesis, deck phased-array link budget, blockage, fading, handover, coastal 5G |
| `src/features.py` | Preprocessing, IMU segmentation, FFT motion descriptors, tokenisation, masks, z-scoring (`Scaler`) |
| `src/model.py` | Physics-biased attention (PBA) network, multi-task heads, Adam training with autograd, ensemble + temperature scaling |
| `src/risk.py` | Decision rule, outcomes with dual-link fallback, binomial p-values, Learn-then-Test certification, sea-state-conditional certification, online adaptation, metrics |
| `src/experiments.py` | Full campaign: data generation, training, baselines, 100-split evaluation, alpha sweep, operating curves, ablations, robustness, OOD |
| `src/make_figures.py` | Figures 4-8, Tables 3-7 and numeric macros for the manuscript |
| `manuscript/` | LaTeX sources, Crossref-verified references, PDF/Word build |

## 2. Equation-to-code map

| Manuscript | Code |
|---|---|
| Eq. (1) SNR budget | `testbed.simulate` (`snr_cs`, `snr_mean`, fading `hfade`, `scint`) |
| Eq. (2) scan angle and scan loss | `testbed.simulate` (`cosang`, `l_scan`) |
| Eq. (3) pointing loss | `testbed.simulate` (`eps`, `l_point`) |
| Eq. (4) rain attenuation | `testbed.simulate` (`l_rain`) |
| Eq. (5) aggressive / robust labels, handover | `testbed.simulate` (`ya`, `yr`, `stays`) |
| Eq. (6)-(7) link choice, score, acceptance | `risk.decide`, `risk.evaluate` |
| Eq. (8)-(10) selective risk, miss rate, goodput | `risk.evaluate` |
| Eq. (11)-(12) objective and miss bound | `risk.calibrate` (selection by coverage among certified pairs) |
| Eq. (13)-(15) segmentation, RMS, dominant period | `features.segment_imu` |
| Eq. (16) token vector | `features.build` (`TOKEN_FEATURES`) |
| Eq. (17) normalisation | `features.Scaler` |
| Eq. (18)-(22) PBA network and heads | `model.forward` (modes `pba`, `attn`, `pool`, `none` for ablations) |
| Eq. (23) multi-task loss | `model.bce_loss` |
| Eq. (24)-(25) temperature, ensemble mean and spread | `model.Ensemble.fit_temperature`, `model.Ensemble.predict` |
| Eq. (26)-(28) counts, binomial p-value, certified selection | `risk.binom_p`, `risk.ltt`, `risk.calibrate` |
| Eq. (29) sea-state-conditional certificate | `risk.calibrate_groups` |
| Eq. (30)-(31) online threshold update | `risk.online_adapt` |
| Theorem 1, Proposition 1 | checked numerically in `tests/test_seasel.py` |

## 3. Data layout

`features.build` returns arrays shaped `(vessels, epochs, K+1, ...)`; `Scaler.transform` flattens them to `(N, K+1, ...)` with
`tok` (N, 9, 12), `ctx` (N, 7), `mask` (N, 9), `geo` (N, 9, 3), labels `y`, `y_rob`, rates `rate`, and side information
(`hs`, `rain`, `el`, `is_ter`, `roll_rms`, `vessel`). Token index 8 is the coastal link.

## 4. Reproducibility

* Fixed seeds: testbed days 0-4, ensemble members `100*seed + m`, evaluation splits seed 123.
* `results/cache/` stores the simulated days and trained ensembles, so reruns skip completed work.
* Set `OMP_NUM_THREADS=1`; the campaign uses process-level parallelism (4 workers).
