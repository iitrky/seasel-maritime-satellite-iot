"""Small end-to-end demo (about 3 minutes): simulate 12 vessels for 6 h, train a 2-member ensemble,
certify alpha = 0.05 and compare with always-aggressive and always-robust operation.

    python scripts/quick_demo.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from testbed import simulate
from features import build, Scaler
from model import train_ensemble
import risk as R

d = simulate(dict(n_vessels=12, hours=6), seed=7)
F = build(d)
tr, va, ho = np.arange(0, 7), np.arange(7, 8), np.arange(8, 12)
sc = Scaler().fit(F, tr)
parts = [sc.transform(F, a) for a in (tr, va, ho)]
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
n1, n2 = len(parts[0]["y"]), len(parts[1]["y"])
ens = train_ensemble(D, np.arange(n1), np.arange(n1, n1 + n2), M=2, epochs=6)
P, S, PR = ens.predict(D)
pool = np.random.default_rng(0).permutation(np.arange(n1 + n2, len(D["y"])))
cal, te = pool[: len(pool) // 2], pool[len(pool) // 2:]
g, lam = R.calibrate(P, S, D, cal, 0.05, 0.1)
for name, (gg, ll) in {"SeaSel certified (alpha=0.05)": (g, lam), "always aggressive": (0, -1.0), "always robust": (0, 9.0)}.items():
    r = R.evaluate(P, S, D, te, gg, ll, PR=PR)
    print(f"{name:30s} coverage {r['coverage']:.2f}  selective risk {r['sel_risk']:.3f}  "
          f"miss {100 * r['miss_rate']:.1f}%  goodput {r['goodput']:.0f} Mbit/epoch")
