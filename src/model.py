"""Physics-biased link-set attention (PBA) and its deep ensemble (Section 6 of the paper).

score_lm = q_l k_m / sqrt(d_k) - softplus(beta) * (1 - u_l . u_m)
The angular term makes links that look at nearby parts of the sky (and therefore share rain cells,
superstructure shadows and motion-induced scan loss) attend to each other.  The network is permutation
equivariant and handles any number of visible satellites through the validity mask.
"""
import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc import flatten


def init_params(ft, fc, rng, d=32, dk=16):
    s = lambda a, b: rng.normal(0, np.sqrt(2.0 / a), (a, b))
    return dict(W1=s(ft + fc, d), b1=np.zeros(d), Wq=s(d, dk), Wk=s(d, dk), Wv=s(d, dk), Wo=s(dk, d) * 0.5,
                beta=np.array([0.5]), W2=s(d, d), b2=np.zeros(d), W3=s(d, d) * 0.5, w=s(d, 2) * 0.5,
                b=np.zeros(2))


def softplus(x):
    return anp.logaddexp(0.0, x)


def forward(p, tok, ctx, mask, geo, mode="pba"):
    B, K, _ = tok.shape
    x = anp.concatenate([tok, anp.repeat(ctx[:, None, :], K, axis=1)], axis=-1)
    h0 = anp.maximum(0.0, anp.dot(x, p["W1"]) + p["b1"])
    if mode in ("pba", "attn"):
        q, k, v = anp.dot(h0, p["Wq"]), anp.dot(h0, p["Wk"]), anp.dot(h0, p["Wv"])
        logits = anp.einsum("bld,bmd->blm", q, k) / np.sqrt(q.shape[-1])
        if mode == "pba":
            cos = np.einsum("blc,bmc->blm", geo, geo)
            logits = logits - softplus(p["beta"][0]) * (1.0 - cos)
        logits = logits + (mask[:, None, :] - 1.0) * 1e9
        logits = logits - anp.max(logits, axis=-1, keepdims=True)
        a = anp.exp(logits)
        a = a / anp.sum(a, axis=-1, keepdims=True)
        h1 = h0 + anp.dot(anp.einsum("blm,bmd->bld", a, v), p["Wo"])
    elif mode == "pool":                                      # DeepSets-style mean pooling (ablation)
        pooled = anp.sum(h0 * mask[..., None], axis=1) / (anp.sum(mask, axis=1)[:, None] + 1e-9)
        h1 = h0 + anp.dot(anp.dot(pooled, p["Wv"]), p["Wo"])[:, None, :]
    else:                                                     # "none": independent per-link MLP
        h1 = h0
    h2 = h1 + anp.dot(anp.maximum(0.0, anp.dot(h1, p["W2"]) + p["b2"]), p["W3"])
    return anp.dot(h2, p["w"]) + p["b"]                       # (B, K, 2): aggressive, robust


def bce_loss(p, tok, ctx, mask, geo, y, mode, l2=1e-4):
    """Multi-task BCE: y[..., 0] aggressive-mode success, y[..., 1] robust-mode success."""
    z = forward(p, tok, ctx, mask, geo, mode)
    ll = anp.sum(anp.logaddexp(0.0, z) - y * z, axis=-1)
    reg = sum(anp.sum(v ** 2) for k, v in p.items() if k.startswith("W"))
    return anp.sum(ll * mask) / anp.sum(mask) + l2 * reg


def train_one(D, idx, seed, mode="pba", epochs=12, bs=256, lr=3e-3, verbose=False):
    rng = np.random.default_rng(seed)
    ft, fc = D["tok"].shape[-1], D["ctx"].shape[-1]
    p = init_params(ft, fc, rng)
    flat, unflat = flatten(p)
    g = grad(lambda f, *a: bce_loss(unflat(f), *a))
    m, v = np.zeros_like(flat), np.zeros_like(flat)
    t = 0
    boot = rng.choice(idx, len(idx), replace=True)
    for ep in range(epochs):
        order = rng.permutation(boot)
        lr_e = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)))
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            gr = g(flat, D["tok"][b], D["ctx"][b], D["mask"][b], D["geo"][b], D["Y2"][b], mode)
            t += 1
            m = 0.9 * m + 0.1 * gr
            v = 0.999 * v + 0.001 * gr ** 2
            flat = flat - lr_e * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if verbose:
            print("  epoch", ep, float(bce_loss(unflat(flat), D["tok"][b], D["ctx"][b], D["mask"][b], D["geo"][b], D["Y2"][b], mode)))
    return unflat(flat)


class Ensemble:
    def __init__(self, members, mode):
        self.members, self.mode, self.T = members, mode, 1.0

    def logits(self, D, idx=None, bs=4096):
        idx = np.arange(len(D["y"])) if idx is None else idx
        out = []
        for p in self.members:
            z = np.concatenate([forward(p, D["tok"][idx[i:i + bs]], D["ctx"][idx[i:i + bs]], D["mask"][idx[i:i + bs]],
                                        D["geo"][idx[i:i + bs]], self.mode) for i in range(0, len(idx), bs)])
            out.append(z)
        return np.stack(out)                                   # (M, N, K, 2)

    def fit_temperature(self, D, idx):
        z = self.logits(D, idx).mean(0)[..., 0]
        y, m = D["y"][idx], D["mask"][idx] > 0
        best = (1e9, 1.0)
        for T in np.exp(np.linspace(np.log(0.4), np.log(3.0), 60)):
            pz = 1 / (1 + np.exp(-z[m] / T))
            nll = -np.mean(y[m] * np.log(pz + 1e-9) + (1 - y[m]) * np.log(1 - pz + 1e-9))
            best = min(best, (nll, T))
        self.T = best[1]
        return self.T

    def predict(self, D, idx=None):
        Z = self.logits(D, idx)
        P = 1 / (1 + np.exp(-Z / self.T))
        # aggressive-mode mean probability, its epistemic spread, robust-mode mean probability
        return P[..., 0].mean(0), P[..., 0].std(0), P[..., 1].mean(0)


def train_ensemble(D, idx_tr, idx_val, M=5, mode="pba", seed=0, epochs=12):
    D["Y2"] = np.stack([D["y"], D["y_rob"]], -1)
    members = [train_one(D, idx_tr, seed * 100 + m, mode=mode, epochs=epochs) for m in range(M)]
    ens = Ensemble(members, mode)
    ens.fit_temperature(D, idx_val)
    return ens
