"""Preprocessing, segmentation and input representation (Section 5 of the paper).

* IMU streams (roll, pitch at 2 Hz) are segmented into non-overlapping windows that end at each decision
  epoch; each window yields RMS roll, RMS pitch, RMS angular rate and the dominant motion period (FFT).
* The rain gauge reading is log-compressed and corrupted with sensor noise.
* Every candidate link becomes a token of geometric, ephemeris and history features; a validity mask
  handles the variable number of visible satellites; the coastal 5G link is an extra typed token.
* Features are z-scored with statistics estimated on the training vessels only.
"""
import numpy as np

TOKEN_FEATURES = ["el", "sin_az", "cos_az", "el_rate", "log_tset", "scan0", "sector_ev", "deficit_prev",
                  "has_prev", "is_ter", "log_rng", "snr_cs"]
CONTEXT_FEATURES = ["roll_rms", "pitch_rms", "rate_rms", "period", "rain_obs", "speed", "log_dport"]


def segment_imu(roll, pitch, t_ep, hz, win_s=60.0):
    """Window [t - win, t) for every epoch t; returns (V, E, 4) motion descriptors."""
    V, nI = roll.shape
    E = len(t_ep)
    w = int(win_s * hz)
    out = np.zeros((V, E, 4))
    freqs = np.fft.rfftfreq(w, d=1.0 / hz)
    for e in range(E):
        i1 = int(t_ep[e] * hz)
        i0 = i1 - w
        if i0 < 0:
            i0, i1 = 0, w
        r, p = roll[:, i0:i1], pitch[:, i0:i1]
        out[:, e, 0] = r.std(1)
        out[:, e, 1] = p.std(1)
        out[:, e, 2] = np.sqrt((np.diff(r, axis=1) ** 2 + np.diff(p, axis=1) ** 2).mean(1)) * hz
        spec = np.abs(np.fft.rfft(r - r.mean(1, keepdims=True), axis=1)) ** 2
        f_dom = freqs[1:][np.argmax(spec[:, 1:], axis=1)]
        out[:, e, 3] = 1.0 / np.maximum(f_dom, 1e-3)
    return out


def build(d, seed=0, imu_noise=0.0, rain_noise=0.3, drop_ctx=0.0, drop_hist=0.0):
    """Return a dict of arrays shaped (V, E, ...) ready for the model."""
    rng = np.random.default_rng(seed + 991)
    cfg = d["cfg"]
    V, E, K1 = d["valid"].shape
    roll = d["roll"] + (rng.normal(0, imu_noise, d["roll"].shape) if imu_noise else 0)
    pitch = d["pitch"] + (rng.normal(0, imu_noise, d["pitch"].shape) if imu_noise else 0)
    mot = segment_imu(roll, pitch, d["t_ep"], cfg["imu_hz"])
    from testbed import PORTS, haversine
    dport = np.stack([haversine(d["lat"][:, e][:, None], d["lon"][:, e][:, None], PORTS[None, :, 0], PORTS[None, :, 1]).min(1)
                      for e in range(E)], 1)
    rain_obs = np.log1p(d["rain"]) + rng.normal(0, rain_noise, d["rain"].shape)
    ctx = np.concatenate([mot, rain_obs[..., None], np.repeat(d["speed"][:, None, None], E, 1),
                          np.log1p(dport)[..., None]], -1)
    if drop_ctx:
        m = rng.random(ctx.shape[:2]) < drop_ctx
        ctx[m] = np.nan                                       # imputed later with training means
    el = d["el"]
    tok = np.stack([
        el / 90.0, np.sin(np.radians(d["az_rel"])), np.cos(np.radians(d["az_rel"])), d["el_rate"],
        np.log1p(np.clip(d["t_set"], 0, 900)), d["scan0"] / 90.0, d["sector_ev"], d["deficit_prev"] / 10.0,
        d["has_prev"], d["is_ter"], np.log1p(d["rng"]),
        np.where(d["is_ter"] > 0, 0.0, (cfg["snr_ref"] - 20 * np.log10(np.maximum(d["rng"], 1) / cfg["alt"])
                                        - 0.3 / np.sin(np.radians(np.maximum(el, 1)))) / 10.0)], -1)
    if drop_hist:
        m = rng.random(tok.shape[:3]) < drop_hist
        tok[..., 6][m] = np.nan
        tok[..., 7][m] = np.nan
        tok[..., 8][m] = 0.0
    geo = np.stack([d["ux"], d["uy"], d["uz"]], -1)
    return dict(tok=tok, ctx=ctx, mask=d["valid"], geo=geo, y=d["y_aggr"], y_rob=d["y_rob"], rate=d["rate"],
                hs=d["hs"], rain=d["rain"], el=d["el"], is_ter=d["is_ter"], roll_rms=mot[..., 0])


class Scaler:
    def fit(self, F, idx):
        tok = F["tok"][idx][F["mask"][idx] > 0]
        self.tm, self.ts = np.nanmean(tok, 0), np.nanstd(tok, 0) + 1e-6
        ctx = F["ctx"][idx].reshape(-1, F["ctx"].shape[-1])
        self.cm, self.cs = np.nanmean(ctx, 0), np.nanstd(ctx, 0) + 1e-6
        return self

    def transform(self, F, idx):
        tok = (F["tok"][idx] - self.tm) / self.ts
        tok = np.where(np.isnan(tok), 0.0, tok) * (F["mask"][idx][..., None] > 0)
        ctx = (F["ctx"][idx] - self.cm) / self.cs
        ctx = np.where(np.isnan(ctx), 0.0, ctx)
        V, E = tok.shape[:2]
        flat = lambda a: a.reshape((V * E,) + a.shape[2:])
        return dict(tok=flat(tok), ctx=flat(ctx), mask=flat(F["mask"][idx]), geo=flat(F["geo"][idx]),
                    y=flat(F["y"][idx]), y_rob=flat(F["y_rob"][idx]), rate=flat(F["rate"][idx]),
                    hs=flat(F["hs"][idx]), rain=flat(F["rain"][idx]), el=flat(F["el"][idx]),
                    is_ter=flat(F["is_ter"][idx]), roll_rms=flat(F["roll_rms"][idx]),
                    vessel=np.repeat(np.asarray(idx), E))
