"""SeaSel sandbox testbed: maritime IoT vessels connected through a LEO constellation and coastal 5G.

Everything a vessel gateway would see (ephemeris, IMU, rain gauge, beacon SNR) and the ground-truth
outcome of transmitting on every candidate link are generated from physical models:

* two Walker shells (550 km/53 deg, 36 x 22; 560 km/97.6 deg polar, 12 x 20) with Earth rotation
* vessels on great-circle routes between 12 major ports, realistic speeds and headings
* sea-state field (latitude climate + moving storms) driving roll/pitch through a JONSWAP-like spectrum
* deck-mounted phased array: scan loss cos^1.3, motion-induced pointing loss 12 (eps/theta3dB)^2
* Ku-band rain attenuation (ITU-R style power law, slant path, reduction factor) from moving rain cells
* superstructure blockage sectors (vessel specific), Rician fading with elevation-dependent K factor
* handover failure when the satellite sets during the 20 s transmission
* coastal 5G links near ports with sea-path loss and slowly varying ducting

Units: angles in degrees unless stated, distances in km, time in s, SNR in dB.
"""
import numpy as np

RE = 6371.0                    # Earth radius (km)
MU = 398600.4418               # Earth gravitational parameter (km^3/s^2)
WE = 7.2921159e-5              # Earth rotation rate (rad/s)

PORTS = np.array([
    [51.95, 4.14], [1.26, 103.84], [31.23, 121.47], [33.74, -118.27], [-23.96, -46.33], [-29.87, 31.03],
    [18.95, 72.84], [-33.86, 151.21], [40.67, -74.04], [6.45, 3.40], [25.00, 55.06], [35.10, 129.04]])

CFG = dict(
    n_vessels=40, hours=24.0, dt=60.0, t_tx=20.0, imu_hz=2.0,
    # constellation
    alt=550.0, shells=((550.0, 53.0, 36, 22), (560.0, 97.6, 12, 20)), el_min=25.0, k_max=8, sat_fail=0.0, sat_loss_db=0.0,
    # link budget
    snr_ref=23.0, scan_exp=1.3, scan_max=70.0, theta3db=3.0, kappa_lat=0.25,
    gamma_aggr=10.0, gamma_robust=-2.0, rate_sat=20.0, rate_ter=40.0, rate_robust=1.0,
    # weather
    storm_prob=0.45, storm_hs=(2.0, 4.0), storm_rain=(15.0, 45.0), storm_radius=180.0, storm_speed=30.0,
    base_rain_cells=60, lat_band=None,
    # coastal 5G
    ter_range=25.0,
)


# ----------------------------------------------------------------------------- geometry helpers
def llh_to_ecef(lat, lon, h=0.0):
    la, lo = np.radians(lat), np.radians(lon)
    r = RE + h
    return np.stack([r * np.cos(la) * np.cos(lo), r * np.cos(la) * np.sin(lo), r * np.sin(la)], -1)


def enu_basis(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    e = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], -1)
    n = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], -1)
    u = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    return e, n, u


def great_circle(p1, p2, f):
    a, b = llh_to_ecef(*p1) / RE, llh_to_ecef(*p2) / RE
    om = np.arccos(np.clip(a @ b, -1, 1))
    v = (np.sin((1 - f) * om)[..., None] * a + np.sin(f * om)[..., None] * b) / np.sin(om)
    return np.degrees(np.arcsin(v[..., 2])), np.degrees(np.arctan2(v[..., 1], v[..., 0])), om * RE


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * RE * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ----------------------------------------------------------------------------- constellation
class Constellation:
    """Walker-delta shells; each shell = (altitude km, inclination deg, planes, satellites per plane)."""

    def __init__(self, cfg, rng):
        a, n, raan, m0, inc = [], [], [], [], []
        for alt, i_deg, P, S in cfg["shells"]:
            aa = RE + alt
            a.append(np.full(P * S, aa)); n.append(np.full(P * S, np.sqrt(MU / aa ** 3)))
            raan.append(np.repeat(np.arange(P) * 2 * np.pi / P, S))
            m0.append(np.tile(np.arange(S) * 2 * np.pi / S, P) + np.repeat(np.arange(P), S) * 2 * np.pi / (P * S))
            inc.append(np.full(P * S, np.radians(i_deg)))
        self.a, self.n, self.raan, self.m0, self.inc = map(np.concatenate, (a, n, raan, m0, inc))
        self.N = len(self.a)
        self.alive = rng.random(self.N) >= cfg["sat_fail"]

    def ecef(self, t):
        u = self.m0 + self.n * t
        x_o, y_o = self.a * np.cos(u), self.a * np.sin(u)
        ci, si = np.cos(self.inc), np.sin(self.inc)
        th = self.raan - WE * t
        x = x_o * np.cos(th) - y_o * ci * np.sin(th)
        y = x_o * np.sin(th) + y_o * ci * np.cos(th)
        z = y_o * si
        return np.stack([x, y, z], -1)


def look_angles(sat_ecef, lat, lon):
    """Elevation, azimuth (deg), range (km) and ENU unit vectors from vessels (V) to satellites (S)."""
    p = llh_to_ecef(lat, lon)
    e, n, u = enu_basis(lat, lon)
    d = sat_ecef[None, :, :] - p[:, None, :]
    rng_ = np.linalg.norm(d, axis=-1)
    de, dn, du = (d * e[:, None]).sum(-1), (d * n[:, None]).sum(-1), (d * u[:, None]).sum(-1)
    el = np.degrees(np.arcsin(du / rng_))
    az = np.degrees(np.arctan2(de, dn)) % 360
    return el, az, rng_, np.stack([de, dn, du], -1) / rng_[..., None]


# ----------------------------------------------------------------------------- weather and sea state
class Weather:
    """Moving storms (rain + waves) placed on vessel tracks, background rain cells, latitude climate."""

    def __init__(self, cfg, rng, tracks):
        T = cfg["hours"] * 3600
        self.storms = []
        for v, (lat, lon) in enumerate(tracks):
            if rng.random() < cfg["storm_prob"]:
                k = rng.integers(len(lat) // 6, 5 * len(lat) // 6)
                ang = rng.uniform(0, 2 * np.pi)
                self.storms.append(dict(lat=lat[k], lon=lon[k], t0=k * cfg["dt"] + rng.uniform(-3, 3) * 3600,
                                        hs=rng.uniform(*cfg["storm_hs"]), rain=rng.uniform(*cfg["storm_rain"]),
                                        r=cfg["storm_radius"] * rng.uniform(0.7, 1.3),
                                        vx=cfg["storm_speed"] * np.cos(ang), vy=cfg["storm_speed"] * np.sin(ang),
                                        dur=rng.uniform(6, 14) * 3600))
        nb = cfg["base_rain_cells"]
        lo_lat, hi_lat = (-55, 60) if cfg["lat_band"] is None else cfg["lat_band"]
        self.cells = dict(lat=rng.uniform(lo_lat, hi_lat, nb), lon=rng.uniform(-180, 180, nb),
                          t0=rng.uniform(0, T, nb), rain=rng.lognormal(1.6, 0.6, nb),
                          r=rng.uniform(20, 60, nb), dur=rng.uniform(1, 4, nb) * 3600)
        self.tropical = cfg["lat_band"] is not None

    def _storm_frac(self, s, lat, lon, t):
        dtt = t - s["t0"]
        clat = s["lat"] + (s["vy"] * dtt / 3600) / 111.0
        clon = s["lon"] + (s["vx"] * dtt / 3600) / (111.0 * max(np.cos(np.radians(s["lat"])), 0.2))
        d = haversine(lat, lon, clat, clon)
        env = np.exp(-0.5 * (dtt / (0.35 * s["dur"])) ** 2)
        return np.exp(-0.5 * (d / s["r"]) ** 2) * env

    def at(self, lat, lon, t):
        hs = 1.0 + 1.6 * np.abs(lat) / 60.0
        rain = np.zeros_like(lat)
        for s in self.storms:
            f = self._storm_frac(s, lat, lon, t)
            hs = hs + s["hs"] * f
            rain = rain + s["rain"] * f
        c = self.cells
        for i in range(len(c["lat"])):
            dtt = t - c["t0"][i]
            if abs(dtt) > 2 * c["dur"][i]:
                continue
            d = haversine(lat, lon, c["lat"][i], c["lon"][i])
            rain = rain + c["rain"][i] * np.exp(-0.5 * (d / c["r"][i]) ** 2) * np.exp(-0.5 * (dtt / (0.4 * c["dur"][i])) ** 2)
        if self.tropical:
            rain = rain * 1.8
        return hs, rain


# ----------------------------------------------------------------------------- main generator
def simulate(cfg_over=None, seed=0):
    """Run the testbed and return a dict of per-epoch, per-candidate observations and outcomes."""
    cfg = dict(CFG)
    cfg.update(cfg_over or {})
    rng = np.random.default_rng(seed)
    V = cfg["n_vessels"]
    nE = int(cfg["hours"] * 3600 / cfg["dt"])
    t_ep = np.arange(nE) * cfg["dt"]

    # ---- vessel routes
    lat_tr, lon_tr, head = np.zeros((V, nE)), np.zeros((V, nE)), np.zeros((V, nE))
    speed = rng.uniform(10, 22, V) * 1.852                # km/h
    for v in range(V):
        while True:
            i, j = rng.choice(len(PORTS), 2, replace=False)
            if cfg["lat_band"] is not None:
                p1 = np.array([rng.uniform(*cfg["lat_band"]), rng.uniform(-180, 180)])
                p2 = np.array([rng.uniform(*cfg["lat_band"]), p1[1] + rng.uniform(15, 40)])
            else:
                p1, p2 = PORTS[i], PORTS[j]
            _, _, L = great_circle(p1, p2, np.array(0.5))
            if L > 1500:
                break
        f0 = 0.0 if rng.random() < 0.3 else rng.uniform(0.05, 0.8)
        f = np.clip(f0 + speed[v] * t_ep / 3600 / L, 0, 1)
        if rng.random() < 0.15:                            # arriving vessels end near port
            f = np.clip(1.0 - speed[v] * (t_ep[-1] - t_ep) / 3600 / L, 0, 1)
        la, lo, _ = great_circle(p1, p2, f)
        lat_tr[v], lon_tr[v] = la, lo
        la2, lo2, _ = great_circle(p1, p2, np.clip(f + 1e-4, 0, 1))
        head[v] = np.degrees(np.arctan2(np.radians(lo2 - lo) * np.cos(np.radians(la)), np.radians(la2 - la))) % 360
    weather = Weather(cfg, rng, list(zip(lat_tr, lon_tr)))

    # ---- vessel-specific properties
    k_roll = rng.uniform(1.5, 3.0, V)                     # deg roll RMS per metre Hs
    k_pitch = rng.uniform(0.5, 1.0, V)
    t_roll = rng.uniform(9, 18, V)                        # natural roll period (s)
    blocks = []
    for v in range(V):
        nb = rng.integers(1, 3)
        blocks.append([(rng.uniform(0, 360), rng.uniform(15, 35), rng.uniform(35, 55)) for _ in range(nb)])
    kappa_v = cfg["kappa_lat"] * rng.uniform(0.7, 1.3, V)

    # ---- IMU (roll, pitch) at imu_hz for the whole day, amplitude following local Hs
    hz = cfg["imu_hz"]
    nI = int(cfg["hours"] * 3600 * hz)
    tI = np.arange(nI) / hz
    hs_ep, rain_ep = np.zeros((V, nE)), np.zeros((V, nE))
    for e in range(nE):
        hs_ep[:, e], rain_ep[:, e] = weather.at(lat_tr[:, e], lon_tr[:, e], t_ep[e])
    roll, pitch = np.zeros((V, nI), np.float32), np.zeros((V, nI), np.float32)
    for v in range(V):
        w0 = 2 * np.pi / t_roll[v]
        ws = w0 * rng.uniform(0.6, 1.6, 24)
        amp = np.exp(-0.5 * ((ws - w0) / (0.25 * w0)) ** 2)
        ph = rng.uniform(0, 2 * np.pi, 24)
        base = (amp[:, None] * np.sin(ws[:, None] * tI[None] + ph[:, None])).sum(0)
        base /= base.std()
        ws2 = 2 * np.pi / rng.uniform(6, 10) * rng.uniform(0.7, 1.4, 24)
        base2 = np.sin(ws2[:, None] * tI[None] + rng.uniform(0, 6.3, 24)[:, None]).sum(0)
        base2 /= base2.std()
        hs_i = np.interp(tI, t_ep, hs_ep[v])
        roll[v] = k_roll[v] * hs_i * base
        pitch[v] = k_pitch[v] * hs_i * base2

    # ---- constellation and look angles
    con = Constellation(cfg, rng)
    K = cfg["k_max"]
    ns = int(cfg["t_tx"] * hz)
    out = {k: np.zeros((V, nE, K + 1)) for k in
           ["el", "az_rel", "rng", "el_rate", "t_set", "scan0", "sector_ev", "deficit_prev", "has_prev",
            "is_ter", "valid", "y_aggr", "y_rob", "rate", "snr_true", "ux", "uy", "uz"]}
    sat_id = -np.ones((V, nE, K + 1), int)
    prev_def = {}                                          # (vessel, sat) -> last observed beacon deficit
    sector_ev = np.zeros((V, 8))                           # EWMA of blockage evidence per relative-azimuth sector
    ter_duct = np.zeros(V)
    for e in range(nE):
        t = t_ep[e]
        S0, S1 = con.ecef(t), con.ecef(t + cfg["t_tx"])
        el, az, rg, uenu = look_angles(S0, lat_tr[:, e], lon_tr[:, e])
        el1, _, _, _ = look_angles(S1, lat_tr[:, e], lon_tr[:, e])
        el = np.where(con.alive[None], el, -90)
        i0 = int(t * hz)
        for v in range(V):
            vis = np.where(el[v] >= cfg["el_min"])[0]
            vis = vis[np.argsort(-el[v, vis])][:K]
            # body-frame conversion: heading then roll/pitch over the transmission window
            h = np.radians(head[v, e])
            r_s = np.radians(roll[v, i0:i0 + ns])
            p_s = np.radians(pitch[v, i0:i0 + ns])
            om = np.sqrt(np.mean(np.diff(roll[v, i0:i0 + ns]) ** 2 + np.diff(pitch[v, i0:i0 + ns]) ** 2)) * hz
            hs, rr = hs_ep[v, e], rain_ep[v, e]
            for k, s in enumerate(vis):
                u = uenu[v, s]
                fwd = u[0] * np.sin(h) + u[1] * np.cos(h)          # body x (bow)
                stb = u[0] * np.cos(h) - u[1] * np.sin(h)          # body y (starboard)
                up = u[2]
                # deck normal after roll (about x) and pitch (about y): n = (-sin p, sin r cos p, cos r cos p)
                cosang = (-np.sin(p_s)) * fwd + (np.sin(r_s) * np.cos(p_s)) * stb + np.cos(r_s) * np.cos(p_s) * up
                scan = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                scan0 = np.degrees(np.arccos(np.clip(up, -1, 1)))
                l_scan = np.where(scan < cfg["scan_max"], -10 * cfg["scan_exp"] * np.log10(np.cos(np.radians(np.minimum(scan, 89.0)))), 40.0)
                eps = kappa_v[v] * om
                l_point = 12 * (eps / cfg["theta3db"]) ** 2
                sel = np.radians(el[v, s])
                l_atm = 0.3 / np.sin(sel)
                ls = 4.0 / np.sin(sel)
                l_rain = 0.0374 * rr ** 1.15 * ls / (1 + ls * np.cos(sel) / 35.0)
                az_rel = (az[v, s] - head[v, e]) % 360
                blocked = any(abs(((az_rel - c + 180) % 360) - 180) < w and el[v, s] < emax for c, w, emax in blocks[v])
                snr_cs = cfg["snr_ref"] - 20 * np.log10(rg[v, s] / cfg["alt"]) - l_atm - cfg["sat_loss_db"]
                snr_mean = snr_cs - np.mean(l_scan) - l_point - l_rain - (25.0 if blocked else 0.0)
                Kf = 10 ** ((4 + 0.2 * (el[v, s] - 25)) / 10)
                hfade = np.abs(np.sqrt(Kf / (Kf + 1)) + np.sqrt(1 / (Kf + 1)) * (rng.normal() + 1j * rng.normal()) / np.sqrt(2)) ** 2
                scint = rng.normal(0, 0.6 + 2.0 / np.sin(sel) * 0.2)
                snr = snr_mean + 10 * np.log10(hfade) + scint
                worst_scan = np.max(l_scan) - np.mean(l_scan)
                stays = el1[v, s] >= cfg["el_min"] - 2.0
                ya = float(snr - 0.5 * worst_scan >= cfg["gamma_aggr"] and stays)
                yr = float(snr - 0.5 * worst_scan >= cfg["gamma_robust"] and stays)
                # observables: beacon at the start of the epoch (noisy) -> deficit vs clear-sky prediction
                beacon = snr_mean + rng.normal(0, 1.0) + 10 * np.log10(np.abs(np.sqrt(Kf / (Kf + 1)) + np.sqrt(1 / (Kf + 1)) * (rng.normal() + 1j * rng.normal()) / np.sqrt(2)) ** 2)
                deficit = snr_cs - beacon
                key = (v, s)
                o = out
                o["el"][v, e, k], o["az_rel"][v, e, k], o["rng"][v, e, k] = el[v, s], az_rel, rg[v, s]
                o["el_rate"][v, e, k] = (el1[v, s] - el[v, s]) / cfg["t_tx"] * 60.0
                o["t_set"][v, e, k] = (el[v, s] - cfg["el_min"]) / max(1e-3, -(el1[v, s] - el[v, s]) / cfg["t_tx"]) if el1[v, s] < el[v, s] else 900.0
                o["scan0"][v, e, k] = scan0
                o["sector_ev"][v, e, k] = sector_ev[v, int(az_rel // 45) % 8]
                o["deficit_prev"][v, e, k] = prev_def.get(key, 0.0)
                o["has_prev"][v, e, k] = float(key in prev_def)
                o["valid"][v, e, k] = 1.0
                o["y_aggr"][v, e, k], o["y_rob"][v, e, k] = ya, yr
                o["rate"][v, e, k] = cfg["rate_sat"]
                o["snr_true"][v, e, k] = snr
                o["ux"][v, e, k], o["uy"][v, e, k], o["uz"][v, e, k] = u
                sat_id[v, e, k] = s
                prev_def[key] = deficit
                if el[v, s] < 55:
                    sec = int(az_rel // 45) % 8
                    sector_ev[v, sec] = 0.85 * sector_ev[v, sec] + 0.15 * float(deficit > 8.0)
            # coastal 5G candidate (token K)
            dport = haversine(lat_tr[v, e], lon_tr[v, e], PORTS[:, 0], PORTS[:, 1]).min()
            ter_duct[v] = 0.95 * ter_duct[v] + np.sqrt(1 - 0.95 ** 2) * rng.normal(0, 3.0)
            if dport < cfg["ter_range"] * 1.6:
                snr_t = 40 - 30 * np.log10(max(dport, 0.5)) + ter_duct[v] - 0.8 * hs + rng.normal(0, 2.0)
                o = out
                o["valid"][v, e, K], o["is_ter"][v, e, K] = 1.0, 1.0
                o["rng"][v, e, K], o["el"][v, e, K] = dport, 0.0
                o["deficit_prev"][v, e, K] = ter_duct[v]
                o["has_prev"][v, e, K] = 1.0
                o["y_aggr"][v, e, K] = float(snr_t >= cfg["gamma_aggr"])
                o["y_rob"][v, e, K] = float(snr_t >= cfg["gamma_robust"])
                o["rate"][v, e, K] = cfg["rate_ter"]
                o["snr_true"][v, e, K] = snr_t
                o["uz"][v, e, K] = 0.0
    out.update(dict(sat_id=sat_id, hs=hs_ep, rain=rain_ep, lat=lat_tr, lon=lon_tr, head=head, speed=speed,
                    roll=roll, pitch=pitch, t_ep=t_ep, cfg=cfg))
    return out
