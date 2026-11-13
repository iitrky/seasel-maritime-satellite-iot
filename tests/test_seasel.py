"""Checks of the claims the paper relies on.  Run:  python -m pytest -q tests   (or python tests/test_seasel.py)"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from testbed import CFG, Constellation, look_angles, llh_to_ecef, RE  # noqa: E402
from model import init_params, forward                               # noqa: E402
import risk as R                                                     # noqa: E402


def test_constellation_geometry():
    """Satellites sit at the shell altitudes; a satellite seen from its sub-point is at 90 deg elevation."""
    con = Constellation(CFG, np.random.default_rng(0))
    X = con.ecef(1234.0)
    alt = np.linalg.norm(X, axis=1) - RE
    assert set(np.round(alt).astype(int)) == {550, 560}
    assert con.N == 36 * 22 + 12 * 20
    s = X[0]
    lat = np.degrees(np.arcsin(s[2] / np.linalg.norm(s)))
    lon = np.degrees(np.arctan2(s[1], s[0]))
    el, _, rng, _ = look_angles(X[:1], np.array([lat]), np.array([lon]))
    assert abs(el[0, 0] - 90) < 1e-6 and abs(rng[0, 0] - alt[0]) < 1e-6


def test_level_deck_scan_angle_equals_zenith_angle():
    """With zero roll and pitch the array scan angle equals 90 deg minus elevation (Eq. 2)."""
    con = Constellation(CFG, np.random.default_rng(1))
    el, az, _, u = look_angles(con.ecef(0.0), np.array([10.0]), np.array([20.0]))
    vis = el[0] > 25
    scan = np.degrees(np.arccos(u[0, vis, 2]))
    assert np.allclose(scan, 90 - el[0, vis], atol=1e-9)


def test_binomial_pvalue_is_valid():
    """Under H0 (true selective risk > alpha) the p-value of Eq. (27) satisfies Pr[pi <= u] <= u."""
    rng = np.random.default_rng(2)
    alpha, n, u = 0.05, 2000, 0.05
    rej = np.mean([R.binom_p(rng.binomial(n, 0.055), n, alpha) <= u for _ in range(4000)])
    assert rej <= u + 0.01, rej


def test_ltt_certificate_controls_risk():
    """Theorem 1 on synthetic data: the certified threshold violates the target in at most delta of the trials."""
    rng = np.random.default_rng(3)
    alpha, delta, viol = 0.05, 0.1, 0
    trials = 150
    for _ in range(trials):
        s = rng.random(6000)
        err = (rng.random(6000) < 0.25 * (1 - s) ** 2).astype(float)      # failure probability falls with score
        lam = R.ltt(s[:3000], err[:3000], alpha, delta)
        if lam is None:
            continue
        acc = s[3000:] >= lam
        p_true = np.mean(0.25 * (1 - s[3000:][acc]) ** 2)                  # exact conditional risk on test law
        viol += p_true > alpha
    assert viol / trials <= delta, viol / trials


def test_attention_is_permutation_equivariant():
    rng = np.random.default_rng(4)
    p = init_params(12, 7, rng)
    tok, ctx = rng.normal(size=(3, 9, 12)), rng.normal(size=(3, 7))
    mask = np.ones((3, 9)); mask[:, 7:] = 0
    geo = rng.normal(size=(3, 9, 3)); geo /= np.linalg.norm(geo, axis=-1, keepdims=True)
    z = forward(p, tok, ctx, mask, geo)
    perm = np.array([3, 0, 6, 1, 5, 2, 4, 7, 8])
    zp = forward(p, tok[:, perm], ctx, mask[:, perm], geo[:, perm])
    assert np.allclose(z[:, perm][:, :7], zp[:, :7])


def test_physics_bias_reduces_to_plain_attention():
    """With softplus(beta) -> 0 the physics-biased logits (Eq. 19) equal standard attention."""
    rng = np.random.default_rng(5)
    p = init_params(12, 7, rng)
    p["beta"] = np.array([-60.0])
    tok, ctx = rng.normal(size=(2, 9, 12)), rng.normal(size=(2, 7))
    mask = np.ones((2, 9))
    geo = rng.normal(size=(2, 9, 3)); geo /= np.linalg.norm(geo, axis=-1, keepdims=True)
    assert np.allclose(forward(p, tok, ctx, mask, geo, "pba"), forward(p, tok, ctx, mask, geo, "attn"))


def test_online_rule_long_run_bound():
    """Eq. (31): the long-run accepted failure rate approaches alpha under a stationary stream."""
    rng = np.random.default_rng(6)
    s = rng.random(200000)
    err = (rng.random(200000) < 0.3 * (1 - s)).astype(float)
    acc, _ = R.online_adapt(s, err, 0.3, 0.05, eta=0.01)
    assert abs(err[acc].mean() - 0.05) < 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
