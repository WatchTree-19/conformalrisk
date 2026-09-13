import numpy as np
import pytest
from scipy.stats import chi2

from conformalrisk import aci, conformal_quantile, conformal_var, kupiec_pof, split_conformal_interval


def test_conformal_quantile_small_sample_by_hand():
    # n=4, alpha=0.25: k = ceil(5 * 0.75) = 4 -> the largest score.
    assert conformal_quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.25) == 4.0
    # alpha=0.5: k = ceil(5*0.5) = 3 -> third smallest.
    assert conformal_quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.5) == 3.0
    # alpha too small for n: infinite interval, guarantee preserved honestly.
    assert conformal_quantile(np.array([1.0, 2.0]), 0.01) == float("inf")


def test_split_conformal_finite_sample_coverage():
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(200):
        calib = rng.standard_t(df=4, size=99)     # heavy tails, no assumption used
        test = rng.standard_t(df=4, size=100)
        lo, hi = split_conformal_interval(np.zeros(100), np.zeros(99), calib, alpha=0.1)
        cover.append(np.mean((test >= lo) & (test <= hi)))
    # Marginal coverage guarantee: mean coverage >= 90% (up to MC noise).
    assert np.mean(cover) >= 0.895


def test_conformal_var_bound_holds():
    rng = np.random.default_rng(1)
    breaches = []
    for _ in range(200):
        calib = rng.standard_t(df=3, size=499) * 0.01
        test = rng.standard_t(df=3, size=250) * 0.01
        var = conformal_var(calib, alpha=0.05)
        breaches.append(np.mean(test < var))
    assert np.mean(breaches) <= 0.055


def test_kupiec_closed_form():
    # 250 obs, 5% VaR, 10 breaches: the standard textbook case.
    out = kupiec_pof(250, 10, 0.05)
    x, n, p = 10, 250, 0.05
    phat = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
               - (n - x) * np.log(1 - phat) - x * np.log(phat))
    assert out["lr"] == pytest.approx(lr, rel=1e-12)
    assert out["p_value"] == pytest.approx(float(chi2.sf(lr, 1)), rel=1e-12)
    assert not out["reject_5pct"]


def test_kupiec_rejects_bad_var():
    out = kupiec_pof(250, 30, 0.05)   # 12% breaches against a 5% line
    assert out["reject_5pct"]


def test_kupiec_zero_breaches_edge():
    out = kupiec_pof(250, 0, 0.05)
    assert out["lr"] == pytest.approx(-2 * 250 * np.log(0.95), rel=1e-12)


def test_aci_recovers_coverage_under_shift():
    rng = np.random.default_rng(2)
    # Volatility doubles halfway: a static interval undercovers the back half.
    y = np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 2, 500)])
    calib = rng.normal(0, 1, 500)                 # calibrated on the QUIET regime

    def static(alpha_t, t):
        q = conformal_quantile(np.abs(calib), alpha_t)
        return -q, q

    static_res = aci(y, None, None, static, alpha=0.1, gamma=0.0)   # gamma 0 = no adaptation
    adaptive_res = aci(y, None, None, static, alpha=0.1, gamma=0.02)
    assert static_res["coverage"] < 0.88          # static breaks under the shift
    assert abs(adaptive_res["coverage"] - 0.9) < 0.02   # ACI holds the line
