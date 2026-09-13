"""Conformal prediction for quant risk: distribution-free intervals and VaR
bounds with finite-sample guarantees, plus the coverage backtests risk teams
actually run.

Three instruments:

  split_conformal_interval : symmetric two-sided intervals around any point
      forecaster, calibrated on held-out absolute residuals. Finite-sample
      marginal coverage >= 1 - alpha, no distributional assumption.
  conformal_var : one-sided lower bound on returns (a VaR line) from a
      calibration set of realised returns or forecast scores. The bound is
      the conformal quantile with the (n+1) small-sample correction, which
      is precisely the correction naive historical-simulation VaR omits.
  aci : Adaptive Conformal Inference (Gibbs and Candes 2021). Recalibrates
      the miscoverage target online, which is what volatility clustering
      demands of any static interval.

Backtest: kupiec_pof, the proportion-of-failures likelihood ratio test.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile of calibration scores.

    This is the quantity that makes split conformal exact in finite samples;
    using the plain (1-alpha) quantile loses the guarantee.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("empty calibration set")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def split_conformal_interval(
    predictions: np.ndarray,
    calib_predictions: np.ndarray,
    calib_actuals: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric intervals prediction +/- q, q from calibration |residuals|."""
    q = conformal_quantile(np.abs(np.asarray(calib_actuals) - np.asarray(calib_predictions)), alpha)
    p = np.asarray(predictions, dtype=float)
    return p - q, p + q


def conformal_var(calib_returns: np.ndarray, alpha: float = 0.05) -> float:
    """One-sided lower bound: P(return < bound) <= alpha, finite sample.

    Scores are losses (negative returns); the bound is minus the conformal
    quantile of losses at level alpha.
    """
    return -conformal_quantile(-np.asarray(calib_returns, dtype=float), alpha)


def aci(
    actuals: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    recalibrate,
    alpha: float = 0.1,
    gamma: float = 0.01,
) -> dict:
    """Adaptive Conformal Inference over a sequence.

    recalibrate(alpha_t, t) -> (lower_t, upper_t) supplies the interval that a
    static method would produce at miscoverage alpha_t (the initial lower and
    upper arrays are ignored beyond their length). alpha_t is updated by
    alpha_{t+1} = alpha_t + gamma (alpha - err_t), err_t = 1{y_t outside}.
    Long-run coverage converges to 1 - alpha whatever the distribution does.
    """
    y = np.asarray(actuals, dtype=float)
    a_t = alpha
    errs, alphas = [], []
    for t in range(len(y)):
        lo, hi = recalibrate(min(max(a_t, 1e-6), 1 - 1e-6), t)
        err = float(not (lo <= y[t] <= hi))
        errs.append(err)
        alphas.append(a_t)
        a_t = a_t + gamma * (alpha - err)
    errs = np.array(errs)
    return {
        "coverage": float(1 - errs.mean()),
        "target": 1 - alpha,
        "alpha_path": np.array(alphas),
        "errors": errs,
    }


def kupiec_pof(n_obs: int, n_breaches: int, alpha: float) -> dict:
    """Kupiec proportion-of-failures test for a VaR line at level alpha.

    H0: the true breach probability equals alpha. LR ~ chi2(1) under H0.
    """
    if not 0 < alpha < 1 or n_obs <= 0 or not 0 <= n_breaches <= n_obs:
        raise ValueError("bad inputs")
    x, n, p = n_breaches, n_obs, alpha
    phat = x / n
    def loglik(prob):
        with np.errstate(divide="ignore"):
            return (n - x) * np.log(1 - prob) + (x * np.log(prob) if x else 0.0)
    lr = -2.0 * (loglik(p) - (loglik(phat) if 0 < phat < 1 else 0.0 if x in (0, n) else loglik(phat)))
    if x == 0:
        lr = -2.0 * (n * np.log(1 - p))
    if x == n:
        lr = -2.0 * (n * np.log(p))
    pval = float(chi2.sf(lr, df=1))
    return {"lr": float(lr), "p_value": pval, "breach_rate": phat, "expected": p,
            "reject_5pct": pval < 0.05}
