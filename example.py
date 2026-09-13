"""A VaR line that breaks, and one that does not. No private data.

    pip install -e .
    python example.py

The story: calibrate a 95 percent VaR on a calm period, run it through a
regime where volatility doubles, and watch the static line fail its own
backtest. Adaptive Conformal Inference recovers the coverage without being
told the regime changed.
"""

import numpy as np

from conformalrisk import aci, conformal_quantile, conformal_var, kupiec_pof

ALPHA = 0.05
rng = np.random.default_rng(7)

calm = rng.standard_t(5, 1500) * 0.008
stressed = rng.standard_t(5, 1500) * 0.016

print("Calibration on the calm regime, 1500 observations")
var_line = conformal_var(calm, alpha=ALPHA)
naive = float(np.quantile(calm, ALPHA))
print("  conformal VaR, (n+1) corrected   %+.5f" % var_line)
print("  naive historical-simulation VaR  %+.5f" % naive)
print("  difference                       %+.6f" % (naive - var_line))
print()

print("Where the (n+1) correction actually bites: short calibration sets.")
print("  n      conformal      naive    conformal is more conservative by")
for n in (30, 60, 125, 250, 500, 1500):
    sub = calm[:n]
    c = conformal_var(sub, alpha=ALPHA)
    nv = float(np.quantile(sub, ALPHA))
    print("  %5d  %+.5f  %+.5f   %+.6f" % (n, c, nv, nv - c))
print("A desk calibrating on one quarter of daily data is at n = 60. The")
print("naive quantile quietly loses the finite-sample guarantee there; the")
print("correction is what buys it back, and it costs nothing to apply.")
print()

for label, sample in (("same regime", calm), ("volatility doubled", stressed)):
    breaches = int((sample < var_line).sum())
    k = kupiec_pof(len(sample), breaches, ALPHA)
    print("Static line on %s" % label)
    print("  breaches %d of %d, rate %.4f against an expected %.4f"
          % (breaches, len(sample), k["breach_rate"], k["expected"]))
    print("  Kupiec LR %.2f, p = %.4g, rejected at 5 percent: %s"
          % (k["lr"], k["p_value"], k["reject_5pct"]))
    print()

# Adaptive: recalibrate the miscoverage target online, from a rolling window
# of what has actually been seen, without being told a regime changed.
stream = np.concatenate([calm[-500:], stressed])
window = 500


def recalibrate(alpha_t, t):
    past = stream[max(0, t - window):t]
    if len(past) < 50:
        return -np.inf, np.inf
    q = conformal_quantile(-past, alpha_t)
    return -q, np.inf


res = aci(stream, np.zeros_like(stream), np.zeros_like(stream),
          recalibrate, alpha=ALPHA, gamma=0.02)
print("Adaptive Conformal Inference over the same stream")
print("  realised coverage  %.4f against a target of %.4f"
      % (res["coverage"], res["target"]))
print("  alpha drifted from %.4f to %.4f as the regime turned"
      % (res["alpha_path"][0], res["alpha_path"][-1]))
print()
print("The static line is not wrong, it is stale: its guarantee was")
print("conditional on the calibration set resembling the future, and Kupiec")
print("is what tells you that stopped being true. The adaptive version keeps")
print("its promise without anyone declaring a regime change, which is the")
print("property a risk system actually needs.")
