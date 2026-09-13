# conformalrisk

Distribution-free uncertainty for quant risk. Prediction intervals and VaR
bounds that carry a **finite-sample** coverage guarantee, with no
distributional assumption, plus the backtest that tells you when the
guarantee has stopped holding.

Quant risk still largely runs on point estimates and on quantiles that assume
the future resembles the calibration window. Conformal prediction gives a
calibrated "I do not know" band instead, and Adaptive Conformal Inference
keeps that band honest when the regime turns.

## Install

    pip install -e .

Python 3.10 or newer, numpy and scipy. Nothing else.

## What is in it

`conformal_quantile(scores, alpha)` is the `ceil((n+1)(1-alpha))/n` empirical
quantile. That `(n+1)` is the whole game: it is what makes split conformal
exact in finite samples, and it is precisely what naive historical-simulation
VaR omits.

`conformal_var(calib_returns, alpha)` is a one-sided lower bound with
`P(return < bound) <= alpha`, finite sample, no asymptotics.

`split_conformal_interval(...)` wraps any point forecaster in symmetric
intervals calibrated on held-out absolute residuals.

`aci(...)` is Adaptive Conformal Inference (Gibbs and Candes 2021), which
updates the miscoverage target online and so converges to the right long-run
coverage whatever the distribution does.

`kupiec_pof(n_obs, n_breaches, alpha)` is the proportion-of-failures
likelihood ratio test, which is how a risk team finds out a VaR line has gone
stale.

## Use

```python
from conformalrisk import conformal_var, kupiec_pof

var_line = conformal_var(calibration_returns, alpha=0.05)
breaches = int((live_returns < var_line).sum())
print(kupiec_pof(len(live_returns), breaches, 0.05))
```

## What the example shows

`python example.py` calibrates a 95 percent VaR on a calm regime, then runs
it through one where volatility doubles.

    Static line on same regime
      breaches 74 of 1500, rate 0.0493 against an expected 0.0500
      Kupiec LR 0.01, p = 0.9055, rejected at 5 percent: False

    Static line on volatility doubled
      breaches 238 of 1500, rate 0.1587 against an expected 0.0500
      Kupiec LR 243.08, p = 8.395e-55, rejected at 5 percent: True

    Adaptive Conformal Inference over the same stream
      realised coverage  0.9490 against a target of 0.9500

The static line is not wrong, it is stale. Its guarantee was conditional on
the calibration set resembling the future, Kupiec is what tells you that
stopped being true, and the adaptive version keeps its promise without anyone
having to declare a regime change.

The example also shows where the `(n+1)` correction actually matters, which
is short calibration sets:

    n      conformal      naive    conformal is more conservative by
       30  -0.02743  -0.01426   +0.013170
       60  -0.01947  -0.01621   +0.003263
      250  -0.01997  -0.01826   +0.001707
     1500  -0.01714  -0.01714   +0.000003

A desk calibrating on one quarter of daily data is at n = 60. At n = 1500 the
correction is worth three millionths and you would never notice; at n = 30 it
is worth more than a percentage point of VaR. It costs nothing to apply
either way.

## Tests

    python -m pytest tests -q

They pin the quantile index by hand, the finite-sample coverage guarantee
under heavy tails, the Kupiec closed form, and ACI restoring coverage under a
volatility doubling that breaks the static interval.

## References

Vovk, V., Gammerman, A. and Shafer, G. (2005). *Algorithmic Learning in a
Random World*.

Gibbs, I. and Candes, E. (2021). Adaptive Conformal Inference Under
Distribution Shift. *NeurIPS*.

Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement
Models. *Journal of Derivatives*.

## Licence

MIT. Sandeep Singh Rai, ORCID 0009-0001-3360-9205.
