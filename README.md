# Polymarket NVIDIA CEO forecast v3e

Forecast the Yes price of every trading market of Polymarket's "Next CEO
of NVIDIA?" event at every minute of the 4 hours after a checkpoint, as
five quantiles (5/25/50/75/95%), for the lowest mean pinball loss.

## Method

Median(h) = x + (x/s0 - x) * w(h) - mm * d * w(h) + 0.2 * t25 * wt(h)
             + 0.3 * (max_drop + 0.01) * w(h) [only if max_drop < -0.01]
             - 0.2 * chg60 * w(h),
with w(h) = 1 - exp(-h/90), wt(h) = 1 - exp(-h/120):

- **Sum reversion** (from v2): the Yes prices of mutually exclusive
  outcomes should sum to 1; the median drifts toward the proportional
  normalization x/s0.
- **Bounce reversal** (from v2): 1-minute changes are negatively
  autocorrelated (bid-ask bounce); the median undoes a share of the
  last tick, with strength from each market's own lag-1 autocorrelation.
- **25-minute trend** (from v3, window shortened 30 -> 25): the
  market's change over the last 25 minutes keeps going, weakly.
  Small weight (0.2).
- **Down-continuation** (new in v3e): after the market's largest
  single-minute drop in the last 120 minutes exceeds 0.01, the median
  keeps falling by 0.3 * (drop + 0.01). Found by residual diagnostics:
  corr(max_drop, residual) is +0.32..+0.34 at h=30/120. A MEDIAN
  continuation, not v3c's killed post-jump continuation trend (that one
  was symmetric and unthresholded).
- **60-minute reversion** (new in v3e): the median pulls back 0.2 *
  (price change over the last 60 minutes). Residual beta -0.198 at
  h=120. Rises over the last hour tend to give some back.

Quantiles = median + z * 0.55 * sigma * h^0.4, where sigma is the
120-minute realized per-minute stdev, capped at 0.01, then **partially
level-scaled**: sigma *= (sqrt(x*(1-x)) / 0.2) ** 0.7, floored at 0.001
and capped at 0.008. **Per-tau asymmetry** (from v3c, retuned): the z
values are multiplied per quantile: q05 x0.9, q25 x1.25, q50 x1.0,
q75 x0.65, q95 x1.3, giving effective z = [-1.702, -0.717, 0, +0.372,
+2.459]. Width 0.6 -> 0.55.

## Practice score (local replica scorer, exact vs the house baseline)

| entry | loss |
|---|---|
| house baseline | 0.363589 |
| v1 | 0.275425 |
| v2 | 0.250542 |
| v3 | 0.246592 |
| v3b | 0.244464 |
| v3c | 0.238956 |
| **this entry (v3e)** | **0.230663** |

median error 0.6465 (v3c: 0.6961), coverage_90 0.8957; per-case
p00 0.3457, p01 0.2829, p02 0.2719, p03 0.2430, p04 0.1744, p05 0.1099.
Beats v3c on 5 of 6 cases (p02 worse: 0.2719 vs 0.2616) and on the mean
by 0.0083 (3.5%).

## Robustness -- read before submitting

- Leave-one-case-out with the FIXED coefficients vs v3c: wins 5/6
  (only p02 worse, +0.0103).
- Leave-one-case-out with the two median coefficients RE-FIT on the
  other 5 cases (the honest out-of-sample test): wins 5/6, mean
  held-out delta -0.0039. So the expected gain on 12 UNSEEN final
  cases over v3c is roughly 0.003 to 0.005, not the in-sample 0.0083.
  The width/asymmetry retune is sub-noise and adds nothing honest on
  top. Real downside on p02-type cases (sum far below 1 after a leader
  collapse): the combo loses up to +0.014 there.

## What did NOT help in this round

Gap momentum/bounce as the median; joint redistribution of a leader's
drop to other markets; logit-space median (0.238750); sum-momentum gate
on sum reversion (catastrophic: p02 0.33-0.62); sum-level gate; a
10-feature OLS fit on the residual (catastrophic out of sample, LOCO
+0.01 to +0.45); down-continuation on a 30-min window; asymmetric
60-min reversion (revert rises only); n_jumps term (+0.0001, noise);
horizon-shaped weights for the new terms.

## On the 0.20 target

NOT reached, and the search says it is not reachable from price
history alone. v3d tried 19 mechanisms and found nothing; this round
tried the 6 remaining structural ideas plus a 10-feature OLS ceiling
test. The two median terms above are the only real finds, worth ~0.004
honest. No single price feature explains more than ~4% of the median
error. The loss lives in a handful of big markets making news-driven
jumps (top 5 markets carry 39% of the h=240 median error). The oracle
diagnostic (median = the actual future price, v3b widths) scores
0.0734, so the median error is the wall, and it only moved from
0.6961 to 0.6465. 0.20 would need the median error cut by ~25%,
which needs news information this offline program cannot access. If the
friend's claimed 0.21 is real on these same 6 cases, it is either a
different measurement or a mechanism that uses information outside the
price history -- no price-history mechanism tried across v1/v2/v3/v3b/
v3c/v3d/v3e gets within 0.02 of it.
