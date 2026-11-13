"""Polymarket NVIDIA CEO forecast v3e: v3c's sum-reverting median + bounce
reversal + short trend + per-tau asymmetry, plus TWO genuinely new median
terms found by residual diagnostics -- down-continuation after a big drop,
and reversion of the 60-minute move. Plus a small width/asymmetry retune.

What is new vs v3c (measured on the 6 sealed practice cases with the
official pinball loss, mean over every case/market/minute/quantile):
  - Down-continuation: after the market's largest single-minute drop in
    the last 120 minutes exceeds 0.01 in size, the median keeps moving
    down by 0.3 * (drop + 0.01), growing with w(h) = 1 - exp(-h/90).
    The residual diagnostic that found it: corr(max_drop, residual) is
    +0.32..+0.34 at h=30/120 -- markets that just dropped hard keep
    falling, and v3c's median sits too high for them. This is a MEDIAN
    continuation, not the post-jump continuation TREND v3c killed (that
    one was symmetric and unthresholded; this one only fires on drops
    bigger than 0.01 and only moves the median down).
  - 60-minute reversion: the median pulls back 0.2 * (price change over
    the last 60 minutes), growing with w(h). Residual beta -0.198 at
    h=120. Rises over the last hour tend to give some back; the 25-min
    trend already in v3c is too short to see it.
  - Width 0.6 -> 0.55 and inner asymmetry q25 x1.5 -> x1.25,
    q75 x0.5 -> x0.65 (the new median shifts the conditional shape
    slightly back toward symmetric). Both sub-noise on their own; kept
    because the package wins the same 5/6 cases as the median terms
    alone with a smaller loss on the one losing case.
  - Nothing else changed: sum reversion, bounce reversal, 25-min trend,
    sigma, caps, outer z values are v3c's exactly.

Practice scores (local replica scorer, validated by reproducing the
house baseline's published 0.363589 exactly; v1's and v2's server
grades matched the replica exactly):
  house baseline (persistence + 0.0025*sqrt(h)) : 0.363589
  v1 (median = last price, vol quantiles)        : 0.275425
  v2 (sum reversion + bounce reversal)           : 0.250542
  v3 (v2 + 30-min trend + full level scaling)    : 0.246592
  v3b (v3 + partial level scaling + retune)      : 0.244464
  v3c (v3b + per-tau asymmetry)                  : 0.238956
  this entry (v3e)                               : 0.230663
  median error 0.6465 (v3c: 0.6961), coverage_90 0.8957
  per-case: p00 0.3457, p01 0.2829, p02 0.2719, p03 0.2430,
  p04 0.1744, p05 0.1099. Beats v3c on 5 of 6 cases (p02 worse:
  0.2719 vs 0.2616) and on the mean by 0.0083 (3.5%).

Robustness (read this before submitting -- it matters):
  - Leave-one-case-out with the FIXED coefficients vs v3c: wins 5/6
    (only p02 worse, +0.0103).
  - Leave-one-case-out with the two median coefficients RE-FIT on the
    other 5 cases (the honest out-of-sample test, median terms only):
    wins 5/6, mean held-out delta -0.0039. So the expected gain on the
    12 UNSEEN final cases over v3c is roughly 0.003 to 0.005, not the
    in-sample 0.0083. The width/asymmetry retune is sub-noise and
    adds nothing honest on top of that. There is a real downside on
    p02-type cases (sum far below 1 after a leader collapse): the
    combo loses up to +0.014 there.

On the 0.20 target: NOT reached, and the search says it is not
reachable from price history alone. v3d tried 19 mechanisms and found
nothing; this round tried the 6 remaining structural ideas (gap
momentum/bounce, down-continuation, small-market explosions, jump-aware
median, estimator ensemble, own jump-frequency width) plus a 10-feature
OLS ceiling test. The two median terms above are the only real finds,
worth ~0.004 honest. The residual diagnostics cap it: no single price
feature explains more than ~4% of the median error, and a 10-feature
OLS fit on the residual is CATASTROPHIC out of sample (LOCO deltas
+0.01 to +0.45) -- the features are collinear noise. The loss lives
in a handful of big markets making news-driven jumps (top 5 markets
carry 39% of the h=240 median error; e.g. p01 candidate-a falls
0.223 in 240 min, p03 johny-srouji rises 0.144). The oracle diagnostic
(median = the actual future price, v3b widths) scores 0.0734, so the
median error is the wall, and it only moved from 0.6961 to 0.6465.
0.20 would need the median error cut by ~25%, which needs news
information this offline program cannot access. If the friend's claimed
0.21 is real on these same 6 cases, it is either a different
measurement or a mechanism that uses information outside the price
history -- no price-history mechanism tried across v1/v2/v3/v3b/v3c/
v3d/v3e gets within 0.02 of it.

What did NOT help in this round (measured, do not retry): gap
momentum/bounce as the median (gap_chg_30 beta +0.201, +0.2% at
h=120; larger coefficients much worse); joint redistribution of a
leader's drop to other markets (all worse); logit-space median
(0.238750, no better than v3c); sum-momentum gate on sum reversion
(catastrophic: p02 0.33-0.62 -- the reversion pull-up after a leader
collapse is load-bearing); sum-level gate (worse); 10-feature OLS on
the residual (catastrophic out of sample); down-continuation on a
30-min window (worse than 120-min); asymmetric 60-min reversion
(revert rises only, worse than symmetric); n_jumps term (+0.0001,
noise); horizon-shaped weights for the new terms (no better than
w(h) with tau 90).

Runs offline, stdlib only, deterministic: same case in, same bytes out.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# v3c base z (outer x1.15 / inner x0.85), then the per-tau multipliers:
# q05 x0.9, q25 x1.25, q50 x1.0, q75 x0.65, q95 x1.3 (v3e retune: the
# inner quantiles move slightly back toward symmetric vs v3c's 1.5/0.5).
Z_BASE = [-1.644854 * 1.15, -0.674490 * 0.85, 0.0, 0.674490 * 0.85,
          1.644854 * 1.15]
Z_MULT = (0.9, 1.25, 1.0, 0.65, 1.3)
# Effective z per tau: [-1.702, -0.717, 0.0, +0.372, +2.459]. Still
# strictly increasing, so every output row stays non-decreasing.
Z = [z * zm for z, zm in zip(Z_BASE, Z_MULT)]

LOOKBACK_MINUTES = 120   # realized-vol window, minutes
SUM_TAU = 90.0           # sum-reversion / reversal horizon, minutes
TREND_TAU = 120.0        # trend horizon curve, minutes
WIDTH = 0.55             # global width scale on sigma (v3c: 0.6)
HORIZON_EXP = 0.4        # spread grows as h**0.4, not sqrt(h)
SIGMA_FLOOR = 0.001      # per-minute sigma floor (flat markets)
SIGMA_RAW_CAP = 0.01     # cap on the realized vol before level scaling
SIGMA_CAP = 0.008        # cap on the level-scaled sigma (shock regimes)
LEVEL_C = 0.2            # sigma *= (sqrt(x*(1-x)) / LEVEL_C) ** LEVEL_GAMMA
LEVEL_GAMMA = 0.7        # partial level scaling (1.0 = v3's full scaling)
MOM_MAX = 1.25           # max bounce-reversal strength
TREND_BETA = 0.2         # weight on the 25-minute trend
TREND_K = 25             # trend window, minutes
DOWN_CONT_BETA = 0.3     # down-continuation weight after a big drop
DOWN_CONT_THR = -0.01    # drop size (per-minute) that triggers it
REVERT60_BETA = -0.2     # weight on reverting the 60-minute move
JUMP_LOOKBACK = 120      # window for the largest single-minute drop


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0).

    Round-half-up, matching the house baseline's Math.round: Python's
    round() is banker's rounding and would differ on exact .5e-6 ties.
    """
    v = min(1.0, max(0.0, x))
    return math.floor(v * 1e6 + 0.5) / 1e6 + 0.0


def realized_sigma(prices, lookback):
    """Per-minute stdev of price changes over the last `lookback` minutes."""
    p = prices[-lookback - 1:] if len(prices) > lookback else prices
    if len(p) < 3:
        return SIGMA_FLOOR
    diffs = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return min(SIGMA_RAW_CAP, max(SIGMA_FLOOR, math.sqrt(var)))


def reversal_strength(prices, lookback):
    """Bounce-reversal strength from the market's own lag-1 autocorrelation
    of 1-minute changes over the last `lookback` minutes. Negative
    autocorrelation means the last tick was mostly bounce noise: undo a
    share of it. Returns a value in [0, MOM_MAX]."""
    q = prices[-lookback - 1:] if len(prices) > lookback else prices
    if len(q) < 4:
        return 0.0
    diffs = [q[i + 1] - q[i] for i in range(len(q) - 1)]
    mean = sum(diffs) / len(diffs)
    v0 = sum((d - mean) ** 2 for d in diffs)
    if v0 <= 0.0:
        return 0.0
    v1 = sum((diffs[i] - mean) * (diffs[i + 1] - mean)
             for i in range(len(diffs) - 1))
    rho = v1 / v0
    return max(0.0, min(MOM_MAX, -rho * 20.0))


def level_scaled_sigma(prices, x):
    """Realized vol scaled by the market's level, partially: prediction-
    market moves scale with sqrt(x*(1-x)), but the full scaling
    over-widens big markets in mature quiet cases, so the scale is
    raised to LEVEL_GAMMA (0.7). Raw vol is capped first, then scaled,
    then floored and capped again."""
    sig = realized_sigma(prices, LOOKBACK_MINUTES)
    scale = (math.sqrt(max(x * (1.0 - x), 1e-4)) / LEVEL_C) ** LEVEL_GAMMA
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sig * scale))


def largest_drop(prices, lookback):
    """Largest single-minute drop (most negative 1-minute change) over the
    last `lookback` minutes. Returns 0.0 or a negative number."""
    q = prices[-lookback - 1:] if len(prices) > lookback else prices
    if len(q) < 2:
        return 0.0
    return min(q[i + 1] - q[i] for i in range(len(q) - 1))


def forecast_market(market, horizon, sum_last):
    """240 rows of 5 quantiles for one market.

    Median drifts toward the sum-normalized price x/sum_last, undoes the
    market's own last-tick bounce, follows its 25-minute trend a little,
    keeps falling after a big single-minute drop, and pulls back part of
    its 60-minute move, with weight w(h) = 1 - exp(-h/90) (trend on its
    own 1 - exp(-h/120) curve). Spread is the partially level-scaled
    realized-vol shape with the v3e asymmetric per-tau z-values. Rows
    stay non-decreasing: z is increasing, spread >= 0, and round-half-up
    is monotonic."""
    prices = market["prices"]
    x = prices[-1] if prices else 0.5
    d = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    t25 = prices[-1] - prices[-1 - TREND_K] if len(prices) > TREND_K else 0.0
    chg60 = prices[-1] - prices[-1 - 60] if len(prices) > 60 else 0.0
    max_drop = largest_drop(prices, JUMP_LOOKBACK)
    down_cont = (DOWN_CONT_BETA * (max_drop - DOWN_CONT_THR)
                 if max_drop < DOWN_CONT_THR else 0.0)
    mm = reversal_strength(prices, LOOKBACK_MINUTES)
    sigma = level_scaled_sigma(prices, x)
    target = x / sum_last if sum_last > 0.0 else x
    rows = []
    for h in range(1, horizon + 1):
        w = 1.0 - math.exp(-h / SUM_TAU)
        wt = 1.0 - math.exp(-h / TREND_TAU)
        med = (x + (target - x) * w - mm * d * w + TREND_BETA * t25 * wt
               + down_cont * w + REVERT60_BETA * chg60 * w)
        s = WIDTH * sigma * (h ** HORIZON_EXP)
        rows.append([price6(med + z * s) for z in Z])
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        case = json.loads(path.read_text(encoding="utf-8"))
        if (len(case["quantiles"]) != len(QUANTILES)
                or any(t != q for t, q in zip(case["quantiles"], QUANTILES))):
            raise ValueError(
                f"{path.name} asks for quantiles {case['quantiles']}, "
                f"not {QUANTILES}")
        horizon = int(case["horizon_minutes"])
        sum_last = sum(m["prices"][-1] for m in case["markets"]
                       if m["prices"])
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, horizon, sum_last)}
            for m in case["markets"]
        ]
        out_path = out_dir / f"{case_id}.out"
        out_path.write_text(json.dumps({"forecasts": forecasts}) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
