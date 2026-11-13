"""Polymarket NVIDIA CEO forecast v2: sum-reverting median + bounce reversal.

Method (measured on the 6 sealed practice cases with the official pinball
loss, mean over every case/market/minute/quantile; lower is better):
  median(h) = x + (x/s0 - x) * w(h) + mm * d * w(h),
    where x is the market's last price, s0 the sum of all markets' last
    prices, w(h) = 1 - exp(-h/120), d the last 1-minute change, and mm the
    market's own bounce-reversal strength (see below);
  quantiles = median + z_tau * 0.7 * sigma * h**0.4, clamped to [0, 1],
    6 decimals, where sigma is the realized per-minute stdev of price
    changes over the last 120 minutes (floored at 0.001, capped at 0.01),
    and z = [-1.644854*1.15, -0.674490*0.9, 0, 0.674490*0.9, 1.644854*1.15]
    (slightly wider outer tails, slightly narrower inner quantiles).

Why these mechanisms (all measured, each wins on most practice cases):
  - Sum reversion: the Yes prices of mutually exclusive outcomes should
    sum to 1, and when they do not the market corrects toward the
    proportional normalization x/s0. Practice sums: p00 1.0785, p01 1.0045,
    p02 0.8075, p03 1.0470, p04 1.0860, p05 1.0195 -- and the gain is
    proportional to |sum - 1|, exactly as the mechanism predicts. This
    alone took v1's 0.275425 to 0.259700.
  - Bounce reversal: 1-minute changes are negatively autocorrelated in
    EVERY practice case (mean lag-1 autocorr -0.038 to -0.049): the last
    tick is mostly bid-ask bounce noise that reverses. mm undoes it, with
    strength per market from its own estimated lag-1 autocorrelation over
    the last 120 minutes: mm = -clip(-rho*20, 0, 1.25), so a market whose
    changes reverse strongly gets a strong undo and a quiet market gets
    almost none. A fixed mm = -1.0 also works (0.253782); the per-market
    version is better (0.250854) because liquid and illiquid candidates
    bounce differently.
  - Sigma cap 0.01: shock-regime volatility must not inflate the spread
    (v1 0.275425 -> 0.270081 on its own). Sigma floor 0.001 (not 0.0005):
    flat markets need a minimum width; confirmed in three independent
    configs. Outer 1.15 / inner 0.9: the true move distribution is
    slightly fatter-tailed than normal at 5%/95% and tighter at 25%/75%.

Practice scores (local replica scorer, validated by reproducing the house
baseline's published 0.363589 exactly; v1's server grade matched the
replica exactly at 0.275425, rank 57/159):
  house baseline (persistence + 0.0025*sqrt(h)) : 0.363589
  v1 (median = last price, vol quantiles)        : 0.275425
  this entry (v2)                                : 0.250542
  median error 0.7122 (v1: 0.7675), coverage_90 0.8935
  per-case: p00 0.4124, p01 0.2934, p02 0.2893, p03 0.2350, p04 0.2055,
  p05 0.1177. Beats v1 on every case and on the mean by 0.0249 (9%).
  On the practice leaderboard 0.2505 would be rank 1 (top is 0.26) --
  but the final is 12 cases on 48 hours of prices nobody has seen, so
  no rank is promised; what is promised is that every mechanism here is
  structural (arbitrage, microstructure, tail shape), not fitted drift:
  no 72h-mean reversion, no EWMA median, no empirical-quantile fitting
  (all tried, all worse).

Caveat on the 0.25 target: 0.250542 is a hair above 0.25, and the
plateau across ~200 measured variants is 0.2505-0.2525 -- the remaining
gap is within noise. The oracle diagnostic (median = actual future
price, v1 widths) scores 0.0765, so 0.2/0.1 are not theoretically
impossible, but they would need a median forecast far better than
last-price + sum reversion + bounce reversal, and no such structure was
found in the data. Do not expect 0.2.

Runs offline, stdlib only, deterministic: same case in, same bytes out.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854 * 1.15, -0.674490 * 0.9, 0.0, 0.674490 * 0.9, 1.644854 * 1.15]

LOOKBACK_MINUTES = 120   # realized-vol window, minutes
SUM_TAU = 120.0          # sum-reversion / reversal horizon, minutes
WIDTH = 0.7              # global width scale on sigma
HORIZON_EXP = 0.4        # spread grows as h**0.4, not sqrt(h)
SIGMA_FLOOR = 0.001      # per-minute sigma floor (flat markets)
SIGMA_CAP = 0.01         # per-minute sigma cap (shock regimes)
MOM_MAX = 1.25           # max bounce-reversal strength


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
    return min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(var)))


def reversal_strength(prices, lookback):
    """Bounce-reversal strength from the market's own lag-1 autocorrelation
    of 1-minute changes over the last `lookback` minutes. Negative
    autocorrelation means the last tick was mostly bounce noise: undo a
    share of it. Returns a value in [0, MOM_MAX] (the strength; the
    median subtracts strength * last change)."""
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


def forecast_market(market, horizon, sum_last):
    """240 rows of 5 quantiles for one market.

    Median drifts toward the sum-normalized price x/sum_last and undoes
    the market's own last-tick bounce, both with weight w(h) = 1 -
    exp(-h/120). Spread is the v1 realized-vol shape with the v2
    quantile z-values. Rows stay non-decreasing: z is increasing,
    spread >= 0, and round-half-up is monotonic."""
    prices = market["prices"]
    x = prices[-1] if prices else 0.5
    d = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    mm = reversal_strength(prices, LOOKBACK_MINUTES)
    sigma = realized_sigma(prices, LOOKBACK_MINUTES)
    target = x / sum_last if sum_last > 0.0 else x
    rows = []
    for h in range(1, horizon + 1):
        w = 1.0 - math.exp(-h / SUM_TAU)
        med = x + (target - x) * w - mm * d * w
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
