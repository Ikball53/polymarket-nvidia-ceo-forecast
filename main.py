"""Polymarket NVIDIA CEO forecast v1: short-horizon realized-volatility quantiles.

Method (measured on the 6 sealed practice cases with the official pinball
loss, mean over every case/market/minute/quantile; lower is better):
  median = the market's last price at the checkpoint (no drift: every
  median-drift variant tried -- EWMA level, 72h-mean reversion, trend --
  scored WORSE, because these prices behave like a random walk with jumps
  and the 72h mean is stale after the Jensen Huang resignation shock);
  quantiles = median + z_tau * sigma * h**0.4, clamped to [0, 1], 6 decimals,
  where sigma is the realized per-minute stdev of price changes over the
  last 120 minutes (floored at 0.0005, capped at 0.02).

Why these numbers (all measured, not guessed):
  - lookback 120 min: the 15-120 min plateau is flat (0.275-0.285); 120 wins
    the full practice mean and, importantly, the shock case p00 (0.4945 vs
    0.5481 for 60 min), because a longer window keeps the shock regime's
    volatility in sigma; lookbacks >= 1440 are catastrophic (0.89-1.05)
    since they import stale high volatility from the markets' first hours.
  - width 0.7 and horizon exponent 0.4 (not the random-walk 0.5): realized
    volatility overstates future moves here, and most of a 240-minute move
    happens early; both knobs win on ALL 6 practice cases individually.
  - no fat-tail outer quantiles (widening them hurts), no sum-to-1
    normalization (helps p00 but hurts p02/p04 -- noise, not a mechanism),
    no winsorized/EWMA sigma (both worse once combined with the narrow
    width), no median drift of any kind.

Practice scores (local replica scorer, validated by reproducing the house
baseline's published 0.363589 exactly):
  house baseline (persistence + 0.0025*sqrt(h)) : 0.363589
  pure persistence (zero width)                 : 0.383752
  this entry (v1)                               : 0.275425
  median error 0.7675 (same as baseline -- same median), coverage_90 0.8221
  per-case: p00 0.4945, p01 0.3145, p02 0.3683, p03 0.2285, p04 0.2067,
  p05 0.1112. Beats the house baseline on every case and on the mean by
  0.088 (24%). Excluding the shock case p00 the mean is 0.2459.

Caveat (the rules say it too): the practice is past prices and pays
nothing, so a practice score is no guide to the final -- the final is 48
hours of prices nobody has seen, and if the CEO is named the window ends
early. This entry is built to be robust to that (short-ish lookback that
adapts to regime, no fitted drift), not to squeeze the practice mean.

Runs offline, stdlib only, deterministic: same case in, same bytes out.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

LOOKBACK_MINUTES = 120   # realized-vol window, minutes
WIDTH = 0.7              # global width scale on sigma
HORIZON_EXP = 0.4        # spread grows as h**0.4, not sqrt(h)
SIGMA_FLOOR = 0.0005     # per-minute sigma floor (flat markets)
SIGMA_CAP = 0.02         # per-minute sigma cap (shock regimes)


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


def forecast_market(market, horizon, spreads):
    """240 rows of 5 quantiles for one market, median = last price.

    `spreads[h-1]` is the half-width scale for minute h (WIDTH * sigma *
    h**HORIZON_EXP), precomputed per market since sigma is per-market.
    Quantiles stay non-decreasing: z is increasing and spread >= 0, and
    round-half-up is monotonic, so the rounded rows stay ordered.
    """
    prices = market["prices"]
    x = prices[-1] if prices else 0.5
    sigma = realized_sigma(prices, LOOKBACK_MINUTES)
    return [[price6(x + z * sigma * spreads[h])
             for z in Z] for h in range(horizon)]


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
        spreads = [WIDTH * (h ** HORIZON_EXP) for h in range(1, horizon + 1)]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, horizon, spreads)}
            for m in case["markets"]
        ]
        out_path = out_dir / f"{case_id}.out"
        out_path.write_text(json.dumps({"forecasts": forecasts}) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
