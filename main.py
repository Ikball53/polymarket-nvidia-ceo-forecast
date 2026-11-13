"""Polymarket "Next CEO of NVIDIA?" -- a 240-minute-ahead forecast of the
Yes price for every candidate, five quantiles per market per minute, scored
by mean pinball loss. This is the v3e method, rewritten with new names and
wording: the forecasts are unchanged, the practice loss is still 0.230663.

How the median is built (one market, one horizon minute h):
  - It starts at the last price and drifts toward the sum-normalized
    price (last price divided by the sum of all candidates' last prices,
    so the medians respect that the Yes prices partition 1), with weight
    1 - exp(-h/90). A market whose price sits above its fair share is
    pulled down, one below its share pulled up.
  - It undoes the market's own last-tick bounce: the lag-1 autocorrelation
    of one-minute changes over the last 120 minutes measures how much of
    the last tick was noise, and a share of that tick (up to 1.25x the
    autocorrelation effect, scaled by 20) is removed, again with the
    1 - exp(-h/90) weight.
  - It follows the 25-minute trend a little (weight 0.2, on its own
    1 - exp(-h/120) curve).
  - It keeps falling after a hard single-minute drop: when the largest
    one-minute fall in the last 120 minutes is bigger than 0.01, the
    median keeps moving down by 0.3 times the excess drop, with the
    1 - exp(-h/90) weight. Markets that just dropped hard tend to keep
    falling; the older median sat too high for them.
  - It pulls back part of the 60-minute move: 0.2 times the price change
    over the last hour, with the 1 - exp(-h/90) weight. Rises over the
    last hour tend to give some back, and the 25-minute trend is too
    short to see that.

How the spread is built: the realized volatility is the per-minute
standard deviation of price changes over the last 120 minutes, floored
at 0.001 and capped at 0.01, then scaled by the market's level
(sqrt(x*(1-x))/0.2 raised to 0.7 -- partial level scaling, because full
scaling over-widens big markets in quiet cases), floored at 0.001 and
capped at 0.008. The spread is 0.55 times that volatility times
h**0.4, multiplied by per-quantile z values: the base normal z scores
(-1.644854, -0.674490, 0, 0.674490, 1.644854) times outer 1.15 and inner
0.85, then per-quantile factors (0.9, 1.25, 1.0, 0.65, 1.3) that pull
the inner quantiles slightly back toward symmetric. The effective z
values are about (-1.702, -0.717, 0, +0.372, +2.459), still strictly
increasing, so every output row stays non-decreasing. Prices are
clamped to [0, 1] and rounded to six decimals with round-half-up (the
house baseline's Math.round rule; Python's round() is banker's
rounding and would differ on exact half ties).

Practice scores on the six sealed cases, local replica scorer (it
reproduces the house baseline's published 0.363589 exactly, and the
server grades of the earlier entries matched the replica exactly):
the house baseline (persistence plus 0.0025*sqrt(h)) scores 0.363589;
the first entry (median equals last price, volatility quantiles)
0.275425; the second (sum reversion plus bounce reversal) 0.250542;
the third (adds a 30-minute trend and full level scaling) 0.246592;
the third-b variant (partial level scaling plus a retune) 0.244464;
the third-c variant (adds per-quantile asymmetry) 0.238956; this
entry 0.230663. Median error 0.6465 (third-c: 0.6961), 90% coverage
0.8957. Per case: p00 0.3457, p01 0.2829, p02 0.2719, p03 0.2430,
p04 0.1744, p05 0.1099. This entry beats the third-c variant on five
of six cases (p02 is worse, 0.2719 against 0.2616) and on the mean
by 0.0083, about 3.5%.

Robustness, read before submitting: with the coefficients fixed,
leaving one case out wins five of six (only p02 worse, by 0.0103);
re-fitting the two new median coefficients on the other five cases
(the honest out-of-sample test) also wins five of six, with a mean
held-out gain of 0.0039. So the expected gain on the twelve unseen
final cases over the third-c variant is roughly 0.003 to 0.005, not
the in-sample 0.0083. The width and asymmetry retune is below the
noise level and adds nothing honest on top. There is a real downside
on p02-type cases (the sum far below 1 after a leader collapses),
where this entry can lose up to 0.014.

On the 0.20 target: not reached, and the search says it cannot be
reached from price history alone. Nineteen mechanisms were tried in
an earlier round and six more structural ideas in this one (gap
momentum and bounce, down-continuation, small-market explosions,
jump-aware medians, estimator ensembles, jump-frequency widths),
plus a ten-feature linear fit as a ceiling test. Only the two median
terms above are real finds, worth about 0.004 honestly. The residual
diagnostics cap it: no single price feature explains more than about
4% of the median error, and the ten-feature fit is catastrophic out
of sample (leave-one-case-out deltas from +0.01 to +0.45) because the
features are collinear noise. The loss lives in a handful of big
markets making news-driven jumps (the top five markets carry 39% of
the 240-minute median error; for example one candidate falls 0.223
in 240 minutes in p01, another rises 0.144 in p03). An oracle that
uses the actual future price as the median (with the third-b widths)
scores 0.0734, so the median error is the wall, and it only moved
from 0.6961 to 0.6465. Reaching 0.20 would need the median error
cut by about 25%, which needs news information this offline program
cannot access.

What did not help (measured, do not retry): gap momentum or bounce
as the median; jointly redistributing a leader's drop to the other
markets; a logit-space median; a sum-momentum gate on sum reversion
(catastrophic: p02 goes to 0.33-0.62, because the pull-up after a
leader collapse is load-bearing); a sum-level gate; the ten-feature
fit on the residual; down-continuation on a 30-minute window (worse
than 120); asymmetric 60-minute reversion (reverting rises only);
a jump-count term (a gain of 0.0001, noise); horizon-shaped weights
for the new terms (no better than the 1 - exp(-h/90) weight).

Runs offline, standard library only, deterministic: the same case
in gives the same bytes out.
"""

import json
import math
import os
from pathlib import Path

INPUT_FOLDER = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_FOLDER = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The five quantiles every forecast row must carry, in order.
TAU_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Base normal z scores, widened at the tails (outer x1.15) and narrowed
# in the middle (inner x0.85), then the per-quantile factors below: the
# lower-middle quantile widens (x1.25) and the upper-middle narrows
# (x0.65), pulling the inner shape slightly back toward symmetric
# against the previous variant's 1.5 / 0.5.
Z_RAW = [-1.644854 * 1.15, -0.674490 * 0.85, 0.0, 0.674490 * 0.85,
         1.644854 * 1.15]
Z_FACTOR = (0.9, 1.25, 1.0, 0.65, 1.3)
Z_VALUES = [z * zm for z, zm in zip(Z_RAW, Z_FACTOR)]

# Tuning constants. The volatility window and the drop window are both
# 120 minutes; the reversion curve runs on a 90-minute horizon and the
# trend on a 120-minute one; the spread grows as h**0.4.
VOL_WINDOW = 120
REVERSION_TAU = 90.0
TREND_HORIZON = 120.0
WIDTH_SCALE = 0.55
SPREAD_EXPONENT = 0.4
VOL_FLOOR = 0.001
VOL_RAW_CAP = 0.01
VOL_CAP = 0.008
LEVEL_CENTER = 0.2
LEVEL_EXPONENT = 0.7
REVERSAL_MAX = 1.25
TREND_WEIGHT = 0.2
TREND_WINDOW = 25
DROP_CONT_WEIGHT = 0.3
DROP_THRESHOLD = -0.01
REVERT60_WEIGHT = -0.2
DROP_WINDOW = 120


def round_price(value):
    """Clamp to [0, 1] and round to six decimals (0, never -0). Round
    half up, matching the house baseline's Math.round: Python's round()
    rounds half to even and would differ on exact half ties."""
    clipped = min(1.0, max(0.0, value))
    return math.floor(clipped * 1e6 + 0.5) / 1e6 + 0.0


def realized_vol(history, window):
    """Per-minute standard deviation of price changes over the last
    `window` minutes. Falls back to the floor when there is almost no
    history, and the raw volatility is floored and capped."""
    recent = history[-window - 1:] if len(history) > window else history
    if len(recent) < 3:
        return VOL_FLOOR
    changes = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    average = sum(changes) / len(changes)
    variance = sum((c - average) ** 2 for c in changes) / (len(changes) - 1)
    return min(VOL_RAW_CAP, max(VOL_FLOOR, math.sqrt(variance)))


def bounce_reversal(history, window):
    """How much of the last tick was bounce noise, from the lag-1
    autocorrelation of one-minute changes. Negative autocorrelation means
    the last tick was mostly noise, so a share of it should be undone.
    Returns a value between 0 and REVERSAL_MAX."""
    recent = history[-window - 1:] if len(history) > window else history
    if len(recent) < 4:
        return 0.0
    changes = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    average = sum(changes) / len(changes)
    sum_sq = sum((c - average) ** 2 for c in changes)
    if sum_sq <= 0.0:
        return 0.0
    lag_cov = sum((changes[i] - average) * (changes[i + 1] - average)
                  for i in range(len(changes) - 1))
    autocorr = lag_cov / sum_sq
    return max(0.0, min(REVERSAL_MAX, -autocorr * 20.0))


def level_adjusted_vol(history, level):
    """Realized volatility scaled by the market's price level, partially.
    Prediction-market moves scale with sqrt(level * (1 - level)), but
    scaling fully over-widens big markets in mature quiet cases, so the
    scale is raised to LEVEL_EXPONENT (0.7). The raw volatility is
    capped first, then scaled, then floored and capped again."""
    vol = realized_vol(history, VOL_WINDOW)
    factor = (math.sqrt(max(level * (1.0 - level), 1e-4))
              / LEVEL_CENTER) ** LEVEL_EXPONENT
    return min(VOL_CAP, max(VOL_FLOOR, vol * factor))


def worst_drop(history, window):
    """The largest single-minute fall (most negative one-minute change)
    over the last `window` minutes. Returns 0.0 or a negative number."""
    recent = history[-window - 1:] if len(history) > window else history
    if len(recent) < 2:
        return 0.0
    return min(recent[i + 1] - recent[i] for i in range(len(recent) - 1))


def forecast_one_market(market, horizon_minutes, total_last):
    """240 rows of five quantiles for one market.

    The median starts at the last price, drifts toward the
    sum-normalized price, undoes the market's own last-tick bounce,
    follows its 25-minute trend a little, keeps falling after a big
    single-minute drop, and pulls back part of its 60-minute move --
    the reversion, bounce and drop terms on a 1 - exp(-h/90) weight,
    the trend on its own 1 - exp(-h/120) curve. The spread is the
    partially level-scaled realized-volatility shape with the
    asymmetric per-quantile z values. Rows stay non-decreasing: the z
    values increase, the spread is never negative, and round-half-up
    is monotonic."""
    history = market["prices"]
    last_price = history[-1] if history else 0.5
    last_tick = history[-1] - history[-2] if len(history) >= 2 else 0.0
    trend25 = (history[-1] - history[-1 - TREND_WINDOW]
               if len(history) > TREND_WINDOW else 0.0)
    move60 = history[-1] - history[-1 - 60] if len(history) > 60 else 0.0
    worst_move = worst_drop(history, DROP_WINDOW)
    drop_push = (DROP_CONT_WEIGHT * (worst_move - DROP_THRESHOLD)
                 if worst_move < DROP_THRESHOLD else 0.0)
    reversal = bounce_reversal(history, VOL_WINDOW)
    vol = level_adjusted_vol(history, last_price)
    normalized = last_price / total_last if total_last > 0.0 else last_price
    forecast_rows = []
    for minute in range(1, horizon_minutes + 1):
        weight = 1.0 - math.exp(-minute / REVERSION_TAU)
        trend_weight = 1.0 - math.exp(-minute / TREND_HORIZON)
        # Median, term by term: start at the last price, pull toward
        # the sum-normalized price, undo the bounce, follow the trend,
        # keep falling after a hard drop, pull back the 60-minute move.
        median = (last_price + (normalized - last_price) * weight
                  - reversal * last_tick * weight
                  + TREND_WEIGHT * trend25 * trend_weight
                  + drop_push * weight + REVERT60_WEIGHT * move60 * weight)
        spread = WIDTH_SCALE * vol * (minute ** SPREAD_EXPONENT)
        forecast_rows.append([round_price(median + z * spread)
                              for z in Z_VALUES])
    return forecast_rows


def main():
    input_folder = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)
    case_files = sorted(f for f in input_folder.iterdir()
                        if f.is_file() and f.name.endswith(".in"))
    for case_path in case_files:
        case_name = case_path.name[: -len(".in")]
        case_data = json.loads(case_path.read_text(encoding="utf-8"))
        if (len(case_data["quantiles"]) != len(TAU_LEVELS)
                or any(got != want for got, want in
                       zip(case_data["quantiles"], TAU_LEVELS))):
            raise ValueError(
                f"{case_path.name} asks for quantiles "
                f"{case_data['quantiles']}, not {TAU_LEVELS}")
        horizon_minutes = int(case_data["horizon_minutes"])
        total_last = sum(mk["prices"][-1] for mk in case_data["markets"]
                         if mk["prices"])
        market_forecasts = [
            {"id": mk["id"],
             "q": forecast_one_market(mk, horizon_minutes, total_last)}
            for mk in case_data["markets"]
        ]
        output_path = output_folder / f"{case_name}.out"
        output_path.write_text(
            json.dumps({"forecasts": market_forecasts}) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
