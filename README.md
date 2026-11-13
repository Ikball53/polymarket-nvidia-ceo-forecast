# Polymarket NVIDIA CEO forecast — v1

Forecast the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event, 4 hours ahead, as five quantiles (5/25/50/75/95%) per minute.

## Method

Median = the market's last price at the checkpoint. No drift: every
median-drift variant tried (EWMA level, 72h-mean reversion, trend) scored
worse, because these prices behave like a random walk with jumps and the
72h mean is stale after the Jensen Huang resignation shock.

Quantiles = median + z_tau * sigma * h**0.4, clamped to [0, 1], rounded to
6 decimals (round-half-up, like the house baseline). sigma is the realized
per-minute stdev of price changes over the last 120 minutes (floored at
0.0005, capped at 0.02).

Parameters, all measured on the 6 sealed practice cases (not guessed):
- lookback 120 min: the 15–120 min plateau is flat (0.275–0.285); 120 wins
  the full mean and the shock case p00. Lookbacks >= 1440 are catastrophic
  (0.89–1.05) — they import stale high volatility from the markets' first
  hours.
- width 0.7 and horizon exponent 0.4 (not the random-walk 0.5): realized
  volatility overstates future moves, and most of a 240-minute move happens
  early. Both knobs win on all 6 practice cases individually.
- No fat-tail outer quantiles (widening them hurts), no sum-to-1
  normalization (helps p00, hurts p02/p04 — noise, not a mechanism), no
  winsorized/EWMA sigma, no median drift.

## Practice scores (official pinball loss, lower is better)

Measured with a local replica scorer that reproduces the house baseline's
published practice loss of 0.363589 exactly (validation that the replica,
including the answers rebuilt from Polymarket's public price history, is
correct).

| entry | loss | median err | coverage_90 |
|---|---|---|---|
| house baseline | 0.363589 | 0.7675 | 0.9496 |
| pure persistence | 0.383752 | 0.7675 | 0.3264 |
| **this entry (v1)** | **0.275425** | 0.7675 | 0.8221 |

Per-case v1: p00 0.4945, p01 0.3145, p02 0.3683, p03 0.2285, p04 0.2067,
p05 0.1112. Beats the house baseline on every case; mean better by 0.088
(24%). Excluding the shock case p00 the mean is 0.2459.

## Honest caveat

The rules say it and it is true: the practice is past prices and pays
nothing, so a practice score is no guide to the final. The final is 48
hours of prices nobody has seen, and if the CEO is named the window ends
early. This entry is built to be robust to that (short-ish lookback that
adapts to regime, no fitted drift), not to squeeze the practice mean.
A prize also needs a loss strictly below the best house entry on the final,
which is a different dataset.

## Run

```sh
python3 main.py  # reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Stdlib only, no network, deterministic: same case in, same bytes out.
Runs in well under a second per case (limit is 8 s).
