# Polymarket NVIDIA CEO forecast — v2

Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, 240 minutes ahead,
as 5 quantiles (5/25/50/75/95%) per market per minute. Pinball loss, lower
is better.

## Method

Median(h) = x + (x/s0 − x)·w(h) − mm·d·w(h), w(h) = 1 − exp(−h/120):
- **x** = the market's last price; **s0** = sum of all markets' last prices.
  Yes prices of mutually exclusive outcomes should sum to 1 — when they
  don't, the market corrects toward the proportional normalization x/s0.
- **d** = the last 1-minute change; **mm** = the market's own
  bounce-reversal strength from its lag-1 autocorrelation over the last
  120 minutes (mm = −clip(−ρ·20, 0, 1.25)). 1-minute changes are
  negatively autocorrelated in every practice case (mean −0.038 to
  −0.049): the last tick is mostly bounce noise that reverses.

Quantiles = median + z·0.7·σ·h^0.4, σ = realized per-minute stdev over the
last 120 minutes (floor 0.001, cap 0.01), z = [−1.644854·1.15,
−0.674490·0.9, 0, 0.674490·0.9, 1.644854·1.15]. Clamped to [0, 1], 6 decimals.

## Measured practice scores (local replica scorer, validated by reproducing
the house baseline 0.363589 exactly; v1's server grade matched the replica
exactly at 0.275425)

| entry | practice loss |
|---|---|
| house baseline | 0.363589 |
| v1 (median = last price) | 0.275425 |
| **v2 (this entry)** | **0.250542** |

Per-case v2: p00 0.4124, p01 0.2934, p02 0.2893, p03 0.2350, p04 0.2055,
p05 0.1177. Beats v1 on every case, mean better by 0.0249 (9%).
Median error 0.7122 (v1: 0.7675), coverage_90 0.8935.

## What was tried and rejected

- 72h-mean reversion, EWMA median, trend: much worse (0.94–1.8).
- Empirical forward-change quantiles (historical simulation): 0.375–0.39.
- Logit-space forecasting: 0.29–0.33. Level-scaled sigma: worse.
- EWMA sigma, MAD sigma, adaptive lookback: no better than plain realized σ.
- Fine parameter grid around v1: v1 was already the best grid point.

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Stdlib only, offline, deterministic: same case in, same bytes out.
Runs the 6 practice cases in ~0.3 s total (limit 8 s/case).
