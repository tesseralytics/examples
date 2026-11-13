# The two-hour lag discount

Companion code for [The two-hour lag discount: building factors you can actually
trade](https://tesseralytics.dev/blog/the-two-hour-lag-discount).

## What it measures

A factor is a column that ranks the market: open interest, funding, this hour's buy/sell
imbalance. Its edge is the cross-sectional rank correlation (IC) between that column and the
forward return. The trap is that the forward return is always scored from the bar's *close*, an
instant nobody can trade, because a bar's close is its last trade.

So the script builds the same label at increasing entry delays:

```
ret(H, k, L) = close(H + L + k) / close(H + L) - 1
```

`L` moves the entry, never the holding period. Then it reports, per mechanism class, how much of
the zero-delay IC survives.

## Run it

```bash
export TESSERA_API_KEY="sk_..."        # free key: https://tesseralytics.dev

# One month, every listed coin: free tier, ~10 minutes
uv run python lag-discount/lag_discount.py --months 2026-10:2026-10

# The thirteen closed months behind the post's tables (~7.7 GB of minute data)
uv run python lag-discount/lag_discount.py --months 2025-10:2026-10
```

The panel is cached to `/tmp/lag-discount/panel.parquet`; add `--cached` to reuse it (the reads are
the entire runtime, the measurement takes seconds).

Reading a whole month for every coin is a few hundred presigned requests, so the script batches
them. If a batch trips the API rate limit you will see a 429. Lower `CHUNK` near the top of the
file.

## What you should see

At a **two-hour** entry, as a share of the zero-delay IC:

| run | state | microstructure | momentum |
|---|---|---|---|
| one month (2026-10, 291 coins) | 98 % | 20 % | 76 % |
| thirteen months (225 coins) | 99 % | 7 % | 58 % |

The ordering is the result; the digits move with the window. A single month is one regime and one
universe, and thirteen months contain more of both, which is what takes the flow class from a
fifth to almost nothing.

The other two tables are noisier on a short window, by design and by arithmetic:

- **The family bar** (how many of the 66 factor × horizon tests clear the search-wide
  Benjamini-Hochberg threshold at each delay) needs a long sample to be readable: wider standard
  errors on 700 hours mean fewer tests clear any threshold. On 2026-10 three of the 66 tests clear
  it at zero delay and none at two hours; over the thirteen months behind the post it is
  39 → 31 → 23 on the study's universe, and 38 → 30 → 22 on this script's slightly wider one
  (the two agree on the ordering and the class medians).
- **The replication table** re-runs the whole grid on time halves and liquidity tiers. One month
  per half leaves some slices with no factor above the materiality floor at all, and a single
  factor's retention can exceed 100 % in a slice (the two-hour IC came out larger than the
  zero-delay one). Read it on the full window, where it is the check that the class contrast is not
  one period's accident.

## The field classes

Fixed by what each number *is*, before the curve is looked at:

- **state**: a level or cumulative snapshot of where the market stands: `log_open_interest`,
  `log_volume`, `funding_rate`, `cum_funding`, `whale_long_account_share`, and friends.
- **microstructure**: a quantity measured *inside* hour H, describing hour H: `close_vwap_dev`,
  `buysell_imbalance`, `d_cvd`, `pnl_imbalance`.
- **momentum**: a price path: `mom_1h`.

The exception worth knowing: `pnl_imbalance` is computed inside the hour but survives the delay,
because it saturates at ±1 and so carries the last hour's *side* rather than a magnitude. It
behaves like a state flag wearing flow clothes.

## Datasets

| dataset | what it is |
|---|---|
| `gold_ohlcv_1m` | one-minute bars with the order flow split out (buy/sell volume, CVD, aggressor delta, realised PnL by side) |
| `gold_funding_1h` | hourly funding, annualised rate, running cumulative total |
| `gold_positioning_1h` | open interest plus what the account base is doing with it (long/short balance, concentration, whale position, new/closed/flipped counts) |

Only the fields the measurement needs are read; every column is projected at the API.
