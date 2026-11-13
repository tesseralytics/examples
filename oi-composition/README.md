# Open interest has no direction. Its composition does.

Companion code for [Open interest has no direction. Its composition
does.](https://tesseralytics.dev/blog/oi-composition).

## What it measures

A perpetual's ledger is two-sided and complete: every contract has one long and one short, so a
one-sided open interest figure counts each contract exactly once and moves by the same amount
whether the position that opened was long or short. The level cannot carry direction. The
composition of the accounts behind the change can.

`gold_positioning_1h` publishes the transitions, and the script turns them into scale-free fields:

| field | definition | what it says |
|---|---|---|
| `doi_rel` | `d_open_interest / prev open interest` | the level change, as a share of the market |
| `n_new_tilt` | `(n_new_longs − n_new_shorts) / arrivals` | which side is arriving |
| `n_close_tilt` | `(n_closed_shorts − n_closed_longs) / exits` | which side is leaving |
| `n_flip_tilt` | `(n_flipped_s2l − n_flipped_l2s) / flips` | which side is being reversed |
| `flip_share` | `flips / all transitions` | how much of the turnover is a reversal |
| `churn` | `transitions / unique accounts` | how much of the base moved at all |

plus two taker-side fields built from the same tile: `oi_flow_rel` is taker open minus taker close
over the previous hour's open interest, and `taker_flip_oi_share` is aggressive flip volume in both
directions over the same denominator. Every one is a ratio, so a $5 M/hour market and a $20 k/hour
market sit on one scale.

Two definitions worth knowing before reading a number. `n_closed_*` **includes** flips (a flip
closes one position and opens another), so the exit side is a superset of the flip bucket, not a
third disjoint one, and a flip legitimately appears in two fields. And "arrivals" means account
counts, not size: a hundred one-lot accounts and one whale hour are different markets, which is why
the taker-side fields exist beside the count ones.

The script prints, in order: the population the floor leaves; the ledger identities (verified, not
assumed, on the rows where each is testable); the gross activity behind the net position change and
the flip-versus-taker correlations; the per-panel rank IC table with a block-bootstrap `t` and one
Benjamini-Hochberg bar across the full family (eight fields × three horizons × two entry lags,
forty-eight tests), with only the two-hour rows displayed; what the arrival tilt is *not* (the level
change, momentum, or a slow property); and the tercile spread of the tilt against its own turnover
cost.

## Run it

```bash
export TESSERA_API_KEY="sk_..."        # free key: https://tesseralytics.dev

# One month, every listed coin: free tier
uv run python oi-composition/oi_composition.py --months 2026-10:2026-10

# The thirteen months behind the post's tables
uv run python oi-composition/oi_composition.py --months 2025-10:2026-10
```

The panel is cached to `/tmp/tessera-oi-composition/panel-v3-<span>.parquet`, keyed by the month
span and the derivation version, and reused when it exists. `--cached` requires that cache and fails
instead of reading if the span has none (useful without a key); `--rebuild` ignores the cache and
re-reads. The reads are the entire runtime; the measurement takes seconds. Every coin is one
presigned request, so the reads are batched; if a batch trips the API rate limit you will see a 429:
lower `--chunk`.

## Two panels, never pooled

The crypto perpetuals and the HIP-3 markets (`xyz:*`, tokens on equities and commodities) are
floored, measured and printed separately, and the script never averages them. That is not
fastidiousness. The two market structures are different: one underlying trades on the same 24/7
book as its token, the other closes at night, and the composition fields respond to them
differently, in one case in opposite directions. Pooling the two panels and reporting a between-class
component turns two opposed results into one average that neither market exhibits.

The floor is set once, on the whole labelled universe, before the split: at least half the hours of
the best-covered market anywhere in the file, and a median hourly traded notional of at least $10 k.
Below it the account counts are single digits and a "tilt" is the rounding of one account. One rule,
applied once, so no panel can be floored into a result.

A one-month run keeps a much larger HIP-3 cohort than the published thirteen months (78 markets
against 35): the coverage floor is relative to the length of the window, and a market that listed
mid-month passes a 743-hour test and fails a 9,503-hour one. The window length explains it; the
method is unchanged.

## What one month can and cannot show

A free-key run over 2026-10 (97 crypto markets on 72,071 coin-hours, 78 HIP-3 markets on 57,070,
743 hours in each panel) prints, against the post's thirteen-month table:

| field, entry +2h | one month, crypto | thirteen months, crypto | agree? |
|---|---|---|---|
| `doi_rel` | −0.007 to +0.001, no horizon significant | nothing at any horizon | yes, a null |
| `n_new_tilt` | +0.0005 / +0.0018 / −0.0039, none significant | +0.0035 at 6h, t 2.59 | **no**, the effect is too small for 720 hours |
| `n_close_tilt` | −0.006 to 0.000, none survives BH | nothing | yes, a null |
| `churn` | −0.028 / −0.050 / −0.085, all t ≤ −3.9 | −0.014 / −0.024 / −0.029, all BH-significant | yes, same sign, larger |
| `taker_flip_oi_share` | −0.016 / −0.028 / −0.045, all BH-significant | −0.007 / −0.012 / −0.020 | yes, same sign |

One field the table leaves out is the sharper warning about a month. `flip_share` reads +0.036
(t 2.50) at 24 hours and clears the 48-test BH bar; over thirteen months the post reads −0.0014
there and finds nothing. A short span does not only lose power: it can mint a significant result
in a field the long run calls a null.

The HIP-3 panel is not readable on a month: every one of its twenty-four lag-2 cells sits inside
`|t|` 2.9 and none survives the family bar, against two of its twenty-four over thirteen months (five
of forty-eight once both entry lags are counted). That panel's whole contribution is a comparison,
and a month does not carry it.

Concretely:

- **The identities: exactly.** They are properties of the ledger and reproduce on any window. Run it
  and you should see `verified == testable` on every row, with a worst discrepancy of 0. If
  `Δn_long ≡ n_new_longs − n_closed_longs` does not hold on your rows, something in your read is
  wrong.
- **The gross/net ratios and flip-versus-taker correlations: exactly.** They are also ledger
  properties, not estimates: the median account transitions per unit of net position change, the
  taker volume opened plus closed per unit of `|ΔOI|`, and the per-coin rank correlation of each flip
  count against aggressive flip volume and against plain closes.
- **`churn` and `taker_flip_oi_share` in crypto: yes**, same sign and same order of magnitude as the
  published table. Their effects are large enough that a month resolves them.
- **The arrival tilt in crypto: no.** The effect is a few thousandths of rank IC; 720 hours cannot
  separate it from zero reliably, and here it does not even keep its sign across horizons. Read it
  on the long window, where the post's table comes from.
- **Anything session-based: no.** 720 hours split three ways is not a session study.

Whether a number here agrees with the post depends on the span. If they disagree on the crypto tilt
or on the HIP-3 comparisons, the post is citing thirteen months. The docstring says which span each
published figure belongs to.

## Not in this script

The session decomposition, the flip-spike event studies (trailing robust z against matched controls),
the OI-move × composition split cells, and the time-and-liquidity robustness splits. The post says
which run each of those numbers belongs to; this script is the part a free key can check.

## Datasets

| dataset | what it is |
|---|---|
| `gold_positioning_1h` | open interest plus what the account base is doing with it: long/short/closed/flipped account counts, unique users, oi opened and closed on the taker side |
| `gold_ohlcv_1m` | one-minute bars; read for the hourly close and the hour's notional, which the liquidity floor uses |

Only the fields the measurement needs are read; every column is projected at the API. If the reads
are your bottleneck, that is the axis to cut, not the width of the cross-section. Rank IC needs
breadth: dropping coins to save on price data turns a measurement into noise.
