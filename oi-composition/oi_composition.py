#!/usr/bin/env python
"""Does open interest tell you who arrived, or only that something did?

A perpetual's open interest is one-sided. Every contract has a long and a short,
and the published figure counts each contract once, so it moves by exactly the
same amount whether the position that opened was long or short. The level cannot
carry direction. The *composition* of the accounts behind the change can.

`gold_positioning_1h` publishes the transitions: how many accounts opened long,
opened short, closed long, closed short, and flipped. This script turns them into
six scale-free fields, then asks whether any of them ranks the forward return.

    doi_rel            the level change, as a share of the previous hour's open interest
    n_new_tilt         (new longs − new shorts) / arrivals: which side is arriving
    n_close_tilt       (closed shorts − closed longs) / exits: which side is leaving
    n_flip_tilt        (shorts flipped out − longs flipped out) / flips
    flip_share         flips / all transitions: how much of the turnover is a reversal
    churn              transitions / unique accounts: how much of the base moved at all

plus two taker-side size fields built from the same tile (`oi_flow_rel`,
`taker_flip_oi_share`). Every one is a ratio, so a $5 M/hour market and a
$20 k/hour market sit on one scale.

Five rules make the answer mean something:

  * **two panels, never pooled.** `xyz:*` markets trade a 24/7 token against an
    underlying whose cash market closes at night. They are labelled, floored
    separately and measured separately: a cross-sectional average over two market
    structures is how a signal gets invented rather than found.
  * **a stated floor.** A market needs half the hours of its panel's best-covered
    member and a median hourly traded notional of at least $10 k. Below that the
    account counts are single digits and a "tilt" is the rounding of one account.
  * **gap-guarded returns.** The forward return is null unless every hour of the
    holding period is published, so a data hole cannot pass as a short return.
  * **a real entry.** The return is entered two hours after the bar closes: the
    first hour-aligned entry a client can take, not the bar's own close.
  * **one significance bar.** The entry-lag test flavour (`t`) is a block
    bootstrap over hours, because overlapping forward returns are serially
    dependent, and Benjamini-Hochberg is applied across the whole family per
    panel, not per row: eight fields × three horizons × both entry lags (0 and
    2), forty-eight tests. Only the two-hour rows are printed, but they are
    judged against all forty-eight.
  * **the ledger checked before it is measured.** The count identities are
    verified on the rows where they are testable, the gross activity behind the
    net change is reported, and flip counts are correlated against aggressive
    flip volume and plain closes before the word "capitulation" is used.

What this script does *not* reproduce: the session decomposition, the event
studies (flip spikes against matched controls), the OI-move × composition split
cells, and the time-and-liquidity robustness splits. The post says which run
each of those numbers belongs to.

Companion code for the post "Open interest has no direction. Its composition
does.":

    https://tesseralytics.dev/blog/oi-composition

Setup:

    export TESSERA_API_KEY="sk_..."     # free key, no card: https://tesseralytics.dev

One month on a free key (the reads dominate the runtime):

    uv run python oi-composition/oi_composition.py --months 2026-10:2026-10

Thirteen months (needs the history bundle behind the 30-day free window):

    uv run python oi-composition/oi_composition.py --months 2025-10:2026-10
"""

from __future__ import annotations

import argparse
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import polars as pl
import tessera

#: Forward holding periods, hours. One, six and twenty-four: the three the post
#: reports, spanning "this hour" to "tomorrow".
HORIZONS = (1, 6, 24)
#: Entry delay, hours. Zero is the bar's own close, an instant nobody can trade.
#: Two is the first hour-aligned entry after the bar becomes readable.
LAG = 2
#: The entry lags in the significance family. Both sit in the family-wide BH bar;
#: only `LAG` rows print, so the zero-lag column cannot be chosen after the fact.
FAMILY_LAGS = (0, LAG)

#: The family. Signed tilts say which way the holder base moved, flip_share and
#: churn how much it turned over; the last two ask that on the taker side.
FIELDS = [
    "doi_rel",
    "n_new_tilt",
    "n_close_tilt",
    "n_flip_tilt",
    "flip_share",
    "churn",
    "oi_flow_rel",
    "taker_flip_oi_share",
]

#: A coin needs this share of its panel's best-covered coin's hours.
MIN_COVERAGE = 0.5
#: Median hourly traded notional, USD. Below this the counts are single digits.
MIN_MED_USD = 10_000.0
#: Rank IC needs breadth: an hour with fewer markets than this is skipped.
MIN_COINS = 25
#: A cost-gate cell needs this many markets before its mean is quoted.
MIN_CELL = 5
#: An IC series shorter than this cannot support a bootstrap.
MIN_HOURS = 24
#: A coin needs this many hourly pairs before its autocorrelation is quoted.
MIN_PAIRS = 100
#: The family's false-discovery rate: one bar across every test in the search.
ALPHA = 0.05
#: Cost gate: basis points charged per unit of one-way turnover.
COST_BPS = 10.0
#: Coins per API read. Every coin is one presigned request, so keep it modest
#: enough not to trip the rate limit on a long run.
CHUNK = 25
#: Bump when the panel's columns or derivation change, so a cache written by an
#: older version of this script is never read as if it were this one.
CACHE_VERSION = 3
#: Cache directory. Files are keyed by span and version: a thirteen-month run
#: must never silently reuse a one-month panel.
CACHE_DIR = Path("/tmp/tessera-oi-composition")

POSITIONING_COLS = [
    "time",
    "open_interest",
    "d_open_interest",
    "n_unique_users",
    "n_long_positions",
    "n_short_positions",
    "n_new_longs",
    "n_new_shorts",
    "n_closed_longs",
    "n_closed_shorts",
    "n_flipped_l2s",
    "n_flipped_s2l",
    "oi_opened",
    "oi_closed",
    "taker_flip_l2s_1h",
    "taker_flip_s2l_1h",
]
#: Only what the floor and the forward return need: the hourly close is the last
#: minute's close, and notional is the only cross-coin comparable size.
OHLCV_COLS = ["time", "close", "volume", "vwap"]

PANELS = {"crypto": "crypto perpetuals", "xyz": "HIP-3 markets (xyz:*)"}


def month_span(text: str) -> tessera.MonthSpan:
    start, _, end = text.partition(":")
    return tessera.MonthSpan(start, end or start)


def median_or_nan(series: pl.Series) -> float:
    """`Series.median` returns None for an empty series; the report needs NaN."""
    value = series.median()
    return float("nan") if value is None else float(value)


def cache_path(span: tessera.MonthSpan) -> Path:
    return CACHE_DIR / f"panel-v{CACHE_VERSION}-{span.start}_{span.end}.parquet"


# Read


def read_asset(
    client: tessera.TesseraClient,
    asset: str,
    span: tessera.MonthSpan,
    columns: list[str],
    chunk: int = CHUNK,
) -> pl.DataFrame:
    """Every published partition of one asset, in batches.

    Read month by month: `client.read(coin=..., month=...)` expands the cartesian
    product, so a coin listed mid-span would 404 on partitions it never had.
    """
    frames: list[pl.DataFrame] = []
    for month in span.months():
        coins = sorted(
            {
                part.coin
                for part in client.partitions(asset, month=month).partitions
                if part.coin
            }
        )
        for i in range(0, len(coins), chunk):
            frames.append(
                client.read(
                    asset, coin=coins[i : i + chunk], month=month, columns=columns
                )
            )
    if not frames:
        raise ValueError(f"{asset} has no partitions in {span.start}:{span.end}")
    data = pl.concat(frames, how="diagonal_relaxed")
    if data.height == 0:
        raise ValueError(f"{asset} partitions in {span.start}:{span.end} are empty")
    return data


# Shape: one row per (coin, hour), with the composition fields on it


def hourly_positioning(pos: pl.DataFrame) -> pl.DataFrame:
    """The decomposition, from the published counts to six scale-free fields."""
    frame = (
        pos.sort("coin", "time")
        .with_columns(hour=pl.col("time").dt.truncate("1h"))
        .sort("coin", "hour")
    )
    duplicates = frame.group_by(["coin", "hour"]).len().filter(pl.col("len") > 1)
    if duplicates.height:
        sample = duplicates.head(3).rows()
        raise ValueError(
            "duplicate coin-hours in the positioning read "
            f"({duplicates.height} groups, e.g. {sample}); a double-counted hour "
            "would silently weight that market twice"
        )
    prev_oi = pl.col("open_interest").cast(pl.Float64) - pl.col("d_open_interest").cast(
        pl.Float64
    )

    # The counts are unsigned: `UInt32 - UInt32` wraps, so a falling position
    # count would become four billion instead of minus one.
    def count(name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Int64)

    def size(name: str) -> pl.Expr:
        # Taker flow is base-unit size, not an account count: casting it to an
        # integer would truncate a 0.3 BTC print to zero.
        return pl.col(name).cast(pl.Float64)

    arrivals = count("n_new_longs") + count("n_new_shorts")
    exits = count("n_closed_longs") + count("n_closed_shorts")
    flips = count("n_flipped_l2s") + count("n_flipped_s2l")
    gross = arrivals + exits
    taker_flips = size("taker_flip_l2s_1h") + size("taker_flip_s2l_1h")

    def safe(num: pl.Expr, den: pl.Expr) -> pl.Expr:
        # A ratio over an empty denominator is missing, not zero: a coin with no
        # accounts in an hour has no tilt, it does not have a neutral one.
        return pl.when(den > 0).then(num / den).otherwise(None)

    return frame.with_columns(
        doi_rel=safe(pl.col("d_open_interest").cast(pl.Float64), prev_oi),
        # `n_closed_*` includes flips, so the exit side is a superset of the flip
        # bucket and a flip legitimately appears in two fields.
        n_new_tilt=safe(count("n_new_longs") - count("n_new_shorts"), arrivals),
        n_close_tilt=safe(count("n_closed_shorts") - count("n_closed_longs"), exits),
        n_flip_tilt=safe(count("n_flipped_s2l") - count("n_flipped_l2s"), flips),
        flip_share=safe(flips, gross),
        churn=safe(gross, pl.col("n_unique_users")),
        oi_flow_rel=safe(size("oi_opened") - size("oi_closed"), prev_oi),
        taker_flip_oi_share=safe(taker_flips, prev_oi),
    ).select(
        "coin",
        "hour",
        "open_interest",
        "d_open_interest",
        "n_long_positions",
        "n_short_positions",
        "n_unique_users",
        "n_new_longs",
        "n_new_shorts",
        "n_closed_longs",
        "n_closed_shorts",
        "n_flipped_l2s",
        "n_flipped_s2l",
        "oi_opened",
        "oi_closed",
        "taker_flip_l2s_1h",
        "taker_flip_s2l_1h",
        *FIELDS,
    )


def hourly_price(minutes: pl.DataFrame) -> pl.DataFrame:
    """1-minute bars → the hourly close, the hour's notional, and 1h momentum."""
    hourly = (
        minutes.sort("coin", "time")
        .group_by_dynamic(
            "time", every="1h", closed="left", label="left", group_by="coin"
        )
        .agg(
            h_close=pl.col("close").last(),
            usd_volume=(pl.col("volume") * pl.col("vwap")).sum(),
        )
        .rename({"time": "hour"})
        .sort("coin", "hour")
    )
    return (
        hourly.with_columns(
            prev_hour=pl.col("hour").shift(1).over("coin"),
            prev_close=pl.col("h_close").shift(1).over("coin"),
        )
        .with_columns(
            mom_1h=pl.when(
                (pl.col("hour") - pl.col("prev_hour")) == pl.duration(hours=1)
            )
            .then(pl.col("h_close") / pl.col("prev_close") - 1.0)
            .otherwise(None)
        )
        .drop("prev_hour", "prev_close")
    )


def build_panel(
    client: tessera.TesseraClient, span: tessera.MonthSpan, chunk: int = CHUNK
) -> pl.DataFrame:
    pos = hourly_positioning(
        read_asset(client, "gold_positioning_1h", span, POSITIONING_COLS, chunk)
    )
    price = hourly_price(read_asset(client, "gold_ohlcv_1m", span, OHLCV_COLS, chunk))
    frame = price.join(pos, on=["coin", "hour"], how="inner").sort("coin", "hour")
    # HIP-3 markets (`xyz:*`) are labelled, not excluded: they are a different
    # market structure and the split is the point, not a robustness afterthought.
    frame = frame.with_columns(
        panel=pl.when(pl.col("coin").str.starts_with("xyz:"))
        .then(pl.lit("xyz"))
        .otherwise(pl.lit("crypto"))
    )
    # Drop each coin's first hour: the attribution there is computed against an
    # empty prior snapshot, so every account looks like a new arrival.
    return (
        frame.with_columns(rank=pl.int_range(pl.len()).over("coin"))
        .filter(pl.col("rank") > 0)
        .drop("rank")
    )


def dense(frame: pl.DataFrame) -> pl.DataFrame:
    """Re-index every coin onto a complete hourly grid, so a hole is a null row.

    Otherwise `shift()` bridges the hole and a `k`-hour label silently spans `k+1`.
    """
    hours = pl.datetime_range(
        frame["hour"].min(), frame["hour"].max(), interval="1h", eager=True
    )
    coins = frame.select("coin", "panel").unique()
    return (
        pl.DataFrame({"hour": hours})
        .join(coins, how="cross")
        # `panel` comes from the coin list; joining the full frame would suffix
        # it to `panel_right` on every reindexed row.
        .join(frame.drop("panel"), on=["coin", "hour"], how="left")
        .sort("coin", "hour")
    )


def floored(frame: pl.DataFrame, panel: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The stated floor and the panel it leaves, with the population reported.

    The floor is measured against the best-covered market anywhere in the file,
    then split, so no panel can be floored into a result.
    """
    per_coin = frame.group_by("coin").agg(
        n_hours=pl.col("h_close").is_not_null().sum(),
        med_usd=pl.col("usd_volume").median(),
        med_users=pl.col("n_unique_users").median(),
    )
    keep = per_coin.filter(
        (pl.col("n_hours") >= MIN_COVERAGE * per_coin["n_hours"].max())
        & (pl.col("med_usd") >= MIN_MED_USD)
    )
    kept = frame.filter(
        (pl.col("panel") == panel) & pl.col("coin").is_in(keep["coin"].to_list())
    )
    panel_coins = kept["coin"].unique().to_list()
    panel_meta = keep.filter(pl.col("coin").is_in(panel_coins))
    # `hours` is the union of hours the panel covers, not the dense grid's length:
    # the grid carries rows for hours before a market listed.
    covered = kept.filter(pl.col("h_close").is_not_null())
    summary = pl.DataFrame(
        {
            "panel": [PANELS[panel]],
            "coins": [kept["coin"].n_unique()],
            "coin_hours": [covered.height],
            "median_usd_per_hour": [median_or_nan(panel_meta["med_usd"])],
            "median_accounts": [median_or_nan(panel_meta["med_users"])],
            "hours": [covered["hour"].n_unique()],
        }
    )
    return kept, summary


# Measure


def return_col(horizon: int, lag: int) -> str:
    return f"ret_{horizon}_{lag}"


def forward_returns(frame: pl.DataFrame) -> pl.DataFrame:
    """Gap-guarded forward returns for every horizon and entry lag in the family.

    The guard covers every hour between bar and exit, so a hole is a null return
    rather than a shorter holding period.
    """
    out = frame.sort(["coin", "hour"])
    for k in HORIZONS:
        for lag in FAMILY_LAGS:
            window = pl.col("h_close").is_not_null()
            for s in range(1, lag + k + 1):
                window = window & pl.col("h_close").shift(-s).is_not_null()
            out = out.with_columns(
                pl.when(window)
                .then(
                    pl.col("h_close").shift(-(lag + k)) / pl.col("h_close").shift(-lag)
                    - 1.0
                )
                .otherwise(None)
                .over("coin")
                .alias(return_col(k, lag))
            )
    return out


def rank_ic(frame: pl.DataFrame, field: str, ret: str) -> pl.DataFrame:
    """Cross-sectional Spearman IC per hour, sorted so the bootstrap sees order."""
    return (
        frame.filter(pl.col(field).is_not_null() & pl.col(ret).is_not_null())
        .group_by("hour")
        .agg(
            n=pl.len(),
            ic=pl.corr(pl.col(field).rank("average"), pl.col(ret).rank("average")),
        )
        .filter(pl.col("n") >= MIN_COINS)
        .drop_nulls("ic")
        .sort("hour")
    )


def block_bootstrap_se(
    x: np.ndarray, block: int, n_boot: int = 2000, seed: int = 0
) -> float:
    """SE of `mean(x)` under a circular block bootstrap.

    Consecutive ICs share overlapping returns, so `block` (holding period + lag)
    sets the resample length; an i.i.d. resample inflates t by about `sqrt(k + LAG)`.
    """
    x = x[np.isfinite(x)]
    n = x.size
    block = max(1, min(block, n))
    if n < 3 or n < 2 * block:
        return float("nan")
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(n_boot, -1)[:, :n]
    se = float(x[idx].mean(axis=1).std(ddof=1))
    return se if np.isfinite(se) and se > 0 else float("nan")


def two_sided_p(t: float | None) -> float:
    if t is None or not np.isfinite(t):
        return float("nan")
    return float(erfc(abs(t) / sqrt(2)))


def bh_fdr(pvals: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Benjamini-Hochberg across the whole family, not per row."""
    p_arr = np.asarray(pvals, dtype=float)
    out = np.zeros(p_arr.shape, dtype=bool)
    ok = np.isfinite(p_arr)
    if not ok.any():
        return out
    order = np.flatnonzero(ok)[np.argsort(p_arr[ok])]
    m = order.size
    passed = p_arr[order] <= alpha * (np.arange(1, m + 1) / m)
    if passed.any():
        out[order[: np.max(np.flatnonzero(passed)) + 1]] = True
    return out


def ic_table(frame: pl.DataFrame) -> pl.DataFrame:
    """Every field × horizon × entry lag, with one BH bar over the whole family."""
    rows = []
    for field in FIELDS:
        for k in HORIZONS:
            for lag in FAMILY_LAGS:
                ics = rank_ic(frame, field, return_col(k, lag))["ic"].to_numpy()
                if ics.size < MIN_HOURS:
                    rows.append(
                        {
                            "field": field,
                            "horizon": k,
                            "lag": lag,
                            "ic": float("nan"),
                            "t": float("nan"),
                            "p": float("nan"),
                            "hours": ics.size,
                        }
                    )
                    continue
                mean = float(ics.mean())
                se = block_bootstrap_se(ics, block=k + lag)
                t = mean / se if np.isfinite(se) else float("nan")
                rows.append(
                    {
                        "field": field,
                        "horizon": k,
                        "lag": lag,
                        "ic": mean,
                        "t": t,
                        "p": two_sided_p(t),
                        "hours": ics.size,
                    }
                )
    table = pl.DataFrame(rows)
    return table.with_columns(bh_pass=pl.Series(bh_fdr(table["p"].to_numpy())))


def identity_check(frame: pl.DataFrame) -> pl.DataFrame:
    """Verify the ledger identities rather than assume them."""
    work = frame.sort(["coin", "hour"]).with_columns(
        prev_hour=pl.col("hour").shift(1).over("coin"),
        prev_n_long=pl.col("n_long_positions").shift(1).over("coin"),
        prev_n_short=pl.col("n_short_positions").shift(1).over("coin"),
    )
    consec = work.filter((pl.col("hour") - pl.col("prev_hour")) == pl.duration(hours=1))
    subset = (pl.col("n_flipped_l2s") <= pl.col("n_closed_longs")) & (
        pl.col("n_flipped_s2l") <= pl.col("n_closed_shorts")
    )
    excess = pl.max_horizontal(
        (
            pl.col("n_flipped_l2s").cast(pl.Int64)
            - pl.col("n_closed_longs").cast(pl.Int64)
        ).clip(lower_bound=0),
        (
            pl.col("n_flipped_s2l").cast(pl.Int64)
            - pl.col("n_closed_shorts").cast(pl.Int64)
        ).clip(lower_bound=0),
    )
    d_long = (
        pl.col("n_long_positions").cast(pl.Int64) - pl.col("prev_n_long").cast(pl.Int64)
    ) - (pl.col("n_new_longs").cast(pl.Int64) - pl.col("n_closed_longs").cast(pl.Int64))
    d_short = (
        pl.col("n_short_positions").cast(pl.Int64)
        - pl.col("prev_n_short").cast(pl.Int64)
    ) - (
        pl.col("n_new_shorts").cast(pl.Int64) - pl.col("n_closed_shorts").cast(pl.Int64)
    )
    long_ok = d_long == 0
    short_ok = d_short == 0
    row = consec.select(
        long_ok=long_ok.sum(),
        short_ok=short_ok.sum(),
        subset=subset.sum(),
        long_testable=d_long.is_not_null().sum(),
        short_testable=d_short.is_not_null().sum(),
        subset_testable=subset.is_not_null().sum(),
        worst_long=d_long.abs().max(),
        worst_short=d_short.abs().max(),
        worst_subset=excess.max(),
    ).row(0, named=True)

    def worst(value: object) -> float | None:
        return None if value is None else float(value)

    return pl.DataFrame(
        {
            "check": [
                "Δn_long = n_new_longs − n_closed_longs",
                "Δn_short = n_new_shorts − n_closed_shorts",
                "flips are a subset of the exit side",
            ],
            "testable": [
                int(row["long_testable"]),
                int(row["short_testable"]),
                int(row["subset_testable"]),
            ],
            "verified": [int(row["long_ok"]), int(row["short_ok"]), int(row["subset"])],
            "worst_discrepancy": [
                worst(row["worst_long"]),
                worst(row["worst_short"]),
                worst(row["worst_subset"]),
            ],
        }
    )


def gross_vs_net(frame: pl.DataFrame) -> pl.DataFrame:
    """How much gross activity sits behind one unit of net position change.

    Per-coin medians, then the median across coins, so one violent hour never
    becomes the market's number.
    """
    work = (
        frame.sort(["coin", "hour"])
        .with_columns(
            prev_hour=pl.col("hour").shift(1).over("coin"),
            d_n_long=pl.col("n_long_positions").cast(pl.Int64)
            - pl.col("n_long_positions").shift(1).over("coin").cast(pl.Int64),
            d_n_short=pl.col("n_short_positions").cast(pl.Int64)
            - pl.col("n_short_positions").shift(1).over("coin").cast(pl.Int64),
        )
        .filter((pl.col("hour") - pl.col("prev_hour")) == pl.duration(hours=1))
    )
    gross = (
        pl.col("n_new_longs").cast(pl.Int64)
        + pl.col("n_new_shorts").cast(pl.Int64)
        + pl.col("n_closed_longs").cast(pl.Int64)
        + pl.col("n_closed_shorts").cast(pl.Int64)
    )
    per_coin = work.group_by("coin").agg(
        acct_ratio=(
            gross / (pl.col("d_n_long").abs() + pl.col("d_n_short").abs()).clip(1)
        ).median(),
        size_ratio=(
            (
                pl.col("oi_opened").cast(pl.Float64)
                + pl.col("oi_closed").cast(pl.Float64)
            )
            / pl.col("d_open_interest").cast(pl.Float64).abs().clip(1e-12)
        ).median(),
    )
    return pl.DataFrame(
        {
            "ratio, per coin (median over its hours)": [
                "account transitions ÷ |net change in position count|",
                "taker volume opened+closed ÷ |ΔOI|",
            ],
            "median over coins": [
                median_or_nan(per_coin["acct_ratio"]),
                median_or_nan(per_coin["size_ratio"]),
            ],
        }
    )


def flip_vs_taker(frame: pl.DataFrame) -> pl.DataFrame:
    """Is a flip count aggressive volume, or just a busy hour's closes?

    Per-coin Spearman of each flip count against the matching taker flip volume
    and against the pure-close residual (`n_closed_* − n_flipped_*`).
    """
    work = frame.with_columns(
        pure_close_long=pl.col("n_closed_longs").cast(pl.Int64)
        - pl.col("n_flipped_l2s").cast(pl.Int64),
        pure_close_short=pl.col("n_closed_shorts").cast(pl.Int64)
        - pl.col("n_flipped_s2l").cast(pl.Int64),
    )
    pairs = [
        ("n_flipped_l2s vs taker_flip_l2s_1h", "n_flipped_l2s", "taker_flip_l2s_1h"),
        ("n_flipped_l2s vs pure long closes", "n_flipped_l2s", "pure_close_long"),
        (
            "n_flipped_s2l vs taker_flip_s2l_1h",
            "n_flipped_s2l",
            "taker_flip_s2l_1h",
        ),
        ("n_flipped_s2l vs pure short closes", "n_flipped_s2l", "pure_close_short"),
    ]
    per_coin = work.group_by("coin").agg(
        [
            pl.corr(pl.col(a).rank("average"), pl.col(b).rank("average")).alias(label)
            for label, a, b in pairs
        ]
    )
    rows = []
    for label, _, _ in pairs:
        valid = per_coin[label].filter(per_coin[label].is_finite())
        rows.append(
            {
                "pair": label,
                "median_rho": median_or_nan(valid),
                "share_positive": (
                    float((valid > 0).mean()) if valid.len() else float("nan")
                ),
            }
        )
    return pl.DataFrame(rows)


def overlap(frame: pl.DataFrame) -> pl.DataFrame:
    """What the arrival tilt is not: the level change, momentum, or a slow property."""
    rows = []
    for rival in ("doi_rel", "mom_1h"):
        per_hour = (
            frame.filter(
                pl.col("n_new_tilt").is_not_null() & pl.col(rival).is_not_null()
            )
            .group_by("hour")
            .agg(
                n=pl.len(),
                rho=pl.corr(
                    pl.col("n_new_tilt").rank("average"), pl.col(rival).rank("average")
                ),
            )
            .filter(pl.col("n") >= MIN_COINS)
        )
        rows.append(
            {
                "question": f"n_new_tilt vs {rival}, cross-section",
                "value": median_or_nan(per_hour["rho"]),
            }
        )
    # Rank autocorrelation, not Pearson: the field is a bounded ratio and the
    # post's 0.07 is Spearman. Needs enough pairs and more than two distinct values.
    pairs = (
        frame.sort(["coin", "hour"])
        .select("coin", "hour", "n_new_tilt")
        .with_columns(next=pl.col("n_new_tilt").shift(-1).over("coin"))
    )
    per_coin = (
        pairs.group_by("coin")
        .agg(
            rho=pl.corr(
                pl.col("n_new_tilt").rank("average"), pl.col("next").rank("average")
            ),
            n=(pl.col("n_new_tilt").is_not_null() & pl.col("next").is_not_null()).sum(),
            ux=pl.col("n_new_tilt").n_unique(),
            uy=pl.col("next").n_unique(),
        )
        .filter((pl.col("n") > MIN_PAIRS) & (pl.col("ux") > 2) & (pl.col("uy") > 2))
    )
    rows.append(
        {
            "question": "n_new_tilt rank autocorrelation, 1h",
            "value": median_or_nan(per_coin["rho"]),
        }
    )
    return pl.DataFrame(rows)


def tercile_cost(frame: pl.DataFrame) -> pl.DataFrame:
    """Top-minus-bottom tercile spread of the arrival tilt inside a real OI move.

    Gross spread in bps, then charged `COST_BPS` per unit of measured turnover. A
    cost gate, not a backtest: it says whether the ranking is worth a cost model.
    """
    abs_rel = pl.col("doi_rel").abs()
    gated = frame.with_columns(
        big=abs_rel >= abs_rel.quantile(0.7).over("coin"),
    )
    # Tercile membership is fixed once and held: otherwise a name near the sample
    # end would sort differently at 24 h than at 1 h and the turnover would be wrong.
    work = (
        gated.filter(pl.col("big") & pl.col("n_new_tilt").is_not_null())
        .with_columns(
            rho=pl.col("n_new_tilt").rank("average").over("hour"),
            width=pl.col("n_new_tilt").is_not_null().sum().over("hour"),
        )
        .filter(pl.col("width") >= MIN_COINS)
        .with_columns(tercile=(pl.col("rho") / pl.col("width") * 3).ceil())
        .sort(["coin", "hour"])
    )
    # "Hourly" is the share of consecutive coin-hours that change tercile: a shift
    # over gated rows would compare hours a day apart and call it one hour.
    turn = (
        work.with_columns(
            prev=pl.col("tercile").shift(1).over("coin"),
            prev_hour=pl.col("hour").shift(1).over("coin"),
        )
        .filter((pl.col("hour") - pl.col("prev_hour")) == pl.duration(hours=1))
        .select(changed=(pl.col("prev") != pl.col("tercile")).mean())
    )
    hourly_turnover = (
        float(turn["changed"][0])
        if turn.height and turn["changed"][0] is not None
        else float("nan")
    )
    rows = []
    for k in HORIZONS:
        horizon = work.filter(pl.col(return_col(k, LAG)).is_not_null()).with_columns(
            # Demean inside each hour: the question is which market outperforms
            # the others, never whether the market rose.
            excess=pl.col(return_col(k, LAG))
            - pl.col(return_col(k, LAG)).mean().over("hour")
        )
        # Sorted by hour: the bootstrap resamples *adjacent* hours, and a shuffled
        # series would make overlapping returns look independent.
        per_hour = (
            horizon.group_by("hour")
            .agg(
                top_n=(pl.col("tercile") == 3).sum(),
                bottom_n=(pl.col("tercile") == 1).sum(),
                top=pl.col("excess").filter(pl.col("tercile") == 3).mean(),
                bottom=pl.col("excess").filter(pl.col("tercile") == 1).mean(),
            )
            .filter((pl.col("top_n") >= MIN_CELL) & (pl.col("bottom_n") >= MIN_CELL))
            .sort("hour")
        )
        diff = (per_hour["top"] - per_hour["bottom"]).drop_nulls().to_numpy()
        if diff.size < MIN_HOURS:
            continue
        spread = float(diff.mean()) * 1e4
        se = block_bootstrap_se(diff, block=k + LAG) * 1e4
        held = min(1.0, hourly_turnover * k)
        rows.append(
            {
                "horizon": k,
                "spread_bps": spread,
                "t": spread / se if se > 0 else float("nan"),
                "hourly_turnover": hourly_turnover,
                "cost_bps": COST_BPS * held,
                "net_bps": spread - COST_BPS * held,
            }
        )
    return pl.DataFrame(rows)


# Report


def fmt(table: pl.DataFrame, digits: int = 4) -> str:
    """Polars' own table, floats rounded to the quoted precision."""
    return str(
        table.with_columns(
            pl.col(pl.Float64).round(digits),
            pl.col(pl.Boolean).cast(pl.Utf8),
        )
    )


def run_panel(frame: pl.DataFrame, panel: str) -> dict:
    sub, population = floored(frame, panel)
    sub = forward_returns(sub)
    table = ic_table(sub)
    display = table.filter(pl.col("lag") == LAG).drop("lag")
    print(f"\n=== {PANELS[panel]} ===")
    print(fmt(population, digits=0))
    print("\nledger identities (rows verified / rows testable):")
    print(fmt(identity_check(sub), digits=0))
    print("\ngross against net (median over coins):")
    print(fmt(gross_vs_net(sub), digits=1))
    print("\nis a flip count aggressive volume or just a busy hour?")
    print(fmt(flip_vs_taker(sub), digits=3))
    finite = int(np.isfinite(table["p"].to_numpy()).sum())
    print(
        f"\ncross-sectional rank IC, entry at +{LAG}h, BH across {finite} testable "
        f"of {table.height} tests (fields × horizons × entry lags {FAMILY_LAGS}):"
    )
    print(fmt(display))
    print("\nwhat the tilt is not:")
    print(fmt(overlap(sub), digits=3))
    cost = tercile_cost(sub)
    if cost.height:
        print(
            f"\narrival tilt, top-minus-bottom tercile inside a real OI move, "
            f"${COST_BPS:.0f} bps per unit of turnover:"
        )
        print(fmt(cost, digits=2))
    return {"panel": panel, "ic": table}


def main() -> None:
    # Every column and every row: a truncated table is a number you cannot check.
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(80)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--months", default="2026-10:2026-10", help="YYYY-MM:YYYY-MM (inclusive)"
    )
    ap.add_argument(
        "--cached",
        action="store_true",
        help="read the cache and fail if the span has none (no API key needed)",
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="ignore the cache and re-read"
    )
    ap.add_argument("--chunk", type=int, default=CHUNK, help="coins per API read")
    args = ap.parse_args()

    if args.cached and args.rebuild:
        ap.error("--cached and --rebuild are mutually exclusive")
    if args.chunk < 1:
        ap.error("--chunk must be at least 1")
    span = month_span(args.months)
    cache = cache_path(span)
    if args.cached and not cache.exists():
        ap.error(f"no cached panel for {span.start}:{span.end} at {cache}")
    if cache.exists() and not args.rebuild:
        print(f"panel: cached {cache} ({span.start}:{span.end})")
        frame = pl.read_parquet(cache)
    else:
        client = tessera.TesseraClient()
        print(f"reading {span.start}:{span.end} (positioning + one-minute prices) …")
        frame = dense(build_panel(client, span, args.chunk))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(cache, compression="zstd")
        print(
            f"panel: {frame.height:,} coin-hours, {frame['coin'].n_unique()} markets "
            f"→ {cache}"
        )

    for panel in PANELS:
        run_panel(frame, panel)

    print(
        "\nRead the panels side by side, never averaged: the same ledger, the same "
        "fields, and the sign of the answer depends on whether the underlying "
        "closes at night."
    )


if __name__ == "__main__":
    main()
