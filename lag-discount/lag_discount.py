#!/usr/bin/env python
"""How much of a factor's edge survives the delay before you can act on it?

A factor is a column that ranks the market: today's open interest, this hour's
buy/sell imbalance, the last 24 hours of momentum. Its "edge" is measured as the
cross-sectional rank correlation (IC) between that column and the return over the
next `k` hours.

The trap: the return is always scored from the price at the *close* of the hour
the factor describes — an instant nobody can trade, because a bar's close is its
last trade and does not exist until the bar ends. Real entries happen later. This
script measures what is left when the entry slides.

    ret(H, k, L) = close(H + L + k) / close(H + L) - 1

`L` moves the *entry*, never the holding period: still a `k`-hour position, just
opened later. Four rules make the resulting curve mean something:

  * gap-guarded — a return is null unless every hour from H to H+L+k is published,
    so a hole can never masquerade as a short return;
  * identical cells at every lag — for a given `k`, a coin-hour is kept only if
    *every* lag has a price, so the 24h lag is not measured on a different sample;
  * one class per mechanism — fields are grouped by what they *measure*, fixed
    before the curve is looked at, so the classification cannot be circular;
  * the class median only over fields with a real zero-delay IC: a retention ratio
    on a field with no edge is noise over noise;
  * an overlap-aware t (circular block bootstrap of length horizon + delay) and one
    Benjamini–Hochberg bar across the whole search — then a replication run on
    disjoint sub-panels, because a class claim has to survive a different slice.

Companion code for the post "The two-hour lag discount: building factors you can
actually trade" — https://tesseralytics.dev/blog/the-two-hour-lag-discount

Setup:

    export TESSERA_API_KEY="sk_..."     # free key, no card: https://tesseralytics.dev

One month on a free key (~10 minutes):

    uv run python lag-discount/lag_discount.py --months 2026-10:2026-10

Thirteen months (needs the history bundle behind the 30-day free window):

    uv run python lag-discount/lag_discount.py --months 2025-10:2026-10

It prints, in order: the class curve, the per-factor ICs, the family bar (how many
tests survive the search-wide threshold at each delay), and the replication check
(the same contrast measured on time halves and liquidity tiers).
"""

from __future__ import annotations

import argparse
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import polars as pl
import tessera

#: Forward holding periods, hours.
HORIZONS = (1, 4, 24)
#: Entry delays, hours. 0 is the number no published dataset can act on; 2 is the
#: first hour-aligned entry after a bar becomes readable.
LAGS = (0, 1, 2, 4, 8, 24)

#: The mechanism each field belongs to, fixed from its definition. State = a
#: level or cumulative snapshot of where the market stands. Microstructure = a
#: quantity measured inside hour H, describing hour H. Momentum = a price path.
CLASSES: dict[str, str] = {
    # -- state -----------------------------------------------------------------
    "log_open_interest": "state",
    "log_volume": "state",
    "funding_rate": "state",
    "annualized_funding": "state",
    "cum_funding": "state",
    "d_funding_rate": "state",
    "account_ls_ratio": "state",
    "whale_net_position": "state",
    "whale_long_account_share": "state",
    "concentration_skew": "state",
    "hhi_gross": "state",
    "top10_skew": "state",
    "d_open_interest": "state",
    # -- microstructure --------------------------------------------------------
    "close_vwap_dev": "microstructure",
    "buysell_imbalance": "microstructure",
    "d_cvd": "microstructure",
    "pnl_imbalance": "microstructure",
    "net_taker_open_1h": "microstructure",
    "pos_aggressor_delta_1h": "microstructure",
    "taker_flip_net": "microstructure",
    "oi_net_flow": "microstructure",
    "n_new_net": "microstructure",
    "flow_vs_snapshot_oi_residual": "microstructure",
    # -- momentum --------------------------------------------------------------
    "mom_1h": "momentum",
}

#: The horizon a class is judged at. A state field or a price path describes a
#: condition that persists, so it is asked about a day ahead; a microstructure
#: quantity answers for its own hour, so it is asked one hour ahead.
DECISION_HORIZON: dict[str, int] = {"state": 24, "microstructure": 1, "momentum": 24}

#: Below this zero-delay |IC| a retention ratio is noise over noise, not a result.
MIN_BASE_IC = 5e-3
#: A coin needs this share of the best-covered coin's hours to enter the universe.
MIN_COVERAGE = 0.5
#: Rank IC needs breadth: an hour with fewer coins than this is skipped.
MIN_COINS = 25
#: An IC series shorter than this cannot support a bootstrap t.
MIN_HOURS = 24
#: The family's false-discovery rate: one bar across every test in the search.
ALPHA = 0.05
#: Coins per API read. Every coin is one presigned request, so keep it modest
#: enough not to trip the rate limit on a long run.
CHUNK = 25

OHLCV_COLS = [
    "time",
    "close",
    "volume",
    "vwap",
    "buy_vol",
    "sell_vol",
    "cvd",
    "closed_pnl_long",
    "closed_pnl_short",
]
FUNDING_COLS = ["time", "funding_rate", "annualized", "cum_funding"]
POSITIONING_COLS = [
    "time",
    "open_interest",
    "account_ls_ratio",
    "top10_long_share",
    "top10_short_share",
    "concentration_skew",
    "hhi_gross",
    "whale_net_position",
    "whale_long_account_share",
    "d_open_interest",
    "n_new_longs",
    "n_new_shorts",
    "aggressor_delta_1h",
    "net_taker_open_1h",
    "oi_opened",
    "oi_closed",
    "flow_vs_snapshot_oi_residual",
    "taker_flip_l2s_1h",
    "taker_flip_s2l_1h",
]


def month_span(text: str) -> tessera.MonthSpan:
    start, _, end = text.partition(":")
    return tessera.MonthSpan(start, end or start)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def coins_for(client: tessera.TesseraClient, asset: str, span: tessera.MonthSpan) -> list[str]:
    """The coins this asset has a partition for, across the span.

    Not `client.datasets()`: that names a coin if it appears in *any* month, and a
    download for a month it has no file in is a 404.
    """
    coins: set[str] = set()
    for month in span.months():
        for part in client.partitions(asset, month=month).partitions:
            if part.coin:
                coins.add(part.coin)
    return sorted(coins)


def read_asset(
    client: tessera.TesseraClient,
    asset: str,
    span: tessera.MonthSpan,
    columns: list[str],
) -> pl.DataFrame:
    """Every coin of one asset, in batches (one presign request per coin)."""
    coins = coins_for(client, asset, span)
    frames = [
        client.read(asset, coin=coins[i : i + CHUNK], month=span, columns=columns)
        for i in range(0, len(coins), CHUNK)
    ]
    return pl.concat(frames, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Shape each asset into one row per (coin, hour)
# ---------------------------------------------------------------------------

def hourly_ohlcv(minutes: pl.DataFrame) -> pl.DataFrame:
    """1-minute bars → hourly bars + the fields measured inside the hour.

    Bar H covers [H, H+1h) and its close is the last minute's close, which is the
    price at the end of the hour: strictly after every input above it.
    """
    hourly = (
        minutes.sort("coin", "time")
        .group_by_dynamic("time", every="1h", closed="left", label="left", group_by="coin")
        .agg(
            close=pl.col("close").last(),
            volume=pl.col("volume").sum(),
            vwap=pl.col("vwap").last(),
            buy_vol=pl.col("buy_vol").sum(),
            sell_vol=pl.col("sell_vol").sum(),
            cvd=pl.col("cvd").last(),
            pnl_long=pl.col("closed_pnl_long").sum(),
            pnl_short=pl.col("closed_pnl_short").sum(),
            # Notional, not base units: the only cross-coin comparable size.
            usd_volume=(pl.col("volume") * pl.col("vwap")).sum(),
        )
        .rename({"time": "hour"})
        .sort("coin", "hour")
    )
    return hourly.with_columns(
        log_volume=(pl.col("volume").clip(lower_bound=0) + 1.0).log(),
        close_vwap_dev=pl.when(pl.col("vwap") > 0)
        .then(pl.col("close") / pl.col("vwap") - 1.0)
        .otherwise(None),
        buysell_imbalance=pl.when(pl.col("buy_vol") + pl.col("sell_vol") > 0)
        .then(
            (pl.col("buy_vol") - pl.col("sell_vol")) / (pl.col("buy_vol") + pl.col("sell_vol"))
        )
        .otherwise(None),
        d_cvd=pl.col("cvd") - pl.col("cvd").shift(1).over("coin"),
        pnl_imbalance=pl.when((pl.col("pnl_long").abs() + pl.col("pnl_short").abs()) > 0)
        .then(
            (pl.col("pnl_short") - pl.col("pnl_long"))
            / (pl.col("pnl_long").abs() + pl.col("pnl_short").abs())
        )
        .otherwise(None),
        # Consecutive panel rows, not a gap-guarded window: the published
        # definition, kept so this measurement and the dataset agree.
        mom_1h=pl.col("close") / pl.col("close").shift(1).over("coin") - 1.0,
    )


def hourly_funding(funding: pl.DataFrame) -> pl.DataFrame:
    return (
        funding.sort("coin", "time")
        .with_columns(hour=pl.col("time").dt.truncate("1h"))
        .sort("coin", "hour")
        .with_columns(
            annualized_funding=pl.col("annualized"),
            d_funding_rate=pl.col("funding_rate") - pl.col("funding_rate").shift(1).over("coin"),
        )
        .select("coin", "hour", "funding_rate", "annualized_funding", "cum_funding", "d_funding_rate")
    )


def hourly_positioning(pos: pl.DataFrame) -> pl.DataFrame:
    return (
        pos.sort("coin", "time")
        .with_columns(hour=pl.col("time").dt.truncate("1h"))
        .sort("coin", "hour")
        .select(
            "coin",
            "hour",
            "account_ls_ratio",
            "concentration_skew",
            "whale_net_position",
            "whale_long_account_share",
            "hhi_gross",
            "d_open_interest",
            "flow_vs_snapshot_oi_residual",
            "net_taker_open_1h",
            top10_skew=(pl.col("top10_long_share") - pl.col("top10_short_share")),
            log_open_interest=(pl.col("open_interest").clip(lower_bound=0) + 1.0).log(),
            oi_net_flow=(pl.col("oi_opened") - pl.col("oi_closed")),
            pos_aggressor_delta_1h=pl.col("aggressor_delta_1h"),
            taker_flip_net=(pl.col("taker_flip_l2s_1h") - pl.col("taker_flip_s2l_1h")),
            n_new_net=(pl.col("n_new_longs").cast(pl.Int64) - pl.col("n_new_shorts").cast(pl.Int64)),
        )
    )


def build_panel(client: tessera.TesseraClient, span: tessera.MonthSpan) -> pl.DataFrame:
    ohlcv = hourly_ohlcv(read_asset(client, "gold_ohlcv_1m", span, OHLCV_COLS))
    funding = hourly_funding(read_asset(client, "gold_funding_1h", span, FUNDING_COLS))
    pos = hourly_positioning(read_asset(client, "gold_positioning_1h", span, POSITIONING_COLS))
    panel = (
        ohlcv.join(funding, on=["coin", "hour"], how="left")
        .join(pos, on=["coin", "hour"], how="left")
        .sort("coin", "hour")
    )
    # Drop each coin's first hour: the positioning attribution there is computed
    # against an empty prior snapshot, so everyone looks like a new account.
    return panel.with_columns(rank=pl.int_range(pl.len()).over("coin")).filter(pl.col("rank") > 0).drop("rank")


def dense(panel: pl.DataFrame) -> pl.DataFrame:
    """Re-index every coin onto a complete hourly grid, so a hole is a null row.

    Without this, `shift()` bridges a missing hour and turns a gap into a return.
    """
    hours = pl.datetime_range(panel["hour"].min(), panel["hour"].max(), interval="1h", eager=True)
    grid = pl.DataFrame({"hour": hours}).join(
        pl.DataFrame({"coin": panel["coin"].unique()}), how="cross"
    )
    return grid.join(panel, on=["coin", "hour"], how="left").sort("coin", "hour")


def universe(panel: pl.DataFrame) -> list[str]:
    """The liquidity floor, applied to the *universe*, never to the cross-section.

    A coin needs half the hours of the best-covered coin and a positive median
    traded notional. Cutting further turns rank IC into noise — the axis to give
    up is time, not breadth.
    """
    per_coin = panel.group_by("coin").agg(
        n_hours=pl.col("close").is_not_null().sum(),
        med_usd_volume=pl.col("usd_volume").median(),
    )
    keep = per_coin.filter(
        (pl.col("n_hours") >= MIN_COVERAGE * per_coin["n_hours"].max())
        & (pl.col("med_usd_volume") > 0)
    )
    return keep["coin"].to_list()


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

def lag_grid(panel: pl.DataFrame) -> pl.DataFrame:
    """One forward return per (horizon, lag), gap-guarded over the whole span."""
    df = panel
    for k in HORIZONS:
        span = max(LAGS) + k
        window = pl.col("close").is_not_null()
        for s in range(1, span + 1):
            window = window & pl.col("close").shift(-s).is_not_null()
        for lag in LAGS:
            df = df.with_columns(
                pl.when(window)
                .then(
                    pl.col("close").shift(-(lag + k)) / pl.col("close").shift(-lag) - 1.0
                )
                .otherwise(None)
                .over("coin")
                .alias(f"ret_{k}_{lag}")
            )
    return df.drop_nulls("close")


def rank_ic(df: pl.DataFrame, field: str, ret: str) -> pl.DataFrame:
    """Cross-sectional Spearman IC per hour (average ranks for ties)."""
    return (
        df.filter(pl.col(field).is_not_null() & pl.col(ret).is_not_null())
        .group_by("hour")
        .agg(
            n=pl.len(),
            ic=pl.corr(pl.col(field).rank("average"), pl.col(ret).rank("average")),
        )
        .filter(pl.col("n") >= MIN_COINS)
        .drop_nulls("ic")
    )


def ic_series(df: pl.DataFrame, field: str, ret: str) -> np.ndarray:
    return rank_ic(df, field, ret)["ic"].to_numpy()


def block_bootstrap_se(
    x: np.ndarray, block: int, n_boot: int = 1000, seed: int = 0
) -> float:
    """Standard error of `mean(x)` under a circular block bootstrap.

    Plain i.i.d. resampling is wrong here: consecutive ICs share overlapping
    forward returns, so they are serially dependent. `block` is that dependence
    length — the holding period `k` *plus* the entry delay `L`, because sliding the
    entry makes the overlap longer than the label. A naive `ICIR * sqrt(n)` inflates
    the t-statistic by roughly `sqrt(k + L)`.

    Returns NaN rather than a number when the series cannot support a resample:
    with fewer than two whole blocks every replicate is the same rotation, the SE
    collapses, and the t-statistic becomes arbitrarily large.
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
    """Normal-theory two-sided p-value for a t-statistic (bootstrap SE carried)."""
    if t is None or not np.isfinite(t):
        return float("nan")
    return float(erfc(abs(t) / sqrt(2)))


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg: True where the null is rejected, across the whole family."""
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


def measure(
    sub: pl.DataFrame, field: str, k: int, lag: int, bootstrap: bool = True
) -> tuple[float | None, float | None, float | None]:
    """Mean IC, overlap-aware t, and the two-sided p for one (field, horizon, lag)."""
    ics = ic_series(sub, field, f"ret_{k}_{lag}")
    if ics.size < MIN_HOURS:
        return None, None, None
    mean = float(ics.mean())
    if not bootstrap:
        return mean, None, None
    se = block_bootstrap_se(ics, block=k + lag)
    t = mean / se if np.isfinite(se) else None
    return mean, (float(t) if t is not None else None), two_sided_p(t)


def evaluate(panel: pl.DataFrame, bootstrap: bool = True) -> pl.DataFrame:
    """The whole grid, one row per (field, decision horizon, lag)."""
    rows = []
    for field, class_ in CLASSES.items():
        for k in HORIZONS:
            # Identical coin-hours at every delay: keep a cell only if every lag priced.
            keep = pl.lit(True)
            for lag in LAGS:
                keep = keep & pl.col(f"ret_{k}_{lag}").is_not_null()
            sub = panel.filter(keep)
            base = measure(sub, field, k, 0, bootstrap=False)[0]
            # The floor gates the class median only — never the significance family.
            # Every test that was run belongs in the search that BH corrects.
            above_floor = base is not None and abs(base) >= MIN_BASE_IC
            for lag in LAGS:
                ic, t, p = measure(sub, field, k, lag, bootstrap=bootstrap)
                rows.append(
                    dict(
                        field=field,
                        class_=class_,
                        horizon=k,
                        lag=lag,
                        ic=ic,
                        base_ic=base,
                        above_floor=above_floor,
                        t=t,
                        p=p,
                        retention=(
                            None if ic is None or not base else ic / base
                        ),
                    )
                )
    out = pl.DataFrame(rows)
    if out.height and bootstrap:
        # One family, one bar: correct across every test in the search, not per row.
        out = out.with_columns(
            bh_pass=pl.Series(bh_fdr(out["p"].to_numpy(), alpha=ALPHA))
        )
    return out


def class_table(rows: pl.DataFrame) -> pl.DataFrame:
    out = []
    for class_, horizon in DECISION_HORIZON.items():
        sub = rows.filter((pl.col("class_") == class_) & (pl.col("horizon") == horizon))
        if sub.height == 0:
            continue
        usable = sub.filter(pl.col("above_floor"))
        fields = usable["field"].n_unique()
        for lag in LAGS:
            at = sub.filter(pl.col("lag") == lag)
            counted = at.filter(pl.col("above_floor"))
            rets = sorted(r for r in counted["retention"].drop_nulls().to_list())
            out.append(
                dict(
                    class_=class_,
                    horizon=horizon,
                    lag=lag,
                    fields=fields,
                    mean_abs_ic=float(at["ic"].abs().mean()),
                    median_retention=(
                        (rets[len(rets) // 2] + rets[(len(rets) - 1) // 2]) / 2 if rets else None
                    ),
                    bh_pass=int(at["bh_pass"].sum()) if "bh_pass" in at.columns else None,
                )
            )
    return pl.DataFrame(out)


def replication(panel: pl.DataFrame) -> pl.DataFrame:
    """Re-run the grid on disjoint sub-panels: is the contrast a class effect?

    A class claim has to survive being measured on a different slice of the same
    data. Time halves catch one period's accident; the liquidity split catches an
    artefact of the most heavily traded coins.
    """
    hours = sorted(panel["hour"].unique().to_list())
    mid = hours[len(hours) // 2]
    per_coin = (
        panel.group_by("coin")
        .agg(med_usd_volume=pl.col("usd_volume").median())
        .sort("med_usd_volume", descending=True)
    )
    ranked = per_coin["coin"].to_list()
    third = max(1, len(ranked) // 3)
    variants = {
        "first half of the hours": panel.filter(pl.col("hour") < mid),
        "second half of the hours": panel.filter(pl.col("hour") >= mid),
        "most liquid third of coins": panel.filter(pl.col("coin").is_in(ranked[:third])),
        "the rest": panel.filter(pl.col("coin").is_in(ranked[third:])),
    }
    out = []
    for name, sub in variants.items():
        if sub.height == 0:
            continue
        table = class_table(evaluate(lag_grid(sub), bootstrap=False))
        for class_, horizon in DECISION_HORIZON.items():
            at = table.filter(
                (pl.col("class_") == class_) & (pl.col("horizon") == horizon) & (pl.col("lag") == 2)
            )
            if at.height == 0:
                continue
            out.append(
                dict(
                    variant=name,
                    class_=class_,
                    fields=int(at["fields"][0]),
                    retention_at_2h=at["median_retention"][0],
                )
            )
    return pl.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2026-10:2026-10", help="start:end, inclusive")
    ap.add_argument("--out", default=None, help="optional parquet of every (field, horizon, lag) row")
    ap.add_argument("--cache", default="/tmp/lag-discount/panel.parquet")
    ap.add_argument(
        "--cached",
        action="store_true",
        help="reuse the cached panel (the reads are the whole runtime)",
    )
    args = ap.parse_args()

    span = month_span(args.months)
    cache = Path(args.cache)
    if args.cached and cache.exists():
        panel = pl.read_parquet(cache)
        print(f"panel: {cache} (cached)")
    else:
        client = tessera.TesseraClient()   # reads $TESSERA_API_KEY
        panel = dense(build_panel(client, span))
        cache.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(cache)

    keep = universe(panel)
    panel = panel.filter(pl.col("coin").is_in(keep))
    print(
        f"panel: {panel['coin'].n_unique()} coins, {panel.height:,} coin-hours, "
        f"{panel['hour'].n_unique()} hours, {panel['hour'].min()} .. {panel['hour'].max()}"
    )

    rows = evaluate(lag_grid(panel))
    if rows.height == 0:
        raise SystemExit("no field cleared the zero-delay materiality floor")
    table = class_table(rows)

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(120)
    print("\nby class, at the class's decision horizon\n")
    print(
        table.select(
            pl.col("class_").alias("class"),
            "horizon",
            "lag",
            pl.col("mean_abs_ic").round(4),
            pl.col("median_retention").round(3).alias("retention"),
            "fields",
        )
    )
    print("\nper field at its class's decision horizon\n")
    decision = pl.DataFrame(
        [
            {"class_": c, "horizon": h}
            for c, h in DECISION_HORIZON.items()
        ]
    )
    rows_at_decision = rows.join(decision, on=["class_", "horizon"], how="semi")
    print(
        rows_at_decision.filter(pl.col("lag").is_in([0, 2, 24]))
        .pivot(on="lag", index=["field", "class_", "horizon"], values="ic")
        .rename({"0": "ic@0h", "2": "ic@2h", "24": "ic@24h"})
        .sort("class_", "field")
    )

    print("\nthe family bar: how many tests clear the joint BH threshold at each delay\n")
    print(
        rows.group_by("lag")
        .agg(
            tests=pl.len(),
            bh_pass=pl.col("bh_pass").sum(),
            share=(pl.col("bh_pass").mean() * 100).round(0),
        )
        .sort("lag")
    )

    print("\nreplication: the class contrast on disjoint sub-panels (retention at a 2h entry)\n")
    print(
        replication(panel)
        .sort("variant", "class_")
        .pivot(on="class_", index="variant", values="retention_at_2h")
        .with_columns(pl.all().exclude("variant").round(3))
    )

    if args.out:
        rows.write_parquet(args.out)


if __name__ == "__main__":
    main()
