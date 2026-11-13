<div align="center">

# Tessera examples

**The code behind our research posts — runnable, on data you can get for free.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

</div>

---

Every analysis published on the [Tessera blog](https://tesseralytics.dev/blog) ships with the
script that produced it. This repository is where those scripts live: each directory is one post,
self-contained, and prints the numbers that post quotes.

The data comes from [Tessera](https://tesseralytics.dev) — order-flow-enriched Hyperliquid OHLCV,
funding and positioning, as Parquet over a REST API. A **free key** needs no card and reaches every
dataset over the trailing 30 days, which is enough to reproduce the shape of every measurement
here. The longer windows behind the published tables need the history bundle; each script says so
in its docstring.

## Setup

```bash
git clone https://github.com/tesseralytics/examples
cd examples

# A free key, from https://tesseralytics.dev
export TESSERA_API_KEY="sk_..."

# uv resolves the environment from pyproject.toml (pip install -e . works too)
uv run python lag-discount/lag_discount.py --months 2026-10:2026-10
```

Python 3.10+. The only dependencies are `tessera-api` and `polars`.

## What's here

| directory | what it measures | post |
|---|---|---|
| [`lag-discount/`](lag-discount/) | How much of a factor's edge survives the delay between a bar closing and you being able to act on it. The full 22-factor panel, gap-guarded returns, cross-sectional rank IC at delays from 0 to 24 hours. | [The two-hour lag discount](https://tesseralytics.dev/blog/the-two-hour-lag-discount) |

## Notes on reading these scripts

They are measurements, not strategies. Every one is written to be checked rather than trusted:

- **Gap-guarded.** A forward return is null unless every hour of the holding period is published,
  so a data hole can never pass as a short return. The panel is re-indexed onto a complete hourly
  grid first — a missing hour has to be a *row* before `shift()` can notice it.
- **One cross-section.** For a given horizon, the coin set is held identical across every delay in
  the sweep, so a decay curve cannot be a sampling artefact of the longest lag.
- **Stated floors.** A factor with a zero-delay IC below the materiality floor is excluded from a
  class median instead of being quietly averaged in.
- **The window is part of the result.** A one-month run and a thirteen-month run give the same
  ordering and different digits. The class curve, the search-wide significance bar and the
  replication split all sharpen with a longer sample; each example's README says what to expect
  from the span you can afford.

If a number here disagrees with a post, the post is citing the longer run — the docstring says
which span each figure belongs to.

## Licence

GPL-3.0, matching the official clients. The scripts are meant to be copied into your own work.
