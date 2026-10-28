# Market data (Plan 2)

Turns minute flat files from Massive into forward-adjusted 30-min bars, a point-in-time universe
and a sector map, hashed in a manifest. The agent loop only ever opens the `dev` build.

## Layout (`$AF_DATA_DIR`)

```
raw/minute/YYYY/MM/YYYY-MM-DD.csv.gz     immutable downloads, with a .sha256 sidecar
raw/ref/<endpoint>/<fetch_date>.json     cached REST responses (rebuilds need no network)
ref/{instruments,symbols,splits,dividends,sic}.parquet
cache/bars_raw/session_date=D/part-0.parquet   unadjusted bars, shared by dev and full builds
build/<dataset_id>/
  bars_30m/session_date=D/part-0.parquet       Plan 1 bar schema
  universe/session_date=D/part-0.parquet
  adj_factors.parquet  sectors.parquet  session_stats.parquet
  manifest.json        written last: no manifest, no build
  qa.json
datasets/dev, datasets/full                    symlinks to the current builds
```

`bars_30m` holds every instrument that is ever a universe member (all of its sessions, so
history exists when it joins). `session_stats.parquet` carries dollar volume per session for the
unmapped-symbol QA check.

## Rules that matter

- **Bars are labelled by end time.** Bar k holds minutes in `[09:30 + 30k, 09:30 + 30(k+1))` New
  York time (13 bars, 7 on half-days). Empty bars have no row; nothing is forward-filled.
- **Adjustment is causal.** `adj_factor = split_factor * div_factor`, using only corporate
  actions up to the session. Actions on or before the first session are ignored, so levels are in
  share units of the dataset's first day. Prices scale by `adj_factor`, volume divides by
  `split_factor`, `raw_close` is untouched.
- **The universe for session d uses data through d-1 only.** Eligible: type CS, listed before d,
  not delisted before d, bars on at least 18 of the previous 20 sessions, previous raw close at
  least $5. Ranked by trailing dollar volume (sum over the previous 20 sessions divided by 20).
  The first 20 sessions have no universe. A name delisting on d is still a member on d.
- **dev_end is inclusive**: the last session in the dev build. Dev and full come from the same
  code, so every dev row must equal the matching full row (QA `dev_full_equal`).
- **Determinism.** Sums that feed hashes (VWAP, dollar volume) use integer arithmetic, because
  parallel float group-by sums can differ in the last bit between runs.

## Commands

`af data download | ref [--offline] | build --kind dev|full|both | qa | manifest | verify`.
`build --kind both` builds full, then dev, then runs QA on both. QA exits 1 on any failed check.
`dev_full_equal` needs both builds, so a lone dev build never passes QA.

## Reading

```python
from alpha_factory.data import MarketData

md = MarketData("dev")  # raises DatasetNotReadyError if QA failed
md.bars(start, end, columns, members_only=True)  # LazyFrame + rank, tradable, is_member
MarketData("full", sealed=True)  # only final_eval; otherwise SealedDatasetError
```

## What is verified, and how

Tested on a synthetic *raw* market (`synthetic/raw.py`: Massive-format minute files and reference
JSON with splits, dividends, a delisting, an ETF and an unmapped symbol). Real-data checks done
so far: the flat-file schema and aggregation speed on one real day (2024-06-10). Not done: the
REST endpoints (field names are unverified, see `data/massive.py`), a 5-year download and the
NVDA split check.
