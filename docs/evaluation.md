# Sandbox and evaluation engine (Plan 3)

Turns an expression string into a verdict: parse, check, compute, score against labels, backtest
with dollar costs, assign one reason code. Everything here is sealed: no agent code, no LLM.

## Modules

| Package | Contents |
|---|---|
| `sandbox` | `load_sandbox` (sandbox.yaml + data.yaml + dev build), `panel` (dense data), `labels`, `causality`, `final_eval` |
| `dsl` | grammar, `registry` (all operators as data), checker, canonicaliser, compiler, `random_expression` |
| `eval` | `stats` (daily IC, Newey-West), `baselines`, `incremental`, `regimes`, `pool`, `context`, `evaluate`, `verdict`, `gate`, `holdout`, `calibration` |
| `backtest` | `costs`, `engine.simulate`, `backtest()` |

## Choices that differ from, or fill gaps in, the plan

- **One column per operator, not one polars expression.** Nested `.over()` windows return silent
  nulls in the installed polars, so every operator is a step that materialises a column.
- **Null is the missing marker.** The panel is dense (one row per ticker per bar over its life),
  so `Ref(x, n)` is always n bars back; a missing bar is a row of nulls and a rolling window
  containing it is null.
- **Cross-sectional minimums scale with the universe**: `min(200, top_n // 5)` names for an IC
  (200 for a 1,000-name universe), `min(100, top_n // 10)` for a condition split.
- **Labels cache** lives in `$AF_DATA_DIR/cache/labels/`, not next to the bars: a build folder is
  hashed, and an unlisted file would fail `af data verify`.
- **LOW_IC / WRONG_SIGN are one decision**: if the declared sign fails check 6 and the reversed
  sign would pass it, the verdict is WRONG_SIGN.
- **`evaluate` stops at ACCEPTED**; `gate_holdout` runs the holdout afterwards and returns the
  result unchanged or with HOLDOUT_FAIL. The holdout IC is never returned.
- **`evaluate` takes `trial: TrialInfo`** (trial count N, null floor, declared refutations,
  parents), because the plan's signature has no way to carry N or the calibrated floor.
- **Fields** are `$open $high $low $close $volume $trades $vwap $ret $dayopen $prevclose`.
- **Constants limit**: only literals used as float operands count toward the 3-constant limit;
  window and `Winsor`/`SignedPower`/`Clip` constants do not. Every literal has at most 2
  significant digits, and the canonical form must itself pass the checker.
- **Costs**: the half-spread is charged in the cost formula, not added to the fill price, so it is
  counted once. `sell_fee_bps` is an assumption (0.3, SEC section 31) to verify.
- **final_eval** takes a `SealedScorer` (version, horizon, combiner_hash, score) because the
  frozen library and LightGBM combiner are Plan 4. Its dataset guard is `MarketData("full",
  sealed=True)`; the dev build's hash is not checkable from the full build alone.

## Verified

Causality (every operator, 500 random compositions, plus negative controls that must be caught),
operators against numpy/scipy references, the backtester to the cent on a hand-worked case,
statistics against independent implementations, the planted IC on the synthetic market.

## Not verified yet (needs real data)

Timing on 4 years of real bars.

## Real-data results (dataset from 2021-10-29)

- Null floors (99th percentile of |IC|, 1,000 random formulas): h1 0.0105, h2 0.0110, h4 0.0151;
  label-shuffle mean IC +0.0012 (limit 0.002).
- Two canaries. The AQuA one (cumulative volume over full-day volume) is flagged LOOKAHEAD but has
  an IC near zero on real data (-0.0075 at h1): it leaks future volume, and volume says little
  about returns, so it cannot be required to score high. The forward-return canary leaks the
  label itself, must be flagged and must score above 5x the floor. `af eval canary` needs both
  flags and the second score.
- `-$ret` has an h1 IC of +0.0143, so `$ret` is -0.0143: short-term reversal.
