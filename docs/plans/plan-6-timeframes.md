# Plan 6: the timeframe as a sandbox parameter (swing trading)

## Goal

Run the same pipeline for holds of a few days to a few weeks: the same DSL, evaluator, gate,
calibration, loop and final evaluation. Add a **swing sandbox on daily bars** next to the
intraday one, with nothing duplicated. After this plan, a new timeframe is a new sandbox file
plus a data build, not new code.

Why now: costs. At $50k, intraday has to clear 7 to 10 bps of capital per day, and Plan 5
measures how much of the signal survives that. A position held for a week pays its costs once
per week. Plan 7's models also need generic labels and a generic evaluator, so this refactor
happens once, before them.

## What is hard-coded to intraday today

| Where | Assumption |
|---|---|
| `contracts/base.py` | `Horizon = Literal["h1", "h2", "h4"]`, used in about 15 modules and the store |
| `contracts/sandbox.py` | `bar_minutes: Literal[30]`; `LabelSpec.hours`, `exit: "k_hours_or_close"`; `PortfolioSpec.rebalance_times`, `exit_at_close` |
| `sandbox/labels.py` | Labels shift over `(tix, sidx)`: they cannot cross a session |
| `backtest/engine.py` | Rebalances at clock times; everything closes at the session's end |
| `dsl/registry.py` | `WINDOWS` in 30-min bars; `SessMean`/`SessSum`; `$dayopen`, `$prevclose`, `$bar` |
| `eval/stats.py` | Daily IC = mean of bar ICs in a session; the Newey-West lag ignores overlapping labels |
| `research/store.py` | `pool`, `beliefs` and `families` are not keyed by sandbox |
| `data/` | Only 30-minute bars are built |

## Decisions for the owner before starting

1. **Swing labels.** Proposal: `d1`, `d3`, `d5` ("a few days") and `d10`, `d20` ("a few weeks"),
   in sessions, entering at the next session's open and exiting at the close of session t+k.
2. **History.** The minute data starts 2021-10-29, which gives about 4 years of search and holdout
   on daily bars. That is enough for d1-d5 but thin for d20 (about 45 independent periods a year
   per name). Options:
   - (a) start with the 5 years we have;
   - (b) add a `daily_long` dataset from Massive daily aggregates back to 2010 or earlier, built
     as a separate kind with its own QA (an optional phase at the end of this plan).
   Recommendation: (a) now, and (b) only if d10/d20 look promising.
3. **The swing book.** Proposal: dollar-neutral, 15 names per side, sector-neutral, rebalanced
   every k sessions at the next open, held overnight, with a borrow fee on shorts (for example
   50 bps a year; hard-to-borrow names cannot be detected, which is a known limit). Alternative:
   long-only against a benchmark, which avoids borrow issues for a real $50k IBKR account. Which
   one matters if you plan to trade it.
4. **Sealed window.** The swing sandbox uses the same sealed dates as intraday
   (2025-10-27 → 2026-10-26). Each sandbox's library is scored once, but the same calendar year
   is looked at by two systems. Recommendation: accept this, and record it in
   `docs/final-eval-criteria.md`.
5. **Hash bump.** Reshaping `SandboxSpec` changes the intraday sandbox hash. That means one more
   intraday recalibration (about 48 min), and earlier trials stay under the old hash.
   Re-evaluating old trials (task 11) must give the same metrics.

## Design

- **Sandbox files.** `configs/sandboxes/<name>.yaml`, starting with `intraday_30m.yaml` (today's
  `sandbox.yaml`) and `swing_1d.yaml`. `loop.yaml` gets `sandbox: intraday_30m`, and every
  `af eval|loop|lib|final-eval` command takes `--sandbox NAME`, defaulting to the one in
  `loop.yaml`. It is config, not an environment variable, so both machines behave the same.
- **One store per sandbox.** The pool, beliefs and families tables are not keyed by sandbox, so
  each sandbox gets its own SQLite file: `AF_STORE_PATH` becomes a directory, holding
  `<store dir>/<sandbox>.sqlite`. Evidence never mixes across timeframes.
- **Generic labels.** `LabelSpec{name, bars, cross_session}`:
  - intraday: `h1 = 2 bars, cross_session false`;
  - swing: `d5 = 5 bars, cross_session true`.
  - `Horizon` becomes a constrained string checked against the sandbox's labels at load time.
- **Timeframe.** `SandboxSpec.timeframe: Literal["30m", "1d"]` replaces `bar_minutes`. The
  validators measure purge and embargo in bars and sessions of that timeframe.
- **Portfolio.** A discriminated union: `IntradayBook` (today's fields) or
  `SwingBook{rebalance_every, positions_per_side, hold_buffer, sector_neutral,
  borrow_bps_per_year}`.

## Tasks

### 1. Contracts

- `Horizon` becomes a constrained string; it is checked against the sandbox in `load_sandbox`,
  the proposal validator and the store.
- `LabelSpec`, `timeframe` and the portfolio union as described above, with validators:
  - purge (in bars) covers the longest label;
  - for a 1d sandbox, the embargo (in sessions) covers the longest label.
- Migration: `intraday_30m.yaml` expresses today's spec in the new shape. A test checks that the
  intraday labels, folds and costs equal today's values field by field.

### 2. Daily bars in the data build

- `bars_1d/session_date=D/part-0.parquet` in every build, aggregated from the unadjusted
  30-minute cache with integer arithmetic (as for VWAP today), then adjusted with the same
  `adj_factor`.
- Columns: OHLC, volume, trades, vwap, raw_close, labelled at the session close.
- The manifest lists it and QA checks it: missing sessions, extreme daily returns on member-days,
  daily close equal to the last 30m bar's close.
- The universe is unchanged (it is already daily).
- `MarketData.bars(..., timeframe="1d")`.

### 3. Panel and fields

- `load_panel` is timeframe-aware. For 1d, one row per (ticker, session), with `sidx == tsi`.
- `$dayopen` and `$prevclose` keep their meaning (`$open` and `Ref($close, 1)` on daily bars).
- `$bar` and the session operators are not available on 1d.

### 4. DSL registry per timeframe

- Each operator and field declares the timeframes it is valid for; the checker rejects the rest.
- `WINDOWS` per timeframe; for 1d: `1, 2, 3, 5, 10, 20, 60, 120, 250`.
- The LLM-facing documentation and the JSON schema are generated per sandbox, so the proposer
  only sees what is valid there.

### 5. Labels across sessions

- When `cross_session` is true, shift over `tix` only.
- The label is null when the entry or exit bar is missing. QA reports the share of labels nulled
  by delistings: with long holds that share becomes a survivorship question, so it needs to be
  visible.
- The Plan 5 corporate-action mask applies to every label window that crosses the event.

### 6. Statistics for overlapping labels

- For 1d, the "daily IC" is the cross-sectional IC of each session.
- The Newey-West lag becomes `max(NW_LAG, label_bars)` for the label being judged.
- Turnover for 1d is the rank autocorrelation between consecutive rebalance sessions.
- Tests: independent implementations on planted series, as for today's stats.

### 7. The swing backtester

- `backtest/swing.py` `simulate_swing`:
  - rebalance every k sessions, filling at the next session's raw open;
  - hold overnight and mark to market at raw closes;
  - costs per order as today, plus borrow per day on short notional;
  - splits during a hold rescale the shares (ratio of `adj_factor`); dividends are credited to
    longs and debited from shorts.
- It returns the same `BacktestResult`, so the gate and reports do not change.
- Golden test, worked out by hand to the cent: one long and one short over 6 sessions, with a split
  and a dividend during the hold.

### 8. Calibration and canaries per sandbox

- `af eval null --sandbox swing_1d`, keyed by that sandbox's hash.
- Both canaries run on 1d. The forward-return canary must score more than 5x the d1 floor.
- The causality property test (500 random compositions plus negative controls) runs for both
  timeframes, including a negative control where the label leaks across a session.

### 9. Loop and agents

- The template proposer gets swing families: medium-term momentum (skip-a-week), 1-week reversal,
  low volatility, volume trend, gap behaviour across days.
- Prompts show the horizon definitions from the sandbox ("d5: enter at the next open, exit at the
  close 5 sessions later").
- `goals/swing_bootstrap.yaml` with three goals.

### 10. Final evaluation per sandbox

- `final_eval`, `LibraryScorer` and `FrozenLibrary` take the sandbox from the library's
  `sandbox_hash`. The request hash already includes it, so ledger entries stay distinct.

### 11. Regression and docs

- `af loop reeval` on 10 old intraday trials under the new intraday sandbox gives the same metrics.
- Update `AGENTS.md`: replace "Multi-day horizons are out of scope" with the two sandboxes, and
  describe the store directory. Update `docs/data.md` (daily bars), `evaluation.md` and
  `research.md`.

### Optional phase: longer daily history (decision 2b)

- A new data kind `daily_long` from Massive daily aggregates, with its own download, QA and point-
  in-time universe, and `swing_1d_long.yaml` using it. Planned separately if needed.

## Exit check

- [ ] Both sandboxes load. The intraday one reproduces the old metrics on re-evaluated trials.
- [ ] Daily bars pass QA in dev and full builds.
- [ ] Swing null calibration saved; both canaries behave as expected on 1d.
- [ ] Causality property tests pass for both timeframes; the swing backtester golden test passes.
- [ ] One swing goal ran at least 5 LLM batches end to end, in its own store.
- [ ] Full suite, pyright strict on the sealed packages, and lint-imports pass.

## Risks

- **Overlapping labels inflate t-stats.** Mitigated by task 6 and the null calibration, which goes
  through the same statistics.
- **Survivorship in long holds.** Measured in task 5, not hidden.
- **Thin d20 history.** Expected; decision 2b is the remedy.
