# Plan 5: Part I verdict on intraday, and Modal ready

## Goal

Close the known gaps in the factor loop, deploy Modal, and run the loop long enough to answer
one question in writing: **can intraday, flat-by-close factors survive costs at a $50k book?**
The answer decides how much effort intraday gets compared to the swing sandbox of Plan 6.

## Starting point (2026-10-30, see AGENTS.md)

- Dev and full builds pass QA. Null calibration done (IC floors h1 0.0105, h2 0.0110, h4 0.0151).
  The forward-return canary scores 94x the floor.
- 36 trials on the reversal goal, 0 accepted. Reversal is real (IC about 0.013, t 4 to 7) but
  every variant is `cost_killed` (net Sharpe -1.5 to -5).
- Modal code written, never deployed. Evaluation runs locally.
- Gaps still in the code:
  - `eval/gate.py` `ic_floor` returns 0 when the calibration is missing (fails open).
  - `af data publish` crashes with a traceback when `MarketData` refuses the build.
  - Warm Modal containers keep the old `_EVALUATOR` after a new dev build is published.
  - The Enovis name-day (2022-04-05) is not masked out of scoring.
  - `sell_fee_bps: 0.3` is a guess.
  - There are no written success or kill criteria for `af final-eval`.

## Out of scope

Multi-day horizons (Plan 6), trainable models (Plan 7), any sealed final evaluation.

## Decisions for the owner before starting

1. **Sealed changes.** Tasks 4 and 5 change `data.yaml` or `sandbox.yaml`, which changes the
   sandbox hash. That needs a new null calibration (about 48 min) and leaves the 36 existing trials
   under the old hash. Do both changes at once so there is only one recalibration.
2. **Modal factor evaluation: default or opt-in.** Decided by task 6's measured cost per batch.
   Recommendation: keep `local` as the default and save Modal credits for Plan 7's GPU training.
3. **Final-eval criteria (task 8).** The agent drafts them; you sign them off.

## Tasks

### 1. A missing calibration fails closed

- `eval/gate.py`: `ic_floor` raises `CalibrationMissingError` when `gate.min_ic_mean` is null and
  there is no floor for the horizon. Remove the fallback to 0.
- `research/runtime.py` `make_backend` and `compute/modal_dev.py` `_evaluator` raise the same
  error, with the message "run `af eval null`", instead of passing `None`.
- Tests: the gate raises without a floor; `af loop run --once` exits 1 with that message on a
  synthetic build with no calibration; a set `min_ic_mean` still works without a calibration.

### 2. `af data publish` reports a refused build cleanly

- `cli_research.py` `data_publish`: also catch `DatasetNotReadyError` and print the QA failure,
  exit 1.
- Test: publishing a build that failed QA exits 1 with no traceback.

### 3. Warm Modal containers pick up a new build

- `compute/modal_dev.py`: `_evaluator(request)` compares the request's `dataset_hash` and
  `sandbox_hash` with the cached evaluator. On a mismatch, reload the volume once, rebuild, and
  compare again. If they still differ, raise `StaleDeploymentError`.
- Test with a fake volume: publish build A, evaluate, publish build B, evaluate. The second call
  uses B with no redeploy.

### 4. Mask the known corporate actions out of scoring (sealed, owner-approved)

- Every `qa.known_corporate_actions` entry nulls the labels of that ticker for the sessions whose
  label window crosses the event date. The DSL inputs stay as they are.
- Test on the synthetic raw market with a planted unadjusted event: the label is null on the
  affected rows and unchanged everywhere else.

### 5. Verify the sell fee (sealed, owner-approved)

- Look up the current SEC section 31 rate and set `sell_fee_bps`. Remove the `ASSUMPTION`
  comment and cite the source and date.
- Then run `af eval null` and `af eval canary` once for tasks 4 and 5 together.

### 6. Deploy the Modal dev app and prove it matches local

- `modal deploy --env dev deploy/modal_dev.py`, then `af data publish --kind dev`.
- `af loop run --once` with `evaluation.backend: modal`.
- `af loop reeval` on 5 trials evaluated locally: the Modal numbers must be identical.
- Record the measured cost of one batch of 12 from the Modal dashboard in `docs/research.md`.
  Choose the default backend from it (decision 2).

### 7. Deploy the sealed app, without using it

- `modal deploy --env sealed deploy/modal_sealed.py`, then `af data publish --kind full`.
- New test: `ModalBackend` only ever resolves functions of `alpha-factory-eval`. The loop path
  never names `alpha-factory-final`. This adds a code-level check to the import-linter contract,
  because both machines now hold a token that can reach the sealed environment.
- Do not run `af final-eval`.

### 8. Write the final-eval criteria before anything can use them

- New `docs/final-eval-criteria.md`, signed off by you. It covers:
  - what a frozen library must show before it may be scored at all (validation net Sharpe, number
    of factors, holdout passed);
  - the success bar on the sealed year (for example IC t-stat above 2 and net Sharpe above 0.5);
  - what counts as a kill.
- It applies to every sandbox (Plan 6) and every model (Plan 7).

### 9. Run the loop for real

- Approve the two remaining bootstrap goals (volume shock, opening gap) and run LLM batches
  within `daily_llm_budget_usd`. Target: about 300 evaluated trials across the three goals.
- Run `af report run` and `af report ab` (template against LLM).

### 10. Cost-sensitivity study (owner-side, outside the loop)

- A new command, `af eval sensitivity FACTOR...`. It takes the 5 best `cost_killed` factors by
  gross IC and backtests them on the search folds under alternative portfolio specs:
  - 1, 2 or 3 rebalances per day;
  - 10 or 20 names per side;
  - $50k or $100k capital;
  - a half-spread of 1, 2 or 3 bps.
- It reports gross and net Sharpe and the break-even cost per spec.
- It lives on the sealed side (`backtest`, `eval`) and is never reachable from the loop. Every run
  is logged with the specs tried, because choosing a portfolio after seeing results is itself a
  form of search.
- Output: a table in the verdict below.

### 11. The verdict and the docs

- New `docs/verdict-intraday.md`:
  - what ran (trials, goals, LLM cost, Modal cost);
  - what was accepted and why the rest failed;
  - the sensitivity table;
  - one paragraph on intraday at $50k: continue as is, change the book, or deprioritise.
- Update `AGENTS.md` "Current state" and `docs/research.md` "Verified, and not".

## Exit check

- [ ] Tasks 1-3 merged, with tests; full suite, pyright and lint-imports pass.
- [ ] One recalibration after tasks 4-5; the canary passes.
- [ ] A Modal batch matches local on 5 re-evaluated trials; cost per batch recorded.
- [ ] The sealed app is deployed and the full build published. `af final-eval` has never run.
- [ ] `docs/final-eval-criteria.md` signed off.
- [ ] About 300 evaluated trials; `docs/verdict-intraday.md` written.
- [ ] Modal spend for this plan under $10.

## Risks

- **Zero accepted factors.** A valid outcome; the verdict says so, and that strengthens the case
  for Plan 6.
- **Overfitting by the sensitivity study.** If a portfolio change is adopted, it becomes a new
  sandbox version with a new calibration, and the study's run count is recorded in the verdict.
