# Alpha Factory: context for agents

Discovers intraday alpha signals for US equities. An LLM proposes factor formulas in a restricted
DSL; sealed, human-written code evaluates them on historical data; validated evidence accumulates
in a research store that steers the next proposals. The design follows AQuA (arXiv 2608.12841).
Deeper docs: `docs/architecture.md`, `data.md`, `evaluation.md`, `research.md`, `contracts.md`,
`invariants.md`.

## Target

- Intraday, flat by close. 30-minute bars, top-1000 point-in-time universe, top-500 tradable.
- Labels: forward return over 1, 2 and 4 hours (`h1`, `h2`, `h4`); entry at the next bar's open.
- Book: $50k, dollar-neutral, 10 names per side, sector-neutral, rebalanced 10:00/12:00/14:00.
- Costs are dollar-accurate (IBKR tiered, $0.35 minimum, 2 bps half-spread). Expect a hurdle of
  roughly 7 to 10 bps of capital per day before any alpha counts. Only net results are judged.
- Multi-day horizons are out of scope: labels, backtester and contracts assume one session.

## Pipeline

1. **Data** (`data/`): Massive minute flat files to forward-adjusted 30-min bars, point-in-time
   universe, sectors, manifest, QA. `af data download | ref | build --kind both | qa`.
   `MarketData("dev")` refuses a build whose QA failed.
2. **Sandbox** (`sandbox/`, sealed): panel, causal labels, folds, loader. Config: `configs/sandbox.yaml`.
3. **Formulas** (`dsl/`): parser, type checker, compiler, operator registry. Causal by construction.
4. **Evaluation** (`eval/`, `backtest/`, sealed): daily IC per fold, regimes, turnover, net
   backtest, gate with deflation by trial count, pool correlation. Verdicts: accepted,
   `low_ic`, `cost_killed`, `correlated`, `subsumed`, `lookahead`, `invalid_*`.
5. **Calibration** (`eval/calibration.py`): null IC floors from 1,000 random formulas, a label
   shuffle check and two leakage canaries. `af eval null | canary | factor`.
6. **Research loop** (`orchestrator/`, `agents/`, `research/`): a state machine per batch:
   PLANNED, PROPOSING, VALIDATING, EVALUATING, GATING, REFLECTING, DONE. A bandit sets the
   directive (how many new/mutate/refine/combine, which families); a proposer fills it.
   `af goal add|list|approve`, `af loop run [--once]`, `af loop status`, `af report run`.
7. **Final evaluation** (`sandbox/final_eval.py`): the only code that reads the sealed test
   window, once per frozen library. `af lib freeze`, `af final-eval`, `af lib import-final`.

## Who does what

- **Goals** (research questions) come from a YAML file (`goals/bootstrap.yaml`, `af goal add`) or
  from the manager LLM (`af goal brief` then `af goal propose`). You approve every goal.
- **Formulas** come from the `template` proposer (`agents/template.py`, no LLM, seeded by batch
  id) or the `llm` proposer (`agents/llm_proposer.py`). Set it in `configs/loop.yaml`.
  The model sees title, question and horizons of the goal, not its `families` or budget.
- **Nothing the LLM writes is trusted.** Every proposal is parsed, type-checked and
  deduplicated by the DSL before the sealed evaluator scores it.

## Hard boundaries

- Dev, validation and test are separated. The loop only opens the `dev` build. Only
  `final_eval` reads the sealed window; `.importlinter` forbids the loop from importing it
  (`uv run lint-imports`). The gate holdout is returned to the loop as pass/fail only.
- `sandbox.yaml`, `data.yaml` and the sealed packages are edited by the owner, not by agents,
  unless asked. Changing either changes the sandbox/dataset hash and invalidates calibration.
- Never read `.env`. Never print keys. Do not push without asking.
- The laptop and the mini PC are interchangeable (invariant 6): same commit, configs and
  credentials; only `.env` paths differ. Never give a machine a role in code, docs or plans.
  Each machine has its own research store, and the final-eval ledgers do not see each other.

## Configuration

`configs/`: `data.yaml` (window, universe, QA thresholds, `qa.known_corporate_actions`),
`sandbox.yaml` (labels, folds, costs, portfolio, gate), `loop.yaml` (proposer, batch size,
budgets, backend), `llm.yaml` (models, prices, temperature, `reasoning_effort`, token caps).
Secrets and paths come from `.env` (see `.env.example`): `LLM_BASE_URL`, `LLM_API_KEY`,
`AF_DATA_DIR`, Massive and Discord keys.

## DSL notes

Fields carry a `$`: `$open $high $low $close $volume $trades $vwap $ret $dayopen $prevclose $bar`.
Operators are capitalised: `Neg Sub Div Mul Pct Delta Mean EMA Decay TsZScore TsRank Rank
ZScore Sum SignedPower ...`. Windows must come from `1, 2, 3, 5, 8, 13, 26, 39, 65, 130`.
The `condition` of a proposal is a boolean DSL expression or null, never prose, and the
refutation `not_concentrated_in_condition` requires one.

## LLM provider

OpenAI-compatible chat completions at `https://api.meta.ai/v1`; the client appends
`/chat/completions`. Model `muse-wonder-1.0` is a reasoning model: hidden reasoning counts as
output tokens, so `reasoning_effort` must be set or replies can come back empty. Strict JSON
schema output works. Cost is metered per call and capped by `daily_llm_budget_usd` (checked
before every PROPOSING). In template mode the loop makes no LLM calls.

## Current state (2026-10-30)

- Data: 2021-10-29 to 2026-10-26, dev through 2025-10-17, sealed test 2025-10-27 to 2026-10-26.
  Dev and full builds pass QA. QA checks `extreme_returns` and `split_continuity` look at
  universe member-days only; one documented exception (Enovis, 2022-04-05, a spin-off the
  adjustment cannot see). The first two calendar days had no vendor files, hence the start date.
- Calibration (real data): IC floors h1 0.0105, h2 0.0110, h4 0.0151; label shuffle +0.0012.
  The forward-return canary scores 94x the floor; the volume-fraction canary is flagged
  LOOKAHEAD but its IC is near zero on real data and is not required to be high.
- One factor evaluates in about 17 s of compute (28 s with loading). A null run of 1,000
  formulas takes about 48 minutes.
- Loop: 36 trials on the reversal goal (12 template, 12 invalid_schema before a prompt fix,
  12 LLM), 0 accepted. Short-term reversal is real (IC about 0.013, t 4 to 7, stable by fold
  and year) but every variant is `cost_killed` (net Sharpe -1.5 to -5). Template and LLM
  proposals look alike so far. Two other bootstrap goals are pending approval.

## Next

Plans 5 (Part I verdict, Modal), 6 (swing timeframes) and 7 (trainable models) are in
`docs/plans/`.

## Not done yet

- Models as proposals (trainable architectures, GPU training on Modal): nothing exists. Modal
  code is written but was never deployed; evaluation runs locally (`evaluation.backend: local`).
- Unverified: `sell_fee_bps: 0.3` (a guess), Discord delivery, systemd unit, unattended runs
  of many batches, any sealed evaluation. Success and kill criteria for `af final-eval` are not
  yet written; the sealed window can be used once.
- The Enovis name-day is not masked out of scoring.

## Working rules

- Run long commands (builds, null calibration, loop batches) in the background and read the log.
- Grep the real text before scripted edits. Commit in small logical commits; ask before pushing.
- Checks: `uv run pytest`, `uv run ruff check src tests`, `uv run pyright`, `uv run lint-imports`.
- Prompt and schema changes break snapshots under `tests/unit/__snapshots__`; update them with
  `--snapshot-update` and read the diff.
- Reports of results must say what was run and what was not; do not claim a check passed that
  was not run.
