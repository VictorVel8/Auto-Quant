# Architecture

Alpha Factory discovers intraday signals for US equities. An LLM proposes factor hypotheses as
expressions in a restricted DSL. Sealed, human-written code evaluates them on historical data, and
validated evidence accumulates in a research store that steers the next proposals. The design
follows AQuA (arXiv 2608.12841), with a technically sealed test set and dollar-accurate costs.

## Hosts and the one boundary

- **Mini PC**: always-on orchestrator and research store. The LLM returns only JSON proposals and
  belief text.
- **Modal**: fan-out evaluation on train and validation data.
- **MacBook**: development; only `final_eval` reads the test volume, once per frozen spec.

The agent loop can propose and read validation results. It can't reach the data build, the test
data or the final evaluator.

## Packages (`src/alpha_factory/`)

| Package | Owner | Role |
|---|---|---|
| `contracts` | Plan 1 | Every Pydantic model that crosses a boundary. Imports nothing else in the package. |
| `settings`, `logging`, `ids`, `hashing`, `cli` | Plan 1 | Config, logs, prefixed ULIDs, canonical hashing, the `af` CLI |
| `synthetic` | Plan 1 | Seeded synthetic market with planted answers |
| `data` | Plan 2 | Massive flat files to Parquet, universe, manifest, QA; see `data.md` |
| `sandbox`, `dsl`, `eval`, `backtest` | Plan 3 | Labels, folds, DSL, evaluator, gate, backtester |
| `research`, `agents`, `orchestrator`, `compute` | Plan 4 | Store, proposers, loop, Modal; see `research.md` |

Import rules (`.importlinter`): `data`, `sandbox`, `dsl`, `eval` and `backtest` never import
`agents`, `orchestrator` or `research`; `contracts` never imports `data`, `dsl`, `eval`, `agents`
or `orchestrator`.

## Plan stack

1. Foundations (this repo state): skeleton, tooling, config, contracts, synthetic market, CLI.
2. Market data: Massive to Parquet, forward adjustment, 30-min bars, point-in-time universe, QA.
3. Sandbox and evaluation engine: labels, folds, DSL, evaluator, gate, backtester, `final_eval`.
4. Research loop: store, orchestrator, proposers, reflector, bandit, pool, Modal, Discord.

## Configuration

Environment variables hold secrets and machine-specific paths; YAML in `configs/` holds behaviour.
`sandbox.yaml` is sealed by ownership and hash, not secrecy. Every run records the hashes of the
configs it used (`af config show`).

## Cost model for a $50k book

Costs decide whether intraday flat-by-close can work at all: expect a hurdle of roughly 7 to 10 bps
of capital per day before any alpha counts (IBKR Pro tiered, $0.35 minimum per order, 2 bps
half-spread). Costs are computed per order in dollars, the gate judges net results only, and a
factor that dies to costs is rejected as `COST_KILLED`.

## Testing bar

Causality properties, golden numbers, planted signal, null calibration and reproducibility.
Pyright strict on the sealed packages and `contracts`, ruff, pre-commit, CI on every push.
