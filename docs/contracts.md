# Contracts

All models live in `alpha_factory.contracts`. One Pydantic model per object generates the JSON
Schema sent to the LLM, validates the reply, and serialises into the store.

## Two layers

- `LLMModel`: what the model writes. `extra="forbid"`, frozen, whitespace stripped, `Literal` or
  `StrEnum` categories, a max length on every string, at most 2 nesting levels.
- `Record`: what code adds (ids, hashes, timestamps, numbers). Frozen, `extra="forbid"`, carries
  `schema_version`. Computed fields are dropped on input so a stored dump round-trips.

The LLM can't forge ids, hashes, metrics or confidence. Field descriptions are written as
instructions: they ship inside the JSON Schema, so they are prompt text. Every `LLMModel` has a
committed snapshot in `tests/unit/__snapshots__/`; a schema change fails CI until the snapshot is
updated on purpose (`uv run pytest --snapshot-update`).

Cross-references are checked with `model_validate(data, context=...)`: `pool_ids` (parents shown
to the LLM), `batch_trial_ids` and `belief_ids`.

## Modules

| Module | Models |
|---|---|
| `base` | `LLMModel`, `Record`, `Slug`, `Horizon`, `Sha256` |
| `proposals` | `RefutationCheck`, `FactorProposal`, `ProposalBatch`, `RawProposalBatch`, `ProposalRecord` |
| `evaluation` | `ReasonCode`, `FoldStat`, `RefutationOutcome`, `EvalResult`, `EvalFeedback`, `GateSpec` |
| `beliefs` | `BeliefUpdate`, `Reflection`, `BeliefStatus`, `Belief` |
| `context` | `ResearchGoal`, `BatchDirective`, `PoolFactorView`, `BeliefView`, `TrialDigest`, `ContextPacket` |
| `sandbox` | `SandboxSpec` and its parts, `FinalEvalRequest`, `FinalEvalReport` |
| `data` | `DataManifest`, `FileEntry`, `QACheck`, `QAReport`, `BAR_SCHEMA` |
| `llm` | `LLMCallRecord` |

Models for later plans (`GoalProposals`, `CombinerSpec`, `LibFactor`, `FrozenLibrary`) will be
added here by Plan 4. Modal request models stay in `compute/`.

## Hashing

`hashing.sha256_canonical` serialises with sorted keys and no whitespace. Every hash in the system
goes through it, so the same object gives the same hash on every machine. A trial is comparable to
another only if both `dataset_hash` and `sandbox_hash` match.

## Bar schema

`contracts.data.BAR_SCHEMA` is the exact column order and dtypes of a bars file. The synthetic
market writes it and Plan 2 must produce it exactly. Prices and volume are forward-adjusted:
`adj_factor` is the product of split ratios on or before the session, adjusted price is
`raw * adj_factor` and adjusted volume is `raw / adj_factor`, so a build never changes when later
data is added.

## Synthetic market

`SyntheticMarket(seed, n_tickers, n_sessions, start, planted_ic, planted_horizon)` writes bars,
universe, sectors and a manifest (plus `answers.json`). Same seed, same bytes. It plants:
short-horizon reversal with a closed-form IC (`-Pct($close, 2)` against `h1`), 5 two-for-one
splits, 10 delistings, 5 listings, 3 half-days with 7 bars, 0.5% missing bars and two
volatility regimes. `h4` cannot be planted (no bar has both an 8-bar lookback and an 8-bar forward
window in a 13-bar session). `SyntheticMarket.answers` returns the planted facts.
