# Alpha Factory

A personal, open-source research system that discovers intraday trading signals for US equities
with minimal human input. An LLM proposes factor hypotheses as expressions in a restricted DSL;
sealed, human-written code evaluates them on historical data; validated evidence accumulates in a
research store that steers the next proposals.

> **Disclaimer:** This is a research project, not investment advice. Nothing here is a
> recommendation to buy or sell any security. Trading involves risk of loss.

## Status

Plan 1 (foundations): repo, config, contracts, synthetic market, CLI.
Plan 2 (market data): Massive flat files to adjusted 30-min bars, universe, sectors, manifest, QA
and the `MarketData` read API (see `docs/data.md`).
Plan 3 (sandbox and evaluation engine): DSL, labels, evaluator, backtester, gate and sealed
`final_eval` (see `docs/evaluation.md`).
Plan 4 (research loop): store, proposers, orchestrator, beliefs, pool, combiners, Modal worker, reports
(see `docs/research.md`; Modal, a real LLM provider and Discord are written but not yet exercised).

## Quick start

```sh
uv sync --all-groups
uv run af --help
uv run af synth build --out /tmp/synth --seed 7
uv run pytest
```

No market data lives in this repo. Tests and CI use a synthetic market.
See `docs/` for the architecture, contracts and the six invariants.
