# Changelog

## Unreleased
- Plan 1: foundations (repo, tooling, config, contracts, synthetic market, CLI).
- Plan 2: market data (downloader, reference tables, bars, causal adjustment, universe, sectors,
  manifest, QA, `MarketData`, `af data` commands).
- Plan 3: sandbox and evaluation engine (labels, DSL with all operators, causality property test,
  evaluator, backtester with dollar costs, verdicts, gate, holdout, null calibration, canary,
  sealed `final_eval`) and `af eval factor|null|canary`.
- Plan 4: research store, template and LLM proposers, orchestrator state machine, bandit, packet,
  reflector and beliefs, goals and the manager, pool rules and combiners, library freeze, Discord,
  HTML reports, Modal worker code, `af loop|goal|lib|family|report|final-eval`.
- DSL: `$bar` field. Contracts: `GoalProposals`, `CombinerSpec`, `LibFactor`, `FrozenLibrary`.
- Settings: empty environment values (a copied `.env.example`) now fall back to defaults.
- Synthetic universe file now uses Plan 2's schema.
