# Plans

Plans 1 to 4 built the factor-discovery system (the paper's Part I). The next three are small
steps, each with an exit check that has to pass before the next one starts.

| Plan | What it delivers | Depends on |
|---|---|---|
| [5](plan-5-part-one-verdict.md) | Part I hardened and run for real on intraday; Modal deployed; a written verdict on intraday at a $50k book | Plans 1-4 |
| [6](plan-6-timeframes.md) | The timeframe becomes a sandbox parameter: a daily-bar swing sandbox (holds of days to weeks) next to the intraday one | 5 |
| [7](plan-7-models.md) | Part II foundations: trainable models (ridge, LightGBM, MLP, GRU, LSTM, TCN) scored by the same sealed evaluator, CPU locally, GPU on Modal under a budget | 6 |

Later, not planned in detail yet:

- **Plan 8**: the autonomous model loop (the LLM proposes config diffs, a model knowledge store),
  the hybrid architecture (conv stem, attention, cross-sectional mixer), and frozen factor
  libraries as model inputs (the paper's "next direction").
- **Plan 9**: sealed final evaluations per sandbox, a walk-forward report, paper trading.

Every plan keeps the six invariants (`docs/invariants.md`). In particular, rule 6: nothing in a
plan is tied to the laptop or the mini PC.
