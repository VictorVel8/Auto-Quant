# Invariants

Six rules that are never broken. Any LLM or coding agent working on this repo must keep them.

1. **The agent never writes executable code.** It emits only DSL expressions and config diffs,
   validated by Pydantic and the DSL parser.
2. **Sealed packages stay sealed.** `data`, `sandbox`, `dsl`, `eval`, `backtest` and `final_eval`
   never import agent code, and agent code can't read the test volume. Enforced by
   `.importlinter` in CI (`uv run lint-imports`).
3. **Every trial is recorded and counted**, including rejects, duplicates and reversed signs.
4. **Code computes every decision-driving number** (metrics, confidence, costs). An LLM never
   writes one.
5. **No market data in the repo.** Tests and CI use the synthetic market. `scripts/data_guard.py`
   fails if a `.parquet`, `.csv.gz` or `.sqlite` file is tracked outside `tests/fixtures/`.
6. **The laptop and the mini PC are interchangeable.** Both run the same commit, the same configs
   and the same credentials, and any command (tests, data builds, the loop, Modal, `final-eval`)
   runs on either. Code never checks which machine it is on, and no doc, help text or docstring
   gives a machine a role. Only paths in `.env` may differ.

What rule 6 means in practice:

- **Each machine has its own research store** (`AF_STORE_PATH`). Trials, beliefs and the pool on
  one machine do not include the other's, so the trial count N used to deflate t-stats covers
  one store only. Run a goal on one machine, or count both when you judge a result.
- **There are two final-eval ledgers**: one on the sealed Modal volume (`af final-eval`) and one
  per machine (`af final-eval --local`, `$AF_DATA_DIR/final_eval_ledger.jsonl`). Each refuses a
  request it has already seen, but none sees the others. Scoring the sealed window once per
  library is your discipline across ledgers, not something the code can enforce.
- **The sealed boundary is in code, not in who holds what**: the import-linter contracts, the
  `MarketData("full", sealed=True)` guard and the separate Modal environments keep the loop away
  from the sealed window. Both machines hold the token that can reach it.

Non-goals for v1: HFT, live auto-trading, multi-user features, investment advice.
