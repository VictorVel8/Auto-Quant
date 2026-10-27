# Invariants

Five rules that are never broken. Any LLM or coding agent working on this repo must keep them.

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

Non-goals for v1: HFT, live auto-trading, multi-user features, investment advice.
