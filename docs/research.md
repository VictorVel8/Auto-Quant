# Research loop (Plan 4)

The loop stores every trial, proposes factors (templates first, then an LLM), learns beliefs from
results, allocates effort across families, freezes factor libraries and reports to you. It is the
only part with LLM code and it never touches sealed data.

## Layout

| Package | Contents |
|---|---|
| `research` | `store` (SQLite, append-only), `config`, `bandit`, `notify` (Discord), `report`, `runtime` (wiring) |
| `agents` | `template` and `llm_proposer`, `llm_client`, `packet`, `feedback`, `prompts/`, `reflect`, `manager`, `flow` (repair) |
| `orchestrator` | `loop` (the state machine), `lock`, `ops` (re-evaluation) |
| `compute` | `specs`, `worker` (Evaluator, LocalBackend), `modal_dev` (ModalBackend, dev image and functions), `modal_sealed` |
| `eval` | `combine` (rank average, ridge) and `library` (LibraryScorer, freeze report): sealed-side, because `final_eval` needs them |
| `deploy/` | Modal app definitions, the systemd unit |

Import-linter enforces: sealed packages never import the loop, and the loop (`agents`,
`orchestrator`, `research`) never imports `sandbox.final_eval` or `eval.library`, even indirectly.

## Running it

```sh
cp .env.example .env                      # fill in the variables you need
uv run af goal add goals/bootstrap.yaml   # the three bootstrap goals
uv run af goal list && uv run af goal approve <id>
uv run af loop run                        # or: sudo systemctl enable --now alpha-factory-loop
uv run af loop status
uv run af report run                      # reports/report_<date>.html
uv run af lib freeze --combiner rank_average
uv run af final-eval lib_v1               # from the MacBook, once
uv run af lib import-final reports/final_eval_lib_v1.json --lib lib_v1
```

`configs/loop.yaml` sets the proposer (`template` first, `llm` for P5), batch size, daily batch
cap, daily LLM budget, the evaluation backend (`local` or `modal`) and the pool cap.
`configs/llm.yaml` needs the model ids and prices. Needed variables: `LLM_BASE_URL`,
`LLM_API_KEY` (LLM), `DISCORD_WEBHOOK_URL` (optional), `MODAL_TOKEN_*` and
`AF_MODAL_ENVIRONMENT` (Modal).

## Where this build departs from the plan

- **Dedupe key.** Trials are keyed by expression **and condition** (`full_hash`), not the
  expression alone: a `refine` proposal reuses its parent's expression, so the plan's key would
  have called every refinement a duplicate.
- **Deflation N.** Every trial in a batch is judged under `n_evaluated + batch size`, so the
  verdict does not depend on order inside the batch. It is stored with the result.
- **Re-evaluation** rebuilds the pool as it stood at the trial's time (the store keeps when each
  factor entered and left), because the CORRELATED verdict depends on it.
- **The holdout runs in the worker** (it needs the data) and returns a bool, also for a
  CORRELATED-only trial that might replace a pool factor.
- **Clusters.** The store has no pairwise correlations, so "correlation cluster" is approximated
  by family in the packet and in `combine` proposals.
- **A contradiction needs a refuted hypothesis.** `contradict` evidence is dropped in code when
  the trial's `hypothesis_refuted` is false (the reflector prompt asks for this; code enforces it).
- **Combiners live in `eval/`**, not `research/`, so the sealed side can import them.
- **LLM client** uses `httpx` against the OpenAI-compatible chat endpoint (not the `openai` SDK),
  which keeps it testable with a mock transport.
- **`af final-eval import`** is `af lib import-final FILE --lib LIB`: Typer cannot take both a
  positional library and a subcommand on one command name.
- **`$bar`** was added to the DSL (the plan's refine templates use it).
- **Modal apps** are defined at module level in `deploy/` (Modal requires it); the shared logic
  is in `compute/modal_dev.py` and `modal_sealed.py`.

## Verified, and not

Verified on the synthetic market and with fakes: the store (append-only triggers, dedupe, atomic
transactions); the state machine, including a kill at each of 14 points; idempotent re-runs;
bandit, directive and family status; the template proposer; the LLM client against a mock
provider (retries, cost, strict schema); repair; packet assembly and trimming; prompt snapshots;
belief application; pool admission, replacement and cap; combiners; library freeze and the sealed
evaluation through the same scorer; reports; the whole loop through the CLI.

**Not verified, because it needs something this machine does not have:**
- Modal: the apps import and register their functions offline, but nothing was deployed or run.
  Whether a token can be restricted to one environment, and that the mini PC's credential cannot
  invoke the sealed app, must be tested on your account (checkpoint P6).
- Your LLM provider: whether it accepts strict JSON schemas, real output quality, real cost.
- Discord delivery (tested against a mock transport only).
- The systemd unit was written but not installed.
- 1,000 unattended template trials on the mini PC, the template-versus-LLM comparison and a
  real sealed evaluation.
