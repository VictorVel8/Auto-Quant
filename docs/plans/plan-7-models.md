# Plan 7: trainable models (Part II foundations)

## Goal

Train non-linear models on the same data and score them with **the same sealed evaluator** as
factors: the same labels, folds, IC statistics, backtester, costs and gate. This plan builds the
paper's model ladder (its Table 2) on our data, for both sandboxes, with CPU training locally and
GPU training on Modal under a hard budget:

ridge → LightGBM → MLP → GRU / LSTM → TCN / small transformer

What is left for Plan 8: the LLM proposing config diffs on its own, the hybrid architecture, and
factor libraries as inputs. Plan 7 makes all of that possible, because every training run is
already a hashed config recorded in a store.

## Why after Plan 6

- A GRU's IC means something only next to ridge and LightGBM under the same evaluator, on the
  timeframe where costs leave room for alpha.
- Neural networks amplify leaks (the paper's Appendix B). The causality and canary guarantees
  have to hold on both timeframes before any GPU money is spent.

## Decisions for the owner before starting

1. **Budget.** Proposal: `total_budget_usd: 100` (out of $130, keeping $30 for Plan 8 and the dev
   app), `max_usd_per_run: 3`, and `default_gpu: L4`. Look up current Modal GPU prices and put
   them in `configs/models.yaml`, the same way `llm.yaml` holds LLM prices.
2. **Model splits.** Proposal:
   - train on folds F1-F3, with the last 10% of their sessions (purged) as inner validation for
     early stopping;
   - report validation on F4;
   - the gate holdout stays pass/fail, and the sealed window is scored only through
     `final_eval`.
   - Walk-forward retraining comes later (Plan 9), because it multiplies the cost by the number
     of folds.
3. **Feature set v1.** Price and volume only, written as DSL expressions (task 2). Frozen
   factor-library inputs are deferred to Plan 8.

## Design

- **Features are DSL expressions.** A feature set is a list of named DSL expressions in
  `configs/models.yaml` (sealed by ownership). The existing compiler computes them, so they are
  causal by construction, and the existing property tests already cover them. Normalisation is a
  cross-sectional rank or z-score at each timestamp, so no statistics are ever fitted on future
  data. This removes the paper's `stats_window: train_only` risk entirely.
- **The model is a config.** `ModelConfig` names a sandbox, a feature set, a label, the history
  length, and blocks from a registry (`linear`, `mlp`, `gru`, `lstm`, `tcn`,
  `transformer_encoder`, plus heads and pooling), the loss, the optimizer, a seed and the compute
  target. Its content hash identifies the run. In Plan 8 the LLM emits only these configs, so
  invariant 1 still holds.
- **Output = a score frame.** A trained model emits `(ticker_id, ts, score)`. That goes through
  the same IC, fold, by-year, backtest and verdict code as a factor or a combined library, so every
  row of the ladder is comparable.
- **Data on the GPU.** The features become a dense float tensor `[T, N, F]`, plus labels and
  masks:
  - intraday: about 7,900 timestamps × 1,000 names × 24 features ≈ 0.8 GB in float32, which fits
    on one L4;
  - swing: about 60 MB, which trains on a CPU or a Mac (MPS).
  - Training windows are gathered by index on the device rather than materialised, so one run
    should take tens of minutes, not hours.

## Tasks

### 1. Package, dependencies and boundaries

- New package `models/`, on the sealed side: it never imports `agents`, `orchestrator` or
  `research`. Add it to the first contract in `.importlinter`.
- New dependency group `models = ["torch", "lightgbm"]`. LightGBM is already a dependency but
  unused.
- Pyright strict on `models/`, except the torch modules, which are checked in basic mode.

### 2. Contracts and configuration

- `contracts/models.py`:
  - `FeatureSetSpec`;
  - `ModelConfig` (a discriminated union for the blocks, with typed parameters and bounded
    ranges);
  - `ComputeSpec{device: cpu|mps|gpu, gpu_type, max_minutes}`;
  - `ModelRunResult`.
- `configs/models.yaml`: feature set `pv_v1`, about 20-25 expressions:
  - returns at several lags;
  - realised volatility;
  - volume z-score;
  - high-low range;
  - distance to VWAP;
  - gap;
  - bar of day (intraday only).
- The same file also holds GPU prices and the budget.
- Tests: the configs round-trip and their hash is stable; an unknown block or an out-of-range
  parameter is rejected.

### 3. The tensor dataset

- `models/data.py` builds `[T, N, F]` features, labels and masks from the dev panel for a
  sandbox. The arrays are cached under `$AF_DATA_DIR/cache/models/<dataset>/<sandbox>/<features>/`
  as `.npy` shards.
- A window sampler returns `[B, H, F]` for (timestamp, names); `T` is the index into the cached
  array.
- Splits come from decision 2, with a purge of `history + longest label` bars between train,
  inner validation and validation.
- Tests on the synthetic market:
  - the planted signal is recoverable by ridge on these tensors;
  - a window never contains a bar after its timestamp (a property test);
  - no training sample's label reaches into the validation range.

### 4. Baselines on the CPU

- Ridge and LightGBM on the last-step features plus a few lags, trained locally.
- Their score frames go through task 6. This is the floor every neural model must beat.

### 5. The torch registry and trainer

- `models/arch.py`: `linear`, `mlp`, `gru`, `lstm`, `tcn`, `transformer_encoder`, pooling
  (`last`, `mean`), and an MLP head.
- `models/train.py`:
  - fixed seeds;
  - bf16 autocast on GPU;
  - losses: cross-sectionally demeaned MSE or Huber, or a differentiable IC;
  - AdamW with a cosine schedule;
  - early stopping on inner-validation IC;
  - keep the best checkpoint, then emit the validation score frame.
- Tests:
  - a tiny GRU overfits a planted pattern on CPU;
  - two CPU runs with the same seed give identical validation IC;
  - a model given a deliberately leaked feature is caught by the canary path (task 6).

### 6. Model evaluation through the sealed evaluator

- `models/evaluate.py` turns a score frame into the usual metrics and a `ReasonCode`:
  - validation IC, its t-stat and by year;
  - net Sharpe from the backtester of the model's sandbox;
  - turnover and cost share.
- The gate holdout runs as pass/fail only, as for factors.
- Both canaries are re-run as "models" with a leaked feature set: they must be flagged.

### 7. Compute backends and the budget guard

- `LocalTrainer` (CPU or MPS) and `ModalTrainer`.
- `deploy/modal_models.py`: app `alpha-factory-models` in the `dev` environment, volume
  `af-models-dev` holding the tensor shards, and one function `train(config_json)` with the GPU
  type taken from the config and the timeout from `max_minutes`.
- `af model publish` uploads the shards for a sandbox and checks their hash on the worker, the
  same way `StaleDeploymentError` works today.
- **Budget guard (fails closed):** before a launch, the worst case (`max_minutes` × price) must
  fit both under `max_usd_per_run` and in the remaining `total_budget_usd`. After a run, the
  actual cost (duration × price) is recorded. If no price is configured, nothing launches.
- Tests: the guard refuses a run over budget and a run with no price; a fake Modal returns a
  result that round-trips.

### 8. Every run is recorded

- Store migration `002_models.sql`: a `model_runs` table holding the config hash and JSON, the
  sandbox and dataset hashes, the code commit, the backend, the GPU, the duration, the cost, the
  metrics and the reason code.
- Every run counts toward a model trial N per sandbox (invariant 3), which deflates the model
  t-stats the same way the factor loop does.

### 9. CLI

- `af model features build --sandbox S`, `af model publish --sandbox S`
- `af model train CONFIG.yaml [--backend local|modal]`
- `af model list`, `af model show RUN`, `af model budget`

### 10. The ladder experiment

- For each sandbox (`intraday_30m`, `swing_1d`): ridge, LightGBM, MLP, GRU, LSTM, TCN, and a small
  transformer. Run the best two with 3 seeds each.
- Intraday sequence models on Modal; swing models locally.
- New `docs/models.md`: the ladder table (validation IC, its t-stat, net Sharpe, turnover,
  cost, wall time), what ran, what did not, and the spend.

## Exit check

- [ ] Window causality, purge and canary tests pass for both timeframes.
- [ ] The ladder table exists for both sandboxes, with seeds reported for the top two.
- [ ] Recorded spend matches the Modal dashboard within 10%; total Plan 7 spend at most $40.
- [ ] Every run is in `model_runs`; `af model budget` shows the remaining credit.
- [ ] Full suite, pyright and lint-imports pass.

## Rough budget (check current prices)

At about $0.80 per hour for an L4, $40 buys about 50 GPU-hours. At 15 to 30 minutes per intraday
run, that is roughly 100 to 200 runs: enough for the ladder plus seeds. Swing runs cost nothing on
Modal. The rest of the $130 stays for Plan 8's autonomous loop.

## Risks

- **Neural models do not beat LightGBM.** That is a valid result, and it tells Plan 8 where to
  search.
- **GPU non-determinism.** Reproducibility is required exactly on CPU and within a tolerance on GPU
  (|ΔIC| < 0.002), and the seed spread is reported.
- **Selection on F4.** Every run is counted, and the sealed year stays untouched until a model is
  frozen against `docs/final-eval-criteria.md`.
