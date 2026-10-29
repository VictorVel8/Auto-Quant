PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE goals (
  goal_id      TEXT PRIMARY KEY,
  status       TEXT NOT NULL CHECK (status IN ('pending','active','done','abandoned')),
  goal_json    TEXT NOT NULL,
  trials_used  INTEGER NOT NULL DEFAULT 0,
  created_at   TEXT NOT NULL,
  approved_at  TEXT,
  closed_at    TEXT,
  outcome      TEXT
);

CREATE TABLE llm_calls (
  call_id    TEXT PRIMARY KEY,
  role       TEXT NOT NULL,
  model      TEXT NOT NULL,
  tokens_in  INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  cost_usd   REAL NOT NULL,
  ok         INTEGER NOT NULL,
  record_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TRIGGER llm_calls_append_only_u BEFORE UPDATE ON llm_calls
  BEGIN SELECT RAISE(ABORT, 'llm_calls are append-only'); END;
CREATE TRIGGER llm_calls_append_only_d BEFORE DELETE ON llm_calls
  BEGIN SELECT RAISE(ABORT, 'llm_calls are append-only'); END;

CREATE TABLE batches (
  batch_id       TEXT PRIMARY KEY,
  goal_id        TEXT NOT NULL REFERENCES goals,
  state          TEXT NOT NULL,
  proposer       TEXT NOT NULL CHECK (proposer IN ('llm','template')),
  directive_json TEXT NOT NULL,
  packet_json    TEXT,
  response_json  TEXT,
  llm_call_id    TEXT REFERENCES llm_calls,
  error          TEXT,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);

CREATE TABLE families (
  family     TEXT PRIMARY KEY,
  status     TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','exhausted','refuted')),
  created_by TEXT NOT NULL CHECK (created_by IN ('seed','llm','manual')),
  created_at TEXT NOT NULL
);

CREATE TABLE proposals (
  proposal_id TEXT PRIMARY KEY,
  batch_id    TEXT NOT NULL REFERENCES batches,
  goal_id     TEXT NOT NULL REFERENCES goals,
  expr_hash   TEXT NOT NULL,
  family      TEXT NOT NULL,
  record_json TEXT NOT NULL,
  created_at  TEXT NOT NULL
);

CREATE TABLE trials (
  trial_id      TEXT PRIMARY KEY,
  batch_id      TEXT REFERENCES batches,
  proposal_id   TEXT REFERENCES proposals,
  source        TEXT NOT NULL CHECK (source IN ('llm','template','sign_flip','manual')),
  expr_hash     TEXT,
  family        TEXT,
  horizon       TEXT,
  sign_used     INTEGER,
  sandbox_hash  TEXT NOT NULL,
  dataset_hash  TEXT NOT NULL,
  code_commit   TEXT NOT NULL,
  reason        TEXT NOT NULL,
  evaluated     INTEGER NOT NULL,      -- 1 if it reached LOW_IC or later; counts toward N
  note          TEXT,                  -- JSON: expression and message, for trials with no proposal
  created_at    TEXT NOT NULL
);
CREATE UNIQUE INDEX trials_once ON trials (sandbox_hash, expr_hash, horizon, sign_used)
  WHERE evaluated = 1;
CREATE INDEX trials_by_family ON trials (family, evaluated);
CREATE INDEX trials_by_batch ON trials (batch_id);
CREATE TRIGGER trials_append_only_u BEFORE UPDATE ON trials
  BEGIN SELECT RAISE(ABORT, 'trials are append-only'); END;
CREATE TRIGGER trials_append_only_d BEFORE DELETE ON trials
  BEGIN SELECT RAISE(ABORT, 'trials are append-only'); END;

CREATE TABLE eval_results (
  trial_id    TEXT PRIMARY KEY REFERENCES trials,
  result_json TEXT NOT NULL,
  n_at_eval   INTEGER NOT NULL,         -- N used for the deflated t bar
  ic_tstat    REAL NOT NULL,
  holdout_ok  INTEGER                   -- NULL: not checked; never a holdout number
);
CREATE TRIGGER eval_results_append_only_u BEFORE UPDATE ON eval_results
  BEGIN SELECT RAISE(ABORT, 'eval_results are append-only'); END;
CREATE TRIGGER eval_results_append_only_d BEFORE DELETE ON eval_results
  BEGIN SELECT RAISE(ABORT, 'eval_results are append-only'); END;

CREATE TABLE beliefs (
  belief_id   TEXT PRIMARY KEY,
  family      TEXT NOT NULL,
  horizon     TEXT NOT NULL,
  statement   TEXT NOT NULL,
  next_test   TEXT,
  supersedes  TEXT REFERENCES beliefs,
  retired     INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);

CREATE TABLE belief_evidence (
  belief_id  TEXT NOT NULL REFERENCES beliefs,
  trial_id   TEXT NOT NULL REFERENCES trials,
  direction  TEXT NOT NULL CHECK (direction IN ('support','contradict')),
  batch_id   TEXT NOT NULL REFERENCES batches,
  PRIMARY KEY (belief_id, trial_id)
);

CREATE TABLE reflections (
  batch_id    TEXT PRIMARY KEY REFERENCES batches,
  record_json TEXT NOT NULL,
  created_at  TEXT NOT NULL
);

CREATE TABLE pool (
  factor_id       TEXT PRIMARY KEY,
  trial_id        TEXT NOT NULL REFERENCES trials,
  expr_hash       TEXT NOT NULL,
  family          TEXT NOT NULL,
  horizon         TEXT NOT NULL,
  sign_used       INTEGER NOT NULL,
  added_at        TEXT NOT NULL,
  removed_at      TEXT,
  removed_reason  TEXT
);

CREATE TABLE factor_libs (
  lib_version  TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL,
  lib_json     TEXT NOT NULL,
  report_json  TEXT,
  frozen_at    TEXT NOT NULL
);

CREATE TABLE final_evals (
  request_hash TEXT PRIMARY KEY,
  lib_version  TEXT NOT NULL REFERENCES factor_libs,
  report_json  TEXT NOT NULL,
  imported_at  TEXT NOT NULL
);
CREATE TRIGGER final_evals_append_only_u BEFORE UPDATE ON final_evals
  BEGIN SELECT RAISE(ABORT, 'final_evals are append-only'); END;
CREATE TRIGGER final_evals_append_only_d BEFORE DELETE ON final_evals
  BEGIN SELECT RAISE(ABORT, 'final_evals are append-only'); END;
