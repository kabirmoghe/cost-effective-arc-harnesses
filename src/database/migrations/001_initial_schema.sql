-- 001: initial schema

CREATE TABLE evals (
    run_id      TEXT PRIMARY KEY,
    system      TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE explorers (
    agent_id    UUID PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES evals(run_id) ON DELETE CASCADE,
    task_id     TEXT NOT NULL,
    agent       TEXT NOT NULL,
    output      JSONB,
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, task_id, agent)
);

CREATE INDEX explorers_run_task_idx ON explorers (run_id, task_id);

CREATE TABLE definers (
    agent_id            UUID PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES evals(run_id) ON DELETE CASCADE,
    task_id             TEXT NOT NULL,
    agent               TEXT NOT NULL,
    parent_explorer_ids UUID[] NOT NULL DEFAULT '{}',
    output              JSONB,
    metadata            JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, task_id, agent)
);

CREATE INDEX definers_run_task_idx ON definers (run_id, task_id);
CREATE INDEX definers_parent_explorers_idx ON definers USING GIN (parent_explorer_ids);
