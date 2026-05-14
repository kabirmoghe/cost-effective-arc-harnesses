-- 002: convert run_id columns from TEXT to UUID
--
-- Prerequisite: consolidate.py must be run first so that all existing
-- run_id values are valid UUID strings.

ALTER TABLE explorers DROP CONSTRAINT explorers_run_id_fkey;
ALTER TABLE definers  DROP CONSTRAINT definers_run_id_fkey;

ALTER TABLE evals     ALTER COLUMN run_id TYPE UUID USING run_id::uuid;
ALTER TABLE explorers ALTER COLUMN run_id TYPE UUID USING run_id::uuid;
ALTER TABLE definers  ALTER COLUMN run_id TYPE UUID USING run_id::uuid;

ALTER TABLE explorers
    ADD CONSTRAINT explorers_run_id_fkey
    FOREIGN KEY (run_id) REFERENCES evals(run_id) ON DELETE CASCADE;

ALTER TABLE definers
    ADD CONSTRAINT definers_run_id_fkey
    FOREIGN KEY (run_id) REFERENCES evals(run_id) ON DELETE CASCADE;
