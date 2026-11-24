-- Table of the Chainlit app's login users.
-- Runs once when the Postgres volume is empty (after 01-init.sql); the seed script
-- (python -m supplyguard_ai.ingestion.seed_users) also runs it, so it must stay re-runnable.
--
-- The rows are NOT inserted here. seed_users fills the table from
-- support-data/config/users.csv and hashes each user's password from .env
-- (USER_PASSWORD_<USER_ID>). Only bcrypt hashes are stored, never passwords.
--
-- role decides what a user may do (approval tiers in support-data/config/policies.json),
-- document_scopes decide which documents retrieval may return to them, and
-- factory_access lists the plants they work with (F1 Stuttgart, F2 Katowice, F3 Lyon).

CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT        PRIMARY KEY,               -- internal id, e.g. u-pm-01 (lowercase)
    name            TEXT        NOT NULL,                  -- login name, e.g. Maria Keller
    role            TEXT        NOT NULL CHECK (role IN (
                        'procurement_manager', 'operations_analyst', 'incident_coordinator',
                        'finance_controller', 'finance_director', 'viewer')),
    document_scopes TEXT[]      NOT NULL,                  -- public, internal, procurement, finance-confidential
    factory_access  TEXT[]      NOT NULL DEFAULT '{}',
    password_hash   TEXT,                                  -- bcrypt; NULL = no login
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Users log in with their name, compared without upper/lower case or extra spaces
-- (the same rule as normalize() in src/supplyguard_ai/auth/users.py).
-- This index makes that lookup fast and forbids two users with the same login name.
CREATE UNIQUE INDEX IF NOT EXISTS users_login_name
    ON users (lower(regexp_replace(btrim(name), '\s+', ' ', 'g')));
