# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
-- Optional PostgreSQL backend for finkey_evals.FeedbackStore.
-- Apply with: psql "$DATABASE_URL" -f schema.sql

CREATE TABLE IF NOT EXISTS feedback (
    id                     UUID PRIMARY KEY,
    company_id             TEXT NOT NULL,
    session_id             TEXT,
    turn_id                TEXT,
    question               TEXT,
    answer                 TEXT,
    rating                 SMALLINT NOT NULL DEFAULT 0,  -- 1=up -1=down 0=neutral
    reason                 TEXT,
    eval_context_relevance DOUBLE PRECISION,
    eval_faithfulness      DOUBLE PRECISION,
    eval_groundedness      DOUBLE PRECISION,
    eval_overall           DOUBLE PRECISION,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_company_time
    ON feedback (company_id, created_at DESC);
