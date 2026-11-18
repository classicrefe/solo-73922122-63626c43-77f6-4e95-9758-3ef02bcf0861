"""PostgreSQL schema definition and initialisation.

Durability model
----------------
* ``batches`` carries every persisted fact about a submission and its
  decision.  Wafers are stored in a child table ``wafer_measurements``.
* A batch is processed inside a *single* transaction that inserts the
  decision and either appends the batch id to ``device_state.ref_batch_ids``
  (released only) or leaves it untouched.  Hence the decision, the reference
  history and data-lake visibility become visible atomically -- there is no
  observable intermediate ("half batch") state and the reference list can
  never be advanced twice.
* Row-level locking (``SELECT ... FOR UPDATE`` on ``device_state`` plus a
  transaction-scoped advisory lock on the device) serialises workers for one
  device while different devices proceed independently.

Wafer-number corrections
------------------------
* ``batches.revision`` counts the corrections applied to a batch; it is
  bumped in the same transaction that renames the wafers, so a correction
  and its revision become visible atomically.
* ``wafer_measurements.original_wafer_no`` keeps the number as uploaded in
  the original CSV; ``wafer_no`` is the current (possibly corrected) number.
  The primary key on ``(batch_id, wafer_no)`` is ``DEFERRABLE`` so a
  correction transaction can swap two numbers within the batch.
* ``correction_requests`` is the idempotency ledger: one row per applied
  correction request, keyed by the caller-supplied request id together with
  a hash of the request content, so a lost response can be replayed and a
  request id reused with different content is rejected.
"""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    id              BIGGENERATED_PLACEHOLDER,
    batch_no        TEXT        NOT NULL,
    device_id       TEXT        NOT NULL,
    sha256          TEXT        NOT NULL,
    size_bytes      BIGINT      NOT NULL,
    storage_path    TEXT        NOT NULL,
    accept_seq      BIGINT      NOT NULL,
    status          TEXT        NOT NULL CHECK (status IN ('PENDING','RELEASED','QUARANTINED')),
    median_nm       DOUBLE PRECISION,
    ref_mean_nm     DOUBLE PRECISION,
    deviation_nm    DOUBLE PRECISION,
    reference_ids   JSONB,
    decided_at      TIMESTAMPTZ,
    decided_by      TEXT,
    attempts        INTEGER     NOT NULL DEFAULT 0,
    revision        INTEGER     NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT batches_pkey PRIMARY KEY (id),
    CONSTRAINT batches_batch_no_uniq UNIQUE (batch_no),
    CONSTRAINT batches_device_accept_uniq UNIQUE (device_id, accept_seq)
);

CREATE TABLE IF NOT EXISTS wafer_measurements (
    batch_id    BIGINT  NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    wafer_no    TEXT    NOT NULL,
    original_wafer_no TEXT NOT NULL,
    thickness_nm INTEGER NOT NULL CHECK (thickness_nm > 0),
    -- Deferrable so a correction transaction can swap wafer numbers within
    -- a batch (uniqueness is then checked at COMMIT, not per row).
    CONSTRAINT wafer_measurements_pkey PRIMARY KEY (batch_id, wafer_no)
        DEFERRABLE INITIALLY IMMEDIATE
);

-- Idempotency ledger for wafer-number corrections (see module docstring).
CREATE TABLE IF NOT EXISTS correction_requests (
    request_id        TEXT PRIMARY KEY,
    batch_id          BIGINT  NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    batch_no          TEXT    NOT NULL,
    request_hash      TEXT    NOT NULL,
    corrections       JSONB   NOT NULL,
    expected_revision INTEGER NOT NULL,
    new_revision      INTEGER NOT NULL,
    applied_count     INTEGER NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_correction_requests_batch
    ON correction_requests (batch_id);

CREATE INDEX IF NOT EXISTS idx_batches_device_seq
    ON batches (device_id, accept_seq);
CREATE INDEX IF NOT EXISTS idx_batches_status
    ON batches (status) WHERE status = 'PENDING';
CREATE INDEX IF NOT EXISTS idx_wafers_batch ON wafer_measurements (batch_id);

-- One row per device: the durable processing cursor (last accepted sequence
-- that has been decided) and the FIFO list of released batch ids that form
-- the rolling reference window (most recent last).
CREATE TABLE IF NOT EXISTS device_state (
    device_id        TEXT PRIMARY KEY,
    last_decided_seq BIGINT NOT NULL DEFAULT 0,
    ref_batch_ids    JSONB  NOT NULL DEFAULT '[]'::jsonb,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# Idempotent upgrades for databases created by the previous schema version
# (fresh databases already have everything via SCHEMA_SQL; each statement
# below is a no-op there).
MIGRATION_SQL = """
ALTER TABLE batches ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 0;

ALTER TABLE wafer_measurements ADD COLUMN IF NOT EXISTS original_wafer_no TEXT;
UPDATE wafer_measurements SET original_wafer_no = wafer_no
 WHERE original_wafer_no IS NULL;
ALTER TABLE wafer_measurements ALTER COLUMN original_wafer_no SET NOT NULL;

-- Recreate the wafer primary key as DEFERRABLE if it predates corrections.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint
                WHERE conname = 'wafer_measurements_pkey'
                  AND NOT condeferrable) THEN
        ALTER TABLE wafer_measurements DROP CONSTRAINT wafer_measurements_pkey;
        ALTER TABLE wafer_measurements
            ADD CONSTRAINT wafer_measurements_pkey
            PRIMARY KEY (batch_id, wafer_no) DEFERRABLE INITIALLY IMMEDIATE;
    END IF;
END $$;
"""


async def init_schema(conn) -> None:
    # BIGSERIAL spelled out so the file stays readable on its own.
    sql = SCHEMA_SQL.replace("BIGGENERATED_PLACEHOLDER",
                             "BIGINT GENERATED ALWAYS AS IDENTITY")
    await conn.execute(sql)
    await conn.execute(MIGRATION_SQL)
