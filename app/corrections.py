"""Wafer-number corrections for released batches.

A data clerk who discovers that wafer numbers of an already *released*
batch were recorded wrongly posts a correction naming the batch, an
idempotency ``request_id``, the ``expected_revision`` the clerk saw, and a
mapping ``{current_wafer_no: new_wafer_no}``.

Guarantees (all enforced inside one transaction per request):

* only ``RELEASED`` batches can be corrected -- pending or quarantined
  batches are rejected with 409;
* optimistic concurrency: the batch row is locked ``FOR UPDATE`` and the
  stored ``revision`` must equal ``expected_revision``; when two instances
  submit different requests against the same revision, exactly one commits
  and the other fails with 409;
* idempotency: the applied result is persisted in ``correction_requests``
  keyed by ``request_id`` together with a hash of the request content.
  Repeating the same request returns the stored result without applying
  anything twice; reusing the request id with different content is a 409;
* atomicity: the wafer renames, the revision bump and the ledger row commit
  together -- a failed request leaves no partial correction behind;
* numbers may be swapped within the batch (the primary key on
  ``(batch_id, wafer_no)`` is deferred to commit time); after the
  correction the numbers must still be unique within the batch;
* thickness, the original CSV, the release decision and the reference
  history are never touched.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

import asyncpg

WAFER_PKEY_CONSTRAINT = "wafer_measurements_pkey"
REQUEST_ID_CONSTRAINT = "correction_requests_pkey"


class BatchNotFoundError(Exception):
    """HTTP 404: no batch with this number exists."""


class CorrectionValidationError(ValueError):
    """HTTP 400: the correction mapping itself is invalid."""


class CorrectionConflictError(Exception):
    """HTTP 409: state conflict (not released / revision / request id)."""

    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = details


def _canonical_hash(batch_no: str, expected_revision: int,
                    corrections: Dict[str, str]) -> str:
    """Stable fingerprint of the request content for idempotency checks."""
    payload = json.dumps(
        {
            "batch_no": batch_no,
            "expected_revision": expected_revision,
            "corrections": corrections,
        },
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_mapping(mapping: Dict[str, str], current: set[str]) -> None:
    """Validate the mapping against the batch's current wafer numbers."""
    unknown = sorted(set(mapping) - current)
    if unknown:
        raise CorrectionValidationError(
            f"晶圆号不存在于该批当前编号中: {unknown} / "
            "unknown current wafer number(s)")
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        raise CorrectionValidationError(
            "更正目标编号互相重复 / duplicate target wafer numbers")
    # Targets may reuse numbers that are themselves being remapped (swaps and
    # chains are fine) but must never collide with an unmapped number.
    collisions = sorted(set(targets) & (current - set(mapping)))
    if collisions:
        raise CorrectionValidationError(
            f"更正目标编号与未更正的现有编号冲突: {collisions} / "
            "target number collides with an unmapped wafer")


def _response(batch_no: str, request_id: str, previous_revision: int,
              new_revision: int, corrections: Dict[str, str],
              replayed: bool) -> Dict[str, Any]:
    return {
        "batch_no": batch_no,
        "request_id": request_id,
        "status": "RELEASED",
        "previous_revision": previous_revision,
        "revision": new_revision,
        "applied": len(corrections),
        "corrections": corrections,
        "replayed": replayed,
        "message": ("相同请求号重试，返回原更正结果 / idempotent replay"
                    if replayed else
                    "晶圆号更正已生效 / wafer numbers corrected"),
    }


def _replay_response(row: asyncpg.Record) -> Dict[str, Any]:
    return _response(
        batch_no=row["batch_no"], request_id=row["request_id"],
        previous_revision=row["expected_revision"],
        new_revision=row["new_revision"],
        corrections=dict(row["corrections"]), replayed=True,
    )


async def correct_wafer_numbers(
    pool: asyncpg.pool.Pool,
    *,
    batch_no: str,
    request_id: str,
    expected_revision: int,
    corrections: Dict[str, str],
) -> Dict[str, Any]:
    """Apply a wafer-number correction, or replay/conflict an existing one."""
    # Normalise: strip whitespace; reject empty keys/targets up front.
    mapping: Dict[str, str] = {}
    for raw_old, raw_new in corrections.items():
        old, new = raw_old.strip(), raw_new.strip()
        if not old or not new:
            raise CorrectionValidationError(
                "晶圆号不能为空 / wafer numbers must be non-empty")
        mapping[old] = new
    if not mapping:
        raise CorrectionValidationError("更正映射为空 / empty correction mapping")
    request_id = request_id.strip()
    if not request_id:
        raise CorrectionValidationError("请求号不能为空 / request id required")

    req_hash = _canonical_hash(batch_no, expected_revision, mapping)
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                return await _apply(
                    conn, batch_no=batch_no, request_id=request_id,
                    expected_revision=expected_revision,
                    mapping=mapping, req_hash=req_hash)
    except asyncpg.UniqueViolationError as exc:
        if exc.constraint_name == REQUEST_ID_CONSTRAINT:
            # Lost a race against the same request id on another instance
            # (only possible across batches; same-batch races serialise on
            # the batch row lock instead).  The winner's row is committed
            # now: replay it if identical, conflict otherwise.
            return await _resolve_request_id_race(pool, request_id, req_hash)
        if exc.constraint_name == WAFER_PKEY_CONSTRAINT:  # pragma: no cover
            # Defensive: the deferred uniqueness check caught a collision our
            # pre-validation missed.  The transaction rolled back, so no
            # partial correction is visible.
            raise CorrectionValidationError(
                "更正后晶圆号不唯一 / wafer numbers not unique after correction")
        raise


async def _apply(conn: asyncpg.Connection, *, batch_no: str, request_id: str,
                 expected_revision: int, mapping: Dict[str, str],
                 req_hash: str) -> Dict[str, Any]:
    # Lock the batch row first: this serialises concurrent corrections of the
    # same batch, so the revision check below is race-free.
    batch = await conn.fetchrow(
        "SELECT id, batch_no, status, revision FROM batches "
        "WHERE batch_no = $1 FOR UPDATE",
        batch_no,
    )
    if batch is None:
        raise BatchNotFoundError(batch_no)

    # Idempotent replay of an already-committed identical request.
    existing = await conn.fetchrow(
        "SELECT * FROM correction_requests WHERE request_id = $1", request_id)
    if existing is not None:
        if existing["request_hash"] != req_hash:
            raise CorrectionConflictError(
                f"请求号 {request_id} 已被不同内容的请求占用 / "
                "request id already used with different content",
                request_id=request_id)
        return _replay_response(existing)

    if batch["status"] != "RELEASED":
        raise CorrectionConflictError(
            f"仅已放行批可更正，当前状态 {batch['status']} / "
            "only RELEASED batches can be corrected",
            status=batch["status"])
    if batch["revision"] != expected_revision:
        raise CorrectionConflictError(
            f"预期修订号 {expected_revision} 与当前修订号 {batch['revision']} "
            "不一致 / revision mismatch",
            current_revision=batch["revision"])

    rows = await conn.fetch(
        "SELECT wafer_no FROM wafer_measurements WHERE batch_id = $1",
        batch["id"])
    _validate_mapping(mapping, {r["wafer_no"] for r in rows})

    # Defer the (batch_id, wafer_no) uniqueness check to COMMIT so numbers
    # can be swapped within the batch.
    await conn.execute(
        f"SET CONSTRAINTS {WAFER_PKEY_CONSTRAINT} DEFERRED")
    # One statement: matching is evaluated against the pre-update row
    # versions, so swaps/chains cannot contaminate each other's filters.
    tag = await conn.execute(
        """
        UPDATE wafer_measurements AS w
           SET wafer_no = m.new_no
          FROM unnest($2::text[], $3::text[]) AS m(old_no, new_no)
         WHERE w.batch_id = $1 AND w.wafer_no = m.old_no
        """,
        batch["id"], list(mapping.keys()), list(mapping.values()),
    )
    if int(tag.split()[-1]) != len(mapping):  # pragma: no cover - defensive
        raise CorrectionValidationError(
            "更正映射与当前编号不一致 / mapping does not match current numbers")
    # The row lock above already guarantees the revision, but keep the guard
    # so the bump is provably applied to the expected revision only.
    new_revision = await conn.fetchval(
        "UPDATE batches SET revision = revision + 1 "
        "WHERE id = $1 AND revision = $2 RETURNING revision",
        batch["id"], expected_revision)
    if new_revision is None:  # pragma: no cover - impossible under the lock
        raise CorrectionConflictError(
            "修订号已变化 / revision changed concurrently")

    await conn.execute(
        """
        INSERT INTO correction_requests
            (request_id, batch_id, batch_no, request_hash, corrections,
             expected_revision, new_revision, applied_count)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8)
        """,
        request_id, batch["id"], batch_no, req_hash, mapping,
        expected_revision, new_revision, len(mapping),
    )
    return _response(batch_no, request_id, expected_revision, new_revision,
                     mapping, replayed=False)


async def _resolve_request_id_race(pool: asyncpg.pool.Pool, request_id: str,
                                   req_hash: str) -> Dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM correction_requests WHERE request_id = $1",
            request_id)
    if row is not None and row["request_hash"] == req_hash:
        return _replay_response(row)
    raise CorrectionConflictError(
        f"请求号 {request_id} 已被不同内容的请求占用 / "
        "request id already used with different content",
        request_id=request_id)
