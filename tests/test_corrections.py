"""Wafer-number correction API: idempotency, revision CAS, atomicity.

Covers the clerk workflow of fixing wrongly recorded wafer numbers on an
already released batch:
* happy path + read models (revision in batch detail, corrected *and*
  original numbers in /data/released, thickness untouched);
* swaps within the same batch are allowed, uniqueness still enforced;
* only RELEASED batches can be corrected (PENDING/QUARANTINED -> 409);
* optimistic concurrency: two different requests against the same revision
  -> exactly one wins;
* idempotent replay by request id; different content reusing the id -> 409;
* failed requests leave no partial correction behind;
* the release decision, reference history and original CSV are unaffected.
"""
import asyncio

import pytest

from tests.conftest import make_csv, submit, uniform_csv, wait_status


async def _release(ac, batch_no, device, rows):
    r = await submit(ac, batch_no, device, make_csv(rows))
    assert r.status_code == 202, r.text
    return await wait_status(ac, batch_no, "RELEASED")


async def _correct(ac, batch_no, request_id, expected_revision, corrections):
    return await ac.post(
        f"/batches/{batch_no}/wafer-corrections",
        json={"request_id": request_id,
              "expected_revision": expected_revision,
              "corrections": corrections})


async def _released_item(ac, batch_no):
    r = await ac.get("/data/released", params={"batch_no": batch_no})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    return items[0]


def _wafer_map(item):
    return {w["wafer_no"]: (w["original_wafer_no"], w["thickness_nm"])
            for w in item["wafers"]}


@pytest.mark.asyncio
async def test_correction_happy_path_and_read_models(client):
    ac = client
    before = await _release(ac, "C-BASE", "EQ-COR",
                            [("W1", 500), ("W2", 501), ("W3", 502), ("W4", 503)])
    assert before["revision"] == 0  # batch detail exposes the revision

    r = await _correct(ac, "C-BASE", "REQ-1", 0, {"W1": "W9", "W2": "W8"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["revision"] == 1 and body["previous_revision"] == 0
    assert body["applied"] == 2 and body["replayed"] is False
    assert body["corrections"] == {"W1": "W9", "W2": "W8"}

    after = await ac.get("/batches/C-BASE")
    after = after.json()
    assert after["revision"] == 1
    # Decision facts are untouched by the correction.
    for field in ("status", "median_nm", "reference_mean_nm", "deviation_nm",
                  "reference_batch_nos", "sha256", "original_storage_path",
                  "attempts", "decided_by", "decided_at"):
        assert after[field] == before[field], field

    item = await _released_item(ac, "C-BASE")
    assert item["revision"] == 1
    assert _wafer_map(item) == {
        "W9": ("W1", 500), "W8": ("W2", 501),
        "W3": ("W3", 502), "W4": ("W4", 503),
    }


@pytest.mark.asyncio
async def test_swap_within_batch_allowed(client):
    ac = client
    await _release(ac, "C-SWAP", "EQ-COR", [("W1", 100), ("W2", 200)])
    r = await _correct(ac, "C-SWAP", "REQ-SWAP", 0, {"W1": "W2", "W2": "W1"})
    assert r.status_code == 200, r.text
    item = await _released_item(ac, "C-SWAP")
    # Numbers swapped; thickness stayed with the physical wafer (original no).
    assert _wafer_map(item) == {"W2": ("W1", 100), "W1": ("W2", 200)}
    detail = (await ac.get("/batches/C-SWAP")).json()
    assert detail["median_nm"] == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_pending_batch_rejected(client, tmp_path):
    ac = client
    # First batch in a fresh test DB gets id 1; mark it slow so it stays
    # PENDING long enough for a deterministic correction attempt.
    (tmp_path / "slow_batches.txt").write_text("1:4\n")
    r = await submit(ac, "C-PEND", "EQ-COR", uniform_csv("cp", 3, 500))
    assert r.status_code == 202 and r.json()["batch_id"] == 1
    r = await _correct(ac, "C-PEND", "REQ-PEND", 0, {"cp-W001": "X1"})
    assert r.status_code == 409, r.text
    assert r.json()["status"] == "PENDING"
    # Judgement proceeds normally afterwards.
    await wait_status(ac, "C-PEND", "RELEASED")


@pytest.mark.asyncio
async def test_quarantined_batch_rejected(client):
    ac = client
    for i in range(1, 4):
        await _release(ac, f"C-QB{i}", "EQ-CORQ",
                       [(f"q{i}a", 500), (f"q{i}b", 500)])
    await submit(ac, "C-QUAR", "EQ-CORQ", uniform_csv("qq", 3, 900))
    await wait_status(ac, "C-QUAR", "QUARANTINED")
    r = await _correct(ac, "C-QUAR", "REQ-QUAR", 0, {"qq-W001": "X1"})
    assert r.status_code == 409, r.text
    assert r.json()["status"] == "QUARANTINED"


@pytest.mark.asyncio
async def test_revision_mismatch_conflict(client):
    ac = client
    await _release(ac, "C-REV", "EQ-COR", [("W1", 10), ("W2", 20)])
    r = await _correct(ac, "C-REV", "REQ-REV", 1, {"W1": "X1"})
    assert r.status_code == 409, r.text
    assert r.json()["current_revision"] == 0
    # Nothing changed.
    assert (await ac.get("/batches/C-REV")).json()["revision"] == 0
    assert set(_wafer_map(await _released_item(ac, "C-REV"))) == {"W1", "W2"}


@pytest.mark.asyncio
async def test_concurrent_same_revision_only_one_wins(client):
    ac = client
    await _release(ac, "C-RACE", "EQ-COR",
                   [("W1", 1), ("W2", 2), ("W3", 3), ("W4", 4)])
    ra, rb = await asyncio.gather(
        _correct(ac, "C-RACE", "RACE-A", 0, {"W1": "X1"}),
        _correct(ac, "C-RACE", "RACE-B", 0, {"W2": "X2"}),
    )
    codes = sorted([ra.status_code, rb.status_code])
    assert codes == [200, 409], (ra.text, rb.text)
    winner = ra if ra.status_code == 200 else rb
    loser = rb if winner is ra else ra
    assert winner.json()["revision"] == 1
    assert loser.json()["current_revision"] == 1

    assert (await ac.get("/batches/C-RACE")).json()["revision"] == 1
    item = await _released_item(ac, "C-RACE")
    mapping = _wafer_map(item)
    if winner.json()["request_id"] == "RACE-A":
        assert "X1" in mapping and "W2" in mapping and "X2" not in mapping
    else:
        assert "X2" in mapping and "W1" in mapping and "X1" not in mapping


@pytest.mark.asyncio
async def test_idempotent_replay_returns_original_result(client):
    ac = client
    await _release(ac, "C-IDEM", "EQ-COR", [("W1", 7), ("W2", 8)])
    first = await _correct(ac, "C-IDEM", "IDEM-1", 0, {"W1": "N1"})
    assert first.status_code == 200, first.text

    # Response lost -> clerk retries the identical request.
    replay = await _correct(ac, "C-IDEM", "IDEM-1", 0, {"W1": "N1"})
    assert replay.status_code == 200, replay.text
    assert replay.json()["replayed"] is True
    assert replay.json()["revision"] == 1
    assert replay.json()["corrections"] == {"W1": "N1"}

    # Not applied twice: revision still 1 and N1 present exactly once.
    assert (await ac.get("/batches/C-IDEM")).json()["revision"] == 1
    item = await _released_item(ac, "C-IDEM")
    assert _wafer_map(item) == {"N1": ("W1", 7), "W2": ("W2", 8)}


@pytest.mark.asyncio
async def test_request_id_reuse_with_different_content_conflicts(client):
    ac = client
    await _release(ac, "C-DUP", "EQ-COR", [("W1", 1), ("W2", 2)])
    await _release(ac, "C-DUP2", "EQ-COR", [("V1", 3), ("V2", 4)])
    ok = await _correct(ac, "C-DUP", "DUP-REQ", 0, {"W1": "A1"})
    assert ok.status_code == 200

    # Same id, different mapping.
    r = await _correct(ac, "C-DUP", "DUP-REQ", 0, {"W1": "A2"})
    assert r.status_code == 409, r.text
    # Same id, different expected revision.
    r = await _correct(ac, "C-DUP", "DUP-REQ", 1, {"W1": "A1"})
    assert r.status_code == 409, r.text
    # Same id, different batch.
    r = await _correct(ac, "C-DUP2", "DUP-REQ", 0, {"V1": "B1"})
    assert r.status_code == 409, r.text

    # The conflicting attempts changed nothing.
    assert (await ac.get("/batches/C-DUP")).json()["revision"] == 1
    assert (await ac.get("/batches/C-DUP2")).json()["revision"] == 0


@pytest.mark.asyncio
async def test_failed_correction_leaves_no_partial_state(client):
    ac = client
    await _release(ac, "C-ATOM", "EQ-COR",
                   [("W1", 1), ("W2", 2), ("W3", 3)])

    # One valid rename plus an unknown number -> whole request rejected.
    r = await _correct(ac, "C-ATOM", "ATOM-1", 0, {"W1": "OK1", "NOPE": "X"})
    assert r.status_code == 400, r.text
    # Target collides with an unmapped existing number.
    r = await _correct(ac, "C-ATOM", "ATOM-2", 0, {"W1": "W3"})
    assert r.status_code == 400, r.text
    # Two wafers mapped to the same target.
    r = await _correct(ac, "C-ATOM", "ATOM-3", 0, {"W1": "DUP", "W2": "DUP"})
    assert r.status_code == 400, r.text
    # Blank target number.
    r = await _correct(ac, "C-ATOM", "ATOM-4", 0, {"W1": "  "})
    assert r.status_code == 400, r.text
    # Schema-level validation failures.
    r = await _correct(ac, "C-ATOM", "ATOM-5", 0, {})
    assert r.status_code == 422
    r = await _correct(ac, "C-ATOM", "", 0, {"W1": "X"})
    assert r.status_code == 422
    r = await _correct(ac, "C-ATOM", "ATOM-6", -1, {"W1": "X"})
    assert r.status_code == 422

    # No partial correction and no revision drift after all those failures.
    detail = (await ac.get("/batches/C-ATOM")).json()
    assert detail["revision"] == 0
    item = await _released_item(ac, "C-ATOM")
    assert _wafer_map(item) == {"W1": ("W1", 1), "W2": ("W2", 2),
                                "W3": ("W3", 3)}


@pytest.mark.asyncio
async def test_chained_corrections_keep_original_numbers(client):
    ac = client
    await _release(ac, "C-CHAIN", "EQ-COR",
                   [("W1", 11), ("W2", 22), ("W3", 33)])
    r1 = await _correct(ac, "C-CHAIN", "CH-1", 0, {"W1": "C1"})
    assert r1.status_code == 200 and r1.json()["revision"] == 1
    # Second correction addresses the *current* numbers and reuses the
    # just-vacated C1 for another wafer (chain), bumping the revision again.
    r2 = await _correct(ac, "C-CHAIN", "CH-2", 1, {"C1": "C2", "W2": "C1"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["revision"] == 2

    item = await _released_item(ac, "C-CHAIN")
    assert item["revision"] == 2
    assert _wafer_map(item) == {
        "C2": ("W1", 11),   # original numbers always preserved
        "C1": ("W2", 22),
        "W3": ("W3", 33),
    }
    assert (await ac.get("/batches/C-CHAIN")).json()["revision"] == 2


@pytest.mark.asyncio
async def test_correction_unknown_batch_404(client):
    r = await _correct(client, "NO-SUCH-BATCH", "REQ-X", 0, {"W1": "X"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_concurrent_identical_request_retries_apply_once(client):
    """Two in-flight copies of the same request (retry after response loss
    racing the original) must both succeed with the same result while the
    correction is applied exactly once."""
    ac = client
    await _release(ac, "C-TWIN", "EQ-COR", [("W1", 5), ("W2", 6)])
    ra, rb = await asyncio.gather(
        _correct(ac, "C-TWIN", "TWIN-1", 0, {"W1": "T1"}),
        _correct(ac, "C-TWIN", "TWIN-1", 0, {"W1": "T1"}),
    )
    assert ra.status_code == 200 and rb.status_code == 200, (ra.text, rb.text)
    bodies = [ra.json(), rb.json()]
    assert all(b["revision"] == 1 for b in bodies)
    assert sorted(b["replayed"] for b in bodies) == [False, True]
    assert (await ac.get("/batches/C-TWIN")).json()["revision"] == 1
    assert _wafer_map(await _released_item(ac, "C-TWIN")) == {
        "T1": ("W1", 5), "W2": ("W2", 6)}


@pytest.mark.asyncio
async def test_concurrent_same_request_id_different_content(client):
    """Same request id racing on two different batches: exactly one request
    may exist under that id."""
    ac = client
    await _release(ac, "C-X1", "EQ-COR", [("W1", 1), ("W2", 2)])
    await _release(ac, "C-X2", "EQ-COR", [("V1", 3), ("V2", 4)])
    ra, rb = await asyncio.gather(
        _correct(ac, "C-X1", "SHARED-ID", 0, {"W1": "X1"}),
        _correct(ac, "C-X2", "SHARED-ID", 0, {"V1": "X2"}),
    )
    assert sorted([ra.status_code, rb.status_code]) == [200, 409], \
        (ra.text, rb.text)


@pytest.mark.asyncio
async def test_replay_of_old_request_after_later_corrections(client):
    """A lost response retried *after* newer corrections still returns the
    original result of its own request, not the current state."""
    ac = client
    await _release(ac, "C-HIST", "EQ-COR", [("W1", 1), ("W2", 2), ("W3", 3)])
    r1 = await _correct(ac, "C-HIST", "HIST-1", 0, {"W1": "H1"})
    assert r1.status_code == 200
    r2 = await _correct(ac, "C-HIST", "HIST-2", 1, {"W2": "H2"})
    assert r2.status_code == 200 and r2.json()["revision"] == 2

    replay1 = await _correct(ac, "C-HIST", "HIST-1", 0, {"W1": "H1"})
    assert replay1.status_code == 200
    assert replay1.json()["replayed"] is True
    assert replay1.json()["revision"] == 1  # original result, not current
    replay2 = await _correct(ac, "C-HIST", "HIST-2", 1, {"W2": "H2"})
    assert replay2.json()["replayed"] is True
    assert replay2.json()["revision"] == 2
    # State reflects both corrections exactly once.
    assert (await ac.get("/batches/C-HIST")).json()["revision"] == 2
    assert _wafer_map(await _released_item(ac, "C-HIST")) == {
        "H1": ("W1", 1), "H2": ("W2", 2), "W3": ("W3", 3)}


@pytest.mark.asyncio
async def test_correction_does_not_affect_judgement_or_history(client):
    ac = client
    for i in range(1, 4):
        await _release(ac, f"C-J{i}", "EQ-CORJ",
                       [(f"j{i}a", 500), (f"j{i}b", 500)])
    # Correct a baseline batch's numbers; judgement inputs are thickness-only.
    r = await _correct(ac, "C-J1", "J-REQ", 0, {"j1a": "j1a-fixed"})
    assert r.status_code == 200

    await submit(ac, "C-J4", "EQ-CORJ",
                 make_csv([("j4a", 504), ("j4b", 504)]))
    j4 = await wait_status(ac, "C-J4", "RELEASED")
    assert j4["reference_batch_nos"] == ["C-J1", "C-J2", "C-J3"]
    assert j4["deviation_nm"] == pytest.approx(4.0)
    assert j4["revision"] == 0  # new batches start at revision 0

    # Original artifact and decision of the corrected batch are unchanged.
    j1 = (await ac.get("/batches/C-J1")).json()
    assert j1["status"] == "RELEASED"
    assert j1["median_nm"] == pytest.approx(500.0)
    assert j1["revision"] == 1
