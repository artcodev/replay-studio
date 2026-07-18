from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.reconstruction_run_contract import (
    ReconstructionRunFence,
    job_matches_fence,
    lease_matches_fence,
    queued_run_from_scene,
    terminal_run_from_scene,
)
from app.reconstruction_run_scene_transition import transition_scene_to_processing
from app.scene_document import reconstruction_input_fingerprint


def _scene(*, status: str = "queued") -> dict:
    scene = {
        "id": "scene-1",
        "title": "Run contract",
        "revision": 4,
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "selectedSegmentId": "segment-1",
                "sourceStart": 0.0,
                "sourceEnd": 4.0,
                "analysisFps": 10.0,
                "processingState": "reconstructing",
                "reconstruction": {
                    "status": status,
                    "processingStatus": status,
                    "runId": "run-1",
                    "runRevision": 3,
                    "model": "person-model",
                },
            }
        },
    }
    scene["payload"]["videoAsset"]["reconstruction"]["inputFingerprint"] = (
        reconstruction_input_fingerprint(scene)
    )
    return scene


def test_queue_contract_owns_token_and_input_validation() -> None:
    scene = _scene()

    queued = queued_run_from_scene(scene)

    assert queued.fence == ReconstructionRunFence(
        scene_id="scene-1",
        run_id="run-1",
        input_fingerprint=reconstruction_input_fingerprint(scene),
    )
    assert queued.input_revision == 3
    assert queued.model == "person-model"

    stale = deepcopy(scene)
    stale["payload"]["videoAsset"]["analysisFps"] = 25.0
    with pytest.raises(ValueError, match="input fingerprint"):
        queued_run_from_scene(stale)


def test_terminal_contract_fails_closed_for_a_different_fence() -> None:
    scene = _scene(status="ready")
    matching = ReconstructionRunFence(
        scene_id="scene-1",
        run_id="run-1",
        input_fingerprint=reconstruction_input_fingerprint(scene),
    )

    terminal = terminal_run_from_scene(scene, matching)

    assert terminal is not None
    assert terminal.scene_status == "ready"
    assert terminal.telemetry_status == "succeeded"
    assert terminal_run_from_scene(
        scene,
        ReconstructionRunFence(
            scene_id="scene-1",
            run_id="other-run",
            input_fingerprint=matching.input_fingerprint,
        ),
    ) is None


def test_processing_transition_is_pure_and_revisioned() -> None:
    scene = _scene()
    before = deepcopy(scene)
    fence = queued_run_from_scene(scene).fence

    claimed = transition_scene_to_processing(
        scene,
        fence,
        current_time=1_000.0,
    )

    assert claimed is not None
    assert scene == before
    assert claimed.revision == 5
    assert claimed.payload["revision"] == 5
    reconstruction = claimed.payload["payload"]["videoAsset"]["reconstruction"]
    assert reconstruction["status"] == "processing"
    assert reconstruction["processingStatus"] == "processing"
    assert reconstruction["startedAt"] == "1970-01-01T00:16:40+00:00"
    assert claimed.payload["payload"]["videoAsset"]["processingState"] == (
        "reconstructing"
    )


def test_job_and_lease_fences_require_every_token_and_live_owner() -> None:
    fence = queued_run_from_scene(_scene()).fence
    job = SimpleNamespace(
        status="processing",
        run_id=fence.run_id,
        input_fingerprint=fence.input_fingerprint,
    )
    lease = SimpleNamespace(
        run_id=fence.run_id,
        input_fingerprint=fence.input_fingerprint,
        owner_id="owner-1",
        expires_at=101.0,
    )

    assert job_matches_fence(job, fence, statuses={"processing"})
    assert lease_matches_fence(
        lease,
        fence,
        owner_id="owner-1",
        current_time=100.0,
    )
    assert not lease_matches_fence(
        lease,
        fence,
        owner_id="other-owner",
        current_time=100.0,
    )
    assert not lease_matches_fence(
        lease,
        fence,
        owner_id="owner-1",
        current_time=101.0,
    )
