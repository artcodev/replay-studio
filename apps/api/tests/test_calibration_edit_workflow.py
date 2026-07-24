from __future__ import annotations

from pathlib import Path

from app.reconstruction_calibration_edit_session import (
    register_pending_calibration_edit,
)
from app.reconstruction_calibration_finalize_command import (
    finalize_scene_pitch_calibration_drafts,
)
from app.reconstruction_calibration_incremental_phase import (
    affected_calibration_samples,
)


def _reconstruction() -> dict:
    return {
        "status": "ready",
        "calibrationProvenance": {
            "calibrationInputFingerprint": "input-1",
            "dataFingerprint": "data-1",
            "artifact": {"sha256": "artifact-1"},
        },
    }


def _override(sample_index: int) -> dict:
    return {
        "sampleIndex": sample_index,
        "sourceFrameIndex": 100 + sample_index,
        "sceneTime": sample_index / 10,
        "preset": "center-circle",
    }


def test_staged_edits_upsert_without_creating_a_run() -> None:
    reconstruction = _reconstruction()
    register_pending_calibration_edit(
        reconstruction,
        _override(2),
        draft_source="manual",
    )
    register_pending_calibration_edit(
        reconstruction,
        _override(2),
        draft_source="borrowed-previous",
    )

    session = reconstruction["pendingCalibrationEditSession"]
    assert session["editedSampleIndices"] == [2]
    assert session["edits"][0]["draftSource"] == "borrowed-previous"
    assert "runId" not in reconstruction


def test_finalization_is_the_explicit_queue_boundary(monkeypatch) -> None:
    reconstruction = _reconstruction()
    register_pending_calibration_edit(
        reconstruction,
        _override(2),
        draft_source="manual",
    )
    scene = {"payload": {"videoAsset": {"reconstruction": reconstruction}}}
    queued: list[dict] = []
    monkeypatch.setattr(
        "app.reconstruction_calibration_finalize_command.queue_reconstruction",
        lambda value, **kwargs: queued.append(kwargs) or value,
    )

    finalize_scene_pitch_calibration_drafts(scene, match_snapshot=None)

    assert queued == [{
        "mode": "calibrate",
        "calibration_trigger": "manual-draft-finalize",
        "match_snapshot": None,
    }]


def test_affected_region_excludes_unchanged_direct_anchors() -> None:
    frames = [(Path(f"/tmp/frame_{index:05d}.jpg"), float(index)) for index in range(6)]
    evidence = [
        {},
        {"temporal": {"anchorSampleIndices": [0]}},
        {},
        {"temporal": {"anchorSampleIndices": [2]}},
        {},
        {"temporal": {"anchorSampleIndices": [2]}},
    ]

    affected = affected_calibration_samples(
        frames,
        evidence,
        edited_sample_indices={2},
        direct_sample_indices={0, 2, 4},
        max_gap_seconds=1.1,
    )

    # Frame 3 is close and depended on sample 2; frame 5 is farther away but
    # still depended on it in the published artifact. Direct samples 0 and 4
    # are immutable and therefore reused.
    assert affected == {1, 2, 3, 5}
