from __future__ import annotations

"""Publish a calibration-only reconstruction stage and its inspection gate.

The reconstruction can run in two stages. A ``calibrate`` run executes only
calibration, then publishes the per-frame calibration evidence and
stops — no tracks, no ball. If any frame failed to resolve a homography the
scene enters a review gate; the operator inspects and fixes (or explicitly
accepts) the unresolved frames before the ``full`` run spends compute on
identity, ball and publication. The job itself still completes normally: the
gate is a scene-level workflow state, never a change to the job/lease machine.
"""

from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from .artifact_store import ArtifactStore, reconstruction_artifact_store
from .reconstruction_artifact_codec import (
    materialized_artifacts,
    set_materialized_artifacts,
)
from .reconstruction_artifact_manifest import ARTIFACT_MANIFEST_SCHEMA_VERSION
from .reconstruction_calibration_artifacts import publish_calibration_frames_artifact
from .reconstruction_calibration_fingerprint import calibration_input_fingerprint
from .reconstruction_calibration_edit_session import (
    clear_pending_calibration_edit_session,
)
from .reconstruction_calibration_snapshot import calibration_data_fingerprint
from .reconstruction_ball_trajectory import normalize_ball_payload
from .reconstruction_detection_contract import CalibrationPhaseResult
from .reconstruction_publish_payloads import (
    build_calibration_contract,
    build_calibration_metadata,
    build_pitch_orientation,
)


# Per-frame solution statuses that count as a usable calibration.
_RESOLVED_SOLUTION_STATUSES = {"direct-accepted", "temporal-accepted"}


def _frame_gate(frame_evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Every sampled frame is emitted so the operator can browse successes too,
    # not just the failures. `unresolvedSamples` is the failing subset.
    frames: list[dict[str, Any]] = []
    for index, evidence in enumerate(frame_evidence):
        status = str(evidence.get("solutionStatus") or "unresolved")
        accepted = (
            "accepted" in status or status in _RESOLVED_SOLUTION_STATUSES
        ) and str(evidence.get("projectionSource") or "none") != "none"
        sample_index = (
            evidence.get("sampleIndex")
            if evidence.get("sampleIndex") is not None
            else index
        )
        frames.append(
            {
                "sampleIndex": int(sample_index),
                "sourceFrameIndex": evidence.get("sourceFrameIndex"),
                "sceneTime": evidence.get("sceneTime"),
                "solutionStatus": status,
                "projectionSource": evidence.get("projectionSource"),
                "resolved": accepted,
                "residualP95": (
                    (evidence.get("alignmentMetrics") or {}).get("residualP95")
                ),
                "rejectionReasons": list(evidence.get("rejectionReasons") or []),
                "acceptedByOperator": bool(evidence.get("acceptedByOperator")),
                "manual": str(evidence.get("solutionStatus") or "").startswith("manual"),
                # The frame-local homography lets the modal overlay the actual
                # projected pitch lines for inspection — resolved frames only, so
                # a red frame never shows a misleading rejected overlay.
                "imageToPitch": evidence.get("imageToPitch") if accepted else None,
                "frameWidth": evidence.get("frameWidth"),
                "frameHeight": evidence.get("frameHeight"),
            }
        )
    unresolved = [frame for frame in frames if not frame["resolved"]]
    total = len(frames)
    return {
        "totalFrames": total,
        "resolvedFrames": total - len(unresolved),
        "unresolvedFrames": len(unresolved),
        "resolvedRatio": round((total - len(unresolved)) / total, 4) if total else 1.0,
        "frames": frames,
        "unresolvedSamples": unresolved,
    }


def calibration_stage_status(gate: Mapping[str, Any]) -> str:
    """A fully resolved calibration is ready; a gap needs operator review."""

    return "ready" if int(gate["unresolvedFrames"]) == 0 else "review"


def publish_calibration_stage(
    scene: dict,
    calibration_result: CalibrationPhaseResult,
    *,
    store: ArtifactStore | None = None,
) -> dict:
    """Write calibration evidence and the inspection gate; publish no tracks."""

    artifact_store = store or reconstruction_artifact_store()
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}

    calibration_contract = build_calibration_contract(calibration_result)
    calibration_metadata = build_calibration_metadata(calibration_result)
    pitch_orientation = build_pitch_orientation(video, calibration_result)
    gate = _frame_gate(calibration_result.frame_evidence)
    stage_status = calibration_stage_status(gate)

    reconstruction["calibration"] = calibration_contract
    reconstruction["pitchCalibration"] = calibration_metadata
    reconstruction["pitchOrientation"] = pitch_orientation
    reconstruction["coordinateSpace"] = (
        "pitch-metric" if calibration_result.metric_calibration else "screen-relative"
    )
    reconstruction["calibrationWarnings"] = list(calibration_result.warnings)
    # Publication is the commit point for staged editor corrections. Their
    # overrides remain calibration inputs; only transient workflow state ends.
    clear_pending_calibration_edit_session(reconstruction)
    current_calibration_input_fingerprint = calibration_input_fingerprint(scene)
    reconstruction["calibrationInputFingerprint"] = (
        current_calibration_input_fingerprint
    )
    data_fingerprint = calibration_data_fingerprint(reconstruction)

    # Calibration is a separate producer stage. A previous reconstruction was
    # computed from a different calibration product and must not remain visible
    # beside this new one as though both belonged to the same run. Keep manual
    # ball input (operator-owned), but invalidate every derived identity/ball
    # reference and clear the rendered identity read model.
    scene["payload"]["tracks"] = []
    scene["payload"]["canonicalPeople"] = []
    raw_ball = scene["payload"].get("ball")
    if (
        isinstance(raw_ball, Mapping)
        and raw_ball.get("mode") == "manual"
        and "manualKeyframes" not in raw_ball
    ):
        raw_ball = {
            **raw_ball,
            "manualKeyframes": list(raw_ball.get("keyframes") or []),
        }
    ball = normalize_ball_payload(raw_ball)
    ball["automaticKeyframes"] = []
    ball["automaticDiagnostics"] = {
        "trajectoryMode": "automatic",
        "source": "automatic-ball-resolver",
        "status": "invalidated-by-calibration",
        "worldProjectionStatus": "awaiting-reconstruction",
    }
    if ball["mode"] == "automatic":
        ball["keyframes"] = []
        ball["diagnostics"] = dict(ball["automaticDiagnostics"])
    scene["payload"]["ball"] = ball
    references: dict[str, Any] = {}
    materialized = materialized_artifacts(reconstruction)
    calibration_frames = publish_calibration_frames_artifact(
        scene,
        reconstruction,
        references,
        materialized,
        store=artifact_store,
    )
    reconstruction["artifactManifest"] = {
        "schemaVersion": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifacts": {"calibrationFrames": dict(calibration_frames.reference)},
    }
    reconstruction["diagnostics"] = {
        key: value
        for key, value in (reconstruction.get("diagnostics") or {}).items()
        if key in {"calibrationUsage", "contactPoint"}
    }
    reconstruction.pop("identityCorrectionDiagnostics", None)
    set_materialized_artifacts(reconstruction, ())
    if calibration_frames.encoding is not None:
        reconstruction["calibration"] = calibration_frames.encoding.compact_calibration
        reconstruction["ballDetection"] = (
            calibration_frames.encoding.compact_ball_detection
        )

    produced_at = datetime.now(UTC).isoformat()
    reconstruction["stage"] = "calibration"
    reconstruction["resultState"] = "calibration-only"
    reconstruction["calibrationProvenance"] = {
        "schemaVersion": 1,
        "runId": reconstruction.get("runId"),
        "producedAt": produced_at,
        "calibrationInputFingerprint": current_calibration_input_fingerprint,
        "dataFingerprint": data_fingerprint,
        "artifact": dict(calibration_frames.reference),
        "samplingFrameRate": reconstruction.get("samplingFrameRate"),
        "directCalibrationMaxGapSeconds": reconstruction.get(
            "directCalibrationMaxGapSeconds"
        ),
        "totalFrames": gate["totalFrames"],
        "resolvedFrames": gate["resolvedFrames"],
        "unresolvedFrames": gate["unresolvedFrames"],
    }
    reconstruction["calibrationReview"] = {
        "status": stage_status,
        # Stamp the inputs this gate was computed for. The full run refuses to
        # start against a gate whose fingerprint no longer matches the scene —
        # a manual anchor added during review changes the fingerprint and
        # forces a recalibration rather than reconstructing on stale evidence.
        "inputFingerprint": reconstruction.get("inputFingerprint"),
        "calibrationInputFingerprint": current_calibration_input_fingerprint,
        **gate,
        "warnings": list(calibration_result.warnings),
    }
    # The calibrate job succeeded: it produced a calibration. The gate lives in
    # calibrationReview.status, not in the job/run status.
    reconstruction["status"] = "ready"
    reconstruction["processingStatus"] = "completed"
    reconstruction["qualityVerdict"] = (
        "pass" if stage_status == "ready" else "review"
    )
    reconstruction["completedAt"] = produced_at
    video["reconstruction"] = reconstruction
    video["processingState"] = "calibration-ready"
    return scene


__all__ = (
    "calibration_stage_status",
    "publish_calibration_stage",
)
