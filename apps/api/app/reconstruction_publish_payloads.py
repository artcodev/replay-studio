"""Pure metadata builders for terminal reconstruction publication."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Mapping

import numpy as np

from .reconstruction_ball_phase import BallTrajectoryPhaseResult
from .reconstruction_detection_contract import (
    CalibrationPhaseResult,
    FrameAnalysisResult,
)
from .reconstruction_identity_phase import IdentityPhaseResult


def identity_runtime_quality(
    frame: FrameAnalysisResult,
    identity: IdentityPhaseResult,
    *,
    jersey_ocr_profile: str,
) -> dict[str, Any]:
    reid_status = str(
        frame.identity_worker_diagnostics.get("status") or "unknown"
    )
    jersey_status = str(
        identity.jersey_ocr_diagnostics.get("status") or "unknown"
    )
    reasons: list[str] = []
    if reid_status not in {"ready", "no-observations"}:
        reasons.append(f"reid-{reid_status}")
    if (
        jersey_ocr_profile != "off"
        and jersey_status
        not in {
            "ready",
            "no-observations",
            "no-crops",
            "no-readable-crops",
        }
    ):
        reasons.append(f"jersey-ocr-{jersey_status}")
    return {
        "status": "degraded" if reasons else "ready",
        "reidStatus": reid_status,
        "jerseyOcrStatus": jersey_status,
        "jerseyOcrProfile": jersey_ocr_profile,
        "reasons": reasons,
        "automaticCrossGapIdentityAvailable": reid_status == "ready",
        "jerseyNumberEvidenceAvailable": jersey_status == "ready",
    }


def build_ball_detection_metadata(
    frame: FrameAnalysisResult,
    ball: BallTrajectoryPhaseResult,
    *,
    backend: str,
    detector_input: Mapping,
) -> dict[str, Any]:
    source = frame.ball_dense_frame_metadata
    tracking = ball.diagnostics
    runtime_model_versions = sorted(
        {
            str(worker["modelVersion"])
            for frame_batch in frame.ball_detection_batches
            for worker in [(frame_batch.get("metadata") or {}).get("worker") or {}]
            if worker.get("modelVersion")
        }
    )
    return {
        "schemaVersion": 1,
        "status": (
            "skipped"
            if source.get("skippedByProfile")
            else "degraded"
            if source.get("failedFrameCount")
            or source.get("fallbackFrameCount")
            or source.get("source") == "sampled-frame-fallback"
            else "ready"
        ),
        "requestedBackend": backend,
        "runtimeModelVersions": runtime_model_versions,
        "input": deepcopy(detector_input),
        "frameSource": deepcopy(source),
        "frameCount": len(frame.ball_frames),
        "candidateCount": sum(len(items) for items, _ in frame.ball_frames),
        "framesWithCandidates": sum(bool(items) for items, _ in frame.ball_frames),
        "fallbackFrameCount": int(source.get("fallbackFrameCount") or 0),
        "failedFrameCount": int(source.get("failedFrameCount") or 0),
        "backendCounts": deepcopy(source.get("backendCounts") or {}),
        "observedFrameCount": tracking.get("observedFrameCount", 0),
        "inferredFrameCount": tracking.get("inferredFrameCount", 0),
        "occludedFrameCount": tracking.get("occludedFrameCount", 0),
        "observedCoverage": tracking.get("observedCoverage"),
        "publishedCoverage": tracking.get("publishedCoverage"),
        "tracking": deepcopy(tracking),
        "frames": frame.ball_detection_batches,
    }


def build_calibration_metadata(result: CalibrationPhaseResult) -> dict[str, Any]:
    calibration = result.calibration
    if calibration is None:
        return {
            "status": "rejected",
            "method": None,
            "pitchSide": None,
            "reason": "No frame produced a calibration that passed geometric QA.",
        }
    quality = result.quality
    metadata = {
        **calibration.as_dict(),
        "status": (
            "ready"
            if quality["verdict"] == "pass"
            else "review"
            if quality["verdict"] == "review"
            else "rejected"
        ),
        "reason": (
            None
            if quality["verdict"] == "pass"
            else "Calibration QA gates did not permit metric coordinates."
        ),
    }
    if result.representative_manual_sample is None:
        return metadata
    override = result.manual_override_by_sample[result.representative_manual_sample]
    metadata.update(
        {
            "method": "manual-pitch-anchors",
            "preset": override.get("preset"),
            "sceneTime": override.get("sceneTime"),
            "frameIndex": override.get("frameIndex"),
            "sourceFrameIndex": override.get("sourceFrameIndex"),
            "alignmentError": override.get("alignmentError"),
            "alignmentMetrics": override.get("alignmentMetrics"),
            "anchors": override.get("anchors") or [],
            "manualFrameAnchorCount": len(result.accepted_manual_direct_by_sample),
        }
    )
    return metadata


def build_calibration_contract(result: CalibrationPhaseResult) -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "summary": result.quality["summary"],
        "frameEvidence": result.frame_evidence,
        "manualFrameAnchors": [
            {
                "id": item.get("id"),
                "sampleIndex": sample_index,
                "sourceFrameIndex": item.get("sourceFrameIndex"),
                "sceneTime": item.get("sceneTime"),
                "status": (
                    "accepted"
                    if sample_index in result.accepted_manual_direct_by_sample
                    else "rejected"
                ),
            }
            for sample_index, item in sorted(result.manual_override_by_sample.items())
        ],
    }


def build_pitch_orientation(
    video: Mapping[str, Any],
    result: CalibrationPhaseResult,
) -> dict[str, Any]:
    existing = (video.get("reconstruction") or {}).get("pitchOrientation") or {}
    existing_source = str(existing.get("visiblePitchSideSource") or "")
    detected_side = result.quality["summary"].get("visiblePitchSide")
    attacking_goal = existing.get("attackingGoal")
    if attacking_goal not in {"left", "right"}:
        attacking_goal = "unknown"
    # A manually chosen pitch side is authoritative user input and part of the
    # reconstruction input fingerprint. Never clobber it with the detected side —
    # doing so changes the fingerprint mid-run and the terminal publish's
    # compare-and-swap would reject the (correct) result as stale.
    if existing_source.startswith("manual"):
        visible_side = existing.get("visiblePitchSide") or "unknown"
        visible_side_source = existing_source
    else:
        visible_side = detected_side or "unknown"
        visible_side_source = "calibration" if detected_side else "unknown"
    return {
        "visiblePitchSide": visible_side,
        "visiblePitchSideSource": visible_side_source,
        "visiblePitchSideAgreement": result.quality["summary"].get(
            "sideAgreement"
        ),
        "attackingGoal": attacking_goal,
        "attackingGoalSource": existing.get("attackingGoalSource")
        or ("manual" if attacking_goal != "unknown" else "unknown"),
        "updatedAt": datetime.now(UTC).isoformat(),
    }


def coordinate_space(result: CalibrationPhaseResult) -> str:
    if result.metric_calibration and result.accepted_manual_direct_by_sample:
        return (
            "pitch-metric-mixed-direct-anchors"
            if result.accepted_automatic_direct_by_sample
            else "pitch-metric-manual-anchors"
        )
    if result.metric_calibration:
        return (
            "pitch-metric-temporal-hypotheses"
            if result.temporal_recovered_frame_count
            else "pitch-metric-per-frame"
        )
    if result.quality["verdict"] == "review":
        return (
            "pitch-metric-temporal-partial-review"
            if result.temporal_recovered_frame_count
            else "pitch-metric-partial-review"
        )
    return "unavailable-calibration-rejected"


def publication_warnings(
    frame: FrameAnalysisResult,
    calibration: CalibrationPhaseResult,
    identity: IdentityPhaseResult,
    ball: BallTrajectoryPhaseResult,
    *,
    ball_mode: str,
) -> list[str]:
    if calibration.metric_calibration and calibration.accepted_manual_direct_by_sample:
        if calibration.accepted_automatic_direct_by_sample:
            calibration_message = (
                f"Metric positions combine {len(calibration.accepted_manual_direct_by_sample)} "
                f"manual frame anchor(s) with {len(calibration.accepted_automatic_direct_by_sample)} "
                "accepted automatic direct observation(s); manual wins only at the same sample."
            )
        else:
            calibration_message = (
                f"Metric positions use {len(calibration.accepted_manual_direct_by_sample)} "
                "accepted manual frame anchor(s) and QA-gated temporal propagation."
            )
    elif calibration.metric_calibration:
        calibration_message = (
            "Metric positions combine direct pitch observations with QA-gated "
            "forward/backward camera hypotheses."
            if calibration.temporal_recovered_frame_count
            else "Metric positions use semantic per-frame homographies; PnLCalib "
            "frames include point-and-line refinement."
        )
    else:
        calibration_message = (
            "Calibration requires review; only accepted metric observations were "
            "published and gaps remain missing."
            if calibration.quality["verdict"] == "review"
            else "Calibration QA rejected this run; no new world-space tracks or "
            "ball trajectory were published."
        )
    warnings = [
        calibration_message,
        *calibration.warnings,
        *identity.warnings,
        *frame.ball_detection_warnings,
    ]
    if (
        calibration.resolved_calibrations_by_sample
        and len(calibration.resolved_calibrations_by_sample) < len(frame.frames)
    ):
        warnings.append(
            f"Metric calibration remains unresolved on "
            f"{len(frame.frames) - len(calibration.resolved_calibrations_by_sample)} "
            f"of {len(frame.frames)} sampled frames; no representative homography "
            "was used to hide those gaps."
        )
    if ball.diagnostics.get("skippedByProfile"):
        warnings.append(
            "Dense ball detection was skipped by the analysis profile; the "
            "manual ball trajectory remains authoritative."
        )
    elif not ball.keyframes:
        warnings.append(
            "No stable automatic ball trajectory was found; active manual keypoints "
            "were preserved."
            if ball_mode == "manual"
            else "No stable ball trajectory was found."
        )
    return warnings


def publication_diagnostics(
    frame: FrameAnalysisResult,
    calibration: CalibrationPhaseResult,
    identity: IdentityPhaseResult,
    ball: BallTrajectoryPhaseResult,
    *,
    ball_mode: str,
    compact_identity: Mapping[str, Any],
    jersey_ocr_profile: str = "automatic",
) -> dict[str, Any]:
    tracking = ball.diagnostics
    accepted = calibration.accepted_frame_calibrations
    calibration_value = calibration.calibration
    return {
        **identity.track_projection_diagnostics,
        "meanPersonDetections": round(float(np.mean(frame.person_counts)), 2),
        "framesWithBall": sum(count > 0 for count in frame.ball_counts),
        "ballCandidateCount": sum(frame.ball_counts),
        "ballObservedFrameCount": tracking.get("observedFrameCount", 0),
        "ballInferredFrameCount": tracking.get("inferredFrameCount", 0),
        "ballOccludedFrameCount": tracking.get("occludedFrameCount", 0),
        "ballObservedCoverage": tracking.get("observedCoverage"),
        "ballPublishedCoverage": tracking.get("publishedCoverage"),
        "ballTracking": tracking,
        "ballTrajectoryMode": ball_mode,
        "rawTrackCount": identity.raw_track_count,
        "canonicalPersonCount": len(identity.canonical_people),
        "stableTrackCount": identity.stable_track_count,
        "acceptedTrackCount": len(identity.tracks),
        "identity": dict(compact_identity),
        "identityRuntime": identity_runtime_quality(
            frame,
            identity,
            jersey_ocr_profile=jersey_ocr_profile,
        ),
        "jerseyOcr": identity.jersey_ocr_diagnostics,
        "personDetectionCache": deepcopy(frame.person_detection_cache_diagnostics),
        "calibrationBackend": calibration_value.method if calibration_value else None,
        "calibrationBackendCounts": {
            method: sum(item.method == method for item in accepted.values())
            for method in sorted({item.method for item in accepted.values()})
        },
        "calibratedFrameCount": len(calibration.resolved_calibrations_by_sample),
        "directCalibratedFrameCount": len(accepted),
        "temporalRecoveredFrameCount": calibration.temporal_recovered_frame_count,
        "temporalAmbiguousFrameCount": sum(
            item.get("solutionStatus") == "ambiguous"
            for item in calibration.frame_evidence
        ),
        "cameraMotionCutCount": sum(
            (item.get("cameraMotion") or {}).get("status") == "cut"
            for item in calibration.frame_evidence
        ),
        "cameraMotionUnreliableCount": sum(
            (item.get("cameraMotion") or {}).get("status") == "unreliable"
            for item in calibration.frame_evidence
        ),
        "calibrationFrameCoverage": calibration.quality["summary"]["usableCoverage"],
        "calibrationDirectCoverage": calibration.quality["summary"]["directCoverage"],
        "calibrationMaxGapSeconds": calibration.quality["summary"]["maxGapSeconds"],
        "calibrationReprojectionP95": calibration.quality["summary"]["reprojectionP95"],
        "calibrationSideAgreement": calibration.quality["summary"]["sideAgreement"],
        "rejectedCalibrationFrames": calibration.rejected_frame_count,
        "screenApproximateSamples": sum(
            keyframe.get("projectionSource") == "screen-approximate"
            for track in identity.tracks
            for keyframe in track.get("keyframes") or []
        )
        + sum(
            keyframe.get("projectionSource") == "screen-approximate"
            for keyframe in ball.keyframes
        ),
        "metricPersonSamples": calibration.metric_person_sample_count,
        "metricBallSamples": calibration.metric_ball_sample_count,
        "calibrationReprojectionError": (
            round(calibration_value.reprojection_error, 3)
            if calibration_value is not None
            and calibration_value.reprojection_error is not None
            else None
        ),
    }


__all__ = (
    "build_ball_detection_metadata",
    "build_calibration_contract",
    "build_calibration_metadata",
    "build_pitch_orientation",
    "coordinate_space",
    "identity_runtime_quality",
    "publication_diagnostics",
    "publication_warnings",
)
