from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Mapping

from .quality_metrics import evaluate_reconstruction_quality
from .artifact_store import ArtifactStore
from .reconstruction_artifact_publication import (
    publish_dense_reconstruction_artifacts,
)
from .reconstruction_identity_artifacts import (
    publish_identity_diagnostics,
)
from .reconstruction_ball_trajectory import publish_automatic_ball_trajectory
from .reconstruction_ball_phase import BallTrajectoryPhaseResult
from .reconstruction_detection_contract import (
    CalibrationPhaseResult,
    FrameAnalysisResult,
)
from .reconstruction_identity_phase import IdentityPhaseResult
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_publish_payloads import (
    build_ball_detection_metadata,
    build_calibration_contract,
    build_calibration_metadata,
    build_pitch_orientation,
    coordinate_space,
    publication_diagnostics,
    publication_warnings,
)
from .reconstruction_status import set_reconstruction_status_in_memory


def publish_reconstruction_phase(
    scene: dict,
    *,
    frame_result: FrameAnalysisResult,
    calibration_result: CalibrationPhaseResult,
    identity_result: IdentityPhaseResult,
    ball_result: BallTrajectoryPhaseResult,
    ball_backend: str,
    ball_detection_input: Mapping,
    progress: ReconstructionProgress,
    artifact_store: ArtifactStore,
) -> dict:
    tracks = identity_result.tracks
    canonical_people = identity_result.canonical_people
    canonical_identity_diagnostics = identity_result.canonical_identity_diagnostics
    track_projection_diagnostics = identity_result.track_projection_diagnostics
    jersey_ocr_diagnostics = identity_result.jersey_ocr_diagnostics
    colors = identity_result.team_colors
    ball = ball_result.keyframes
    ball_tracking_diagnostics = ball_result.diagnostics

    frames = frame_result.frames
    person_detection_cache_diagnostics = frame_result.person_detection_cache_diagnostics

    calibration_quality = calibration_result.quality
    frame_evidence = calibration_result.frame_evidence

    scene["payload"]["tracks"] = tracks
    scene["payload"]["canonicalPeople"] = canonical_people
    ball_payload = publish_automatic_ball_trajectory(
        scene,
        ball,
        ball_tracking_diagnostics,
    )
    active_ball_keyframes = ball_payload["keyframes"]
    ball_detection_metadata = build_ball_detection_metadata(
        frame_result,
        ball_result,
        backend=ball_backend,
        detector_input=ball_detection_input,
    )
    for team in scene["payload"]["teams"]:
        team["color"] = colors.get(team["id"], team["color"])
    video = scene["payload"]["videoAsset"]
    video["processingState"] = (
        "tracks-ready"
        if tracks
        else "identities-ready"
        if canonical_people
        else "frames-ready"
    )
    calibration_metadata = build_calibration_metadata(calibration_result)
    calibration_contract = build_calibration_contract(scene, calibration_result)
    pitch_orientation = build_pitch_orientation(video, calibration_result)
    artifact_manifest, compact_identity_diagnostics = publish_identity_diagnostics(
        canonical_identity_diagnostics,
        store=artifact_store,
    )
    working_reconstruction = video.get("reconstruction") or {}
    working_reconstruction["artifactManifest"] = artifact_manifest
    working_reconstruction["diagnostics"] = {
        **(working_reconstruction.get("diagnostics") or {}),
        **track_projection_diagnostics,
        "identity": compact_identity_diagnostics,
        "jerseyOcr": jersey_ocr_diagnostics,
        "personDetectionCache": deepcopy(person_detection_cache_diagnostics),
        "ballTracking": ball_tracking_diagnostics,
        "ballTrajectoryMode": ball_payload["mode"],
    }
    working_reconstruction["ballDetection"] = ball_detection_metadata
    working_reconstruction["calibration"] = calibration_contract
    video["reconstruction"] = working_reconstruction
    quality = evaluate_reconstruction_quality(scene, frame_evidence)
    quality["processingStatus"] = "completed"
    quality["calibrationQuality"] = calibration_quality
    verdict_rank = {"pass": 0, "review": 1, "reject": 2}
    quality["verdict"] = max(
        (quality["verdict"], calibration_quality["verdict"]),
        key=lambda value: verdict_rank[value],
    )
    publish_dense_reconstruction_artifacts(scene, store=artifact_store)
    compact_reconstruction = video["reconstruction"]
    compact_calibration_contract = compact_reconstruction["calibration"]
    compact_ball_detection_metadata = compact_reconstruction["ballDetection"]
    progress.update(
        "finalizing",
        6,
        "Saving reconstruction",
        "Writing tracks, calibration diagnostics, and orientation metadata.",
        97,
        100,
        completed=0,
        total=1,
    )
    completed_progress = progress.complete(len(tracks), len(active_ball_keyframes))
    set_reconstruction_status_in_memory(
        scene,
        "ready",
        processingStatus="completed",
        qualityVerdict=quality["verdict"],
        quality=quality,
        completedAt=datetime.now(UTC).isoformat(),
        frameCount=len(frames),
        trackCount=len(tracks),
        canonicalPersonCount=len(canonical_people),
        ballSamples=len(active_ball_keyframes),
        ballBackend=ball_backend,
        ballDetectionInput=ball_detection_input,
        ballDetection=compact_ball_detection_metadata,
        # v3 makes canonical video identity authoritative even when no
        # renderable metric 3D trajectory exists. v1/v2 remain readable.
        trackObservationSchemaVersion=3,
        coordinateSpace=coordinate_space(calibration_result),
        pitchCalibration=calibration_metadata,
        calibration=compact_calibration_contract,
        pitchOrientation=pitch_orientation,
        cameraMotionCompensated=any(
            (item.get("cameraMotion") or {}).get("status") == "estimated"
            for item in frame_evidence
        ),
        progress=completed_progress,
        warnings=publication_warnings(
            frame_result,
            calibration_result,
            identity_result,
            ball_result,
            ball_mode=ball_payload["mode"],
        ),
        inputRange={
            "sourceStart": float(video.get("sourceStart") or 0.0),
            "sourceEnd": float(video.get("sourceEnd") or scene["duration"]),
            "firstFrameTime": round(float(frames[0][1]), 3),
            "lastFrameTime": round(float(frames[-1][1]), 3),
        },
        diagnostics=publication_diagnostics(
            frame_result,
            calibration_result,
            identity_result,
            ball_result,
            ball_mode=ball_payload["mode"],
            compact_identity=compact_identity_diagnostics,
        ),
    )
    return scene
