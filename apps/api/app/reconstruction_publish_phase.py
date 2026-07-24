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
from .reconstruction_ball_trajectory import (
    normalize_ball_payload,
    publish_automatic_ball_trajectory,
)
from .reconstruction_ball_phase import BallTrajectoryPhaseResult
from .reconstruction_detection_contract import (
    CalibrationPhaseResult,
    FrameAnalysisResult,
)
from .reconstruction_identity_phase import IdentityPhaseResult
from .reconstruction_progress import ReconstructionProgress
from .reconstruction_publish_payloads import (
    build_ball_detection_metadata,
    coordinate_space,
    identity_runtime_quality,
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
    ball_detection_profile: str = "automatic",
    jersey_ocr_profile: str = "automatic",
    progress: ReconstructionProgress,
    artifact_store: ArtifactStore,
    calibration_usage: Mapping,
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
    if ball_detection_profile == "skip-manual-authoritative":
        # The automatic channel was intentionally not recomputed: keep both
        # stored channels untouched and publish the authoritative manual path.
        ball_payload = normalize_ball_payload(scene["payload"].get("ball"))
        scene["payload"]["ball"] = ball_payload
    else:
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
    artifact_manifest, compact_identity_diagnostics = publish_identity_diagnostics(
        canonical_identity_diagnostics,
        store=artifact_store,
    )
    working_reconstruction = video.get("reconstruction") or {}
    calibration_metadata = deepcopy(
        working_reconstruction.get("pitchCalibration") or {}
    )
    calibration_contract = deepcopy(
        working_reconstruction.get("calibration") or {}
    )
    pitch_orientation = deepcopy(
        working_reconstruction.get("pitchOrientation") or {}
    )
    working_reconstruction["artifactManifest"] = artifact_manifest
    working_reconstruction["diagnostics"] = {
        **(working_reconstruction.get("diagnostics") or {}),
        **track_projection_diagnostics,
        "identity": compact_identity_diagnostics,
        "jerseyOcr": jersey_ocr_diagnostics,
        "personDetectionCache": deepcopy(person_detection_cache_diagnostics),
        "ballTracking": ball_tracking_diagnostics,
        "ballTrajectoryMode": ball_payload["mode"],
        "calibrationUsage": deepcopy(dict(calibration_usage)),
        **(
            {
                "contactPoint": deepcopy(
                    calibration_result.contact_point_diagnostics
                )
            }
            if calibration_result.contact_point_diagnostics is not None
            else {}
        ),
    }
    working_reconstruction["ballDetection"] = ball_detection_metadata
    working_reconstruction["calibration"] = calibration_contract
    video["reconstruction"] = working_reconstruction
    quality = evaluate_reconstruction_quality(scene, frame_evidence)
    quality["processingStatus"] = "completed"
    quality["calibrationQuality"] = calibration_quality
    identity_runtime = identity_runtime_quality(
        frame_result,
        identity_result,
        jersey_ocr_profile=jersey_ocr_profile,
    )
    quality["identityRuntime"] = identity_runtime
    verdict_rank = {"pass": 0, "review": 1, "reject": 2}
    quality["verdict"] = max(
        (quality["verdict"], calibration_quality["verdict"]),
        key=lambda value: verdict_rank[value],
    )
    if identity_runtime["status"] == "degraded":
        quality["verdict"] = max(
            (quality["verdict"], "review"),
            key=lambda value: verdict_rank[value],
        )
        quality.setdefault("summary", {})[
            "identityRuntimeReview"
        ] = list(identity_runtime["reasons"])
        quality.setdefault("gates", []).append(
            {
                "id": "identity-runtime",
                "label": "Identity inference dependencies",
                "status": "review",
                "required": True,
                "value": None,
                "unit": "status",
                "evidence": "runtime-readiness-and-batch-result",
                "thresholds": {"pass": "all-enabled-dependencies-ready"},
                "note": ", ".join(identity_runtime["reasons"]),
            }
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
        96,
        100,
        completed=0,
        total=1,
    )
    completed_progress = progress.complete(len(tracks), len(active_ball_keyframes))
    set_reconstruction_status_in_memory(
        scene,
        "ready",
        stage="reconstruction",
        resultState="current",
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
            jersey_ocr_profile=jersey_ocr_profile,
        )
        | {
            "calibrationUsage": deepcopy(dict(calibration_usage)),
            **(
                {
                    "contactPoint": deepcopy(
                        calibration_result.contact_point_diagnostics
                    )
                }
                if calibration_result.contact_point_diagnostics is not None
                else {}
            ),
        },
    )
    completed_reconstruction = video["reconstruction"]
    calibration_artifact = (
        (completed_reconstruction.get("calibrationArtifactInput") or {}).get(
            "artifact"
        )
        or {}
    )
    completed_reconstruction["reconstructionProvenance"] = {
        "schemaVersion": 1,
        "producerRunId": completed_reconstruction.get("runId"),
        "inputFingerprint": completed_reconstruction.get("inputFingerprint"),
        "producedAt": completed_reconstruction.get("completedAt"),
        "calibrationProducerRunId": (
            completed_reconstruction.get("calibrationArtifactInput") or {}
        ).get("producerRunId"),
        "calibrationDataFingerprint": (
            completed_reconstruction.get("calibrationArtifactInput") or {}
        ).get("dataFingerprint"),
        "calibrationArtifactSha256": calibration_artifact.get("sha256"),
        "identityTimelineArtifactSha256": (
            (
                completed_reconstruction.get("artifactManifest") or {}
            ).get("artifacts")
            or {}
        ).get("identityTimeline", {}).get("sha256"),
    }
    return scene
