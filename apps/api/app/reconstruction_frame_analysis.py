from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import get_settings
from .person_detection_candidate_selection import parse_person_prediction
from .person_detector_provenance import person_detection_input
from .person_detection_provider_factory import build_person_detection_provider
from .temporal_homography import homography_disagreement_metres
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_errors import ReconstructionError
from .reconstruction_frame_ball_analysis import (
    detect_frame_balls,
    project_frame_ball_candidates,
)
from .reconstruction_frame_context import camera_transform_to_frame
from .reconstruction_frame_calibration_projection import (
    project_inspection_people,
    published_frame_evidence,
    resolve_frame_calibration,
    selected_hypothesis_matrix,
)
from .reconstruction_frame_identity_projection import project_frame_people
from .reconstruction_identity_read_model import saved_pitch_calibration
from .reconstruction_inputs import frame_paths, load_model
from .reconstruction_motion import stabilize_detections
from .reconstruction_person_annotations import (
    apply_person_annotations,
    frame_annotations,
)


def _stabilization_debug(
    raw_coordinates: list[tuple[float, float]],
    people: list,
) -> list[dict]:
    return [
        {
            "rawX": round(raw_x, 1),
            "rawY": round(raw_y, 1),
            "stabilizedX": round(item.x, 1),
            "stabilizedY": round(item.y, 1),
        }
        for (raw_x, raw_y), item in zip(raw_coordinates, people)
    ]


def persist_frame_analysis(payload: dict, scene_id: str, frame_index: int) -> str | None:
    """Best-effort dump of one analysis to the run-log directory family."""

    directory = (
        Path(get_settings().analysis_run_log_directory).expanduser()
        / "frame-analysis"
    )
    target = directory / f"{scene_id}-frame-{frame_index:05d}.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.replace(temporary, target)
    except OSError:
        return None
    return str(target)


def build_frame_analysis_debug(
    scene: dict,
    *,
    scene_time: float,
    frames: list,
    target_index: int,
    target_path,
    frame_time: float,
    model_name: str,
    model,
    person_provider_info: dict,
    person_class_names: dict[int, str],
    raw_detection_log: list[dict],
    frame_index: int,
    camera_transform,
    stabilization_debug: list[dict],
    calibration,
    calibration_metadata: dict,
    calibration_source: str,
    reconstruction: dict,
    frame_size: tuple[int, int],
    timings: dict,
) -> dict:
    """Every per-step fact of one Analyze Frame call, ready for JSON."""

    published_evidence = published_frame_evidence(reconstruction, frame_index)
    disagreement_metres = None
    evidence_summary = None
    if published_evidence is not None:
        evidence_summary = {
            "observationStatus": published_evidence.get("observationStatus"),
            "solutionStatus": published_evidence.get("solutionStatus"),
            "projectionSource": published_evidence.get("projectionSource"),
            "selectedHypothesisId": published_evidence.get("selectedHypothesisId"),
            "rejectionReasons": published_evidence.get("rejectionReasons"),
            "residualP95": (
                (published_evidence.get("alignmentMetrics") or {}).get("residualP95")
            ),
        }
        selected_matrix = selected_hypothesis_matrix(published_evidence)
        if calibration is not None and selected_matrix is not None:
            disagreement_metres = homography_disagreement_metres(
                calibration.image_to_pitch,
                np.asarray(selected_matrix, dtype=np.float64),
                frame_size[0],
                frame_size[1],
            )
    return {
        "schemaVersion": 1,
        "analyzedAt": datetime.now(UTC).isoformat(),
        "frameSelection": {
            "requestedTime": round(float(scene_time), 3),
            "chosenSampleIndex": target_index,
            "frameFile": target_path.name,
            "frameTime": round(float(frame_time), 3),
            "sampleCount": len(frames),
        },
        "detector": {
            **person_detection_input(
                model_name,
                model,
                provider_info=person_provider_info,
            ),
            "classNames": dict(person_class_names),
        },
        "rawDetections": raw_detection_log,
        "rawDetectionVerdictCounts": {
            verdict: sum(
                1
                for record in raw_detection_log
                if record.get("verdict") == verdict
            )
            for verdict in sorted(
                {record.get("verdict") for record in raw_detection_log}
            )
            if verdict
        },
        "annotationsApplied": [
            str(annotation.get("id"))
            for annotation in frame_annotations(scene, frame_index)
        ],
        "cameraTransform": {
            "matrix": np.asarray(camera_transform, dtype=np.float64)
            .round(6)
            .tolist(),
            "isIdentity": bool(
                np.allclose(camera_transform, np.eye(3), atol=1e-9)
            ),
        },
        "stabilization": stabilization_debug,
        "calibration": {
            "analyzeFrameUses": calibration_source,
            "status": calibration_metadata.get("status"),
            "confidence": calibration_metadata.get("confidence"),
            "available": calibration is not None,
            "publishedFrameEvidence": evidence_summary,
            # Kept as a regression signal for older artifacts. A current
            # inspector must report published-per-frame-homography and zero
            # disagreement against the matrix used for its projection.
            "representativeVsPublishedDisagreementMetres": (
                round(float(disagreement_metres), 3)
                if disagreement_metres is not None
                else None
            ),
        },
        "timingsMs": timings,
    }


def analyze_scene_frame(scene: dict, scene_time: float) -> dict:
    started = perf_counter()
    timings: dict[str, float] = {}
    hydrate_scene_reconstruction(scene)
    frames = frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    target_index = min(
        range(len(frames)), key=lambda index: abs(frames[index][1] - scene_time)
    )
    target_path, frame_time = frames[target_index]
    video = scene.get("payload", {}).get("videoAsset", {})
    reconstruction = video.get("reconstruction") or {}
    model_name = str(reconstruction.get("model") or get_settings().reconstruction_model)
    model = load_model(model_name)
    person_provider = build_person_detection_provider(model_name, model)
    timings["prepareMs"] = round((perf_counter() - started) * 1000, 1)
    stage = perf_counter()
    prediction = person_provider.predict(target_path)
    timings["inferenceMs"] = round((perf_counter() - stage) * 1000, 1)
    frame_size = (prediction.image_bgr.shape[1], prediction.image_bgr.shape[0])
    stage = perf_counter()
    raw_detection_log: list[dict] = []
    people, generic_ball_candidates = (
        parse_person_prediction(
            prediction,
            debug_log=raw_detection_log,
        )
    )
    timings["parseMs"] = round((perf_counter() - stage) * 1000, 1)
    ball_backend = str(
        reconstruction.get("ballBackend")
        or get_settings().ball_detection_backend
    )
    ball_detection = detect_frame_balls(
        model=model,
        frames=frames,
        target_index=target_index,
        target_path=target_path,
        frame_time=frame_time,
        frame_size=frame_size,
        backend=ball_backend,
        detector_input=reconstruction.get("ballDetectionInput"),
        generic_candidates=generic_ball_candidates,
    )
    frame_index = int(target_path.stem.split("_")[-1])
    people = apply_person_annotations(
        prediction.image_bgr,
        people,
        frame_annotations(scene, frame_index),
    )
    camera_transform = camera_transform_to_frame(
        frames,
        target_index,
        prediction.image_bgr,
    )
    raw_coordinates = [(item.x, item.y) for item in people]
    representative_calibration = saved_pitch_calibration(scene)
    calibration, calibration_source = resolve_frame_calibration(
        reconstruction,
        frame_index,
        representative_calibration,
    )
    pitch = scene["payload"]["pitch"]
    projected_people, raw_projected_people = project_inspection_people(
        people,
        frame_size=frame_size,
        pitch=pitch,
        calibration=calibration,
    )
    identity = project_frame_people(
        scene,
        people=people,
        projected_people=projected_people,
        raw_projected_people=raw_projected_people,
        frame_index=frame_index,
        frame_time=frame_time,
        calibration=calibration,
    )
    ball_candidates, primary_ball = project_frame_ball_candidates(
        ball_detection,
        frame_size=frame_size,
        pitch=pitch,
        calibration=calibration,
        scene=scene,
    )
    # Stabilization is diagnostic; metric projection used raw pixels.
    stabilize_detections(people, ball_detection.balls, camera_transform)
    stabilization_debug = _stabilization_debug(raw_coordinates, people)
    calibration_metadata = reconstruction.get("pitchCalibration") or {}
    source_start = float(scene["payload"]["videoAsset"].get("sourceStart") or 0.0)

    debug = build_frame_analysis_debug(
        scene,
        scene_time=scene_time,
        frames=frames,
        target_index=target_index,
        target_path=target_path,
        frame_time=frame_time,
        model_name=model_name,
        model=model,
        person_provider_info=person_provider.info(),
        person_class_names=prediction.names,
        raw_detection_log=raw_detection_log,
        frame_index=frame_index,
        camera_transform=camera_transform,
        stabilization_debug=stabilization_debug,
        calibration=calibration,
        calibration_metadata=calibration_metadata,
        calibration_source=calibration_source,
        reconstruction=reconstruction,
        frame_size=frame_size,
        timings=timings,
    )
    response = {
        "sceneId": scene["id"],
        "requestedTime": round(float(scene_time), 3),
        "sceneTime": round(float(frame_time), 3),
        "ballSceneTime": round(ball_detection.frame_time, 3),
        "ballFrameIndex": ball_detection.frame_index,
        "sourceTime": round(source_start + float(frame_time), 3),
        "frameIndex": frame_index,
        "frameWidth": frame_size[0],
        "frameHeight": frame_size[1],
        "model": model_name,
        "ballBackend": ball_backend,
        "projectionMode": reconstruction.get("coordinateSpace") or "screen-relative",
        "calibrationStatus": calibration_metadata.get("status") or "fallback",
        "identityLinking": {
            "mode": (
                "canonical-observations"
                if identity.observation_schema_version is not None
                and identity.observation_schema_version >= 3
                else "persisted-observations"
                if identity.has_observation_schema
                else "rebuild-required"
            ),
            "schemaVersion": identity.observation_schema_version,
        },
        "people": identity.detections,
        "annotations": identity.annotations,
        "correctionSummary": {
            "confirmed": sum(item["action"] == "confirm" for item in identity.annotations),
            "excluded": sum(item["action"] == "exclude" for item in identity.annotations),
            "merged": sum(item["action"] == "merge" for item in identity.annotations),
            "split": sum(item["action"] == "split" for item in identity.annotations),
        },
        "ballCandidates": ball_candidates,
        "matchedTracks": sum(
            item["matchedTrackId"] is not None for item in identity.detections
        ),
        "matchedCanonicalPeople": sum(
            item.get("canonicalPersonId") is not None
            for item in identity.detections
        ),
        "warnings": [
            *(
                [
                    "This scene has no authoritative video-track observations; rebuild tracks before using video ↔ 3D identity linking."
                ]
                if not identity.has_observation_schema
                else []
            ),
            *(
                ["Positions use an approximate visible-half projection."]
                if calibration_metadata.get("status") == "approximate"
                else []
            ),
            *(
                ["No reliable ball candidate was found on this frame."]
                if primary_ball is None
                else []
            ),
            *(
                [
                    "Specialized ball detector degraded to fallback: "
                    f"{ball_detection.warning}"
                ]
                if ball_detection.warning
                else []
            ),
        ],
        "debug": debug,
    }
    debug["timingsMs"]["totalMs"] = round((perf_counter() - started) * 1000, 1)
    debug["debugFile"] = persist_frame_analysis(
        response, str(scene["id"]), frame_index
    )
    return response
