from __future__ import annotations

import app.ultralytics_person_inference as ultralytics_person_inference

from .config import get_settings
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_errors import ReconstructionError
from .reconstruction_frame_ball_analysis import (
    detect_frame_balls,
    project_frame_ball_candidates,
)
from .reconstruction_frame_context import camera_transform_to_frame
from .reconstruction_frame_identity_projection import project_frame_people
from .reconstruction_identity_read_model import saved_pitch_calibration
from .reconstruction_inputs import frame_paths, load_model
from .reconstruction_motion import stabilize_detections
from .reconstruction_person_annotations import (
    apply_person_annotations,
    frame_annotations,
)
from .reconstruction_pitch_projection import project_pitch_point


def analyze_scene_frame(scene: dict, scene_time: float) -> dict:
    hydrate_scene_reconstruction(scene)
    frames = frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    target_index = min(
        range(len(frames)),
        key=lambda index: abs(frames[index][1] - scene_time),
    )
    target_path, frame_time = frames[target_index]
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    model_name = str(
        reconstruction.get("model") or get_settings().reconstruction_model
    )
    model = load_model(model_name)
    prediction = ultralytics_person_inference.predict_frame(model, target_path)
    frame_size = (prediction.orig_img.shape[1], prediction.orig_img.shape[0])
    people, generic_ball_candidates = (
        ultralytics_person_inference.parse_person_detections(prediction)
    )
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
        prediction.orig_img,
        people,
        frame_annotations(scene, frame_index),
    )
    camera_transform = camera_transform_to_frame(
        frames,
        target_index,
        prediction.orig_img,
    )
    stabilize_detections(people, ball_detection.balls, camera_transform)
    calibration = saved_pitch_calibration(scene)
    pitch = scene["payload"]["pitch"]
    projected_people = [
        project_pitch_point(item.x, item.y, *frame_size, pitch, calibration)
        for item in people
    ]
    identity = project_frame_people(
        scene,
        people=people,
        projected_people=projected_people,
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
    calibration_metadata = reconstruction.get("pitchCalibration") or {}
    source_start = float(scene["payload"]["videoAsset"].get("sourceStart") or 0.0)
    return {
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
    }
