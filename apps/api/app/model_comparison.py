from __future__ import annotations

from datetime import UTC, datetime
from math import hypot
from time import perf_counter

import numpy as np

from .config import get_settings
from .pitch_calibration import PitchCalibration
from .reconstruction import (
    METRIC_CALIBRATION_THRESHOLD,
    Detection,
    TrackState,
    _camera_step,
    _frame_paths,
    _load_model,
    _person_detections,
    _project_unclamped,
    _saved_pitch_calibration,
    _scene_tracks,
    _stabilize_detections,
    _team_clusters,
    _track_people,
)


BASELINE_MODEL = "yolo26n.pt"
CANDIDATE_MODEL = "yolo26m.pt"
MODEL_COMPARISON_IMAGE_SIZE = 1280
MODEL_COMPARISON_CONFIDENCE = 0.035
OUTSIDE_PITCH_MARGIN_METRES = 1.5


def _inside_pitch(position: tuple[float, float], pitch: dict, margin: float = 0.0) -> bool:
    half_length = float(pitch["length"]) / 2
    half_width = float(pitch["width"]) / 2
    return (
        -half_length - margin <= position[0] <= half_length + margin
        and -half_width - margin <= position[1] <= half_width + margin
    )


def _observation(
    detection: Detection,
    frame_index: int,
    pitch: dict,
    frame_size: tuple[int, int],
    calibration: PitchCalibration | None,
) -> dict:
    width, height = frame_size
    pitch_position = _project_unclamped(
        detection.x,
        detection.y,
        width,
        height,
        pitch,
        calibration,
    )
    return {
        "frame": frame_index,
        "x": detection.x,
        "y": detection.y,
        "height": detection.height,
        "confidence": detection.confidence,
        "pitch": pitch_position,
        "insidePitch": _inside_pitch(pitch_position, pitch, OUTSIDE_PITCH_MARGIN_METRES),
        "wouldClamp": not _inside_pitch(pitch_position, pitch),
    }


def _mostly_outside_track_count(
    tracks: list[dict],
    pitch: dict,
    frame_size: tuple[int, int],
    calibration: PitchCalibration | None,
) -> int:
    count = 0
    for track in tracks:
        keyframes = track.get("keyframes") or []
        if not keyframes:
            continue
        boundary_points = sum(
            abs(float(point["x"])) >= float(pitch["length"]) / 2 - 0.01
            or abs(float(point["z"])) >= float(pitch["width"]) / 2 - 0.01
            for point in keyframes
            if point.get("observed") is not False
            and float(point.get("confidence") or 0.0) > 0.12
        )
        observed = sum(
            point.get("observed") is not False
            and float(point.get("confidence") or 0.0) > 0.12
            for point in keyframes
        )
        if observed and boundary_points / observed >= 0.6:
            count += 1
    return count


def _run_model(scene: dict, model_name: str) -> tuple[dict, list[list[dict]]]:
    frames = _frame_paths(scene)
    if not frames:
        raise ValueError("No sampled frames are available for model comparison")
    model = _load_model(model_name)
    pitch = scene["payload"]["pitch"]
    calibration = _saved_pitch_calibration(scene)
    person_frames: list[tuple[list[Detection], float]] = []
    observations: list[list[dict]] = []
    frame_counts: list[int] = []
    inference_seconds = 0.0
    previous_image: np.ndarray | None = None
    camera_transform = np.eye(3, dtype=np.float64)
    frame_size = (960, 540)

    for frame_index, (path, time) in enumerate(frames):
        started = perf_counter()
        result = model.predict(
            str(path),
            imgsz=MODEL_COMPARISON_IMAGE_SIZE,
            conf=MODEL_COMPARISON_CONFIDENCE,
            classes=[0, 32],
            device=get_settings().reconstruction_device,
            verbose=False,
        )[0]
        inference_seconds += perf_counter() - started
        frame_size = (result.orig_img.shape[1], result.orig_img.shape[0])
        people, balls = _person_detections(result)
        if previous_image is not None:
            camera_transform = camera_transform @ _camera_step(previous_image, result.orig_img)
        _stabilize_detections(people, balls, camera_transform)
        current_observations = [
            _observation(person, frame_index, pitch, frame_size, calibration)
            for person in people
        ]
        observations.append(current_observations)
        person_frames.append((people, time))
        frame_counts.append(len(people))
        previous_image = result.orig_img

    raw_tracks = _track_people(person_frames)
    minimum = max(5, round(len(frames) * 0.24))
    stable_tracks: list[TrackState] = [track for track in raw_tracks if len(track.points) >= minimum]
    mapping, colors = _team_clusters(stable_tracks, frame_size[0])
    accepted_tracks = _scene_tracks(raw_tracks, mapping, colors, frame_size, scene, calibration)
    flattened = [item for frame in observations for item in frame]
    outside = sum(not item["insidePitch"] for item in flattened)
    would_clamp = sum(item["wouldClamp"] for item in flattened)
    total = len(flattened)
    return (
        {
            "model": model_name,
            "frameCount": len(frames),
            "totalDetections": total,
            "meanDetectionsPerFrame": round(float(np.mean(frame_counts)), 2),
            "minimumDetectionsInFrame": min(frame_counts),
            "maximumDetectionsInFrame": max(frame_counts),
            "inPitchDetections": total - outside,
            "outsidePitchDetections": outside,
            "wouldClampDetections": would_clamp,
            "lowConfidenceDetections": sum(item["confidence"] < 0.15 for item in flattened),
            "rawTrackCount": len(raw_tracks),
            "stableTrackCount": len(stable_tracks),
            "acceptedTrackCount": len(accepted_tracks),
            "boundaryRiskTrackCount": _mostly_outside_track_count(
                accepted_tracks,
                pitch,
                frame_size,
                calibration,
            ),
            "inferenceSeconds": round(inference_seconds, 2),
            "meanInferenceMilliseconds": round(inference_seconds / len(frames) * 1000, 1),
        },
        observations,
    )


def _pair_frame_observations(
    baseline: list[dict],
    candidate: list[dict],
) -> tuple[int, list[dict], list[dict]]:
    pairs: list[tuple[float, int, int]] = []
    for baseline_index, left in enumerate(baseline):
        for candidate_index, right in enumerate(candidate):
            distance = hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))
            limit = max(14.0, min(float(left["height"]), float(right["height"])) * 0.72)
            if distance <= limit:
                pairs.append((distance / limit, baseline_index, candidate_index))

    baseline_matches: set[int] = set()
    candidate_matches: set[int] = set()
    for _, baseline_index, candidate_index in sorted(pairs):
        if baseline_index in baseline_matches or candidate_index in candidate_matches:
            continue
        baseline_matches.add(baseline_index)
        candidate_matches.add(candidate_index)
    return (
        len(baseline_matches),
        [item for index, item in enumerate(baseline) if index not in baseline_matches],
        [item for index, item in enumerate(candidate) if index not in candidate_matches],
    )


def _comparison_summary(
    baseline: dict,
    candidate: dict,
    baseline_frames: list[list[dict]],
    candidate_frames: list[list[dict]],
) -> dict:
    shared = 0
    baseline_only: list[dict] = []
    candidate_only: list[dict] = []
    for left, right in zip(baseline_frames, candidate_frames):
        matched, unmatched_left, unmatched_right = _pair_frame_observations(left, right)
        shared += matched
        baseline_only.extend(unmatched_left)
        candidate_only.extend(unmatched_right)

    baseline_only_in_pitch = sum(item["insidePitch"] for item in baseline_only)
    candidate_only_in_pitch = sum(item["insidePitch"] for item in candidate_only)
    in_pitch_gain = candidate_only_in_pitch - baseline_only_in_pitch
    outside_delta = int(candidate["outsidePitchDetections"]) - int(baseline["outsidePitchDetections"])
    stable_delta = int(candidate["stableTrackCount"]) - int(baseline["stableTrackCount"])
    frame_count = max(1, int(candidate["frameCount"]))
    meaningful_gain = max(3, round(frame_count * 0.15))

    if in_pitch_gain >= meaningful_gain and outside_delta <= meaningful_gain and stable_delta >= -1:
        verdict = "candidate"
    elif (
        in_pitch_gain <= -meaningful_gain
        and outside_delta >= -meaningful_gain
        and stable_delta < 0
    ):
        verdict = "baseline"
    else:
        verdict = "review"

    rationale = [
        f"The candidate adds {candidate_only_in_pitch} in-pitch observations not paired with the baseline.",
        f"The baseline retains {baseline_only_in_pitch} in-pitch observations not paired with the candidate.",
        f"Outside-pitch observations change by {outside_delta:+d}; stable tracks change by {stable_delta:+d}.",
    ]
    return {
        "sharedDetections": shared,
        "baselineOnlyDetections": len(baseline_only),
        "candidateOnlyDetections": len(candidate_only),
        "baselineOnlyInPitchDetections": baseline_only_in_pitch,
        "candidateOnlyInPitchDetections": candidate_only_in_pitch,
        "inPitchObservationGain": in_pitch_gain,
        "outsidePitchDetectionDelta": outside_delta,
        "stableTrackDelta": stable_delta,
        "acceptedTrackDelta": int(candidate["acceptedTrackCount"]) - int(baseline["acceptedTrackCount"]),
        "verdict": verdict,
        "rationale": rationale,
    }


def compare_scene_models(scene: dict) -> dict:
    baseline, baseline_frames = _run_model(scene, BASELINE_MODEL)
    candidate, candidate_frames = _run_model(scene, CANDIDATE_MODEL)
    calibration = _saved_pitch_calibration(scene)
    metric_projection = calibration is not None and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD
    warnings = [
        "Candidate-only observations are not ground truth; inspect the source frame before treating them as recovered players."
    ]
    if not metric_projection:
        warnings.append("Pitch-boundary metrics use an approximate projection because metric calibration is unavailable.")
    return {
        "sceneId": scene["id"],
        "completedAt": datetime.now(UTC).isoformat(),
        "frameCount": baseline["frameCount"],
        "settings": {
            "imageSize": MODEL_COMPARISON_IMAGE_SIZE,
            "confidence": MODEL_COMPARISON_CONFIDENCE,
            "device": get_settings().reconstruction_device,
            "outsidePitchMarginMetres": OUTSIDE_PITCH_MARGIN_METRES,
            "metricProjection": metric_projection,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _comparison_summary(
            baseline,
            candidate,
            baseline_frames,
            candidate_frames,
        ),
        "warnings": warnings,
    }
