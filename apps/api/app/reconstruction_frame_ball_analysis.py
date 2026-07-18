from __future__ import annotations

"""Single-frame ball inference and candidate projection."""

from copy import deepcopy
from dataclasses import dataclass
from math import hypot
from pathlib import Path

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_ball_detector_selection import configured_ball_detectors
from .reconstruction_identity_read_model import interpolate_scene_keyframes
from .reconstruction_pitch_projection import project_pitch_point


@dataclass(frozen=True)
class FrameBallDetection:
    balls: list[dict]
    raw_balls: list[dict]
    backend: str
    warning: str | None
    frame_time: float
    frame_index: int


def _scaled_detections(batch, frame_size: tuple[int, int]) -> list[dict]:
    frame_width, frame_height = frame_size
    detections = batch.as_reconstruction_detections()
    source_width, source_height = batch.image_size
    scale_x = frame_width / max(1.0, float(source_width))
    scale_y = frame_height / max(1.0, float(source_height))
    for item in detections:
        source_x, source_y = float(item["x"]), float(item["y"])
        item["sourceImagePosition"] = {
            "x": source_x,
            "y": source_y,
            "width": int(source_width),
            "height": int(source_height),
        }
        item["x"] = source_x * scale_x
        item["y"] = source_y * scale_y
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            item["bbox"] = [
                float(bbox[0]) * scale_x,
                float(bbox[1]) * scale_y,
                float(bbox[2]) * scale_x,
                float(bbox[3]) * scale_y,
            ]
    return detections


def detect_frame_balls(
    *,
    model: object,
    frames: list[tuple[Path, float]],
    target_index: int,
    target_path: Path,
    frame_time: float,
    frame_size: tuple[int, int],
    backend: str,
    detector_input: dict | None,
    generic_candidates: list[dict],
) -> FrameBallDetection:
    detector, fallback = configured_ball_detectors(model, backend, detector_input)
    sampled_paths = [Path(path) for path, _ in frames]
    context_paths = (
        sampled_paths[max(0, target_index - 1)],
        sampled_paths[min(len(sampled_paths) - 1, target_index + 1)],
    )
    warning: str | None = None
    try:
        batch = detector.detect(
            target_path,
            frame_index=target_index,
            timestamp=frame_time,
            context_frames=context_paths,
        )
        balls = _scaled_detections(batch, frame_size)
        for rank, item in enumerate(balls, start=1):
            item["candidateId"] = f"frame-ball-{target_index:05d}-{rank:02d}"
            item["provenance"] = {
                "backend": item.get("detectorBackend") or batch.backend,
                "detectorMetadata": deepcopy(item.get("detectorMetadata") or {}),
                "batchMetadata": deepcopy(dict(batch.metadata)),
            }
    except Exception as exc:
        warning = f"{type(exc).__name__}: {exc}"
        balls = generic_candidates
        if fallback is not None:
            try:
                fallback_batch = fallback.detect(
                    target_path,
                    frame_index=target_index,
                    timestamp=frame_time,
                    context_frames=context_paths,
                )
                balls = _scaled_detections(fallback_batch, frame_size)
            except Exception as fallback_exc:
                warning += (
                    f"; fallback {type(fallback_exc).__name__}: {fallback_exc}"
                )
    return FrameBallDetection(
        balls=balls,
        raw_balls=[{**item} for item in balls],
        backend=backend,
        warning=warning,
        frame_time=float(frame_time),
        frame_index=target_index,
    )


def project_frame_ball_candidates(
    detection: FrameBallDetection,
    *,
    frame_size: tuple[int, int],
    pitch: dict,
    calibration: PitchCalibration | None,
    scene: dict,
) -> tuple[list[dict], int | None]:
    frame_width, frame_height = frame_size
    projected = [
        project_pitch_point(
            float(item.get("stabilizedX", item["x"])),
            float(item.get("stabilizedY", item["y"])),
            frame_width,
            frame_height,
            pitch,
            calibration,
        )
        for item in detection.balls
    ]
    keyframes = scene.get("payload", {}).get("ball", {}).get("keyframes") or []
    ball_position = (
        interpolate_scene_keyframes(keyframes, detection.frame_time)
        if keyframes
        and float(keyframes[0]["t"])
        <= detection.frame_time
        <= float(keyframes[-1]["t"])
        else None
    )
    primary: int | None = None
    if (
        projected
        and ball_position is not None
        and float(ball_position.get("confidence") or 0.0) > 0.12
    ):
        primary = min(
            range(len(projected)),
            key=lambda index: hypot(
                projected[index][0] - float(ball_position["x"]),
                projected[index][1] - float(ball_position["z"]),
            ),
        )
    elif projected:
        strongest = max(
            range(len(detection.balls)),
            key=lambda index: detection.balls[index]["confidence"],
        )
        if float(detection.balls[strongest]["confidence"]) >= 0.25:
            primary = strongest
    return (
        [
            {
                "id": f"ball-{index + 1}",
                "confidence": round(float(item["confidence"]), 3),
                "image": {"x": round(raw["x"], 2), "y": round(raw["y"], 2)},
                "pitch": {"x": round(position[0], 2), "z": round(position[1], 2)},
                "primary": index == primary,
                "backend": item.get("detectorBackend") or detection.backend,
            }
            for index, (item, raw, position) in enumerate(
                zip(detection.balls, detection.raw_balls, projected)
            )
        ],
        primary,
    )
