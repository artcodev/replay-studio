from __future__ import annotations

"""Build a frame-local draft from neighboring published calibration evidence."""

from copy import deepcopy
from typing import Mapping

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_geometry import ANCHOR_PRESETS
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_calibration_draft import calibration_draft
from .reconstruction_calibration_frame_context import sampled_frame_context
from .reconstruction_errors import ReconstructionError


def _matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(float(np.linalg.det(matrix))) < 1e-10
    ):
        raise ReconstructionError(f"{label} is not a valid homography")
    return matrix


def _normalized(matrix: np.ndarray, label: str) -> np.ndarray:
    scale = float(matrix[2, 2])
    if abs(scale) < 1e-10:
        scale = float(np.linalg.norm(matrix))
    if abs(scale) < 1e-10:
        raise ReconstructionError(f"{label} cannot be normalized")
    result = matrix / scale
    if result[2, 2] < 0:
        result = -result
    return result


def _accepted(item: Mapping) -> bool:
    return (
        "accepted" in str(item.get("solutionStatus") or "")
        and str(item.get("projectionSource") or "none") != "none"
        and item.get("imageToPitch") is not None
    )


def _camera_to_reference(item: Mapping) -> np.ndarray:
    camera = item.get("cameraMotion")
    if not isinstance(camera, Mapping):
        raise ReconstructionError("Stored frame has no camera-motion transform")
    return _matrix(
        camera.get("currentToReference"),
        "Stored camera-motion transform",
    )


def _stabilized_homography(item: Mapping) -> np.ndarray:
    image_to_pitch = _matrix(
        item.get("imageToPitch"),
        "Stored frame calibration",
    )
    try:
        return _normalized(
            image_to_pitch @ np.linalg.inv(_camera_to_reference(item)),
            "Stabilized frame calibration",
        )
    except np.linalg.LinAlgError as exc:
        raise ReconstructionError(
            "Stored camera-motion transform could not be inverted"
        ) from exc


def _require_contiguous_camera_reference(
    evidence: list,
    start_index: int,
    end_index: int,
) -> None:
    """Reject a borrow that crosses a reset camera-motion reference."""

    first, last = sorted((start_index, end_index))
    for index in range(first + 1, last + 1):
        item = evidence[index]
        camera = item.get("cameraMotion") if isinstance(item, Mapping) else None
        status = (
            str(camera.get("status") or "")
            if isinstance(camera, Mapping)
            else ""
        )
        if status != "estimated":
            source_index = (
                item.get("sourceFrameIndex")
                if isinstance(item, Mapping)
                else None
            )
            label = (
                f"source frame #{source_index}"
                if source_index is not None
                else f"calibration sample {index}"
            )
            raise ReconstructionError(
                "Cannot borrow calibration across a camera-motion boundary: "
                f"{label} starts a new reference ({status or 'missing motion evidence'}). "
                "Run PnLCalib or calibrate this frame manually."
            )


def _pitch_calibration(item: Mapping, matrix: np.ndarray, method: str) -> PitchCalibration:
    return PitchCalibration(
        image_to_pitch=matrix,
        confidence=float(item.get("confidence") or 0.75),
        supported_lines=int(item.get("supportedLines") or 4),
        mean_line_score=float(item.get("meanLineScore") or 0.0),
        rectangle=str(item.get("rectangle") or "manual"),
        matched_curves=int(item.get("matchedCurves") or 0),
        method=method,
        keypoint_count=int(item.get("keypointCount") or 4),
        detected_keypoint_count=int(item.get("detectedKeypointCount") or 0),
        inlier_count=int(item.get("inlierCount") or 4),
        inlier_ratio=(
            float(item["inlierRatio"])
            if item.get("inlierRatio") is not None
            else None
        ),
        reprojection_error=(
            float(item["reprojectionError"])
            if item.get("reprojectionError") is not None
            else None
        ),
        reprojection_p95=(
            float(item["reprojectionP95"])
            if item.get("reprojectionP95") is not None
            else None
        ),
        raw_keypoints=tuple(
            deepcopy(item.get("keypoints") or item.get("rawKeypoints") or [])
        ),
        raw_lines=tuple(deepcopy(item.get("rawLines") or [])),
        confidence_kind="borrowed-frame-calibration-score",
    )


def borrow_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    source: str,
    requested_preset: str | None = None,
) -> dict:
    hydrated = deepcopy(scene)
    hydrate_scene_reconstruction(hydrated, names=("calibrationFrames",))
    frame_index, frame_time, image = sampled_frame_context(
        hydrated,
        scene_time,
    )
    reconstruction = hydrated["payload"]["videoAsset"].get("reconstruction") or {}
    calibration = reconstruction.get("calibration") or {}
    evidence = calibration.get("frameEvidence") or []
    if not (0 <= frame_index < len(evidence)):
        raise ReconstructionError("The selected frame is outside calibration evidence")

    previous = next(
        (
            (index, evidence[index])
            for index in range(frame_index - 1, -1, -1)
            if isinstance(evidence[index], Mapping) and _accepted(evidence[index])
        ),
        None,
    )
    following = next(
        (
            (index, evidence[index])
            for index in range(frame_index + 1, len(evidence))
            if isinstance(evidence[index], Mapping) and _accepted(evidence[index])
        ),
        None,
    )
    if source == "previous":
        selected = previous
        draft_source = "borrowed-previous"
    elif source == "next":
        selected = following
        draft_source = "borrowed-next"
    elif source == "interpolation":
        selected = None
        draft_source = "borrowed-interpolation"
    else:
        raise ReconstructionError("Unknown neighboring calibration source")

    target = evidence[frame_index]
    if not isinstance(target, Mapping):
        raise ReconstructionError("The selected frame calibration is malformed")
    target_camera = _camera_to_reference(target)
    borrowed_indices: list[int]
    if source == "interpolation":
        if previous is None or following is None:
            raise ReconstructionError(
                "Interpolation requires resolved frames on both sides"
            )
        _require_contiguous_camera_reference(
            evidence,
            previous[0],
            following[0],
        )
        previous_time = float(previous[1].get("sceneTime") or 0.0)
        next_time = float(following[1].get("sceneTime") or 0.0)
        span = next_time - previous_time
        alpha = (
            0.5
            if span <= 1e-9
            else max(0.0, min(1.0, (float(frame_time) - previous_time) / span))
        )
        previous_stabilized = _stabilized_homography(previous[1])
        next_stabilized = _stabilized_homography(following[1])
        if float(np.sum(previous_stabilized * next_stabilized)) < 0:
            next_stabilized = -next_stabilized
        stabilized = _normalized(
            previous_stabilized * (1.0 - alpha) + next_stabilized * alpha,
            "Interpolated frame calibration",
        )
        reference = previous[1] if alpha <= 0.5 else following[1]
        borrowed_indices = [previous[0], following[0]]
    else:
        if selected is None:
            direction = "previous" if source == "previous" else "next"
            raise ReconstructionError(
                f"No resolved {direction} frame is available"
            )
        _require_contiguous_camera_reference(
            evidence,
            frame_index,
            selected[0],
        )
        stabilized = _stabilized_homography(selected[1])
        reference = selected[1]
        borrowed_indices = [selected[0]]

    current_to_pitch = _normalized(
        stabilized @ target_camera,
        "Borrowed current-frame calibration",
    )
    preset = requested_preset
    if preset not in ANCHOR_PRESETS:
        visible_side = str(reference.get("visiblePitchSide") or "")
        preset = (
            f"penalty-area-{visible_side}"
            if visible_side in {"left", "right"}
            else "center-circle"
        )
    draft = calibration_draft(
        hydrated,
        frame_index,
        frame_time,
        image,
        _pitch_calibration(reference, current_to_pitch, draft_source),
        preset,
        draft_source,
        warnings=[
            (
                f"Prepared from calibration sample(s) "
                f"{', '.join(str(index) for index in borrowed_indices)} using "
                "stored camera-motion transforms; review the overlay before saving."
            )
        ],
    )
    draft["borrowedFromSampleIndices"] = borrowed_indices
    return draft


__all__ = ("borrow_scene_pitch_calibration",)
