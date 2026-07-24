"""Pose-derived ground contact points for the `pose-feet` profile.

The bbox bottom-centre is a systematically wrong foot point for tackling,
jumping, leaning and partially occluded players; after the homography that
pixel error becomes metres. This module reads the already-extracted person
crops from the crop store, runs RTMPose (Halpe26 — the schema with explicit
toe/heel keypoints) on each eligible crop, and stamps a detector-space
contact point onto the detection. Metric projection then prefers that point.

Every degradation is explicit and per-observation: a missing pose runtime,
an undersized crop, low-confidence feet or an implausible pose all fall back
to the observed bbox bottom-centre with a counted reason — the run never
fails because of this optional evidence layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np

from .reconstruction_person_detection_contract import Detection


# Halpe26 keypoint indices (feet evidence).
_LEFT_ANKLE, _RIGHT_ANKLE = 15, 16
_LEFT_FOOT = (20, 22, 24)  # big toe, small toe, heel
_RIGHT_FOOT = (21, 23, 25)

# crop BGR image -> (keypoints[K, 2] in crop pixels, scores[K])
PoseBackend = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray] | None]


@dataclass(frozen=True, slots=True)
class ContactPointPolicy:
    minimum_crop_height: int = 48
    minimum_keypoint_score: float = 0.35
    maximum_bbox_deviation_ratio: float = 0.35


def contact_point_policy_from_settings() -> ContactPointPolicy:
    from .config import get_settings

    settings = get_settings()
    return ContactPointPolicy(
        minimum_crop_height=int(settings.pose_contact_min_crop_height),
        minimum_keypoint_score=float(settings.pose_contact_min_keypoint_score),
        maximum_bbox_deviation_ratio=float(
            settings.pose_contact_max_bbox_deviation_ratio
        ),
    )


def rtmpose_backend() -> PoseBackend | None:
    """Build the real RTMPose backend, or None when the runtime is missing."""

    from .config import get_settings

    settings = get_settings()
    try:
        from rtmlib import RTMPose

        model = RTMPose(
            onnx_model=str(settings.pose_contact_model_url),
            model_input_size=tuple(settings.pose_contact_model_input_size),
            backend="onnxruntime",
            device=str(settings.pose_contact_device),
        )
    except Exception:
        return None

    def infer(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        keypoints, scores = model(crop_bgr)
        if keypoints is None or len(keypoints) == 0:
            return None
        return (
            np.asarray(keypoints[0], dtype=np.float64),
            np.asarray(scores[0], dtype=np.float64),
        )

    return infer


def _foot_point(
    keypoints: np.ndarray,
    scores: np.ndarray,
    foot_indices: tuple[int, ...],
    ankle_index: int,
    minimum_score: float,
) -> tuple[float, float, float] | None:
    """The lowest confident toe/heel of one foot, ankle as fallback."""

    candidates = [
        (float(keypoints[index][0]), float(keypoints[index][1]), float(scores[index]))
        for index in foot_indices
        if float(scores[index]) >= minimum_score
    ]
    if candidates:
        x, y, score = max(candidates, key=lambda item: item[1])
        return x, y, score
    if float(scores[ankle_index]) >= minimum_score:
        return (
            float(keypoints[ankle_index][0]),
            float(keypoints[ankle_index][1]),
            float(scores[ankle_index]),
        )
    return None


def contact_point_from_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    policy: ContactPointPolicy,
) -> tuple[float, float, float] | None:
    """Crop-space ground point: midpoint of confident feet, or one foot."""

    if keypoints.shape[0] < 26 or scores.shape[0] < 26:
        return None
    left = _foot_point(
        keypoints, scores, _LEFT_FOOT, _LEFT_ANKLE, policy.minimum_keypoint_score
    )
    right = _foot_point(
        keypoints, scores, _RIGHT_FOOT, _RIGHT_ANKLE, policy.minimum_keypoint_score
    )
    feet = [point for point in (left, right) if point is not None]
    if not feet:
        return None
    x = sum(point[0] for point in feet) / len(feet)
    y = sum(point[1] for point in feet) / len(feet)
    score = min(point[2] for point in feet)
    return x, y, score


def _count(diagnostics: dict, field: str) -> None:
    diagnostics[field] = int(diagnostics.get(field, 0)) + 1


def _resolve_one(
    detection: Detection,
    crop_record,
    backend: PoseBackend,
    policy: ContactPointPolicy,
    diagnostics: dict,
) -> None:
    quality = detection.crop_quality or {}
    if int(quality.get("cropHeight") or 0) < policy.minimum_crop_height:
        _count(diagnostics, "cropTooSmallCount")
        return
    crop = cv2.imdecode(
        np.frombuffer(crop_record.crop_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if crop is None:
        _count(diagnostics, "cropUndecodableCount")
        return
    try:
        pose = backend(crop)
    except Exception:
        _count(diagnostics, "poseBackendErrorCount")
        return
    if pose is None:
        _count(diagnostics, "noPoseCount")
        return
    contact = contact_point_from_pose(pose[0], pose[1], policy=policy)
    if contact is None:
        _count(diagnostics, "lowConfidenceFeetCount")
        return
    crop_x, crop_y, score = contact
    x1, y1, _x2, _y2 = crop_record.padded_rect
    frame_x = float(x1) + crop_x
    frame_y = float(y1) + crop_y
    bbox_x = float(
        detection.image_x if detection.image_x is not None else detection.x
    )
    bbox_y = float(
        detection.image_y if detection.image_y is not None else detection.y
    )
    limit_x = float(detection.width) * policy.maximum_bbox_deviation_ratio
    limit_y = float(detection.height) * policy.maximum_bbox_deviation_ratio
    if abs(frame_x - bbox_x) > limit_x or abs(frame_y - bbox_y) > limit_y:
        _count(diagnostics, "implausibleFeetCount")
        return
    detection.contact_image_x = frame_x
    detection.contact_image_y = frame_y
    detection.contact_source = "pose-feet"
    detection.contact_score = round(float(score), 4)
    _count(diagnostics, "poseFeetCount")


def resolve_pose_contact_points(
    person_frames: Sequence[tuple[list[Detection], float]],
    *,
    policy: ContactPointPolicy,
    backend_factory: Callable[[], PoseBackend | None] = rtmpose_backend,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Stamp pose-feet contact points; return explicit coverage diagnostics."""

    from .person_crop_store import lookup_person_crop_envelope, person_crop_store_runtime

    diagnostics: dict = {
        "profile": "pose-feet",
        "requestedObservationCount": 0,
        "poseFeetCount": 0,
    }
    backend = backend_factory()
    if backend is None:
        diagnostics["status"] = "pose-runtime-unavailable"
        return diagnostics
    store_directory, crop_policy = person_crop_store_runtime()
    envelopes: dict[str, dict | None] = {}
    total = sum(
        1
        for people, _time in person_frames
        for detection in people
        if detection.observation_id
    )
    for people, _time in person_frames:
        for detection in people:
            if not detection.observation_id:
                continue
            diagnostics["requestedObservationCount"] += 1
            if on_progress is not None and (
                diagnostics["requestedObservationCount"] % 50 == 0
                or diagnostics["requestedObservationCount"] == total
            ):
                on_progress(diagnostics["requestedObservationCount"], total)
            frame_sha = detection.crop_frame_sha256
            if not frame_sha or not detection.crop_sha256:
                _count(diagnostics, "cropStoreUnavailableCount")
                continue
            if frame_sha not in envelopes:
                envelopes[frame_sha] = lookup_person_crop_envelope(
                    store_directory,
                    frame_sha256=frame_sha,
                    policy=crop_policy,
                )
            record = (envelopes[frame_sha] or {}).get(
                str(detection.observation_id)
            )
            if record is None or not record.crop_jpeg:
                _count(diagnostics, "cropStoreUnavailableCount")
                continue
            _resolve_one(detection, record, backend, policy, diagnostics)
    requested = diagnostics["requestedObservationCount"]
    diagnostics["status"] = "ready" if requested else "no-observations"
    diagnostics["poseFeetRatio"] = (
        round(diagnostics["poseFeetCount"] / requested, 3) if requested else 0.0
    )
    return diagnostics


__all__ = (
    "ContactPointPolicy",
    "PoseBackend",
    "contact_point_from_pose",
    "contact_point_policy_from_settings",
    "resolve_pose_contact_points",
    "rtmpose_backend",
)
