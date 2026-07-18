"""Resolve and encode one persisted identity observation crop."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import cv2

from .config import get_settings
from .identity_review_errors import IdentityReviewError
from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .video_media_paths import video_generation_directory
from .video_processing_contract import VideoProcessingError


def _identity_observation(
    scene: Mapping[str, Any],
    observation_id: str,
) -> Mapping[str, Any]:
    match: Mapping[str, Any] | None = None
    for person in scene.get("payload", {}).get("canonicalPeople") or []:
        for observation in person.get("observations") or []:
            identifier = str(
                observation.get("observationId") or observation.get("id") or ""
            )
            if identifier != observation_id:
                continue
            if match is not None:
                raise IdentityReviewError(
                    "Observation ID is ambiguous across canonical people"
                )
            match = observation
    if match is None:
        raise IdentityReviewError("Identity observation was not found")
    return match


def _source_frame_index(observation: Mapping[str, Any]) -> int:
    value = observation.get("sourceFrameIndex")
    if value is None:
        value = observation.get("frameIndex")
    if isinstance(value, bool):
        raise IdentityReviewError("Identity observation has no source frame")
    try:
        frame_index = int(value)
    except (TypeError, ValueError) as exc:
        raise IdentityReviewError("Identity observation has no source frame") from exc
    if frame_index < 0:
        raise IdentityReviewError("Identity observation has no source frame")
    return frame_index


def identity_observation_crop(
    scene: Mapping[str, Any],
    observation_id: str,
    *,
    media_root: Path | None = None,
    padding_ratio: float = 0.12,
) -> bytes:
    """Return the source crop identified by a persisted canonical observation."""

    scene = deepcopy(scene)
    hydrate_scene_reconstruction(scene, names=("identityTimeline",))
    observation = _identity_observation(scene, observation_id)
    video = scene.get("payload", {}).get("videoAsset") or {}
    asset_id = str(video.get("id") or "")
    if not asset_id:
        raise IdentityReviewError("Identity observation has no source frame")
    frame_index = _source_frame_index(observation)
    root = Path(media_root or get_settings().media_root).resolve()
    generation_key = str(video.get("generationKey") or "")
    try:
        frame_path = (
            video_generation_directory(
                asset_id,
                generation_key,
                media_root=root,
            )
            / "frames"
            / f"frame_{frame_index:05d}.jpg"
        ).resolve()
    except VideoProcessingError as exc:
        raise IdentityReviewError(str(exc)) from exc
    if not frame_path.is_relative_to(root):
        raise IdentityReviewError("Identity source frame path is invalid")
    image = cv2.imread(str(frame_path))
    if image is None:
        raise IdentityReviewError("Identity source frame is unavailable")
    bbox = observation.get("bbox") or {}
    x = float(bbox.get("x") or 0.0)
    y = float(bbox.get("y") or 0.0)
    width = float(bbox.get("width") or 0.0)
    height = float(bbox.get("height") or 0.0)
    if width <= 0.0 or height <= 0.0:
        raise IdentityReviewError("Identity observation has an invalid bbox")
    pad_x = width * max(0.0, min(0.5, padding_ratio))
    pad_y = height * max(0.0, min(0.5, padding_ratio))
    image_height, image_width = image.shape[:2]
    x1 = max(0, min(image_width, int(x - pad_x)))
    y1 = max(0, min(image_height, int(y - pad_y)))
    x2 = max(0, min(image_width, int(x + width + pad_x + 0.999)))
    y2 = max(0, min(image_height, int(y + height + pad_y + 0.999)))
    if x2 <= x1 or y2 <= y1:
        raise IdentityReviewError("Identity observation crop is empty")
    crop = image[y1:y2, x1:x2]
    encoded, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not encoded:
        raise IdentityReviewError("Identity observation crop could not be encoded")
    return bytes(buffer)


__all__ = ("identity_observation_crop",)
