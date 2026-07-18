"""Publish immutable per-frame calibration reconstruction evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .artifact_store import ArtifactStore
from .reconstruction_artifact_codec import CalibrationFramesEncoding, encode_calibration_frames
from .reconstruction_artifact_manifest import (
    CALIBRATION_FRAMES_ARTIFACT_KIND,
    CALIBRATION_FRAMES_SCHEMA_VERSION,
    existing_artifact_reference,
)


@dataclass(frozen=True)
class PublishedCalibrationFrames:
    reference: dict[str, Any]
    encoding: CalibrationFramesEncoding | None


def publish_calibration_frames_artifact(
    scene: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
    references: Mapping[str, Any],
    materialized: set[str],
    *,
    store: ArtifactStore,
) -> PublishedCalibrationFrames:
    """Publish materialized frame evidence, or retain its immutable ref."""

    calibration = (
        reconstruction.get("calibration")
        if isinstance(reconstruction.get("calibration"), Mapping)
        else {}
    )
    ball_detection = (
        reconstruction.get("ballDetection")
        if isinstance(reconstruction.get("ballDetection"), Mapping)
        else {}
    )
    reference = existing_artifact_reference(references, "calibrationFrames")
    needs_publication = (
        reference is None
        or "calibrationFrames" in materialized
        or "frameEvidence" in calibration
    )
    if not needs_publication:
        assert reference is not None
        return PublishedCalibrationFrames(reference, None)

    encoding = encode_calibration_frames(
        str(scene.get("id") or ""),
        calibration,
        ball_detection,
    )
    reference = store.put_json(
        kind=CALIBRATION_FRAMES_ARTIFACT_KIND,
        schema_version=CALIBRATION_FRAMES_SCHEMA_VERSION,
        payload=encoding.payload,
    )
    return PublishedCalibrationFrames(reference, encoding)


__all__ = ("PublishedCalibrationFrames", "publish_calibration_frames_artifact")
