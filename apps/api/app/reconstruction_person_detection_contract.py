from __future__ import annotations

"""Detector-space person observation consumed by reconstruction phases."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    feature: np.ndarray
    annotation_id: str | None = None
    annotation_kind: str | None = None
    annotation_label: str | None = None
    external_player_id: str | None = None
    pitch_x: float | None = None
    pitch_z: float | None = None
    projection_source: str | None = None
    calibration_frame_index: int | None = None
    position_uncertainty_metres: float | None = None
    association_cost: float | None = None
    association_margin: float | None = None
    association_diagnostics: dict | None = None
    tracking_decision: str | None = None
    metric_projection_reason: str | None = None
    raw_pitch_x: float | None = None
    raw_pitch_z: float | None = None
    # The tracker later moves x/y into stabilized camera coordinates. Preserve
    # the detector-space foot point and frame for published observations.
    source_frame_index: int | None = None
    image_x: float | None = None
    image_y: float | None = None
    # Soccer ReID and the short-horizon HSV feature are distinct vector spaces.
    reid_feature: np.ndarray | None = None
    reid_quality: dict | None = None
    # Crop identity published by the detection pass (person crop store). The
    # crop digest is the cache key for downstream ReID/OCR model results.
    crop_frame_sha256: str | None = None
    crop_sha256: str | None = None
    crop_quality: dict | None = None
    crop_rejection_reasons: tuple[str, ...] = ()
    # Optional pose-derived ground contact point in frozen detector space.
    # When absent, metric projection uses the bbox bottom-centre (x, y).
    contact_image_x: float | None = None
    contact_image_y: float | None = None
    contact_source: str | None = None
    contact_score: float | None = None
    reid_evidence_fingerprint: str | None = None
    reid_role: str | None = None
    reid_role_confidence: float | None = None
    observation_id: str | None = None
    annotation_ids: set[str] = field(default_factory=set)
    identity_tombstone_annotation_ids: set[str] = field(default_factory=set)
    roster_binding_annotation_ids: set[str] = field(default_factory=set)
    roster_binding_state: str | None = None
    manual_semantic_key: tuple[int, str, int, str] | None = None
    manual_identity_owner_ids: set[str] = field(default_factory=set)
    # A roster unbind anchors canonical remapping but is not positive identity
    # evidence merely because it is stored as a confirm-shaped correction.
    annotation_is_identity_evidence: bool = True
