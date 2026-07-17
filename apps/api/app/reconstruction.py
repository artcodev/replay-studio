from __future__ import annotations

import os
from bisect import bisect_left
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
from math import cos, exp, hypot, isfinite, log1p, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable, Mapping
from uuid import uuid4

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .ball_detection import (
    BallDetector,
    BallDetectorConfig,
    build_ball_detector,
)
from .ball_detection_cache import (
    BallDetectionCacheError,
    load_ball_detection_cache,
    store_clean_ball_detection_cache,
)
from .ball_frames import DenseBallFramesError, dense_ball_frame_paths
from .ball_tracking import BallTrackingConfig, resolve_ball_trajectory
from .calibration_worker import CalibrationWorkerError, calibrate_frames_with_worker
from .config import get_settings
from .field_keypoints import calibration_from_pose_result
from .identity_decisions import rejected_roster_candidate_ids
from .identity_resolver import IdentityTracklet, resolve_global_identities
from .identity_worker import (
    IdentityWorkerError,
    embed_identity_frames,
    identity_worker_readiness,
)
from .jersey_ocr_fusion import (
    JerseyEvidenceSummary,
    JerseyFusionConfig,
    JerseyOcrObservation,
    RosterPlayer,
    aggregate_canonical_people,
    aggregate_tracklets,
    normalize_jersey_number,
)
from .jersey_ocr_worker import (
    JerseyCropRequest,
    JerseyOcrWorkerError,
    analyze_jersey_crops,
    jersey_ocr_worker_readiness,
)
from .pitch_calibration import (
    ANCHOR_PRESETS,
    PitchCalibration,
    calibrate_pitch,
    calibration_alignment_metrics,
    calibration_alignment_error,
    calibration_horizon,
    calibration_from_anchors,
    canonicalize_penalty_side,
    flip_pitch_calibration,
    opposite_pitch_preset,
    pitch_side,
    projected_pitch_markings,
    semantic_line_evidence,
)
from .person_detection_cache import (
    PERSON_DETECTION_CACHE_SCHEMA_VERSION,
    PersonDetectionCacheError,
    frame_content_sha256,
    lookup_person_detection_cache,
    store_person_detection_cache,
)
from .quality_metrics import evaluate_reconstruction_quality
from .roster_identity_resolver import (
    AttributeEvidence as RosterAttributeEvidence,
    CanonicalPersonEvidence as RosterCanonicalPersonEvidence,
    PersistedRosterPlayer,
    resolve_closed_set_roster,
)
from .store import reconstruction_input_fingerprint, scene_store
from .temporal_calibration import (
    CameraMotionEstimate,
    TemporalCalibrationFrame,
    solve_calibration_sequence,
)


class ReconstructionError(RuntimeError):
    pass


class IdentityCorrectionError(ReconstructionError):
    """A fail-closed identity correction with machine-readable diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        correction_id: str,
        action: str,
        status: str,
        reason: str,
        source_track_id: str | None = None,
        target_id: str | None = None,
        candidates: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = {
            "correctionId": correction_id,
            "action": action,
            "status": status,
            "reason": reason,
            "message": message,
            "sourceTrackId": source_track_id,
            "targetId": target_id,
            "candidates": candidates or [],
        }


class StaleReconstructionRun(ReconstructionError):
    """The worker no longer owns the scene revision it started from."""

    pass


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
    # `x`/`y` are later moved into the stabilized camera coordinate system.
    # Keep the detector-space foot point and source frame immutable so a
    # published track can retain the exact video observation that created it.
    source_frame_index: int | None = None
    image_x: float | None = None
    image_y: float | None = None
    # Soccer-specific ReID is deliberately separate from the small HSV feature
    # used by the short-horizon online tracker.  The two vector spaces must
    # never be mixed in a similarity graph.
    reid_feature: np.ndarray | None = None
    reid_quality: dict | None = None
    reid_evidence_fingerprint: str | None = None
    reid_role: str | None = None
    reid_role_confidence: float | None = None
    observation_id: str | None = None
    # More than one persisted correction may legitimately refer to the same
    # detector observation (for example an older person confirmation plus a
    # later roster unbind). Keep the complete set even though the legacy
    # singular fields below still expose the highest-precedence annotation.
    annotation_ids: set[str] = field(default_factory=set)
    identity_tombstone_annotation_ids: set[str] = field(default_factory=set)
    roster_binding_annotation_ids: set[str] = field(default_factory=set)
    roster_binding_state: str | None = None
    manual_semantic_key: tuple[int, str, int, str] | None = None
    manual_identity_owner_ids: set[str] = field(default_factory=set)
    # A roster unbind is a durable negative decision.  It still needs an
    # annotation anchor for canonical-id remapping, but it must not turn that
    # identity into a manually confirmed person merely because it is stored as
    # a confirm-shaped legacy frame annotation.
    annotation_is_identity_evidence: bool = True


@dataclass
class TrackState:
    id: int
    points: list[dict] = field(default_factory=list)
    feature_sum: np.ndarray | None = None
    feature_count: int = 0
    last_frame: int = 0
    last_height: float = 0.0
    role: str | None = None
    annotation_ids: set[str] = field(default_factory=set)
    identity_tombstone_ids: set[str] = field(default_factory=set)
    roster_binding_annotation_ids: set[str] = field(default_factory=set)
    roster_binding_state: str | None = None
    manual_identity_owner_ids: set[str] = field(default_factory=set)
    manual_kind: str | None = None
    manual_label: str | None = None
    manual_semantic_key: tuple[int, str, int, str] | None = None
    manual_external_player_id: str | None = None
    source_tracklet_ids: set[str] = field(default_factory=set)
    reid_feature_sum: np.ndarray | None = None
    reid_feature_count: int = 0
    reid_observation_count: int = 0
    reid_observation_ids: set[str] = field(default_factory=set)
    reid_evidence_fingerprints: set[str] = field(default_factory=set)
    reid_duplicate_evidence_count: int = 0
    reid_samples: list[np.ndarray] = field(default_factory=list)
    reid_sample_candidates: list[dict] = field(default_factory=list)
    reid_selected_metadata: list[dict] = field(default_factory=list)
    canonical_person_id: str | None = None
    identity_status: str = "unresolved"
    identity_confidence: float | None = None
    identity_evidence: list[dict] = field(default_factory=list)
    identity_conflicts: list[dict] = field(default_factory=list)
    identity_group_id: str | None = None
    reid_role_votes: dict[str, float] = field(default_factory=dict)
    # Manual split barriers are applied after automatic identity resolution.
    # Both partitions carry the same correction key with different values so a
    # later legacy scene-document merge cannot join them again.
    identity_split_partitions: dict[str, str] = field(default_factory=dict)

    @property
    def feature(self) -> np.ndarray:
        assert self.feature_sum is not None
        return self.feature_sum / max(1, self.feature_count)

    @property
    def local_tracklet_id(self) -> str:
        return f"tracklet-{self.id:04d}"

    @property
    def positive_annotation_ids(self) -> set[str]:
        """Manual annotations that positively assert this identity.

        Tombstones remain in ``annotation_ids`` so a rebuild can preserve the
        canonical id and cannot resurrect an old roster binding.  Keeping the
        two concepts separate prevents that persistence mechanism from being
        reported as positive manual identity evidence.
        """

        return self.annotation_ids - self.identity_tombstone_ids

    @property
    def reid_feature(self) -> np.ndarray | None:
        if self.reid_feature_sum is None or self.reid_feature_count <= 0:
            return None
        value = self.reid_feature_sum / self.reid_feature_count
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 1e-8 else None

    def _select_reid_samples(self) -> None:
        """Keep quality-ranked, genuinely time-separated ReID views."""

        selected: list[dict] = []
        selected_fingerprints: set[str] = set()
        for candidate in sorted(
            self.reid_sample_candidates,
            key=lambda item: (
                -float(item["quality"]),
                float(item["time"]),
                int(item["frameIndex"]),
            ),
        ):
            fingerprint = str(candidate.get("evidenceFingerprint") or "")
            if fingerprint and fingerprint in selected_fingerprints:
                continue
            if any(
                abs(float(candidate["time"]) - float(previous["time"])) < 0.45
                for previous in selected
            ):
                continue
            selected.append(candidate)
            if fingerprint:
                selected_fingerprints.add(fingerprint)
            if len(selected) >= 12:
                break
        selected.sort(key=lambda item: (float(item["time"]), int(item["frameIndex"])))
        self.reid_samples = [item["vector"].copy() for item in selected]
        self.reid_selected_metadata = [
            {
                "time": float(item["time"]),
                "frameIndex": int(item["frameIndex"]),
                "quality": float(item["quality"]),
                "evidenceFingerprint": item.get("evidenceFingerprint"),
            }
            for item in selected
        ]
        self.reid_feature_count = len(self.reid_samples)
        self.reid_feature_sum = (
            np.sum(np.stack(self.reid_samples), axis=0)
            if self.reid_samples
            else None
        )

    def _add_reid_sample(
        self,
        vector: np.ndarray,
        detection: Detection,
        frame_index: int,
        time: float,
    ) -> bool:
        quality = detection.reid_quality or {}
        crop_width = max(0.0, float(quality.get("cropWidth") or detection.width))
        crop_height = max(0.0, float(quality.get("cropHeight") or detection.height))
        crop_area_score = min(1.0, sqrt(crop_width * crop_height) / 120.0)
        sharpness = max(0.0, float(quality.get("sharpness") or 0.0))
        sharpness_score = min(1.0, log1p(sharpness) / log1p(500.0))
        detector_score = max(0.0, min(1.0, float(detection.confidence)))
        quality_score = (
            0.42 * detector_score
            + 0.33 * crop_area_score
            + 0.25 * sharpness_score
            - (0.08 if quality.get("borderClipped") else 0.0)
        )
        # Retain one best candidate per short temporal bin. A larger bounded
        # pool lets later sharp crops replace the first blurry observations.
        temporal_bin = int(float(time) / 0.25)
        candidate = {
            "time": float(time),
            "frameIndex": int(
                detection.source_frame_index
                if detection.source_frame_index is not None
                else frame_index
            ),
            "quality": round(max(0.0, min(1.0, quality_score)), 6),
            "temporalBin": temporal_bin,
            "observationId": str(
                detection.observation_id
                or f"{self.local_tracklet_id}:{int(frame_index)}"
            ),
            "evidenceFingerprint": str(
                detection.reid_evidence_fingerprint
                or "observation:" + str(
                    detection.observation_id
                    or f"{self.local_tracklet_id}:{int(frame_index)}"
                )
            ),
            "vector": vector.copy(),
        }
        self.reid_observation_ids.add(str(candidate["observationId"]))
        evidence_fingerprint = str(candidate["evidenceFingerprint"])
        if evidence_fingerprint in self.reid_evidence_fingerprints:
            self.reid_duplicate_evidence_count += 1
            self.reid_observation_count = len(self.reid_evidence_fingerprints)
            return False
        self.reid_evidence_fingerprints.add(evidence_fingerprint)
        existing_index = next(
            (
                index
                for index, item in enumerate(self.reid_sample_candidates)
                if int(item["temporalBin"]) == temporal_bin
            ),
            None,
        )
        if existing_index is None:
            self.reid_sample_candidates.append(candidate)
        elif (
            candidate["quality"],
            -candidate["frameIndex"],
        ) > (
            self.reid_sample_candidates[existing_index]["quality"],
            -self.reid_sample_candidates[existing_index]["frameIndex"],
        ):
            self.reid_sample_candidates[existing_index] = candidate
        self.reid_sample_candidates = sorted(
            self.reid_sample_candidates,
            key=lambda item: (-float(item["quality"]), int(item["frameIndex"])),
        )[:64]
        self.reid_observation_count = len(self.reid_evidence_fingerprints)
        self._select_reid_samples()
        return True

    def append(self, detection: Detection, frame_index: int, time: float) -> None:
        combined_manual_owners = (
            set(self.manual_identity_owner_ids)
            | set(detection.manual_identity_owner_ids)
        )
        if len(combined_manual_owners) > 1:
            # Check before mutating points/features so a failed reconstruction
            # cannot retain half of a conflicting observation in memory.
            raise ReconstructionError(
                "Conflicting explicit canonical identities reached one raw track"
            )
        self.source_tracklet_ids.add(self.local_tracklet_id)
        image_x = detection.image_x if detection.image_x is not None else detection.x
        image_y = detection.image_y if detection.image_y is not None else detection.y
        source_frame_index = (
            detection.source_frame_index
            if detection.source_frame_index is not None
            else frame_index
        )
        observation_id = detection.observation_id or (
            f"{self.local_tracklet_id}:{source_frame_index}"
        )
        point = {
            "t": time,
            "px": detection.x,
            "py": detection.y,
            "confidence": detection.confidence,
            "frameIndex": source_frame_index,
            "observationId": observation_id,
            "sourceTrackletId": self.local_tracklet_id,
            "bbox": {
                "x": image_x - detection.width / 2,
                "y": image_y - detection.height,
                "width": detection.width,
                "height": detection.height,
            },
            "annotationId": detection.annotation_id,
            **(
                {"annotationIds": sorted(detection.annotation_ids)}
                if detection.annotation_ids
                else {}
            ),
            "_appearanceFeature": detection.feature.copy(),
            **(
                {"annotationIsIdentityEvidence": detection.annotation_is_identity_evidence}
                if detection.annotation_id
                else {}
            ),
        }
        if detection.pitch_x is not None and detection.pitch_z is not None:
            point["pitchX"] = detection.pitch_x
            point["pitchZ"] = detection.pitch_z
            point["projectionSource"] = detection.projection_source or "direct"
            point["calibrationFrameIndex"] = detection.calibration_frame_index
            point["positionUncertaintyMetres"] = detection.position_uncertainty_metres
        if detection.association_cost is not None:
            point["associationCost"] = round(detection.association_cost, 4)
            point["associationMargin"] = (
                round(detection.association_margin, 4)
                if detection.association_margin is not None
                else None
            )
        self.points.append(point)
        if self.feature_sum is None:
            self.feature_sum = detection.feature.copy()
        else:
            self.feature_sum += detection.feature
        self.feature_count += 1
        independent_reid_evidence = False
        if detection.reid_feature is not None:
            vector = np.asarray(detection.reid_feature, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.ndim == 1 and vector.size and np.isfinite(vector).all() and norm > 1e-8:
                vector = vector / norm
                point["_hasReidEvidence"] = True
                point["_reidEvidenceFingerprint"] = str(
                    detection.reid_evidence_fingerprint
                    or "observation:" + str(observation_id)
                )
                independent_reid_evidence = self._add_reid_sample(
                    vector, detection, frame_index, time
                )
        if (
            independent_reid_evidence
            and
            detection.reid_role in {"player", "goalkeeper", "referee", "other"}
            and detection.reid_role_confidence is not None
            and float(detection.reid_role_confidence) >= 0.60
        ):
            role = str(detection.reid_role)
            point["_reidRole"] = role
            point["_reidRoleConfidence"] = float(detection.reid_role_confidence)
            self.reid_role_votes[role] = self.reid_role_votes.get(role, 0.0) + float(
                detection.reid_role_confidence
            )
            if not self.manual_kind:
                self.role = max(
                    self.reid_role_votes,
                    key=lambda value: (self.reid_role_votes[value], value),
                )
        self.last_frame = frame_index
        self.last_height = detection.height
        if detection.annotation_id:
            detection_annotation_ids = set(detection.annotation_ids) | {
                detection.annotation_id
            }
            self.annotation_ids.update(detection_annotation_ids)
            tombstone_ids = set(detection.identity_tombstone_annotation_ids)
            if not detection.annotation_is_identity_evidence:
                tombstone_ids.add(detection.annotation_id)
            self.identity_tombstone_ids.update(tombstone_ids)
            self.identity_tombstone_ids.intersection_update(self.annotation_ids)
            self.manual_identity_owner_ids.update(
                detection.manual_identity_owner_ids
            )
            if (
                detection.manual_semantic_key is not None
                and (
                    self.manual_semantic_key is None
                    or detection.manual_semantic_key >= self.manual_semantic_key
                )
            ):
                self.manual_kind = detection.annotation_kind
                self.manual_label = detection.annotation_label
                self.manual_semantic_key = detection.manual_semantic_key
            # The dedicated unbind correction is authoritative for roster
            # identity even if an older generic confirm is encountered later
            # in frame order. A later intentional rebind replaces the
            # tombstone correction in persisted input, so it does not pass
            # through this branch with a tombstone still present.
            if detection.roster_binding_state in {"bound", "unbound"}:
                incoming_external_id = detection.external_player_id
                if (
                    self.roster_binding_state is not None
                    and (
                        self.roster_binding_state != detection.roster_binding_state
                        or self.manual_external_player_id != incoming_external_id
                    )
                ):
                    raise ReconstructionError(
                        "Conflicting dedicated roster corrections reached one raw track"
                    )
                self.roster_binding_state = detection.roster_binding_state
                self.roster_binding_annotation_ids.update(
                    detection.roster_binding_annotation_ids
                    or {detection.annotation_id}
                )
                self.manual_external_player_id = incoming_external_id
            elif self.roster_binding_state is None:
                incoming_external_id = detection.external_player_id
                if (
                    incoming_external_id
                    and self.manual_external_player_id
                    and incoming_external_id != self.manual_external_player_id
                ):
                    raise ReconstructionError(
                        "Conflicting legacy roster confirmations reached one raw track"
                    )
                if incoming_external_id and not self.identity_tombstone_ids:
                    self.manual_external_player_id = incoming_external_id


_models: dict[str, object] = {}
_model_lock = Lock()
_checkpoint_digest_cache: dict[tuple[str, int, int], str] = {}
_checkpoint_digest_lock = Lock()
METRIC_CALIBRATION_THRESHOLD = 0.75
CALIBRATION_PASS_COVERAGE = 0.90
CALIBRATION_REVIEW_COVERAGE = 0.75
CALIBRATION_PASS_MAX_GAP_SECONDS = 0.60
CALIBRATION_REVIEW_MAX_GAP_SECONDS = 1.00
CALIBRATION_PASS_REPROJECTION_P95 = 8.0
CALIBRATION_REVIEW_REPROJECTION_P95 = 15.0
CALIBRATION_SHOT_REVIEW_REPROJECTION_P95 = 20.0
CALIBRATION_PASS_SIDE_AGREEMENT = 0.90
CALIBRATION_REVIEW_SIDE_AGREEMENT = 0.80
TEMPORAL_PASS_UNCERTAINTY_METRES = 2.50
TEMPORAL_REVIEW_UNCERTAINTY_METRES = 5.00
DENSE_BALL_INTERPOLATION_MAX_GAP_SECONDS = 0.25
PARTIAL_VIEW_REPROJECTION_P95_LIMIT = 20.0
PARTIAL_VIEW_REPROJECTION_P50_LIMIT = 5.0
PARTIAL_VIEW_ALIGNMENT_F1_MINIMUM = 0.15
DETECTOR_IMAGE_SIZE = 1280
DETECTOR_CONFIDENCE = 0.035
DETECTOR_PROVIDER_NMS_IOU = 0.70
DETECTOR_MAX_DETECTIONS = 300
PERSON_LOCAL_NMS_IOU = 0.48
NEW_TRACK_CONFIDENCE = 0.12
MINIMUM_PERSON_FOOT_Y = 0.18
SHALLOW_PERSON_FOOT_Y = 0.34
SHALLOW_PERSON_CONFIDENCE = 0.12
SHALLOW_PERSON_GRASS_RATIO = 0.52
PERSON_FILTER_POLICY_VERSION = "pitch-person-v3"
APPEARANCE_FEATURE_SCHEMA_VERSION = "hsv-histogram-v1"
LEGACY_BALL_FILTER_POLICY_VERSION = "legacy-coco-ball-v2"
RECONSTRUCTION_PHASES = [
    ("preparing", "Prepare inputs"),
    ("calibration", "Calibrate pitch"),
    ("detection", "Detect objects"),
    ("tracking", "Build tracks"),
    ("projection", "Reconstruct 3D"),
    ("finalizing", "Save result"),
]


def _semantic_alignment_passes_review(alignment) -> bool:
    """Allow a small unsupported tail only with strong central line evidence.

    Partial broadcast views often contain a projected marking that is mostly
    occluded or just outside the crop. The ordinary review gate remains
    authoritative; this second branch tolerates only that tail and requires a
    low median residual plus a substantially stronger bidirectional F1 score.
    """

    if alignment is None:
        return False
    ordinary_review = (
        alignment.residual_p95 <= CALIBRATION_REVIEW_REPROJECTION_P95
        and alignment.f1 >= 0.08
    )
    partial_view_review = (
        alignment.residual_p95 <= PARTIAL_VIEW_REPROJECTION_P95_LIMIT
        and alignment.residual_p50 <= PARTIAL_VIEW_REPROJECTION_P50_LIMIT
        and alignment.f1 >= PARTIAL_VIEW_ALIGNMENT_F1_MINIMUM
    )
    return ordinary_review or partial_view_review


def _phase_rows(current_index: int, complete: bool = False) -> list[dict]:
    return [
        {
            "id": phase_id,
            "label": label,
            "status": (
                "completed"
                if complete or index < current_index
                else "current"
                if index == current_index
                else "pending"
            ),
        }
        for index, (phase_id, label) in enumerate(RECONSTRUCTION_PHASES, start=1)
    ]


def _queued_progress(frame_count: int) -> dict:
    return {
        "phase": "preparing",
        "phaseIndex": 1,
        "phaseCount": len(RECONSTRUCTION_PHASES),
        "label": "Waiting to start",
        "detail": f"Queued {frame_count} sampled frames for analysis.",
        "completed": 0,
        "total": frame_count,
        "phasePercent": 0,
        "overallPercent": 0,
        "elapsedSeconds": 0.0,
        "etaSeconds": None,
        "updatedAt": datetime.now(UTC).isoformat(),
        "phases": _phase_rows(1),
    }


class ReconstructionProgress:
    def __init__(
        self,
        scene: dict,
        listener: Callable[[dict], None] | None = None,
        expected_run_id: str | None = None,
        expected_input_fingerprint: str | None = None,
        expected_lease_owner_id: str | None = None,
    ) -> None:
        self.scene = scene
        self.listener = listener
        self.expected_run_id = expected_run_id
        self.expected_input_fingerprint = expected_input_fingerprint
        self.expected_lease_owner_id = expected_lease_owner_id
        self.started = monotonic()
        self.phase_started = self.started
        self.phase = ""

    def update(
        self,
        phase: str,
        phase_index: int,
        label: str,
        detail: str,
        overall_start: float,
        overall_end: float,
        completed: int = 0,
        total: int = 0,
        fraction: float | None = None,
        eta_padding: float = 0.0,
    ) -> dict:
        now = monotonic()
        if phase != self.phase:
            self.phase = phase
            self.phase_started = now
        if fraction is None:
            fraction = completed / total if total > 0 else 0.0
        fraction = max(0.0, min(1.0, float(fraction)))
        phase_elapsed = max(0.0, now - self.phase_started)
        eta = None
        if fraction > 0.0 and fraction < 1.0:
            eta = phase_elapsed * (1.0 - fraction) / fraction + eta_padding
        elif fraction >= 1.0:
            eta = eta_padding
        payload = {
            "phase": phase,
            "phaseIndex": phase_index,
            "phaseCount": len(RECONSTRUCTION_PHASES),
            "label": label,
            "detail": detail,
            "completed": int(completed),
            "total": int(total),
            "phasePercent": round(fraction * 100),
            "overallPercent": round(overall_start + (overall_end - overall_start) * fraction),
            "elapsedSeconds": round(max(0.0, now - self.started), 1),
            "etaSeconds": round(eta, 1) if eta is not None else None,
            "updatedAt": datetime.now(UTC).isoformat(),
            "phases": _phase_rows(phase_index),
        }
        video = self.scene["payload"]["videoAsset"]
        reconstruction = video.get("reconstruction") or {}
        reconstruction["status"] = "processing"
        reconstruction["processingStatus"] = "processing"
        reconstruction["progress"] = payload
        video["reconstruction"] = reconstruction
        _persist_reconstruction_state(
            self.scene,
            self.expected_run_id,
            self.expected_input_fingerprint,
            self.expected_lease_owner_id,
        )
        if self.listener is not None:
            self.listener(deepcopy(payload))
        return payload

    def complete(self, track_count: int, ball_samples: int) -> dict:
        now = monotonic()
        payload = {
            "phase": "complete",
            "phaseIndex": len(RECONSTRUCTION_PHASES),
            "phaseCount": len(RECONSTRUCTION_PHASES),
            "label": "Analysis complete",
            "detail": f"Saved {track_count} tracks and {ball_samples} ball samples.",
            "completed": 1,
            "total": 1,
            "phasePercent": 100,
            "overallPercent": 100,
            "elapsedSeconds": round(max(0.0, now - self.started), 1),
            "etaSeconds": 0.0,
            "updatedAt": datetime.now(UTC).isoformat(),
            "phases": _phase_rows(len(RECONSTRUCTION_PHASES), complete=True),
        }
        if self.listener is not None:
            self.listener(deepcopy(payload))
        return payload

    def failed(self, message: str) -> dict:
        current = (
            self.scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
            .get("progress")
            or _queued_progress(0)
        )
        return {
            **current,
            "phase": "failed",
            "label": "Analysis failed",
            "detail": message,
            "etaSeconds": 0.0,
            "updatedAt": datetime.now(UTC).isoformat(),
        }


def _load_model(model_name: str | None = None):
    name = model_name or get_settings().reconstruction_model
    with _model_lock:
        if name not in _models:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/replay-studio-matplotlib")
            from ultralytics import YOLO

            _models[name] = YOLO(name)
    return _models[name]


BALL_DETECTION_BACKENDS = {
    "generic-ultralytics",
    "dedicated-ultralytics",
    "wasb-service",
}


def _ball_checkpoint_identity(path: str | Path) -> dict:
    checkpoint = Path(path).expanduser().resolve()
    identity: dict = {"name": checkpoint.name}
    if checkpoint.is_file():
        stat = checkpoint.stat()
        identity.update({"size": int(stat.st_size), "mtimeNs": int(stat.st_mtime_ns)})
    return identity


def _verify_queued_ball_checkpoint(path: str | Path, expected: object) -> None:
    if not isinstance(expected, dict):
        return
    actual = _ball_checkpoint_identity(path)
    mismatches = [
        key
        for key in ("name", "size", "mtimeNs")
        if expected.get(key) is not None and expected.get(key) != actual.get(key)
    ]
    if mismatches:
        raise ReconstructionError(
            "Queued ball checkpoint no longer matches the local file "
            f"({', '.join(mismatches)} changed); queue a new reconstruction run."
        )


def _ball_detection_input(backend: str | None = None) -> dict:
    settings = get_settings()
    selected = str(backend or settings.ball_detection_backend)
    if selected not in BALL_DETECTION_BACKENDS:
        raise ReconstructionError(f"Unsupported ball detection backend: {selected}")

    generic_input = {
        "backend": "generic-ultralytics",
        "modelSource": "reconstruction-model",
        "classId": 32,
        "confidence": float(DETECTOR_CONFIDENCE),
        "imageSize": int(DETECTOR_IMAGE_SIZE),
        "nmsIou": float(settings.ball_detection_nms_iou),
    }
    dedicated_input = {
        "backend": "dedicated-ultralytics",
        "checkpoint": _ball_checkpoint_identity(settings.ball_detection_model),
        "classId": 0,
        "confidence": float(settings.ball_detection_confidence),
        "imageSize": int(settings.ball_detection_image_size),
        "tileSize": int(settings.ball_detection_tile_size),
        "tileOverlap": float(settings.ball_detection_tile_overlap),
        "inferenceBatchSize": int(settings.ball_detection_inference_batch_size),
        "nmsIou": float(settings.ball_detection_nms_iou),
    }

    value = {
        "schemaVersion": 1,
        "backend": selected,
        "maxCandidates": int(settings.ball_detection_max_candidates),
        "analysisFrameRate": float(settings.ball_analysis_frame_rate),
        "failurePolicy": str(settings.ball_detection_failure_policy),
    }
    if selected == "generic-ultralytics":
        value.update({key: item for key, item in generic_input.items() if key != "backend"})
    elif selected == "dedicated-ultralytics":
        value.update({key: item for key, item in dedicated_input.items() if key != "backend"})
        if str(settings.ball_detection_failure_policy) == "fallback":
            value["fallback"] = generic_input
    else:
        value.update(
            {
                "workerEndpoint": settings.ball_wasb_worker_url,
                "timeoutSeconds": float(settings.ball_wasb_timeout),
                "temporalWindowFrames": 3,
                "temporalContext": "previous-current-next",
                "fallback": (
                    dedicated_input
                    if str(settings.ball_detection_failure_policy) == "fallback"
                    else None
                ),
            }
        )
    return value


def _configured_ball_detectors(
    person_model: object,
    backend: str,
    detection_input: dict | None = None,
) -> tuple[BallDetector, BallDetector | None]:
    """Build the requested detector and an explicit last-resort fallback."""

    settings = get_settings()
    contract = deepcopy(detection_input) if isinstance(detection_input, dict) else _ball_detection_input(backend)
    if str(contract.get("backend") or "") != backend:
        raise ReconstructionError(
            "Queued ball detector input does not match its requested backend"
        )
    policy = str(contract.get("failurePolicy") or settings.ball_detection_failure_policy)
    if policy not in {"raise", "fallback"}:
        raise ReconstructionError(
            "BALL_DETECTION_FAILURE_POLICY must be raise or fallback"
        )
    max_candidates = int(
        contract.get("maxCandidates", settings.ball_detection_max_candidates)
    )
    fallback_contract = contract.get("fallback")
    if not isinstance(fallback_contract, dict):
        fallback_contract = {}
    generic_contract = (
        contract
        if backend == "generic-ultralytics"
        else fallback_contract
        if fallback_contract.get("backend") == "generic-ultralytics"
        else {}
    )
    generic = build_ball_detector(
        BallDetectorConfig(
            backend="generic-ultralytics",
            device=settings.reconstruction_device,
            confidence=float(generic_contract.get("confidence", DETECTOR_CONFIDENCE)),
            image_size=int(generic_contract.get("imageSize", DETECTOR_IMAGE_SIZE)),
            max_candidates=max_candidates,
            nms_iou=float(
                generic_contract.get("nmsIou", settings.ball_detection_nms_iou)
            ),
        ),
        model=person_model,
    )
    if backend == "generic-ultralytics":
        return generic, None

    dedicated: BallDetector | None = None
    if backend == "dedicated-ultralytics" or policy == "fallback":
        dedicated_contract = (
            contract if backend == "dedicated-ultralytics" else fallback_contract
        )
        _verify_queued_ball_checkpoint(
            settings.ball_detection_model,
            dedicated_contract.get("checkpoint"),
        )
        dedicated = build_ball_detector(
            BallDetectorConfig(
                backend="dedicated-ultralytics",
                checkpoint_path=settings.ball_detection_model,
                device=settings.reconstruction_device,
                confidence=float(
                    dedicated_contract.get(
                        "confidence", settings.ball_detection_confidence
                    )
                ),
                image_size=int(
                    dedicated_contract.get(
                        "imageSize", settings.ball_detection_image_size
                    )
                ),
                max_candidates=max_candidates,
                tile_size=(
                    int(
                        dedicated_contract.get(
                            "tileSize", settings.ball_detection_tile_size
                        )
                    ),
                    int(
                        dedicated_contract.get(
                            "tileSize", settings.ball_detection_tile_size
                        )
                    ),
                ),
                tile_overlap=float(
                    dedicated_contract.get(
                        "tileOverlap", settings.ball_detection_tile_overlap
                    )
                ),
                inference_batch_size=int(
                    dedicated_contract.get(
                        "inferenceBatchSize",
                        settings.ball_detection_inference_batch_size,
                    )
                ),
                nms_iou=float(
                    dedicated_contract.get(
                        "nmsIou", settings.ball_detection_nms_iou
                    )
                ),
            ),
            model=_load_model(settings.ball_detection_model),
        )
    if backend == "dedicated-ultralytics":
        assert dedicated is not None
        return dedicated, generic if policy == "fallback" else None
    if backend == "wasb-service":
        # Keep the service adapter strict.  Reconstruction owns the fallback
        # and circuit breaker so a worker outage is recorded once instead of
        # causing one long timeout per dense frame.
        detector = build_ball_detector(
            BallDetectorConfig(
                backend="wasb-service",
                device=settings.reconstruction_device,
                max_candidates=max_candidates,
                nms_iou=float(contract.get("nmsIou", settings.ball_detection_nms_iou)),
                wasb_service_url=contract.get("workerEndpoint")
                or settings.ball_wasb_worker_url,
                wasb_timeout=float(
                    contract.get("timeoutSeconds", settings.ball_wasb_timeout)
                ),
                failure_policy="raise",
            ),
        )
        return detector, dedicated if policy == "fallback" else None
    raise ReconstructionError(f"Unsupported ball detection backend: {backend}")


def _persist_reconstruction_state(
    scene: dict,
    expected_run_id: str | None = None,
    expected_input_fingerprint: str | None = None,
    expected_lease_owner_id: str | None = None,
) -> None:
    if expected_run_id is None:
        scene_store.put(scene)
        return
    if not expected_input_fingerprint:
        raise ReconstructionError("A guarded reconstruction run is missing its input fingerprint")
    arguments = [scene, expected_run_id, expected_input_fingerprint]
    # Preserve direct/internal test callers that intentionally exercise the
    # pre-lease CAS contract. Production processing always supplies owner.
    if expected_lease_owner_id is not None:
        arguments.append(expected_lease_owner_id)
    if not scene_store.put_if_reconstruction_run(*arguments):
        raise StaleReconstructionRun(
            f"Reconstruction run {expected_run_id} was superseded by a newer scene revision or inputs"
        )


def set_reconstruction_status(
    scene: dict,
    status: str,
    *,
    expected_run_id: str | None = None,
    expected_input_fingerprint: str | None = None,
    expected_lease_owner_id: str | None = None,
    **values,
) -> dict:
    video = scene.get("payload", {}).get("videoAsset")
    if video is None:
        raise ReconstructionError("Scene has no source video")
    current = video.get("reconstruction") or {}
    model_name = values.pop("model", None) or current.get("model") or get_settings().reconstruction_model
    video["reconstruction"] = {
        **current,
        "status": status,
        "model": model_name,
        **values,
    }
    _persist_reconstruction_state(
        scene,
        expected_run_id,
        expected_input_fingerprint,
        expected_lease_owner_id,
    )
    return scene


def queue_reconstruction(
    scene: dict,
    model_name: str | None = None,
    *,
    ball_backend: str | None = None,
    expected_scene_fingerprint: str | None = None,
    persist: bool = True,
) -> dict:
    video = scene.get("payload", {}).get("videoAsset")
    if video is None:
        raise ReconstructionError("Scene has no source video")
    if video.get("multiPass"):
        raise ReconstructionError(
            "Multi-pass composites must be rebuilt with multi-angle analysis, "
            "not single-pass reconstruction"
        )
    previous = video.get("reconstruction") or {}
    selected_model = model_name or previous.get("model") or get_settings().reconstruction_model
    selected_ball_backend = str(
        ball_backend
        or previous.get("ballBackend")
        or get_settings().ball_detection_backend
    )
    ball_detection_input = _ball_detection_input(selected_ball_backend)
    base_run_id = str(previous.get("runId") or "")
    base_input_fingerprint = (
        expected_scene_fingerprint or reconstruction_input_fingerprint(scene)
    )
    guard_existing_scene = scene_store.get(str(scene.get("id") or "")) is not None
    run_id = uuid4().hex
    run_revision = int(previous.get("runRevision") or 0) + 1
    previous_result = {
        "completedAt": previous.get("completedAt"),
        "trackCount": len(scene.get("payload", {}).get("tracks") or []),
        "ballSamples": len(scene.get("payload", {}).get("ball", {}).get("keyframes") or []),
        "calibrationStatus": (previous.get("pitchCalibration") or {}).get("status"),
    }
    input_frames = _frame_paths(scene)
    video["processingState"] = "reconstructing"
    # The model is a reconstruction input, so include the selected value in
    # the immutable fingerprint recorded for this run.
    previous_diagnostics = {
        key: value
        for key, value in (previous.get("diagnostics") or {}).items()
        if key != "identityCorrections"
    }
    video["reconstruction"] = {
        **previous,
        "model": selected_model,
        "ballBackend": selected_ball_backend,
        "ballDetectionInput": ball_detection_input,
        "identityCorrectionDiagnostics": [],
        "diagnostics": previous_diagnostics,
    }
    input_fingerprint = reconstruction_input_fingerprint(scene)
    status_values = {
        "processingStatus": "queued",
        "qualityVerdict": "pending",
        "quality": None,
        "model": selected_model,
        "ballBackend": selected_ball_backend,
        "ballDetectionInput": ball_detection_input,
        "runId": run_id,
        "runRevision": run_revision,
        "inputFingerprint": input_fingerprint,
        "error": None,
        "startedAt": None,
        "completedAt": None,
        "frameCount": len(input_frames),
        "trackCount": previous_result["trackCount"],
        "ballSamples": previous_result["ballSamples"],
        "warnings": [],
        "previousResult": previous_result,
        "progress": _queued_progress(len(input_frames)),
    }
    if not persist:
        # Project-level match updates prepare every affected run first, then
        # commit all scene documents in one store transaction. Persisting here
        # would expose a partially updated project between sibling writes.
        video["reconstruction"] = {
            **(video.get("reconstruction") or {}),
            "status": "queued",
            **status_values,
        }
        return scene
    return set_reconstruction_status(
        scene,
        "queued",
        expected_run_id=base_run_id if guard_existing_scene else None,
        expected_input_fingerprint=(
            base_input_fingerprint if guard_existing_scene else None
        ),
        **status_values,
    )


def _frame_paths(scene: dict) -> list[tuple[Path, float]]:
    video = scene["payload"]["videoAsset"]
    analysis_fps = float(video.get("analysisFps") or 10.0)
    source_start = float(video.get("sourceStart") or 0.0)
    source_end = float(video.get("sourceEnd") or source_start + scene["duration"])
    sample_fps = min(analysis_fps, get_settings().reconstruction_frame_rate)
    step = max(1, round(analysis_fps / sample_fps))
    first = max(1, int(source_start * analysis_fps) + 1)
    last = max(first, int(source_end * analysis_fps) + 1)
    frames = Path(get_settings().media_root).resolve() / video["id"] / "frames"
    return [
        (frames / f"frame_{index:05d}.jpg", max(0.0, (index - 1) / analysis_fps - source_start))
        for index in range(first, last + 1, step)
        if (frames / f"frame_{index:05d}.jpg").exists()
    ]


def _frame_context(
    scene: dict,
    scene_time: float,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    frames = _frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    target_index = min(range(len(frames)), key=lambda index: abs(frames[index][1] - scene_time))
    previous_image: np.ndarray | None = None
    target_image: np.ndarray | None = None
    camera_transform = np.eye(3, dtype=np.float64)
    for index, (path, _) in enumerate(frames[: target_index + 1]):
        image = cv2.imread(str(path))
        if image is None:
            continue
        if previous_image is not None:
            motion = _camera_motion_estimate(previous_image, image)
            camera_transform = (
                camera_transform @ motion.matrix
                if motion.reliable
                else np.eye(3, dtype=np.float64)
            )
        previous_image = image
        if index == target_index:
            target_image = image
    if target_image is None:
        raise ReconstructionError("The sampled frame could not be read")
    return target_index, float(frames[target_index][1]), target_image, camera_transform


def _seed_pitch_anchors(preset: str, width: int, height: int) -> list[dict]:
    layouts = {
        "center-circle": {
            "circle-left": (0.34, 0.57),
            "circle-top": (0.50, 0.40),
            "circle-right": (0.66, 0.57),
            "circle-bottom": (0.50, 0.75),
        },
        "penalty-area-right": {
            "front-far": (0.24, 0.36),
            "front-near": (0.16, 0.76),
            "goal-far": (0.74, 0.35),
            "goal-near": (0.90, 0.78),
        },
        "goal-area-right": {
            "front-far": (0.38, 0.43),
            "front-near": (0.31, 0.69),
            "goal-far": (0.74, 0.42),
            "goal-near": (0.86, 0.71),
        },
        "penalty-area-left": {
            "goal-far": (0.10, 0.35),
            "goal-near": (0.26, 0.78),
            "front-far": (0.76, 0.36),
            "front-near": (0.84, 0.76),
        },
        "goal-area-left": {
            "goal-far": (0.14, 0.42),
            "goal-near": (0.26, 0.71),
            "front-far": (0.62, 0.43),
            "front-near": (0.69, 0.69),
        },
    }
    layout = layouts[preset]
    return [
        {
            "id": anchor_id,
            "label": label,
            "image": {
                "x": round(layout[anchor_id][0] * width, 2),
                "y": round(layout[anchor_id][1] * height, 2),
            },
            "pitch": {"x": pitch[0], "z": pitch[1]},
            "source": "seed",
        }
        for anchor_id, label, pitch in ANCHOR_PRESETS[preset]
    ]


def _project_preset_anchors(
    calibration: PitchCalibration,
    preset: str,
    width: int,
    height: int,
) -> list[dict]:
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return _seed_pitch_anchors(preset, width, height)
    anchors = []
    for anchor_id, label, pitch in ANCHOR_PRESETS[preset]:
        projected = pitch_to_image @ np.array([pitch[0], pitch[1], 1.0], dtype=np.float64)
        if abs(float(projected[2])) < 1e-8:
            return _seed_pitch_anchors(preset, width, height)
        image_x = float(projected[0] / projected[2])
        image_y = float(projected[1] / projected[2])
        if not np.isfinite([image_x, image_y]).all():
            return _seed_pitch_anchors(preset, width, height)
        anchors.append(
            {
                "id": anchor_id,
                "label": label,
                "image": {"x": round(image_x, 2), "y": round(image_y, 2)},
                "pitch": {"x": pitch[0], "z": pitch[1]},
            }
        )
    inside = sum(
        -width * 0.08 <= anchor["image"]["x"] <= width * 1.08
        and -height * 0.08 <= anchor["image"]["y"] <= height * 1.08
        for anchor in anchors
    )
    if inside < 3:
        return _seed_pitch_anchors(preset, width, height)
    for anchor in anchors:
        anchor["source"] = "projected"
    return anchors


def _calibration_draft(
    scene: dict,
    frame_index: int,
    frame_time: float,
    image: np.ndarray,
    calibration: PitchCalibration,
    preset: str,
    source: str,
    anchors: list[dict] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    height, width = image.shape[:2]
    alignment_metrics = calibration_alignment_metrics(image, calibration)
    alignment_error = (
        round(alignment_metrics.residual_p50, 2)
        if alignment_metrics is not None
        else None
    )
    if anchors is None:
        anchors = _project_preset_anchors(calibration, preset, width, height)
    quality = (
        "good"
        if alignment_metrics is not None
        and alignment_metrics.residual_p95 <= CALIBRATION_PASS_REPROJECTION_P95
        and alignment_metrics.f1 >= 0.15
        else "review"
        if _semantic_alignment_passes_review(alignment_metrics)
        else "poor"
    )
    draft_warnings = list(warnings or [])
    if any(anchor.get("source") == "seed" for anchor in anchors):
        draft_warnings.append(
            "Anchor projection was outside the frame; the shown anchors are an unverified manual seed."
        )
    if alignment_error is None:
        draft_warnings.append("Not enough visible white markings to score the overlay.")
    elif alignment_error > 9.0:
        draft_warnings.append("Pitch overlay is still far from the detected markings; move the anchors.")
    return {
        "sceneId": scene["id"],
        "sceneTime": round(frame_time, 3),
        "frameIndex": frame_index + 1,
        "frameWidth": width,
        "frameHeight": height,
        "source": source,
        "preset": preset,
        "confidence": round(calibration.confidence, 3),
        "alignmentError": alignment_error,
        "alignmentMetrics": alignment_metrics.as_dict() if alignment_metrics is not None else None,
        "horizon": calibration_horizon(calibration, width),
        "quality": quality,
        "anchors": anchors,
        "markings": projected_pitch_markings(calibration, width, height),
        "imageToPitch": [
            [round(float(value), 10) for value in row]
            for row in calibration.image_to_pitch
        ],
        "warnings": draft_warnings,
    }


def propose_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    requested_preset: str | None = None,
) -> dict:
    frame_index, frame_time, image, _ = _frame_context(scene, scene_time)
    frames = _frame_paths(scene)
    path = frames[frame_index][0]
    source_frame_index = _source_frame_index(path)
    calibration: PitchCalibration | None = None
    selected_evidence: dict | None = None
    attempts: list[dict] = []
    warnings: list[str] = []

    settings = get_settings()
    automatic, automatic_warnings = _automatic_frame_calibrations(
        [(path, frame_time)],
        worker_timeout=settings.calibration_frame_worker_timeout,
    )
    warnings.extend(automatic_warnings)
    keypoint_candidate = automatic.get(source_frame_index)
    if keypoint_candidate is not None:
        keypoint_candidate = canonicalize_penalty_side(keypoint_candidate, image.shape[1])
        keypoint_evidence = _frame_calibration_evidence(
            scene,
            frame_index,
            frame_time,
            image,
            keypoint_candidate,
            projection_source="direct",
        )
        attempts.append(_calibration_attempt_payload(keypoint_evidence))
        calibration = keypoint_candidate
        selected_evidence = keypoint_evidence

    # A line/curve fit is deliberately a fallback, not a replacement for a
    # healthy semantic-keypoint result. It is expensive, but this operation is
    # scoped to one explicitly requested frame and therefore remains useful for
    # diagnosing model misses without rebuilding the full shot.
    if selected_evidence is None or selected_evidence.get("status") != "accepted":
        height, width = image.shape[:2]
        scale = min(1.0, 640.0 / max(1, width))
        fallback_image = (
            image
            if scale == 1.0
            else cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
        fallback_diagnostics: dict = {
            "inputWidth": fallback_image.shape[1],
            "inputHeight": fallback_image.shape[0],
        }
        fallback_started = monotonic()
        line_candidate = calibrate_pitch(
            fallback_image,
            max_quad_candidates=240,
            deadline=fallback_started + 5.0,
            diagnostics=fallback_diagnostics,
        )
        fallback_diagnostics.update(
            {
                "inputWidth": fallback_image.shape[1],
                "inputHeight": fallback_image.shape[0],
                "elapsedSeconds": round(monotonic() - fallback_started, 3),
            }
        )
        if fallback_diagnostics.get("deadlineExceeded"):
            warnings.append(
                "The bounded line/curve fallback reached its five-second deadline; its best-so-far result was retained when available."
            )
        if fallback_diagnostics.get("candidateLimitReached"):
            warnings.append(
                "The bounded line/curve fallback reached its candidate search limit before the deadline; its best-so-far result was retained when available."
            )
        if (
            fallback_diagnostics.get("budgetExhausted")
            and not fallback_diagnostics.get("deadlineExceeded")
            and not fallback_diagnostics.get("candidateLimitReached")
        ):
            # Compatibility with an older/external fallback implementation that
            # exposes only the aggregate flag.
            warnings.append(
                "The bounded line/curve fallback reached its configured search budget; its best-so-far result was retained when available."
            )
        if line_candidate is not None:
            if scale != 1.0:
                full_to_small = np.asarray(
                    [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                lifted = line_candidate.image_to_pitch @ full_to_small
                lifted /= lifted[2, 2]
                line_candidate = replace(line_candidate, image_to_pitch=lifted)
            line_candidate = canonicalize_penalty_side(line_candidate, image.shape[1])
            line_evidence = _frame_calibration_evidence(
                scene,
                frame_index,
                frame_time,
                image,
                line_candidate,
                projection_source="direct",
            )
            line_evidence["backendDiagnostics"] = fallback_diagnostics
            attempts.append(_calibration_attempt_payload(line_evidence))
            if selected_evidence is None or _calibration_evidence_rank(line_evidence) > _calibration_evidence_rank(
                selected_evidence
            ):
                calibration = line_candidate
                selected_evidence = line_evidence
        elif keypoint_candidate is None:
            warnings.append("Neither semantic keypoints nor the line/curve fallback found a camera fit.")

    preset = requested_preset
    if preset is None:
        calibrated_side = pitch_side(calibration.rectangle) if calibration is not None else None
        preset = (
            calibration.rectangle
            if calibration is not None and calibration.rectangle in ANCHOR_PRESETS
            else f"penalty-area-{calibrated_side}"
            if calibrated_side in {"left", "right"}
            else "center-circle"
        )
    if preset not in ANCHOR_PRESETS:
        raise ReconstructionError("Unsupported pitch anchor preset")
    if calibration is None:
        anchors = _seed_pitch_anchors(preset, image.shape[1], image.shape[0])
        seed = calibration_from_anchors(anchors, preset, confidence=0.35)
        warnings.append("Automatic pitch fit failed; align the four anchors manually.")
        selected_evidence = _frame_calibration_evidence(
            scene,
            frame_index,
            frame_time,
            image,
            None,
            projection_source="none",
        )
        draft = _calibration_draft(
            scene,
            frame_index,
            frame_time,
            image,
            seed,
            preset,
            "manual-seed",
            anchors,
            warnings,
        )
    else:
        assert selected_evidence is not None
        if selected_evidence.get("status") != "accepted":
            warnings.append(
                "The best current-frame candidate failed geometric QA; inspect the reasons or refine its anchors manually."
            )
        draft = _calibration_draft(
            scene,
            frame_index,
            frame_time,
            image,
            calibration,
            preset,
            "frame-evidence",
            warnings=warnings,
        )

    assert selected_evidence is not None
    selected_evidence["attempts"] = attempts
    draft.update(
        {
            "requestedSceneTime": round(float(scene_time), 3),
            "sampleIndex": frame_index,
            "sourceFrameIndex": source_frame_index,
            "sourceTime": selected_evidence.get("sourceTime"),
            "status": selected_evidence["status"],
            "solutionStatus": selected_evidence["solutionStatus"],
            "method": selected_evidence.get("backend"),
            "backend": selected_evidence.get("backend"),
            "confidenceKind": selected_evidence.get("confidenceKind"),
            "keypointCount": selected_evidence.get("keypointCount", 0),
            "detectedKeypointCount": selected_evidence.get("detectedKeypointCount", 0),
            "inlierCount": selected_evidence.get("inlierCount", 0),
            "inlierRatio": selected_evidence.get("inlierRatio"),
            "reprojectionP95": selected_evidence.get("reprojectionP95"),
            "visiblePitchSide": selected_evidence.get("visiblePitchSide"),
            "rejectionReasons": selected_evidence.get("rejectionReasons") or [],
            "qualityGates": selected_evidence.get("qualityGates") or [],
            "keypoints": selected_evidence.get("keypoints") or [],
            "detectedKeypoints": selected_evidence.get("keypoints") or [],
            "rawLines": selected_evidence.get("rawLines") or [],
            "attempts": attempts,
            "evidence": selected_evidence,
        }
    )
    _persist_frame_calibration_preview(scene, selected_evidence)
    return draft


def preview_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    preset: str,
    anchors: list[dict],
) -> dict:
    frame_index, frame_time, image, _ = _frame_context(scene, scene_time)
    resolved_anchors = deepcopy(anchors)
    rough = calibration_from_anchors(resolved_anchors, preset, confidence=0.9)
    canonical = canonicalize_penalty_side(rough, image.shape[1])
    resolved_preset = preset
    if canonical.rectangle != rough.rectangle and canonical.rectangle in ANCHOR_PRESETS:
        resolved_preset = canonical.rectangle
        for anchor in resolved_anchors:
            anchor["pitch"]["x"] = -float(anchor["pitch"]["x"])
        rough = canonical
    alignment_error = calibration_alignment_error(image, rough)
    # Manual anchors are not automatically trustworthy. Confidence must fall with
    # measured alignment instead of being forced above the metric threshold.
    confidence = (
        0.35
        if alignment_error is None
        else 0.55 + 0.43 * exp(-float(alignment_error) / 6.0)
    )
    calibration = calibration_from_anchors(resolved_anchors, resolved_preset, confidence=confidence)
    return _calibration_draft(
        scene,
        frame_index,
        frame_time,
        image,
        calibration,
        resolved_preset,
        "manual",
        resolved_anchors,
    )


def _manual_override_key(override: dict) -> tuple[str, int | float]:
    if override.get("sourceFrameIndex") is not None:
        return "source-frame", int(override["sourceFrameIndex"])
    if override.get("sampleIndex") is not None:
        return "sample", int(override["sampleIndex"])
    return "scene-time", round(float(override.get("sceneTime") or 0.0), 3)


def _manual_pitch_calibration_overrides(reconstruction: dict) -> list[dict]:
    """Read the multi-anchor contract and migrate the legacy single value in memory."""

    result: list[dict] = []
    seen: set[tuple[str, int | float]] = set()
    collection = reconstruction.get("pitchCalibrationOverrides")
    if isinstance(collection, list):
        for item in collection:
            if not isinstance(item, dict) or not item.get("imageToPitch"):
                continue
            key = _manual_override_key(item)
            if key in seen:
                continue
            result.append(deepcopy(item))
            seen.add(key)
    legacy = reconstruction.get("pitchCalibrationOverride")
    if isinstance(legacy, dict) and legacy.get("imageToPitch"):
        key = _manual_override_key(legacy)
        if key not in seen:
            result.append(deepcopy(legacy))
    result.sort(
        key=lambda item: (
            float(item.get("sceneTime") or 0.0),
            int(item.get("sourceFrameIndex") or 0),
        )
    )
    return result


def apply_scene_pitch_calibration(
    scene: dict,
    scene_time: float,
    preset: str,
    anchors: list[dict],
) -> dict:
    draft = preview_scene_pitch_calibration(scene, scene_time, preset, anchors)
    if draft.get("quality") == "poor" or draft.get("alignmentMetrics") is None:
        raise ReconstructionError(
            "Manual calibration does not align with enough pitch markings; refine the anchors before applying."
        )
    _, _, _, camera_transform = _frame_context(scene, draft["sceneTime"])
    current_to_pitch = np.asarray(draft["imageToPitch"], dtype=np.float64)
    try:
        stabilized_to_pitch = current_to_pitch @ np.linalg.inv(camera_transform)
    except np.linalg.LinAlgError as exc:
        raise ReconstructionError("Camera motion transform could not be inverted") from exc
    stabilized_to_pitch /= stabilized_to_pitch[2, 2]
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    resolved_anchors = draft.get("anchors") or anchors
    resolved_preset = draft.get("preset") or preset
    sampled_frames = _frame_paths(scene)
    sampled_index = int(draft["frameIndex"]) - 1
    source_frame_index = (
        _source_frame_index(sampled_frames[sampled_index][0])
        if sampled_frames and 0 <= sampled_index < len(sampled_frames)
        else None
    )
    override = {
        "id": (
            f"manual-frame-{source_frame_index}"
            if source_frame_index is not None
            else f"manual-time-{float(draft['sceneTime']):.3f}"
        ),
        "status": "ready" if draft.get("quality") == "good" else "review",
        "validationStatus": draft.get("quality") or "poor",
        "method": "manual-pitch-anchors",
        "confidence": draft["confidence"],
        "supportedLines": len(resolved_anchors),
        "matchedCurves": 1 if resolved_preset == "center-circle" else 0,
        "meanLineScore": 0.0,
        "preset": resolved_preset,
        "pitchSide": pitch_side(resolved_preset),
        "sceneTime": draft["sceneTime"],
        "frameIndex": draft["frameIndex"],
        "sampleIndex": sampled_index,
        "alignmentError": draft["alignmentError"],
        "alignmentMetrics": draft.get("alignmentMetrics"),
        "horizon": draft.get("horizon"),
        "sourceFrameIndex": source_frame_index,
        "anchors": resolved_anchors,
        "coordinateSpace": "stabilized-reference-image",
        "imageToPitch": [
            [round(float(value), 10) for value in row]
            for row in stabilized_to_pitch
        ],
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    overrides = [
        item
        for item in _manual_pitch_calibration_overrides(reconstruction)
        if _manual_override_key(item) != _manual_override_key(override)
    ]
    overrides.append(override)
    overrides.sort(
        key=lambda item: (
            float(item.get("sceneTime") or 0.0),
            int(item.get("sourceFrameIndex") or 0),
        )
    )
    reconstruction["pitchCalibrationOverrides"] = overrides
    # Compatibility alias for old clients. The collection above is authoritative.
    reconstruction["pitchCalibrationOverride"] = override
    resolved_side = pitch_side(resolved_preset)
    if resolved_side:
        current_orientation = reconstruction.get("pitchOrientation") or {}
        reconstruction["pitchOrientation"] = {
            **current_orientation,
            "visiblePitchSide": resolved_side,
            "visiblePitchSideSource": "manual-calibration",
            "attackingGoal": current_orientation.get("attackingGoal") or "unknown",
            "attackingGoalSource": current_orientation.get("attackingGoalSource") or "unknown",
            "updatedAt": datetime.now(UTC).isoformat(),
        }
    video["reconstruction"] = reconstruction
    scene_store.put(scene)
    return queue_reconstruction(scene)


def _flip_pitch_metadata(metadata: dict, source: str) -> dict:
    result = deepcopy(metadata)
    matrix = result.get("imageToPitch")
    if matrix:
        flipped = np.asarray(matrix, dtype=np.float64)
        flipped[0, :] *= -1.0
        result["imageToPitch"] = [
            [round(float(value), 10) for value in row]
            for row in flipped
        ]
    for key in ("rectangle", "preset"):
        if result.get(key):
            result[key] = opposite_pitch_preset(str(result[key]))
    for anchor in result.get("anchors") or []:
        pitch = anchor.get("pitch") or {}
        if "x" in pitch:
            pitch["x"] = -float(pitch["x"])
    resolved_side = pitch_side(result.get("preset") or result.get("rectangle"))
    if resolved_side:
        result["pitchSide"] = resolved_side
    result["orientationSource"] = source
    return result


def _current_scene_pitch_side(reconstruction: dict) -> str | None:
    orientation = reconstruction.get("pitchOrientation") or {}
    if orientation.get("visiblePitchSide") in {"left", "right"}:
        return str(orientation["visiblePitchSide"])
    calibration = reconstruction.get("pitchCalibration") or {}
    if calibration.get("pitchSide") in {"left", "right"}:
        return str(calibration["pitchSide"])
    return pitch_side(calibration.get("preset") or calibration.get("rectangle"))


def set_scene_pitch_side(scene: dict, target_side: str) -> dict:
    if target_side not in {"left", "right"}:
        raise ReconstructionError("Pitch side must be left or right")
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError("Wait for reconstruction to finish before changing pitch side")
    current_orientation = reconstruction.get("pitchOrientation") or {}
    reconstruction["pitchOrientation"] = {
        **current_orientation,
        "attackingGoal": target_side,
        "attackingGoalSource": "manual",
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    video["reconstruction"] = reconstruction
    scene_store.put(scene)
    return scene


def _ball_keyframe_documents(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [deepcopy(item) for item in value if isinstance(item, dict)]


def _manual_ball_diagnostics(keyframes: list[dict]) -> dict:
    return {
        "trajectoryMode": "manual",
        "source": "manual-keypoints",
        "confidenceKind": "manual-authoritative",
        "manualKeyframeCount": len(keyframes),
        "interpolationSegmentCount": max(0, len(keyframes) - 1),
        "interpolation": {
            "method": "linear-between-keypoints",
            "scope": "between-first-and-last-keypoint",
        },
        # Coverage values from the detector are intentionally not copied onto
        # a user-authored path.  They describe a different evidence source.
        "observedCoverage": None,
        "publishedCoverage": None,
        "worldProjectionStatus": "published" if keyframes else "no-manual-keypoints",
        "provenance": {
            "source": "manual",
            "method": "user-pitch-keypoint",
        },
    }


def _normalize_ball_payload(value: object) -> dict:
    """Upgrade legacy ``ball.keyframes`` documents without losing data.

    ``keyframes`` remains the active playback contract.  The two explicit
    collections retain the latest detector output and the user-authored path
    so changing modes is reversible.
    """

    ball = deepcopy(value) if isinstance(value, dict) else {}
    explicit_mode = ball.get("mode")
    mode = explicit_mode if explicit_mode in {"automatic", "manual"} else "automatic"
    legacy_keyframes = _ball_keyframe_documents(ball.get("keyframes"))

    if isinstance(ball.get("automaticKeyframes"), list):
        automatic_keyframes = _ball_keyframe_documents(ball["automaticKeyframes"])
    elif mode == "automatic":
        automatic_keyframes = legacy_keyframes
    else:
        automatic_keyframes = []

    if isinstance(ball.get("manualKeyframes"), list):
        manual_keyframes = _ball_keyframe_documents(ball["manualKeyframes"])
    elif explicit_mode == "manual":
        # Compatibility for an early/manual document that only exposed active
        # keyframes and did not yet have the split collections.
        manual_keyframes = legacy_keyframes
    else:
        manual_keyframes = []

    existing_diagnostics = (
        deepcopy(ball.get("diagnostics"))
        if isinstance(ball.get("diagnostics"), dict)
        else {}
    )
    automatic_diagnostics = (
        deepcopy(ball.get("automaticDiagnostics"))
        if isinstance(ball.get("automaticDiagnostics"), dict)
        else existing_diagnostics
        if mode == "automatic"
        else {}
    )
    automatic_diagnostics = {
        **automatic_diagnostics,
        "trajectoryMode": "automatic",
        "source": automatic_diagnostics.get("source") or "automatic-ball-resolver",
    }
    manual_diagnostics = _manual_ball_diagnostics(manual_keyframes)

    ball.update(
        {
            "mode": mode,
            "automaticKeyframes": automatic_keyframes,
            "manualKeyframes": manual_keyframes,
            "automaticDiagnostics": automatic_diagnostics,
            "manualDiagnostics": manual_diagnostics,
            "keyframes": (
                manual_keyframes if mode == "manual" else automatic_keyframes
            ),
            "diagnostics": (
                manual_diagnostics if mode == "manual" else automatic_diagnostics
            ),
        }
    )
    return ball


def _manual_ball_keyframes(scene: dict, values: list[dict]) -> list[dict]:
    duration = float(scene.get("duration") or 0.0)
    pitch = scene.get("payload", {}).get("pitch") or {}
    try:
        pitch_length = float(pitch["length"])
        pitch_width = float(pitch["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError("Scene pitch dimensions are unavailable") from exc
    if not all(
        isfinite(item) and item > 0.0
        for item in (duration, pitch_length, pitch_width)
    ):
        raise ReconstructionError("Scene duration and pitch dimensions must be finite and positive")

    def number(item: dict, key: str, *, required: bool = True) -> float | None:
        raw = item.get(key)
        if raw is None and not required:
            return None
        if isinstance(raw, bool):
            raise ReconstructionError(f"Manual ball keyframe {key} must be a finite number")
        try:
            result = float(raw)
        except (TypeError, ValueError) as exc:
            raise ReconstructionError(
                f"Manual ball keyframe {key} must be a finite number"
            ) from exc
        if not isfinite(result):
            raise ReconstructionError(f"Manual ball keyframe {key} must be a finite number")
        return result

    by_time: dict[float, dict] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ReconstructionError("Manual ball keyframes must be objects")
        time = number(item, "t")
        x = number(item, "x")
        z = number(item, "z")
        y = number(item, "y", required=False)
        assert time is not None and x is not None and z is not None
        if time < 0.0 or time > duration:
            raise ReconstructionError(
                f"Manual ball keyframe time must be between 0 and {duration:g} seconds"
            )
        if abs(x) > pitch_length / 2.0:
            raise ReconstructionError(
                f"Manual ball x must be within the pitch bounds [-{pitch_length / 2:g}, {pitch_length / 2:g}]"
            )
        if abs(z) > pitch_width / 2.0:
            raise ReconstructionError(
                f"Manual ball z must be within the pitch bounds [-{pitch_width / 2:g}, {pitch_width / 2:g}]"
            )

        normalized_time = round(time, 3)
        resolved_y = 0.22 if y is None else round(y, 3)
        by_time[normalized_time] = {
            "id": f"manual-ball-{int(round(normalized_time * 1000)):06d}",
            "t": normalized_time,
            "x": round(x, 3),
            "y": resolved_y,
            "z": round(z, 3),
            "confidence": 1.0,
            "confidenceKind": "manual-authoritative",
            "detectionConfidence": None,
            "trajectoryConfidence": 1.0,
            "state": "observed",
            "observed": True,
            "positionSource": "manual-keypoint",
            "heightSource": "manual" if y is not None else "rendering-placeholder",
            "projectionSource": "manual-pitch-coordinate",
            "positionUncertaintyMetres": 0.0,
            "projection": {
                "source": "manual-pitch-coordinate",
                "calibrationFrameIndex": None,
                "uncertaintyMetres": 0.0,
            },
            "provenance": {
                "source": "manual",
                "method": "user-pitch-keypoint",
            },
        }
    return [by_time[time] for time in sorted(by_time)]


def _publish_automatic_ball_trajectory(
    scene: dict,
    keyframes: list[dict],
    diagnostics: dict,
) -> dict:
    payload = scene.setdefault("payload", {})
    ball = _normalize_ball_payload(payload.get("ball"))
    automatic_keyframes = deepcopy(keyframes)
    automatic_diagnostics = {
        **deepcopy(diagnostics),
        "trajectoryMode": "automatic",
        "source": "automatic-ball-resolver",
    }
    ball.update(
        {
            "automaticKeyframes": automatic_keyframes,
            "automaticDiagnostics": automatic_diagnostics,
            "automaticUpdatedAt": datetime.now(UTC).isoformat(),
        }
    )
    if ball["mode"] == "manual":
        ball["keyframes"] = deepcopy(ball["manualKeyframes"])
        ball["diagnostics"] = deepcopy(ball["manualDiagnostics"])
    else:
        ball["keyframes"] = deepcopy(automatic_keyframes)
        ball["diagnostics"] = deepcopy(automatic_diagnostics)
    payload["ball"] = ball
    return ball


def set_scene_ball_trajectory(
    scene: dict,
    mode: str,
    keyframes: list[dict] | None = None,
    *,
    persist: bool = True,
) -> dict:
    if mode not in {"automatic", "manual"}:
        raise ReconstructionError("Ball trajectory mode must be automatic or manual")
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for reconstruction to finish before editing the ball trajectory"
        )
    if mode == "automatic" and keyframes is not None:
        raise ReconstructionError(
            "Keyframes can only be supplied for manual ball trajectory mode"
        )

    payload = scene.setdefault("payload", {})
    ball = _normalize_ball_payload(payload.get("ball"))
    if keyframes is not None:
        ball["manualKeyframes"] = _manual_ball_keyframes(scene, keyframes)
        ball["manualUpdatedAt"] = datetime.now(UTC).isoformat()
    ball["mode"] = mode
    ball["manualDiagnostics"] = _manual_ball_diagnostics(ball["manualKeyframes"])
    if mode == "manual":
        ball["keyframes"] = deepcopy(ball["manualKeyframes"])
        ball["diagnostics"] = deepcopy(ball["manualDiagnostics"])
    else:
        ball["keyframes"] = deepcopy(ball["automaticKeyframes"])
        ball["diagnostics"] = deepcopy(ball["automaticDiagnostics"])
    payload["ball"] = ball
    if persist:
        scene_store.put(scene)
    return scene


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1.0, left_area + right_area - intersection)


def _green_ratio(hsv: np.ndarray, x: float, y: float, radius_x: int, radius_y: int) -> float:
    height, width = hsv.shape[:2]
    x1, x2 = max(0, int(x) - radius_x), min(width, int(x) + radius_x)
    y1, y2 = max(0, int(y) - radius_y), min(height, int(y) + radius_y)
    patch = hsv[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    green = (
        (patch[:, :, 0] > 25)
        & (patch[:, :, 0] < 100)
        & (patch[:, :, 1] > 35)
        & (patch[:, :, 2] > 25)
    )
    return float(green.mean())


def _appearance_feature(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    crop = image[
        max(0, int(y1 + height * 0.12)):max(1, int(y1 + height * 0.62)),
        max(0, int(x1 + width * 0.18)):max(1, int(x2 - width * 0.18)),
    ]
    if crop.size == 0:
        return np.zeros(12, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    vivid = pixels[(pixels[:, 1] > 55) & (pixels[:, 2] > 45)]
    histogram = np.zeros(8, dtype=np.float32)
    if len(vivid):
        histogram, _ = np.histogram(vivid[:, 0], bins=8, range=(0, 180))
        histogram = histogram.astype(np.float32)
        histogram /= max(1.0, float(histogram.sum()))
    white_ratio = float(((pixels[:, 1] < 55) & (pixels[:, 2] > 135)).mean())
    dark_ratio = float((pixels[:, 2] < 72).mean())
    mean_saturation = float(pixels[:, 1].mean() / 255.0)
    mean_value = float(pixels[:, 2].mean() / 255.0)
    return np.concatenate(
        [histogram, np.array([white_ratio, dark_ratio, mean_saturation, mean_value], dtype=np.float32)]
    )


def _is_pitch_person(
    hsv: np.ndarray,
    box: tuple[float, float, float, float],
    confidence: float,
) -> bool:
    """Reject spectators and graphics without losing small players near the far touchline."""
    x1, y1, x2, y2 = box
    height, _ = hsv.shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    if box_height < 14 or box_width < 5 or box_height < box_width * 1.05:
        return False
    if y2 < height * MINIMUM_PERSON_FOOT_Y:
        return False
    center_x = (x1 + x2) / 2
    pitch_ratio = _green_ratio(
        hsv,
        center_x,
        y2 + min(4.0, box_height * 0.06),
        max(8, int(box_width)),
        max(6, int(box_height * 0.16)),
    )
    if pitch_ratio < 0.38:
        # Goalkeepers and players standing on painted lines or in a crowded box
        # can have little grass immediately below the bbox. Keep only strong
        # detector evidence in that ambiguous band; low-confidence observations
        # cannot start tracks and are handled by the second association pass.
        if not (
            confidence >= 0.55
            and y2 >= height * 0.24
            and pitch_ratio >= 0.15
        ):
            return False
    if y2 < height * SHALLOW_PERSON_FOOT_Y:
        return confidence >= SHALLOW_PERSON_CONFIDENCE and (
            pitch_ratio >= SHALLOW_PERSON_GRASS_RATIO or confidence >= 0.55
        )
    return True


def _annotation_team(kind: str | None) -> str | None:
    if kind in {"home-player", "home-goalkeeper"}:
        return "home"
    if kind in {"away-player", "away-goalkeeper"}:
        return "away"
    if kind == "referee":
        return "officials"
    if kind == "other":
        return "unknown"
    return None


def _annotation_role(kind: str | None) -> str | None:
    if kind in {"home-goalkeeper", "away-goalkeeper"}:
        return "goalkeeper"
    if kind == "referee":
        return "referee"
    if kind == "other":
        return "other"
    if kind in {"home-player", "away-player"}:
        return "player"
    return None


def _annotation_action(annotation: dict) -> str:
    """Normalize legacy `ignore` labels into the identity-correction contract."""

    action = str(annotation.get("action") or "").strip().lower()
    if action in {"confirm", "exclude", "merge", "split"}:
        return action
    if annotation.get("kind") == "ignore":
        return "exclude"
    return "confirm"


def _annotation_scope(annotation: dict) -> str:
    """Keep legacy frame labels observation-scoped unless scope was explicit."""

    scope = str(annotation.get("scope") or "").strip().lower()
    return scope if scope in {"observation", "range", "identity"} else "observation"


def _is_identity_unbind_tombstone(annotation: dict | None) -> bool:
    """Return whether an annotation is an explicit negative roster decision."""

    return bool(
        annotation
        and annotation.get("correctionKind") == "canonical-roster-binding-v1"
        and annotation.get("rosterBindingState") == "unbound"
        and annotation.get("externalPlayerId") is None
    )


def _annotation_manual_semantic_key(
    annotation: dict,
) -> tuple[int, str, int, str]:
    """Order role/label edits by authoring metadata, never video time."""

    updated_at = str(annotation.get("updatedAt") or "").strip()
    is_dedicated_roster = int(
        annotation.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION
    )
    return (
        int(bool(updated_at)),
        updated_at,
        is_dedicated_roster,
        str(annotation.get("id") or ""),
    )


def _annotation_box(annotation: dict) -> tuple[float, float, float, float]:
    bbox = annotation["bbox"]
    return (
        float(bbox["x"]),
        float(bbox["y"]),
        float(bbox["x"]) + float(bbox["width"]),
        float(bbox["y"]) + float(bbox["height"]),
    )


def _detection_box(detection: Detection) -> tuple[float, float, float, float]:
    return (
        detection.x - detection.width / 2,
        detection.y - detection.height,
        detection.x + detection.width / 2,
        detection.y,
    )


def _frame_annotations(scene: dict, frame_index: int) -> list[dict]:
    reconstruction = scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    annotations = deepcopy(list(reconstruction.get("frameAnnotations") or []))
    split_owner_aliases = {
        str(annotation.get("splitCanonicalPersonId")): str(
            _annotation_source_identity(annotation) or ""
        )
        for annotation in annotations
        if _annotation_action(annotation) == "split"
        and annotation.get("splitCanonicalPersonId")
        and _annotation_source_identity(annotation)
    }

    def pre_split_owner(owner: str) -> str:
        visited: set[str] = set()
        current = owner
        while current in split_owner_aliases and current not in visited:
            visited.add(current)
            current = split_owner_aliases[current]
        return current

    for annotation in annotations:
        owner = str(annotation.get("canonicalPersonId") or "").strip()
        if owner in split_owner_aliases:
            annotation["preSplitCanonicalOwnerId"] = pre_split_owner(owner)
    dedicated_owner_keys = {
        str(value)
        for annotation in annotations
        if annotation.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION
        and annotation.get("rosterBindingState") in {"bound", "unbound"}
        for value in (
            annotation.get("canonicalPersonId"),
            annotation.get("sourceTrackId"),
        )
        if str(value or "").strip()
    }
    result: list[dict] = []
    for annotation in annotations:
        if (
            annotation.get("frameIndex") is None
            or int(annotation["frameIndex"]) != frame_index
        ):
            continue
        owner_keys = {
            str(value)
            for value in (
                annotation.get("canonicalPersonId"),
                annotation.get("sourceTrackId"),
            )
            if str(value or "").strip()
        }
        if (
            annotation.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION
            and annotation.get("externalPlayerId") is not None
            and (
                _annotation_action(annotation) == "merge"
                or bool(owner_keys & dedicated_owner_keys)
            )
        ):
            result.append(
                {
                    **annotation,
                    "externalPlayerId": None,
                    "rosterValueSupersededByDedicatedCorrection": True,
                }
            )
        else:
            result.append(annotation)
    return result


def _annotation_detection_index(detections: list[Detection], annotation: dict) -> int | None:
    target = _annotation_box(annotation)
    candidates: list[tuple[float, int]] = []
    for index, detection in enumerate(detections):
        box = _detection_box(detection)
        overlap = _iou(target, box)
        center_inside = target[0] <= detection.x <= target[2] and target[1] <= detection.y <= target[3]
        if overlap >= 0.12 or center_inside:
            candidates.append((overlap + (0.25 if center_inside else 0.0), index))
    return max(candidates)[1] if candidates else None


def _apply_person_annotations(
    image: np.ndarray,
    detections: list[Detection],
    annotations: list[dict],
) -> list[Detection]:
    result = list(detections)
    for annotation in annotations:
        # A split edits the persisted identity graph, not detector evidence. In
        # particular, it must never synthesize a person when the selected
        # observation cannot be remapped on a later detector run.
        if _annotation_action(annotation) == "split":
            continue
        detection_index = _annotation_detection_index(result, annotation)
        if _annotation_action(annotation) == "exclude":
            if _annotation_scope(annotation) == "identity":
                # Keep one exact negative anchor until association is complete.
                # The raw track carrying this annotation id can then be removed
                # deterministically even if a new calibration shifts metric
                # coordinates. Observation-scoped excludes still remove only
                # the current detection.
                if detection_index is not None:
                    detection = result[detection_index]
                    detection.annotation_id = annotation["id"]
                    detection.annotation_kind = "ignore"
                    detection.annotation_label = None
                    detection.external_player_id = None
                continue
            if detection_index is not None:
                result.pop(detection_index)
            continue
        if detection_index is None:
            x1, y1, x2, y2 = _annotation_box(annotation)
            detection = Detection(
                x=(x1 + x2) / 2,
                y=y2,
                width=x2 - x1,
                height=y2 - y1,
                confidence=1.0,
                feature=_appearance_feature(image, (x1, y1, x2, y2)),
            )
            result.append(detection)
        else:
            detection = result[detection_index]
            detection.confidence = max(detection.confidence, 0.98)
        annotation_id = str(annotation["id"])
        detection.annotation_ids.add(annotation_id)
        if (
            _annotation_action(annotation) == "confirm"
            and _annotation_scope(annotation) == "identity"
        ):
            manual_owner_id = str(
                annotation.get("preSplitCanonicalOwnerId")
                or annotation.get("canonicalPersonId")
                or annotation.get("sourceTrackId")
                or ""
            ).strip()
            if manual_owner_id:
                detection.manual_identity_owner_ids.add(manual_owner_id)
                if len(detection.manual_identity_owner_ids) > 1:
                    raise ReconstructionError(
                        "Conflicting explicit canonical identities target one observation"
                    )
        is_unbind_tombstone = _is_identity_unbind_tombstone(annotation)
        is_roster_binding = (
            annotation.get("correctionKind")
            == CANONICAL_ROSTER_BINDING_CORRECTION
            and annotation.get("rosterBindingState") in {"bound", "unbound"}
        )
        if is_roster_binding:
            incoming_state = str(annotation["rosterBindingState"])
            incoming_external_id = annotation.get("externalPlayerId")
            if (
                detection.roster_binding_state is not None
                and (
                    detection.roster_binding_state != incoming_state
                    or detection.external_player_id != incoming_external_id
                )
            ):
                raise ReconstructionError(
                    "Conflicting dedicated roster corrections target one observation"
                )
            detection.roster_binding_state = incoming_state
            detection.roster_binding_annotation_ids.add(annotation_id)
        if is_unbind_tombstone:
            detection.identity_tombstone_annotation_ids.add(annotation_id)
        semantic_key = _annotation_manual_semantic_key(annotation)
        if (
            detection.manual_semantic_key is None
            or semantic_key >= detection.manual_semantic_key
        ):
            detection.annotation_id = annotation["id"]
            detection.annotation_kind = annotation["kind"]
            detection.annotation_label = annotation.get("label")
            detection.annotation_is_identity_evidence = not is_unbind_tombstone
            detection.manual_semantic_key = semantic_key
        incoming_external_id = annotation.get("externalPlayerId")
        if is_roster_binding:
            detection.external_player_id = incoming_external_id
        elif detection.roster_binding_state is None and incoming_external_id:
            if (
                detection.external_player_id
                and detection.external_player_id != incoming_external_id
            ):
                raise ReconstructionError(
                    "Conflicting legacy roster confirmations target one observation"
                )
            detection.external_player_id = incoming_external_id
    return result


def _predict_frame(model, path: Path | str):
    return model.predict(
        str(path),
        imgsz=DETECTOR_IMAGE_SIZE,
        conf=DETECTOR_CONFIDENCE,
        classes=[0, 32],
        iou=DETECTOR_PROVIDER_NMS_IOU,
        max_det=DETECTOR_MAX_DETECTIONS,
        device=get_settings().reconstruction_device,
        verbose=False,
    )[0]


def _checkpoint_sha256(path: Path) -> str:
    stat = path.stat()
    cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    with _checkpoint_digest_lock:
        cached = _checkpoint_digest_cache.get(cache_key)
    if cached is not None:
        return cached
    digest = frame_content_sha256(path)
    with _checkpoint_digest_lock:
        _checkpoint_digest_cache[cache_key] = digest
    return digest


def _person_checkpoint_identity(model_name: str, model: object | None = None) -> dict:
    """Return auditable checkpoint provenance without depending on Ultralytics internals."""

    candidates: list[Path] = []
    for raw in (model_name, getattr(model, "ckpt_path", None)):
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
    identity: dict = {"requested": str(model_name)}
    if checkpoint is None:
        identity["contentAvailable"] = False
        return identity
    stat = checkpoint.stat()
    identity.update(
        {
            "contentAvailable": True,
            "name": checkpoint.name,
            "size": int(stat.st_size),
            "mtimeNs": int(stat.st_mtime_ns),
            "sha256": _checkpoint_sha256(checkpoint),
        }
    )
    return identity


def _installed_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _person_detection_input(model_name: str, model: object | None = None) -> dict:
    """Complete cache-invalidating contract for sampled-frame base inference."""

    return {
        "schemaVersion": 1,
        "provider": {
            "backend": "ultralytics-yolo",
            "version": _installed_package_version("ultralytics"),
        },
        "postprocessRuntime": {
            "opencv": str(cv2.__version__),
            "numpy": str(np.__version__),
        },
        "checkpoint": _person_checkpoint_identity(model_name, model),
        "classes": {"person": 0, "legacyGenericBall": 32},
        "inference": {
            "imageSize": int(DETECTOR_IMAGE_SIZE),
            "confidence": float(DETECTOR_CONFIDENCE),
            "providerNmsIou": float(DETECTOR_PROVIDER_NMS_IOU),
            "maxDetections": int(DETECTOR_MAX_DETECTIONS),
            # CPU/GPU kernels can differ at tight thresholds. Treat device as
            # part of provenance instead of claiming bit-identical evidence.
            "device": str(get_settings().reconstruction_device),
        },
        "personFilter": {
            "version": PERSON_FILTER_POLICY_VERSION,
            "localNmsIou": float(PERSON_LOCAL_NMS_IOU),
            "minimumFootYRatio": float(MINIMUM_PERSON_FOOT_Y),
            "shallowFootYRatio": float(SHALLOW_PERSON_FOOT_Y),
            "shallowConfidence": float(SHALLOW_PERSON_CONFIDENCE),
            "shallowGrassRatio": float(SHALLOW_PERSON_GRASS_RATIO),
            "appearanceFeatureSchema": APPEARANCE_FEATURE_SCHEMA_VERSION,
        },
        "legacyBallFilter": {
            "version": LEGACY_BALL_FILTER_POLICY_VERSION,
            "minimumCenterYRatio": 0.30,
            "maximumBoxSizePixels": 24.0,
            "minimumGrassRatio": 0.24,
            "deduplicationRadiusPixels": 10.0,
        },
    }


def _base_detection_cache_diagnostics(frame_count: int, detector_input: dict) -> dict:
    return {
        "schemaVersion": 1,
        "artifactSchemaVersion": PERSON_DETECTION_CACHE_SCHEMA_VERSION,
        "frameCount": int(frame_count),
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "errors": 0,
        "corruptArtifacts": 0,
        "providerCalls": 0,
        "input": deepcopy(detector_input),
    }


def _base_detection_payload(detection: Detection) -> dict:
    """Serialize only immutable pre-annotation detector evidence."""

    return {
        "x": float(detection.x),
        "y": float(detection.y),
        "width": float(detection.width),
        "height": float(detection.height),
        "confidence": float(detection.confidence),
        "feature": np.asarray(detection.feature, dtype=np.float32).tolist(),
    }


def _base_detection_from_payload(payload: Mapping) -> Detection:
    return Detection(
        x=float(payload["x"]),
        y=float(payload["y"]),
        width=float(payload["width"]),
        height=float(payload["height"]),
        confidence=float(payload["confidence"]),
        feature=np.asarray(payload["feature"], dtype=np.float32).copy(),
    )


def _cached_base_frame_detections(
    model: object,
    path: Path,
    asset_directory: Path,
    detector_input: Mapping,
    diagnostics: dict,
) -> tuple[np.ndarray, list[Detection], list[dict]]:
    """Load or compute one frame's base detections before all manual state."""

    frame_digest: str | None = None
    try:
        frame_digest = frame_content_sha256(path)
        lookup = lookup_person_detection_cache(
            asset_directory,
            frame_sha256=frame_digest,
            detector_input=detector_input,
        )
    except (OSError, PersonDetectionCacheError, ValueError) as exc:
        lookup = None
        diagnostics["errors"] += 1
        diagnostics.setdefault("errorDetails", []).append(
            {"frame": path.name, "stage": "lookup", "detail": str(exc)}
        )

    if lookup is not None and lookup.entry is not None:
        image = cv2.imread(str(path))
        if image is not None and (image.shape[1], image.shape[0]) == lookup.entry.image_size:
            people_payload, balls, _ = lookup.entry.as_pipeline_data()
            diagnostics["hits"] += 1
            return (
                image,
                [_base_detection_from_payload(item) for item in people_payload],
                balls,
            )
        diagnostics["errors"] += 1
        diagnostics.setdefault("errorDetails", []).append(
            {
                "frame": path.name,
                "stage": "decode",
                "detail": "cached image size does not match the current decoded JPEG",
            }
        )
    elif lookup is not None and lookup.status in {"corrupt", "error"}:
        diagnostics["errors"] += 1
        if lookup.status == "corrupt":
            diagnostics["corruptArtifacts"] += 1
        diagnostics.setdefault("errorDetails", []).append(
            {
                "frame": path.name,
                "stage": "lookup",
                "detail": lookup.error or lookup.status,
            }
        )

    diagnostics["misses"] += 1
    diagnostics["providerCalls"] += 1
    result = _predict_frame(model, path)
    image = result.orig_img
    people, balls = _person_detections(result)
    if frame_digest is not None:
        try:
            stored = store_person_detection_cache(
                asset_directory,
                frame_sha256=frame_digest,
                detector_input=detector_input,
                image_size=(int(image.shape[1]), int(image.shape[0])),
                people=[_base_detection_payload(item) for item in people],
                legacy_ball_candidates=balls,
            )
        except (OSError, PersonDetectionCacheError, ValueError) as exc:
            diagnostics["errors"] += 1
            diagnostics.setdefault("errorDetails", []).append(
                {"frame": path.name, "stage": "write", "detail": str(exc)}
            )
        else:
            if stored is not None:
                diagnostics["writes"] += 1
    return image, people, balls


def _source_frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _best_pitch_calibration(
    calibrations: dict[int, PitchCalibration],
) -> PitchCalibration | None:
    if not calibrations:
        return None
    return max(
        calibrations.values(),
        key=lambda item: (
            2
            if item.method.startswith("pnlcalib")
            else 1
            if item.method == "roboflow-field-keypoints"
            else 0,
            item.confidence,
            item.inlier_count,
            item.keypoint_count,
            -(item.reprojection_error if item.reprojection_error is not None else 999.0),
        ),
    )


def _positive_image_size(value) -> int | None:
    if isinstance(value, (tuple, list)):
        values = [_positive_image_size(item) for item in value]
        values = [item for item in values if item is not None]
        return max(values) if values else None
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if resolved > 0 else None


def _pitch_keypoint_inference_size(model, configured_size: int | None) -> int:
    """Use an explicit override, otherwise honor the checkpoint training size."""

    explicit = _positive_image_size(configured_size)
    if explicit is not None:
        return explicit
    metadata_sources = (
        getattr(model, "overrides", None),
        getattr(getattr(model, "model", None), "args", None),
        getattr(getattr(model, "model", None), "yaml", None),
    )
    for metadata in metadata_sources:
        if isinstance(metadata, dict):
            native = _positive_image_size(metadata.get("imgsz") or metadata.get("img_size"))
        else:
            native = _positive_image_size(
                getattr(metadata, "imgsz", None) or getattr(metadata, "img_size", None)
            )
        if native is not None:
            return native
    # This is also the native size of the bundled Roboflow Sports checkpoint.
    return 640


def _local_frame_calibrations(
    frames: list[tuple[Path, float]],
    requested_indices: set[int] | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[int, PitchCalibration]:
    settings = get_settings()
    model_path = Path(settings.pitch_keypoint_model)
    if not model_path.is_file():
        return {}
    selected = [
        (path, _source_frame_index(path))
        for path, _ in frames
        if requested_indices is None or _source_frame_index(path) in requested_indices
    ]
    if not selected:
        return {}
    model = _load_model(str(model_path))
    inference_size = _pitch_keypoint_inference_size(
        model,
        settings.pitch_keypoint_image_size,
    )
    result: dict[int, PitchCalibration] = {}
    for start in range(0, len(selected), 4):
        batch = selected[start : start + 4]
        predictions = model.predict(
            [str(path) for path, _ in batch],
            imgsz=inference_size,
            device=settings.reconstruction_device,
            verbose=False,
        )
        for prediction, (_, source_index) in zip(predictions, batch):
            calibration = calibration_from_pose_result(prediction, source_index)
            if calibration is not None:
                result[source_index] = calibration
        if on_progress is not None:
            on_progress(min(len(selected), start + len(batch)), len(selected), len(result))
    return result


def _automatic_frame_calibrations(
    frames: list[tuple[Path, float]],
    on_progress: Callable[[str, int, int, float, int], None] | None = None,
    *,
    worker_timeout: float | None = None,
) -> tuple[dict[int, PitchCalibration], list[str]]:
    indexed = [(_source_frame_index(path), path) for path, _ in frames]
    warnings: list[str] = []
    calibrations: dict[int, PitchCalibration] = {}
    worker_configured = bool(get_settings().calibration_worker_url)
    worker_failed = False
    if worker_configured:
        if on_progress is not None:
            on_progress("pnlcalib", 0, len(indexed), 0.0, 0)
        try:
            calibrations.update(
                calibrate_frames_with_worker(
                    indexed,
                    on_progress=(
                        lambda completed, total, valid: on_progress(
                            "pnlcalib",
                            completed,
                            total,
                            0.9 * completed / max(1, total),
                            valid,
                        )
                        if on_progress is not None
                        else None
                    ),
                    timeout=worker_timeout,
                )
            )
        except CalibrationWorkerError as exc:
            worker_failed = True
            warnings.append(str(exc))
    missing = {index for index, _ in indexed} - set(calibrations)
    if missing:
        if on_progress is not None:
            on_progress("local-keypoints", 0, len(missing), 0.0 if worker_failed else 0.9, len(calibrations))
        local = _local_frame_calibrations(
            frames,
            missing,
            on_progress=(
                lambda completed, total, valid: on_progress(
                    "local-keypoints",
                    completed,
                    total,
                    completed / max(1, total)
                    if worker_failed or not worker_configured
                    else 0.9 + 0.1 * completed / max(1, total),
                    len(calibrations) + valid,
                )
                if on_progress is not None
                else None
            ),
        )
        calibrations.update(local)
        if worker_configured and local:
            warnings.append(
                f"Local semantic-keypoint fallback calibrated {len(local)} frames missed by PnLCalib."
            )
    if on_progress is not None:
        backend = "local-keypoints" if worker_failed or not worker_configured else "pnlcalib"
        on_progress(backend, len(indexed), len(indexed), 1.0, len(calibrations))
    return calibrations, warnings


def _project_metric_point(
    x: float,
    y: float,
    calibration: PitchCalibration,
    pitch: dict,
) -> tuple[float, float] | None:
    projected = calibration.image_to_pitch @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(float(projected[2])) < 1e-8:
        return None
    pitch_x = float(projected[0] / projected[2])
    pitch_z = float(projected[1] / projected[2])
    half_length = float(pitch["length"]) / 2.0
    half_width = float(pitch["width"]) / 2.0
    if not np.isfinite([pitch_x, pitch_z]).all():
        return None
    if not (-half_length - 4.0 <= pitch_x <= half_length + 4.0):
        return None
    if not (-half_width - 4.0 <= pitch_z <= half_width + 4.0):
        return None
    return (
        max(-half_length, min(half_length, pitch_x)),
        max(-half_width, min(half_width, pitch_z)),
    )


def _attach_metric_positions(
    people: list[Detection],
    balls: list[dict],
    calibration: PitchCalibration | None,
    pitch: dict,
    *,
    projection_source: str = "direct",
    calibration_frame_index: int | None = None,
    position_uncertainty_metres: float | None = None,
) -> None:
    if calibration is None:
        return
    for detection in people:
        position = _project_metric_point(detection.x, detection.y, calibration, pitch)
        if position is not None:
            detection.pitch_x, detection.pitch_z = position
            detection.projection_source = projection_source
            detection.calibration_frame_index = calibration_frame_index
            detection.position_uncertainty_metres = position_uncertainty_metres
    for ball in balls:
        position = _project_metric_point(ball["x"], ball["y"], calibration, pitch)
        if position is not None:
            ball["pitchX"], ball["pitchZ"] = position
            ball["projectionSource"] = projection_source
            ball["calibrationFrameIndex"] = calibration_frame_index
            ball["positionUncertaintyMetres"] = position_uncertainty_metres


def _calibration_person_support(
    people: list[Detection],
    calibration: PitchCalibration,
    pitch: dict,
) -> tuple[int, int]:
    if not people:
        return 0, 0
    supported = sum(
        _project_metric_point(person.x, person.y, calibration, pitch) is not None
        for person in people
    )
    return supported, len(people)


def _calibration_uncertainty_metres(
    calibration: PitchCalibration,
    alignment_error: float | None = None,
) -> float:
    """Return an explicit engineering estimate, not a statistical confidence interval."""
    pixel_error = (
        alignment_error
        if alignment_error is not None
        else calibration.reprojection_error
    )
    if pixel_error is not None:
        return round(max(0.25, min(12.0, float(pixel_error) * 0.25)), 2)
    return round(max(0.75, min(8.0, 1.0 + (1.0 - calibration.confidence) * 8.0)), 2)


def _merge_direct_calibration_anchors(
    automatic: dict[int, PitchCalibration],
    manual: dict[int, PitchCalibration],
) -> dict[int, PitchCalibration]:
    """Merge immutable direct observations; manual wins at the same sample.

    This intentionally performs no matrix interpolation. Inter-frame recovery
    remains the temporal solver's job and must cross only QA-approved motion
    edges.
    """

    merged = dict(automatic)
    merged.update(manual)
    return merged


def _resolve_temporal_frame_calibrations(
    frames: list[tuple[Path, float]],
    frame_sizes: dict[int, tuple[int, int]],
    direct_calibrations: dict[int, PitchCalibration],
    motion_edges: dict[int, CameraMotionEstimate],
    frame_evidence: list[dict],
    person_frames: list[tuple[list[Detection], float]],
    pitch: dict,
    *,
    max_gap_seconds: float = 2.0,
) -> tuple[
    dict[int, PitchCalibration],
    dict[int, int],
    dict[int, float],
    int,
]:
    """Resolve a shot in both temporal directions and publish auditable evidence.

    The direct detector observations remain immutable under ``observation``.
    A rejected or missing observation may get a metric solution from an earlier
    or later direct anchor, but only through QA-approved camera-motion edges.
    Target-frame line/person checks can still veto a propagated hypothesis.
    """

    descriptors = [
        TemporalCalibrationFrame(
            sample_index=sample_index,
            source_frame_index=_source_frame_index(path),
            scene_time=float(scene_time),
            width=frame_sizes[sample_index][0],
            height=frame_sizes[sample_index][1],
        )
        for sample_index, (path, scene_time) in enumerate(frames)
    ]
    resolutions = solve_calibration_sequence(
        descriptors,
        direct_calibrations,
        motion_edges,
        max_gap_seconds=max_gap_seconds,
    )
    resolved: dict[int, PitchCalibration] = {}
    anchor_frames: dict[int, int] = {}
    uncertainties: dict[int, float] = {}
    recovered_count = 0

    for descriptor, evidence in zip(descriptors, frame_evidence):
        sample_index = descriptor.sample_index
        observation_status = str(evidence.get("status") or "missing")
        observation_source = str(evidence.get("projectionSource") or "none")
        direct_observation = observation_source in {"direct", "manual-direct"}
        evidence["observationStatus"] = (
            "direct-accepted"
            if observation_status == "accepted" and direct_observation
            else "direct-rejected"
            if observation_status == "rejected" and direct_observation
            else "missing"
        )
        evidence["observation"] = {
            "status": observation_status,
            "source": evidence.get("source"),
            "projectionSource": observation_source,
            "backend": evidence.get("backend"),
            "confidence": evidence.get("confidence"),
            "imageToPitch": evidence.get("imageToPitch"),
            "visiblePitchSide": evidence.get("visiblePitchSide"),
            "rejectionReasons": list(evidence.get("rejectionReasons") or []),
        }

        resolution = resolutions[sample_index]
        hypothesis_payloads = resolution.hypotheses_payload()
        if observation_status == "rejected" and evidence.get("imageToPitch") is not None:
            hypothesis_payloads.append(
                {
                    "id": f"direct-rejected-s{sample_index}",
                    "rank": len(hypothesis_payloads) + 1,
                    "selected": False,
                    "origin": "direct-rejected",
                    "eligibility": "rejected-observation",
                    "score": round(float(evidence.get("confidence") or 0.0), 5),
                    "scoreKind": evidence.get("confidenceKind"),
                    "visiblePitchSide": evidence.get("visiblePitchSide"),
                    "anchorFrameIndices": [descriptor.source_frame_index],
                    "anchorSampleIndices": [sample_index],
                    "motionEdgeIndices": [],
                    "temporalDistanceSeconds": 0.0,
                    "motionConfidence": None,
                    "uncertaintyP95Metres": None,
                    "disagreementMetres": None,
                    "imageToPitch": evidence.get("imageToPitch"),
                    "rejectionReasons": list(evidence.get("rejectionReasons") or []),
                }
            )
        evidence["hypotheses"] = hypothesis_payloads
        evidence["ambiguityMargin"] = (
            round(float(resolution.ambiguity_margin), 5)
            if resolution.ambiguity_margin is not None
            else None
        )
        selected = resolution.selected
        if selected is None:
            solver_reasons = list(resolution.rejection_reasons)
            evidence["solutionStatus"] = (
                "ambiguous"
                if "conflicting-temporal-hypotheses" in solver_reasons
                else "unresolved"
            )
            evidence["selectedHypothesisId"] = None
            evidence["projectionSource"] = "none"
            evidence["temporal"] = None
            evidence["uncertainty"] = None
            evidence["rejectionReasons"] = list(
                dict.fromkeys([*(evidence.get("rejectionReasons") or []), *solver_reasons])
            )
            continue

        calibration = selected.calibration
        if resolution.projection_source == "direct":
            resolved[sample_index] = calibration
            anchor_frames[sample_index] = selected.anchor_source_frame_index
            uncertainties[sample_index] = selected.uncertainty_metres
            evidence["solutionStatus"] = "direct-accepted"
            evidence["selectedHypothesisId"] = selected.id
            evidence["uncertainty"] = {
                "kind": "engineering-p95",
                "p95Metres": round(float(selected.uncertainty_metres), 3),
                "temporalDistanceSeconds": 0.0,
                "motionConfidence": 1.0,
            }
            evidence["positionUncertaintyMetres"] = round(
                float(selected.uncertainty_metres), 3
            )
            evidence["temporal"] = None
            continue

        validation_reasons: list[str] = []
        alignment = None
        target_uncertainty_penalty = 0.0
        image = cv2.imread(str(frames[sample_index][0]))
        if image is None:
            validation_reasons.append("temporal-target-frame-unreadable")
        else:
            alignment_metrics = calibration_alignment_metrics(image, calibration)
            alignment = alignment_metrics.as_dict() if alignment_metrics is not None else None
            if alignment_metrics is None:
                target_uncertainty_penalty += 1.25
            if (
                alignment_metrics is not None
                and not _semantic_alignment_passes_review(alignment_metrics)
            ):
                validation_reasons.append("temporal-semantic-line-alignment-poor")

        people = person_frames[sample_index][0]
        person_support = None
        if len(people) >= 4:
            supported_people, total_people = _calibration_person_support(
                people,
                calibration,
                pitch,
            )
            support_ratio = supported_people / max(1, total_people)
            person_support = {
                "supported": supported_people,
                "total": total_people,
                "ratio": round(support_ratio, 3),
            }
            if supported_people < 4 or support_ratio < 0.55:
                validation_reasons.append("temporal-insufficient-person-pitch-support")
        else:
            target_uncertainty_penalty += 0.50

        target_uncertainty = selected.uncertainty_metres + target_uncertainty_penalty
        if target_uncertainty > TEMPORAL_REVIEW_UNCERTAINTY_METRES:
            validation_reasons.append("temporal-target-uncertainty-too-high")

        selected_payload = next(
            (item for item in hypothesis_payloads if item.get("id") == selected.id),
            None,
        )
        if selected_payload is not None:
            selected_payload["targetValidation"] = {
                "alignmentMetrics": alignment,
                "personSupport": person_support,
                "uncertaintyPenaltyMetres": round(target_uncertainty_penalty, 3),
                "uncertaintyP95Metres": round(target_uncertainty, 3),
                "rejectionReasons": validation_reasons,
            }
        if validation_reasons:
            if selected_payload is not None:
                selected_payload["selected"] = False
                selected_payload["rejectionReasons"] = list(
                    dict.fromkeys(
                        [*(selected_payload.get("rejectionReasons") or []), *validation_reasons]
                    )
                )
            evidence["solutionStatus"] = "temporal-rejected"
            evidence["selectedHypothesisId"] = None
            evidence["projectionSource"] = "none"
            evidence["temporal"] = None
            evidence["uncertainty"] = None
            evidence["rejectionReasons"] = list(
                dict.fromkeys([*(evidence.get("rejectionReasons") or []), *validation_reasons])
            )
            continue

        consensus = (
            resolution.projection_source == "temporal-bidirectional"
            and len(resolution.hypotheses) > 1
        )
        consensus_peer = (
            next(
                (
                    item
                    for item in resolution.hypotheses
                    if item.id != selected.id
                    and item.direction != selected.direction
                    and item.disagreement_metres is not None
                    and item.disagreement_metres <= 2.5
                ),
                None,
            )
            if consensus
            else None
        )
        contributing = (
            (selected, consensus_peer)
            if consensus_peer is not None
            else (selected,)
        )
        anchor_source_indices = list(
            dict.fromkeys(item.anchor_source_frame_index for item in contributing)
        )
        anchor_sample_indices = list(
            dict.fromkeys(item.anchor_sample_index for item in contributing)
        )
        resolved[sample_index] = calibration
        anchor_frames[sample_index] = selected.anchor_source_frame_index
        uncertainties[sample_index] = target_uncertainty
        recovered_count += 1
        calibration_payload = calibration.as_dict()
        evidence.update(
            {
                "status": "accepted",
                "solutionStatus": "temporal-accepted",
                "source": calibration.method,
                "projectionSource": resolution.projection_source,
                "backend": "temporal-camera-graph",
                "confidence": round(float(selected.score), 3),
                "confidenceKind": calibration_payload.get("confidenceKind"),
                "imageToPitch": _matrix_payload(calibration.image_to_pitch),
                "reprojectionError": (
                    alignment.get("residualP50") if alignment is not None else None
                ),
                "reprojectionP95": (
                    alignment.get("residualP95") if alignment is not None else None
                ),
                "groundErrorP50Metres": None,
                "groundErrorP95Metres": None,
                "visiblePitchSide": pitch_side(calibration.rectangle),
                "rectangle": calibration.rectangle,
                "alignmentMetrics": alignment,
                "horizon": calibration_horizon(calibration, descriptor.width),
                "rejectionReasons": [],
                "personSupport": person_support,
                "selectedHypothesisId": selected.id,
                "temporal": {
                    "direction": (
                        "bidirectional"
                        if resolution.projection_source == "temporal-bidirectional"
                        else selected.direction
                    ),
                    "anchorFrameIndices": anchor_source_indices,
                    "anchorSampleIndices": anchor_sample_indices,
                    "anchorSceneTimes": [
                        round(float(item.anchor_scene_time), 3) for item in contributing
                    ],
                    "motionEdgeIndices": list(selected.motion_edge_indices),
                    "temporalDistanceSeconds": round(
                        float(selected.temporal_distance_seconds), 3
                    ),
                    "motionConfidence": round(float(selected.motion_confidence), 5),
                },
                "uncertainty": {
                    "kind": "engineering-p95",
                    "p95Metres": round(float(target_uncertainty), 3),
                    "temporalDistanceSeconds": round(
                        float(selected.temporal_distance_seconds), 3
                    ),
                    "motionConfidence": round(float(selected.motion_confidence), 5),
                },
                "positionUncertaintyMetres": round(
                    float(target_uncertainty), 3
                ),
            }
        )

    return resolved, anchor_frames, uncertainties, recovered_count


def _matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


def _keypoint_evidence(calibration: PitchCalibration) -> list[dict]:
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return [dict(item) for item in calibration.raw_keypoints]
    result: list[dict] = []
    for raw in calibration.raw_keypoints:
        item = deepcopy(raw)
        image_point = item.get("image") or {}
        pitch_point = item.get("pitch") or {}
        try:
            projected = pitch_to_image @ np.array(
                [float(pitch_point["x"]), float(pitch_point["z"]), 1.0],
                dtype=np.float64,
            )
            if abs(float(projected[2])) < 1e-8:
                raise ValueError
            projected_x = float(projected[0] / projected[2])
            projected_y = float(projected[1] / projected[2])
            observed_x = float(image_point["x"])
            observed_y = float(image_point["y"])
            dx = projected_x - observed_x
            dy = projected_y - observed_y
            item["projectedImage"] = {
                "x": round(projected_x, 3),
                "y": round(projected_y, 3),
            }
            item["residualVector"] = {
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "magnitude": round(hypot(dx, dy), 3),
            }
        except (KeyError, TypeError, ValueError):
            item["projectedImage"] = None
            item["residualVector"] = None
        result.append(item)
    return result


def _calibration_quality_gate(
    gate_id: str,
    label: str,
    status: str,
    *,
    value=None,
    threshold=None,
    reason: str | None = None,
) -> dict:
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "value": value,
        "threshold": threshold,
        "reason": reason,
    }


def _direct_calibration_qa(
    image: np.ndarray,
    calibration: PitchCalibration,
    *,
    people: list[Detection] | None = None,
    pitch: dict | None = None,
    manual: bool = False,
) -> dict:
    """Apply the same auditable direct-observation gates in previews and rebuilds."""

    rejection_reasons: list[str] = []
    gates: list[dict] = []
    matrix = calibration.image_to_pitch
    finite_matrix = matrix.shape == (3, 3) and bool(np.isfinite(matrix).all())
    gates.append(
        _calibration_quality_gate(
            "finite-homography",
            "Finite 3×3 homography",
            "pass" if finite_matrix else "fail",
            value=finite_matrix,
            threshold={"required": True},
            reason=None if finite_matrix else "invalid-homography",
        )
    )
    if not finite_matrix:
        rejection_reasons.append("invalid-homography")
        non_singular = False
    else:
        non_singular = abs(float(np.linalg.det(matrix))) >= 1e-10
        if not non_singular:
            rejection_reasons.append("singular-homography")
    gates.append(
        _calibration_quality_gate(
            "non-singular-homography",
            "Invertible homography",
            "pass" if non_singular else "fail",
            value=abs(float(np.linalg.det(matrix))) if finite_matrix else None,
            threshold={"atLeast": 1e-10},
            reason=None if non_singular else "singular-homography",
        )
    )

    confidence_pass = calibration.confidence >= METRIC_CALIBRATION_THRESHOLD
    if not confidence_pass:
        rejection_reasons.append("confidence-below-metric-threshold")
    gates.append(
        _calibration_quality_gate(
            "metric-confidence",
            "Metric confidence",
            "pass" if confidence_pass else "fail",
            value=round(float(calibration.confidence), 5),
            threshold={"atLeast": METRIC_CALIBRATION_THRESHOLD},
            reason=None if confidence_pass else "confidence-below-metric-threshold",
        )
    )

    detected_keypoints = (
        calibration.detected_keypoint_count or calibration.keypoint_count
    )
    inlier_ratio = calibration.inlier_ratio
    if inlier_ratio is None and detected_keypoints:
        inlier_ratio = calibration.inlier_count / detected_keypoints
    candidate_p95 = calibration.reprojection_p95
    if candidate_p95 is None:
        candidate_p95 = calibration.reprojection_error
    is_line_fallback = calibration.method == "pitch-lines-ransac"
    raw_partial_view_support = (
        not manual
        and not is_line_fallback
        and candidate_p95 is not None
        and float(candidate_p95) <= 25.0
        and calibration.reprojection_error is not None
        and float(calibration.reprojection_error) <= 8.0
        and detected_keypoints >= 6
        and inlier_ratio is not None
        and inlier_ratio >= 0.65
    )
    raw_reprojection_pass = (
        candidate_p95 is None
        or float(candidate_p95) <= CALIBRATION_REVIEW_REPROJECTION_P95
        or raw_partial_view_support
    )
    if not raw_reprojection_pass:
        rejection_reasons.append("reprojection-error-too-high")
    gates.append(
        _calibration_quality_gate(
            "model-reprojection-p95",
            "Model reprojection p95",
            "not-available"
            if candidate_p95 is None
            else "pass"
            if raw_reprojection_pass
            else "fail",
            value=round(float(candidate_p95), 3) if candidate_p95 is not None else None,
            threshold={
                "atMostPixels": CALIBRATION_REVIEW_REPROJECTION_P95,
                "partialViewAtMostPixels": 25.0,
            },
            reason=None if raw_reprojection_pass else "reprojection-error-too-high",
        )
    )

    alignment_metrics = calibration_alignment_metrics(image, calibration)
    alignment = alignment_metrics.as_dict() if alignment_metrics is not None else None
    semantic_pass = _semantic_alignment_passes_review(alignment_metrics)
    if alignment_metrics is None:
        rejection_reasons.append("semantic-line-alignment-unscored")
    elif not semantic_pass:
        rejection_reasons.append("semantic-line-alignment-poor")
    gates.append(
        _calibration_quality_gate(
            "semantic-line-alignment",
            "Projected markings match observed pitch lines",
            "not-available"
            if alignment_metrics is None
            else "pass"
            if semantic_pass
            else "fail",
            value=alignment,
            threshold={
                "residualP95AtMostPixels": CALIBRATION_REVIEW_REPROJECTION_P95,
                "f1AtLeast": 0.08,
                "partialViewResidualP95AtMostPixels": PARTIAL_VIEW_REPROJECTION_P95_LIMIT,
                "partialViewResidualP50AtMostPixels": PARTIAL_VIEW_REPROJECTION_P50_LIMIT,
                "partialViewF1AtLeast": PARTIAL_VIEW_ALIGNMENT_F1_MINIMUM,
            },
            reason=(
                "semantic-line-alignment-unscored"
                if alignment_metrics is None
                else None
                if semantic_pass
                else "semantic-line-alignment-poor"
            ),
        )
    )

    if manual:
        gates.append(
            _calibration_quality_gate(
                "direct-observation-support",
                "Manual anchor support",
                "pass",
                value={"anchorCount": calibration.supported_lines},
                threshold={"atLeast": 4},
            )
        )
    elif is_line_fallback:
        line_support_pass = calibration.supported_lines >= 4
        curve_support_pass = calibration.matched_curves >= 1
        if not line_support_pass:
            rejection_reasons.append("insufficient-supported-lines")
        if not curve_support_pass:
            rejection_reasons.append("missing-curve-evidence")
        gates.extend(
            [
                _calibration_quality_gate(
                    "supported-pitch-lines",
                    "Supported pitch markings",
                    "pass" if line_support_pass else "fail",
                    value=calibration.supported_lines,
                    threshold={"atLeast": 4},
                    reason=None if line_support_pass else "insufficient-supported-lines",
                ),
                _calibration_quality_gate(
                    "curve-evidence",
                    "Penalty arc or centre-circle evidence",
                    "pass" if curve_support_pass else "fail",
                    value=calibration.matched_curves,
                    threshold={"atLeast": 1},
                    reason=None if curve_support_pass else "missing-curve-evidence",
                ),
            ]
        )
    else:
        keypoint_pass = detected_keypoints >= 6
        inlier_pass = inlier_ratio is not None and inlier_ratio >= 0.65
        if not keypoint_pass:
            rejection_reasons.append("insufficient-detected-keypoints")
        if not inlier_pass:
            rejection_reasons.append("insufficient-keypoint-inlier-ratio")
        gates.extend(
            [
                _calibration_quality_gate(
                    "semantic-keypoints",
                    "Detected semantic pitch keypoints",
                    "pass" if keypoint_pass else "fail",
                    value=detected_keypoints,
                    threshold={"atLeast": 6},
                    reason=None if keypoint_pass else "insufficient-detected-keypoints",
                ),
                _calibration_quality_gate(
                    "keypoint-inlier-ratio",
                    "Semantic keypoint inlier ratio",
                    "pass" if inlier_pass else "fail",
                    value=round(float(inlier_ratio), 5) if inlier_ratio is not None else None,
                    threshold={"atLeast": 0.65},
                    reason=None if inlier_pass else "insufficient-keypoint-inlier-ratio",
                ),
            ]
        )

    person_support = None
    if people is not None and pitch is not None and len(people) >= 4:
        supported_people, total_people = _calibration_person_support(
            people,
            calibration,
            pitch,
        )
        support_ratio = supported_people / max(1, total_people)
        person_support = {
            "supported": supported_people,
            "total": total_people,
            "ratio": round(support_ratio, 3),
        }
        person_support_pass = supported_people >= 4 and support_ratio >= 0.55
        if not person_support_pass:
            rejection_reasons.append("insufficient-person-pitch-support")
        gates.append(
            _calibration_quality_gate(
                "person-pitch-support",
                "Detected people project inside the pitch",
                "pass" if person_support_pass else "fail",
                value=person_support,
                threshold={"supportedAtLeast": 4, "ratioAtLeast": 0.55},
                reason=None if person_support_pass else "insufficient-person-pitch-support",
            )
        )

    return {
        "rejectionReasons": list(dict.fromkeys(rejection_reasons)),
        "alignmentMetrics": alignment,
        "personSupport": person_support,
        "qualityGates": gates,
        "detectedKeypointCount": detected_keypoints,
        "inlierRatio": inlier_ratio,
    }


def _frame_calibration_evidence(
    scene: dict,
    sample_index: int,
    scene_time: float,
    image: np.ndarray,
    calibration: PitchCalibration | None,
    *,
    projection_source: str,
    people: list[Detection] | None = None,
    pitch: dict | None = None,
    source_frame_index: int | None = None,
    manual: bool = False,
) -> dict:
    if source_frame_index is None:
        frames = _frame_paths(scene)
        if 0 <= sample_index < len(frames):
            source_frame_index = _source_frame_index(frames[sample_index][0])
    source_frame_index = int(source_frame_index or 0)
    height, width = image.shape[:2]
    source_start = float(
        scene.get("payload", {}).get("videoAsset", {}).get("sourceStart") or 0.0
    )
    if calibration is None:
        return {
            "sourceFrameIndex": source_frame_index,
            "sampleIndex": sample_index,
            "sceneTime": round(float(scene_time), 3),
            "sourceTime": round(source_start + float(scene_time), 3),
            "frameWidth": width,
            "frameHeight": height,
            "status": "missing",
            "solutionStatus": "unresolved",
            "source": "none",
            "projectionSource": "none",
            "backend": None,
            "confidence": None,
            "confidenceKind": None,
            "imageToPitch": None,
            "keypointCount": 0,
            "detectedKeypointCount": 0,
            "completedKeypointCount": 0,
            "inlierCount": 0,
            "inlierRatio": None,
            "rawLineCount": 0,
            "rawKeypoints": [],
            "keypoints": [],
            "reprojectionError": None,
            "reprojectionP95": None,
            "groundErrorP50Metres": None,
            "groundErrorP95Metres": None,
            "visiblePitchSide": None,
            "rectangle": None,
            "alignmentMetrics": None,
            "horizon": None,
            "markings": [],
            "rejectionReasons": ["no-automatic-calibration-candidate"],
            "personSupport": None,
            "qualityGates": [],
        }

    qa = _direct_calibration_qa(
        image,
        calibration,
        people=people,
        pitch=pitch,
        manual=manual,
    )
    payload = calibration.as_dict()
    alignment = qa["alignmentMetrics"]
    accepted = not qa["rejectionReasons"]
    frame_reprojection_error = (
        alignment.get("residualP50")
        if alignment is not None
        else payload.get("reprojectionError")
    )
    frame_reprojection_p95 = (
        alignment.get("residualP95")
        if alignment is not None
        else payload.get("reprojectionP95") or payload.get("reprojectionError")
    )
    return {
        "sourceFrameIndex": source_frame_index,
        "sampleIndex": sample_index,
        "sceneTime": round(float(scene_time), 3),
        "sourceTime": round(source_start + float(scene_time), 3),
        "frameWidth": width,
        "frameHeight": height,
        "status": "accepted" if accepted else "rejected",
        "solutionStatus": "direct-accepted" if accepted else "direct-rejected",
        "source": calibration.method,
        "projectionSource": projection_source,
        "backend": calibration.method,
        "confidence": round(float(calibration.confidence), 3),
        "confidenceKind": payload.get("confidenceKind"),
        "backendDiagnostics": deepcopy(payload.get("backendDiagnostics")),
        "imageToPitch": _matrix_payload(calibration.image_to_pitch),
        "keypointCount": payload.get("keypointCount", 0),
        "detectedKeypointCount": payload.get("detectedKeypointCount", 0),
        "completedKeypointCount": payload.get("completedKeypointCount", 0),
        "inlierCount": payload.get("inlierCount", 0),
        "inlierRatio": payload.get("inlierRatio"),
        "rawLineCount": payload.get("rawLineCount", 0),
        "rawKeypoints": payload.get("rawKeypoints", []),
        "rawLines": semantic_line_evidence(calibration),
        "keypoints": _keypoint_evidence(calibration),
        "reprojectionError": frame_reprojection_error,
        "reprojectionP95": frame_reprojection_p95,
        "groundErrorP50Metres": payload.get("groundErrorP50Metres"),
        "groundErrorP95Metres": payload.get("groundErrorP95Metres"),
        "visiblePitchSide": pitch_side(calibration.rectangle),
        "rectangle": calibration.rectangle,
        "alignmentMetrics": alignment,
        "horizon": calibration_horizon(calibration, width),
        "markings": projected_pitch_markings(calibration, width, height),
        "rejectionReasons": qa["rejectionReasons"],
        "personSupport": qa["personSupport"],
        "qualityGates": qa["qualityGates"],
    }


def _calibration_attempt_payload(evidence: dict) -> dict:
    return {
        "backend": evidence.get("backend"),
        "status": evidence.get("status"),
        "confidence": evidence.get("confidence"),
        "reprojectionError": evidence.get("reprojectionError"),
        "reprojectionP95": evidence.get("reprojectionP95"),
        "visiblePitchSide": evidence.get("visiblePitchSide"),
        "rejectionReasons": list(evidence.get("rejectionReasons") or []),
        "backendDiagnostics": deepcopy(evidence.get("backendDiagnostics")),
    }


def _calibration_backend_rank(evidence: dict) -> float:
    backend = str(evidence.get("backend") or evidence.get("source") or "")
    if backend.startswith("pnlcalib"):
        return 3.0
    if "keypoint" in backend:
        return 2.0
    if backend == "pitch-lines-ransac":
        return 0.0
    return 1.0


def _calibration_evidence_rank(evidence: dict) -> tuple[float, float, float, float, float]:
    alignment = evidence.get("alignmentMetrics") or {}
    residual = evidence.get("reprojectionP95")
    return (
        1.0 if evidence.get("status") == "accepted" else 0.0,
        _calibration_backend_rank(evidence),
        float(alignment.get("f1") or 0.0),
        float(evidence.get("confidence") or 0.0),
        -float(residual) if residual is not None else -1e9,
    )


def _persist_frame_calibration_preview(scene: dict, evidence: dict) -> None:
    """Persist diagnostic evidence without changing reconstruction lifecycle state."""

    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    calibration_contract = dict(reconstruction.get("calibration") or {})
    calibration_contract.setdefault("schemaVersion", 1)
    previews = [
        deepcopy(item)
        for item in calibration_contract.get("framePreviews") or []
        if int(item.get("sourceFrameIndex") or -1)
        != int(evidence.get("sourceFrameIndex") or -2)
    ]
    persisted = {
        **deepcopy(evidence),
        "previewedAt": datetime.now(UTC).isoformat(),
    }
    previews.append(persisted)
    previews.sort(
        key=lambda item: (
            float(item.get("sceneTime") or 0.0),
            int(item.get("sourceFrameIndex") or 0),
        )
    )
    calibration_contract["framePreviews"] = previews[-240:]
    calibration_contract["lastFramePreview"] = persisted
    reconstruction["calibration"] = calibration_contract
    video["reconstruction"] = reconstruction
    scene_store.put(scene)


def _calibration_summary(frame_evidence: list[dict]) -> dict:
    total = len(frame_evidence)
    accepted = [item for item in frame_evidence if item.get("status") == "accepted"]
    rejected = [item for item in frame_evidence if item.get("status") == "rejected"]
    missing = [item for item in frame_evidence if item.get("status") == "missing"]
    direct = [
        item
        for item in accepted
        if item.get("projectionSource") in {"direct", "manual-direct"}
    ]
    temporal = [
        item
        for item in accepted
        if str(item.get("projectionSource") or "").startswith("temporal-")
    ]
    ambiguous = [
        item for item in frame_evidence if item.get("solutionStatus") == "ambiguous"
    ]
    temporal_uncertainties = sorted(
        float(value)
        for item in temporal
        if (
            value := (item.get("uncertainty") or {}).get("p95Metres")
            or item.get("positionUncertaintyMetres")
        )
        is not None
    )
    motion_edges = [
        item.get("cameraMotion") or {}
        for item in frame_evidence
        if (item.get("cameraMotion") or {}).get("status") != "first-frame"
    ]
    motion_estimated = sum(item.get("status") == "estimated" for item in motion_edges)
    motion_unreliable = sum(item.get("status") == "unreliable" for item in motion_edges)
    motion_cuts = sum(item.get("status") == "cut" for item in motion_edges)
    trackable_motion_edges = motion_estimated + motion_unreliable
    accepted_times = [float(item["sceneTime"]) for item in accepted]
    all_times = [float(item["sceneTime"]) for item in frame_evidence]
    max_gap = None
    if all_times:
        if accepted_times:
            gaps = [
                max(0.0, accepted_times[0] - all_times[0]),
                max(0.0, all_times[-1] - accepted_times[-1]),
                *(
                    accepted_times[index] - accepted_times[index - 1]
                    for index in range(1, len(accepted_times))
                ),
            ]
            max_gap = max(gaps)
        else:
            max_gap = max(0.0, all_times[-1] - all_times[0])
    median_errors = sorted(
        float(item["reprojectionError"])
        for item in accepted
        if item.get("reprojectionError") is not None
    )
    p95_errors = sorted(
        float(item.get("reprojectionP95") or item.get("reprojectionError"))
        for item in accepted
        if item.get("reprojectionP95") is not None or item.get("reprojectionError") is not None
    )
    alignment_f1 = sorted(
        float((item.get("alignmentMetrics") or {})["f1"])
        for item in accepted
        if (item.get("alignmentMetrics") or {}).get("f1") is not None
    )
    orientation_observations = direct
    if not orientation_observations:
        # Recovered frames inherit their anchor's rectangle and are therefore
        # not independent orientation votes. Keep one vote per anchor only for
        # legacy/manual evidence that has no explicit direct frame.
        seen_orientation_anchors: set[tuple] = set()
        orientation_observations = []
        for item in accepted:
            anchor_key = tuple(
                (item.get("temporal") or {}).get("anchorFrameIndices") or []
            ) or (str(item.get("projectionSource") or "unknown"),)
            if anchor_key in seen_orientation_anchors:
                continue
            seen_orientation_anchors.add(anchor_key)
            orientation_observations.append(item)
    known_sides = [
        str(item["visiblePitchSide"])
        for item in orientation_observations
        if item.get("visiblePitchSide") in {"left", "right"}
    ]
    side_counts = {side: known_sides.count(side) for side in {"left", "right"}}
    visible_side = max(side_counts, key=side_counts.get) if known_sides else None
    side_agreement = (
        side_counts[visible_side] / len(known_sides)
        if visible_side is not None
        else None
    )
    return {
        "sampledFrameCount": total,
        "acceptedFrameCount": len(accepted),
        "rejectedFrameCount": len(rejected),
        "missingFrameCount": len(missing),
        "directFrameCount": len(direct),
        "temporalRecoveredFrameCount": len(temporal),
        "temporalAmbiguousFrameCount": len(ambiguous),
        "directCoverage": round(len(direct) / total, 3) if total else 0.0,
        "usableCoverage": round(len(accepted) / total, 3) if total else 0.0,
        "maxGapSeconds": round(max_gap, 3) if max_gap is not None else None,
        "reprojectionP50": (
            round(float(np.percentile(median_errors, 50)), 3) if median_errors else None
        ),
        "reprojectionP95": (
            round(float(np.percentile(p95_errors, 95)), 3) if p95_errors else None
        ),
        "alignmentF1P10": (
            round(float(np.percentile(alignment_f1, 10)), 3) if alignment_f1 else None
        ),
        "visiblePitchSide": visible_side,
        "sideAgreement": round(side_agreement, 3) if side_agreement is not None else None,
        "sideVotes": side_counts,
        "temporalUncertaintyP95Metres": (
            round(float(np.percentile(temporal_uncertainties, 95)), 3)
            if temporal_uncertainties
            else None
        ),
        "cameraMotionReliability": (
            round(motion_estimated / trackable_motion_edges, 3)
            if trackable_motion_edges
            else None
        ),
        "cameraMotionEstimatedEdgeCount": motion_estimated,
        "cameraMotionUnreliableEdgeCount": motion_unreliable,
        "cameraMotionCutCount": motion_cuts,
    }


def _calibration_gate(
    gate_id: str,
    label: str,
    value: float | None,
    pass_limit: float,
    review_limit: float,
    *,
    higher_is_better: bool,
    unit: str,
    unavailable_status: str = "review",
) -> dict:
    if value is None:
        status = unavailable_status
    elif higher_is_better:
        status = "pass" if value >= pass_limit else "review" if value >= review_limit else "reject"
    else:
        status = "pass" if value <= pass_limit else "review" if value <= review_limit else "reject"
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "value": value,
        "unit": unit,
        "passThreshold": pass_limit,
        "reviewThreshold": review_limit,
        "higherIsBetter": higher_is_better,
    }


def _evaluate_calibration_quality(frame_evidence: list[dict]) -> dict:
    summary = _calibration_summary(frame_evidence)
    gates = [
        _calibration_gate(
            "calibration-coverage",
            "Usable calibrated frames",
            summary["usableCoverage"],
            CALIBRATION_PASS_COVERAGE,
            CALIBRATION_REVIEW_COVERAGE,
            higher_is_better=True,
            unit="ratio",
        ),
        _calibration_gate(
            "calibration-gap",
            "Longest gap between calibrated frames",
            summary["maxGapSeconds"],
            CALIBRATION_PASS_MAX_GAP_SECONDS,
            CALIBRATION_REVIEW_MAX_GAP_SECONDS,
            higher_is_better=False,
            unit="seconds",
        ),
        _calibration_gate(
            "reprojection-error",
            "Reprojection error p95",
            summary["reprojectionP95"],
            CALIBRATION_PASS_REPROJECTION_P95,
            CALIBRATION_SHOT_REVIEW_REPROJECTION_P95,
            higher_is_better=False,
            unit="pixels",
        ),
        _calibration_gate(
            "orientation-stability",
            "Visible-side agreement",
            summary["sideAgreement"],
            CALIBRATION_PASS_SIDE_AGREEMENT,
            CALIBRATION_REVIEW_SIDE_AGREEMENT,
            higher_is_better=True,
            unit="ratio",
            unavailable_status="not-applicable",
        ),
        _calibration_gate(
            "semantic-line-alignment",
            "Bidirectional semantic-line F1 p10",
            summary["alignmentF1P10"],
            0.15,
            0.08,
            higher_is_better=True,
            unit="ratio",
        ),
    ]
    if summary["temporalRecoveredFrameCount"]:
        gates.append(
            _calibration_gate(
                "temporal-uncertainty",
                "Recovered calibration uncertainty p95",
                summary["temporalUncertaintyP95Metres"],
                TEMPORAL_PASS_UNCERTAINTY_METRES,
                TEMPORAL_REVIEW_UNCERTAINTY_METRES,
                higher_is_better=False,
                unit="metres",
            )
        )
    ranked = {"pass": 0, "not-applicable": 0, "review": 1, "reject": 2}
    verdict = max(gates, key=lambda gate: ranked[gate["status"]])["status"]
    if verdict == "not-applicable":
        verdict = "pass"
    failed = [gate["id"] for gate in gates if gate["status"] in {"review", "reject"}]
    return {
        "schemaVersion": 1,
        "verdict": verdict,
        "summary": summary,
        "gates": gates,
        "failedGateIds": failed,
        "limitations": [
            "Uncertainty is an engineering estimate derived from image residuals, not a calibrated probability interval.",
            "Single-view ground projection does not recover player pose or airborne ball height.",
        ],
    }


def _person_detections(result) -> tuple[list[Detection], list[dict]]:
    image = result.orig_img
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, _ = image.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    people: list[tuple[float, tuple[float, float, float, float]]] = []
    balls: list[dict] = []

    for box, class_id, confidence in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = (float(value) for value in box)
        box_width, box_height = x2 - x1, y2 - y1
        center_x = (x1 + x2) / 2
        if class_id == 0:
            if not _is_pitch_person(hsv, (x1, y1, x2, y2), float(confidence)):
                continue
            people.append((float(confidence), (x1, y1, x2, y2)))
        elif class_id == 32:
            center_y = (y1 + y2) / 2
            radius = max(7, int(max(box_width, box_height) * 1.8))
            if center_y > height * 0.3 and max(box_width, box_height) < 24:
                context = _green_ratio(hsv, center_x, center_y, radius, radius)
                if context >= 0.24:
                    balls.append({"x": center_x, "y": center_y, "confidence": float(confidence)})

    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    for confidence, box in sorted(people, reverse=True):
        if all(_iou(box, existing) < PERSON_LOCAL_NMS_IOU for _, existing in kept):
            kept.append((confidence, box))

    detections = []
    for confidence, (x1, y1, x2, y2) in kept:
        detections.append(
            Detection(
                x=(x1 + x2) / 2,
                y=y2,
                width=x2 - x1,
                height=y2 - y1,
                confidence=confidence,
                feature=_appearance_feature(image, (x1, y1, x2, y2)),
            )
        )
    unique_balls: list[dict] = []
    for ball in sorted(balls, key=lambda item: item["confidence"], reverse=True):
        if any(hypot(ball["x"] - kept["x"], ball["y"] - kept["y"]) < 10.0 for kept in unique_balls):
            continue
        unique_balls.append(ball)
    return detections, unique_balls


def _scene_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    previous_hsv = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)
    current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    previous_hist = cv2.calcHist([previous_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    current_hist = cv2.calcHist([current_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(previous_hist, previous_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    cv2.normalize(current_hist, current_hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return float(cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA))


def _unreliable_motion(
    reason: str,
    scene_change_score: float,
    *,
    tracked_count: int = 0,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    residual_p50: float | None = None,
    residual_p95: float | None = None,
    forward_backward_p95: float | None = None,
    coverage_ratio: float = 0.0,
) -> CameraMotionEstimate:
    cut = scene_change_score > 0.18 and (
        tracked_count < 12
        or inlier_ratio < 0.20
        or (forward_backward_p95 is not None and forward_backward_p95 > 8.0)
    )
    return CameraMotionEstimate(
        matrix=np.eye(3, dtype=np.float64),
        status="cut" if cut else "unreliable",
        confidence=0.0,
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        residual_p50=residual_p50,
        residual_p95=residual_p95,
        forward_backward_p95=forward_backward_p95,
        coverage_ratio=coverage_ratio,
        scene_change_score=scene_change_score,
        reason=reason,
    )


def _camera_motion_estimate(previous: np.ndarray, current: np.ndarray) -> CameraMotionEstimate:
    """Estimate a QA-scored projective transform from current to previous.

    A successful static-camera estimate remains ``estimated`` even when its
    matrix is nearly identity. Failed flow and shot cuts are explicit graph
    barriers, so temporal calibration can never silently cross them.
    """

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    previous_hsv = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV)
    scene_change = _scene_change_score(previous, current)
    field_mask = cv2.inRange(previous_hsv, np.array([25, 35, 25]), np.array([100, 255, 255]))
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=500,
        qualityLevel=0.012,
        minDistance=7,
        mask=field_mask,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return _unreliable_motion(
            "insufficient-pitch-features",
            scene_change,
            tracked_count=0 if points is None else len(points),
        )
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    )
    if moved is None or status is None:
        return _unreliable_motion("forward-optical-flow-failed", scene_change)
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        moved,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01),
    )
    if returned is None or backward_status is None:
        return _unreliable_motion("backward-optical-flow-failed", scene_change)
    forward_valid = status.ravel() == 1
    backward_valid = backward_status.ravel() == 1
    forward_backward_error = np.linalg.norm(
        returned.reshape(-1, 2) - points.reshape(-1, 2),
        axis=1,
    )
    flow_candidates = (
        forward_valid & backward_valid & np.isfinite(forward_backward_error)
    )
    valid = flow_candidates & (forward_backward_error <= 2.5)
    previous_points = points.reshape(-1, 2)[valid]
    current_points = moved.reshape(-1, 2)[valid]
    tracked_count = len(previous_points)
    fb_p95 = (
        float(np.percentile(forward_backward_error[valid], 95))
        if int(valid.sum())
        else None
    )
    if tracked_count < 16:
        return _unreliable_motion(
            "insufficient-forward-backward-tracks",
            scene_change,
            tracked_count=tracked_count,
            forward_backward_p95=fb_p95,
        )
    height, width = previous_gray.shape
    coverage = float(cv2.contourArea(cv2.convexHull(previous_points.astype(np.float32)))) / max(
        1.0, float(width * height)
    )
    matrix, inliers = cv2.findHomography(
        current_points,
        previous_points,
        cv2.RANSAC,
        2.5,
        maxIters=2000,
        confidence=0.995,
    )
    if (
        matrix is None
        or inliers is None
        or not np.isfinite(matrix).all()
        or abs(float(matrix[2, 2])) < 1e-10
        or abs(float(np.linalg.det(matrix))) < 1e-10
    ):
        return _unreliable_motion(
            "projective-motion-fit-failed",
            scene_change,
            tracked_count=tracked_count,
            forward_backward_p95=fb_p95,
            coverage_ratio=coverage,
        )
    matrix /= matrix[2, 2]
    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / max(1, tracked_count)
    projected = cv2.perspectiveTransform(
        current_points[inlier_mask].reshape(-1, 1, 2).astype(np.float32),
        matrix,
    ).reshape(-1, 2)
    residuals = np.linalg.norm(projected - previous_points[inlier_mask], axis=1)
    residual_p50 = float(np.percentile(residuals, 50)) if len(residuals) else None
    residual_p95 = float(np.percentile(residuals, 95)) if len(residuals) else None
    corners = np.float32(
        [[[0.0, 0.0]], [[width, 0.0]], [[width, height]], [[0.0, height]]]
    )
    warped_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    warped_area = abs(float(cv2.contourArea(warped_corners.astype(np.float32))))
    area_ratio = warped_area / max(1.0, float(width * height))
    plausible_corners = bool(
        np.isfinite(warped_corners).all()
        and np.all(warped_corners[:, 0] > -width * 1.5)
        and np.all(warped_corners[:, 0] < width * 2.5)
        and np.all(warped_corners[:, 1] > -height * 1.5)
        and np.all(warped_corners[:, 1] < height * 2.5)
        and 0.40 <= area_ratio <= 2.50
    )
    rejection = None
    if inlier_count < 16:
        rejection = "insufficient-projective-inliers"
    elif inlier_ratio < 0.52:
        rejection = "projective-inlier-ratio-too-low"
    elif coverage < 0.02:
        rejection = "motion-features-too-concentrated"
    elif residual_p95 is None or residual_p95 > 3.5:
        rejection = "projective-motion-residual-too-high"
    elif fb_p95 is None:
        rejection = "forward-backward-flow-error-too-high"
    elif not plausible_corners:
        rejection = "implausible-projective-frame-warp"
    if rejection:
        return _unreliable_motion(
            rejection,
            scene_change,
            tracked_count=tracked_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            residual_p50=residual_p50,
            residual_p95=residual_p95,
            forward_backward_p95=fb_p95,
            coverage_ratio=coverage,
        )

    confidence = (
        0.30 * min(1.0, inlier_ratio / 0.80)
        + 0.20 * min(1.0, inlier_count / 60.0)
        + 0.20 * exp(-float(residual_p95) / 2.5)
        + 0.20 * exp(-float(fb_p95) / 1.5)
        + 0.10 * min(1.0, coverage / 0.12)
    )
    return CameraMotionEstimate(
        matrix=matrix,
        status="estimated",
        confidence=max(0.0, min(0.99, confidence)),
        tracked_count=tracked_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        residual_p50=residual_p50,
        residual_p95=residual_p95,
        forward_backward_p95=fb_p95,
        coverage_ratio=coverage,
        scene_change_score=scene_change,
    )


def _camera_step(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    estimate = _camera_motion_estimate(previous, current)
    return estimate.matrix if estimate.reliable else np.eye(3, dtype=np.float64)


def _stabilize_point(x: float, y: float, transform: np.ndarray) -> tuple[float, float]:
    projected = transform @ np.array([x, y, 1.0], dtype=np.float64)
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def _stabilize_detections(
    detections: list[Detection],
    balls: list[dict],
    transform: np.ndarray,
) -> None:
    for detection in detections:
        detection.x, detection.y = _stabilize_point(detection.x, detection.y, transform)
    for ball in balls:
        # Preserve detector-space coordinates for video overlays and expose the
        # compensated position separately for temporal association.  Mutating
        # x/y used to make the saved candidate impossible to audit against the
        # source frame.
        stable_x, stable_y = _stabilize_point(ball["x"], ball["y"], transform)
        ball["stabilizedX"] = stable_x
        ball["stabilizedY"] = stable_y


@dataclass(frozen=True)
class _DenseBallProjectionContext:
    """Auditable camera/calibration state for one dense ball frame."""

    calibration: PitchCalibration | None
    camera_transform: np.ndarray
    target_size: tuple[int, int]
    nearest_sample_index: int
    calibration_frame_index: int | None
    projection_source: str
    position_uncertainty_metres: float | None
    provenance: dict


def _normalise_interpolation_homography(
    matrix: np.ndarray,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    """Normalise and reject homographies that are unsafe at pitch-foot probes."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        return None, "matrix-shape-invalid"
    if not np.isfinite(value).all():
        return None, "matrix-non-finite"
    scale = float(value[2, 2])
    if abs(scale) < 1e-10:
        return None, "matrix-scale-degenerate"
    value = value / scale
    try:
        singular_values = np.linalg.svd(value, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, "matrix-svd-failed"
    if (
        not np.isfinite(singular_values).all()
        or float(singular_values[-1]) <= 1e-12 * max(1.0, float(singular_values[0]))
        or float(singular_values[0] / singular_values[-1]) > 1e12
    ):
        return None, "matrix-near-singular"

    width, height = frame_size
    probes = np.asarray(
        [
            [width * x_fraction, height * y_fraction, 1.0]
            for y_fraction in (0.58, 0.78, 0.96)
            for x_fraction in (0.08, 0.50, 0.92)
        ],
        dtype=np.float64,
    ).T
    projected = value @ probes
    denominator_scale = np.linalg.norm(value[2, :]) * np.linalg.norm(probes, axis=0)
    if np.any(
        np.abs(projected[2, :])
        <= 1e-9 * np.maximum(1.0, denominator_scale)
    ):
        return None, "matrix-probe-at-infinity"
    projected = projected[:2, :] / projected[2:3, :]
    if not np.isfinite(projected).all():
        return None, "matrix-projection-non-finite"
    return value, None


def _interpolate_homography_bounded(
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray | None, str | None]:
    """Linearly blend consistently scaled nearby H matrices, then revalidate."""

    if not 0.0 < alpha < 1.0:
        return None, "alpha-outside-open-interval"
    lower_value, lower_reason = _normalise_interpolation_homography(lower, frame_size)
    if lower_value is None:
        return None, f"lower-{lower_reason}"
    upper_value, upper_reason = _normalise_interpolation_homography(upper, frame_size)
    if upper_value is None:
        return None, f"upper-{upper_reason}"
    candidate = lower_value * (1.0 - alpha) + upper_value * alpha
    candidate, candidate_reason = _normalise_interpolation_homography(
        candidate,
        frame_size,
    )
    if candidate is None:
        return None, f"interpolated-{candidate_reason}"
    return candidate, None


def _dense_ball_projection_context(
    scene_time: float,
    sampled_times: list[float],
    frame_sizes: dict[int, tuple[int, int]],
    resolved_calibrations_by_sample: dict[int, PitchCalibration],
    calibration_anchor_by_sample: dict[int, int],
    calibration_uncertainty_by_sample: dict[int, float],
    frame_evidence: list[dict],
    camera_transforms: dict[int, np.ndarray],
    *,
    max_interpolation_gap_seconds: float = DENSE_BALL_INTERPOLATION_MAX_GAP_SECONDS,
) -> _DenseBallProjectionContext:
    """Choose an exact, interpolated, or explicitly nearest dense-frame state."""

    if not sampled_times:
        raise ValueError("Dense ball projection requires at least one sampled frame")
    if len(sampled_times) != len(frame_evidence):
        raise ValueError("Sample times and calibration evidence must have equal length")

    nearest_sample_index = min(
        range(len(sampled_times)),
        key=lambda index: (abs(float(sampled_times[index]) - scene_time), index),
    )

    def source_frame_index(sample_index: int) -> int:
        raw = frame_evidence[sample_index].get("sourceFrameIndex")
        return int(raw) if raw is not None else sample_index

    def accepted(sample_index: int) -> bool:
        return (
            frame_evidence[sample_index].get("status") == "accepted"
            and sample_index in resolved_calibrations_by_sample
        )

    def nearest_context(
        reason: str | None,
        attempted_sample_indices: list[int],
        alpha: float | None,
        *,
        exact: bool = False,
    ) -> _DenseBallProjectionContext:
        index = nearest_sample_index
        target_size = frame_sizes[index]
        evidence = frame_evidence[index]
        calibration = resolved_calibrations_by_sample.get(index) if accepted(index) else None
        calibration_validation_reason = None
        if calibration is not None:
            _, calibration_validation_reason = _normalise_interpolation_homography(
                calibration.image_to_pitch,
                target_size,
            )
            if calibration_validation_reason is not None:
                calibration = None
        else:
            calibration_validation_reason = "not-qa-accepted"

        transform = camera_transforms.get(source_frame_index(index))
        transform_method = "nearest-sample"
        transform_validation_reason = "matrix-missing" if transform is None else None
        if transform is not None:
            transform, transform_validation_reason = _normalise_interpolation_homography(
                transform,
                target_size,
            )
        if transform is None:
            transform = np.eye(3, dtype=np.float64)
            transform_method = "identity-fallback"

        base_uncertainty = calibration_uncertainty_by_sample.get(index)
        if base_uncertainty is None and calibration is not None:
            base_uncertainty = _calibration_uncertainty_metres(calibration)
        time_offset = abs(float(sampled_times[index]) - scene_time)
        uncertainty = (
            round(min(12.0, float(base_uncertainty) + time_offset * 2.0), 3)
            if base_uncertainty is not None
            else None
        )
        method = "exact-calibration-sample" if exact else "nearest-qa-sample-fallback"
        final_reason = reason
        if calibration_validation_reason is not None:
            final_reason = (
                f"nearest-calibration-{calibration_validation_reason}"
                if final_reason is None
                else f"{final_reason};nearest-calibration-{calibration_validation_reason}"
            )
        if transform_validation_reason is not None:
            final_reason = (
                f"nearest-camera-{transform_validation_reason}"
                if final_reason is None
                else f"{final_reason};nearest-camera-{transform_validation_reason}"
            )
        used_fallback = (
            not exact
            or calibration is None
            or transform_method == "identity-fallback"
        )
        provenance = {
            "method": method,
            "sampleIndices": attempted_sample_indices or [index],
            "sourceFrameIndices": [
                source_frame_index(sample_index)
                for sample_index in (attempted_sample_indices or [index])
            ],
            "alpha": round(float(alpha), 6) if alpha is not None else None,
            "nearestSampleIndex": index,
            "nearestSourceFrameIndex": source_frame_index(index),
            "sampleTime": round(float(sampled_times[index]), 6),
            "sceneTime": round(float(scene_time), 6),
            "timeOffsetSeconds": round(time_offset, 6),
            "fallback": used_fallback,
            "fallbackReason": final_reason,
            "calibrationMethod": (
                "exact-sample" if exact else "nearest-sample"
            ),
            "cameraTransformMethod": transform_method,
            "positionUncertaintyMetres": uncertainty,
        }
        return _DenseBallProjectionContext(
            calibration=calibration,
            camera_transform=transform,
            target_size=target_size,
            nearest_sample_index=index,
            calibration_frame_index=calibration_anchor_by_sample.get(index),
            projection_source=(
                str(evidence.get("projectionSource") or "none")
                if calibration is not None
                else "none"
            ),
            position_uncertainty_metres=uncertainty,
            provenance=provenance,
        )

    if any(
        float(sampled_times[index]) >= float(sampled_times[index + 1])
        for index in range(len(sampled_times) - 1)
    ):
        return nearest_context("sample-times-not-strictly-increasing", [], None)

    insertion_index = bisect_left(sampled_times, scene_time)
    exact_index = None
    for candidate_index in (insertion_index - 1, insertion_index):
        if (
            0 <= candidate_index < len(sampled_times)
            and abs(float(sampled_times[candidate_index]) - scene_time) <= 1e-6
        ):
            exact_index = candidate_index
            break
    if exact_index is not None:
        nearest_sample_index = exact_index
        return nearest_context(None, [exact_index], 0.0, exact=True)

    if insertion_index <= 0 or insertion_index >= len(sampled_times):
        return nearest_context("dense-frame-outside-sample-bracket", [], None)

    lower_index, upper_index = insertion_index - 1, insertion_index
    lower_time = float(sampled_times[lower_index])
    upper_time = float(sampled_times[upper_index])
    interval = upper_time - lower_time
    alpha = (scene_time - lower_time) / interval
    attempted_indices = [lower_index, upper_index]
    if interval > max_interpolation_gap_seconds:
        return nearest_context(
            "sample-bracket-exceeds-interpolation-bound",
            attempted_indices,
            alpha,
        )
    if not accepted(lower_index) or not accepted(upper_index):
        return nearest_context(
            "bracket-calibration-not-qa-accepted",
            attempted_indices,
            alpha,
        )
    lower_size, upper_size = frame_sizes[lower_index], frame_sizes[upper_index]
    if lower_size != upper_size:
        return nearest_context("bracket-frame-size-mismatch", attempted_indices, alpha)
    upper_motion = frame_evidence[upper_index].get("cameraMotion") or {}
    if upper_motion.get("status") != "estimated":
        return nearest_context(
            "bracket-camera-motion-edge-not-reliable",
            attempted_indices,
            alpha,
        )

    lower_calibration = resolved_calibrations_by_sample[lower_index]
    upper_calibration = resolved_calibrations_by_sample[upper_index]
    calibration_matrix, calibration_reason = _interpolate_homography_bounded(
        lower_calibration.image_to_pitch,
        upper_calibration.image_to_pitch,
        alpha,
        lower_size,
    )
    if calibration_matrix is None:
        return nearest_context(
            f"calibration-interpolation-{calibration_reason}",
            attempted_indices,
            alpha,
        )

    lower_transform = camera_transforms.get(source_frame_index(lower_index))
    upper_transform = camera_transforms.get(source_frame_index(upper_index))
    if lower_transform is None or upper_transform is None:
        return nearest_context(
            "camera-transform-endpoint-missing",
            attempted_indices,
            alpha,
        )
    camera_transform, camera_reason = _interpolate_homography_bounded(
        lower_transform,
        upper_transform,
        alpha,
        lower_size,
    )
    if camera_transform is None:
        return nearest_context(
            f"camera-transform-interpolation-{camera_reason}",
            attempted_indices,
            alpha,
        )

    nearest_endpoint = lower_index if alpha <= 0.5 else upper_index
    interpolated_calibration = replace(
        lower_calibration,
        image_to_pitch=calibration_matrix,
        confidence=(
            float(lower_calibration.confidence) * (1.0 - alpha)
            + float(upper_calibration.confidence) * alpha
        ),
        method="dense-bounded-bracket-interpolation",
        frame_index=source_frame_index(nearest_endpoint),
        confidence_kind="bounded-temporal-interpolation-score",
    )
    lower_uncertainty = float(
        calibration_uncertainty_by_sample.get(
            lower_index,
            _calibration_uncertainty_metres(lower_calibration),
        )
    )
    upper_uncertainty = float(
        calibration_uncertainty_by_sample.get(
            upper_index,
            _calibration_uncertainty_metres(upper_calibration),
        )
    )
    motion_confidence = max(0.0, min(1.0, float(upper_motion.get("confidence") or 0.0)))
    midpoint_weight = 4.0 * alpha * (1.0 - alpha)
    interpolation_penalty = (
        0.15
        + 0.60 * midpoint_weight * interval / max(1e-6, max_interpolation_gap_seconds)
        + 0.75 * (1.0 - motion_confidence)
    )
    uncertainty = round(
        min(
            12.0,
            lower_uncertainty * (1.0 - alpha)
            + upper_uncertainty * alpha
            + interpolation_penalty,
        ),
        3,
    )
    anchor_frame_indices = list(
        dict.fromkeys(
            frame_index
            for frame_index in (
                calibration_anchor_by_sample.get(lower_index),
                calibration_anchor_by_sample.get(upper_index),
            )
            if frame_index is not None
        )
    )
    provenance = {
        "method": "bounded-bracketing-homography-interpolation",
        "sampleIndices": attempted_indices,
        "sourceFrameIndices": [
            source_frame_index(lower_index),
            source_frame_index(upper_index),
        ],
        "anchorFrameIndices": anchor_frame_indices,
        "sampleTimes": [round(lower_time, 6), round(upper_time, 6)],
        "sceneTime": round(float(scene_time), 6),
        "alpha": round(float(alpha), 6),
        "intervalSeconds": round(interval, 6),
        "maxIntervalSeconds": round(max_interpolation_gap_seconds, 6),
        "nearestSampleIndex": nearest_endpoint,
        "fallback": False,
        "fallbackReason": None,
        "calibrationMethod": "normalised-matrix-linear-interpolation",
        "cameraTransformMethod": "normalised-matrix-linear-interpolation",
        "endpointProjectionSources": [
            str(frame_evidence[lower_index].get("projectionSource") or "none"),
            str(frame_evidence[upper_index].get("projectionSource") or "none"),
        ],
        "motionConfidence": round(motion_confidence, 6),
        "positionUncertaintyMetres": uncertainty,
    }
    return _DenseBallProjectionContext(
        calibration=interpolated_calibration,
        camera_transform=camera_transform,
        target_size=lower_size,
        nearest_sample_index=nearest_endpoint,
        calibration_frame_index=calibration_anchor_by_sample.get(nearest_endpoint),
        projection_source="dense-bracket-interpolated",
        position_uncertainty_metres=uncertainty,
        provenance=provenance,
    )


def _apply_dense_ball_projection(
    balls: list[dict],
    context: _DenseBallProjectionContext,
    pitch: dict,
    dense_frame_index: int,
) -> int:
    """Scale, project, and stabilise candidates while retaining full provenance."""

    target_width, target_height = context.target_size
    for ball in balls:
        source_width = float(ball.get("imageWidth") or target_width)
        source_height = float(ball.get("imageHeight") or target_height)
        source_x, source_y = float(ball["x"]), float(ball["y"])
        ball["sourceImagePosition"] = {
            "x": source_x,
            "y": source_y,
            "width": source_width,
            "height": source_height,
        }
        ball["x"] = source_x * target_width / max(1.0, source_width)
        ball["y"] = source_y * target_height / max(1.0, source_height)
        ball.pop("pitchX", None)
        ball.pop("pitchZ", None)
        ball["nearestCalibrationSampleIndex"] = context.nearest_sample_index
        ball["calibrationSampleIndices"] = list(
            context.provenance.get("sampleIndices") or []
        )
        ball["calibrationInterpolationAlpha"] = context.provenance.get("alpha")
        ball["calibrationProjectionMethod"] = context.provenance.get("method")
        ball["projectionProvenance"] = deepcopy(context.provenance)
        provenance = ball.get("provenance")
        provenance = dict(provenance) if isinstance(provenance, dict) else {}
        provenance["projection"] = deepcopy(context.provenance)
        ball["provenance"] = provenance
        ball["denseFrameIndex"] = dense_frame_index

    _attach_metric_positions(
        [],
        balls,
        context.calibration,
        pitch,
        projection_source=context.projection_source,
        calibration_frame_index=context.calibration_frame_index,
        position_uncertainty_metres=context.position_uncertainty_metres,
    )
    _stabilize_detections([], balls, context.camera_transform)
    return sum(ball.get("pitchX") is not None for ball in balls)


def _capture_detection_observations(
    detections: list[Detection],
    source_frame_index: int,
) -> None:
    """Freeze detector-space evidence before camera stabilization mutates x/y.

    Automatic observation identifiers are content-addressed. Detector result
    ordering is not an identity signal and may change across providers or
    rebuilds, so an array index must never become a correction/cache key.
    """

    generated_ids: set[str] = set()
    for detection in detections:
        detection.source_frame_index = int(source_frame_index)
        detection.image_x = float(detection.x)
        detection.image_y = float(detection.y)
        if not detection.observation_id:
            if detection.annotation_id:
                stable_source = f"annotation-{detection.annotation_id}"
            else:
                feature = np.asarray(detection.feature, dtype=np.float32).reshape(-1)
                fingerprint = "|".join(
                    [
                        "person-observation-v2",
                        str(int(source_frame_index)),
                        f"{float(detection.image_x):.6f}",
                        f"{float(detection.image_y):.6f}",
                        f"{float(detection.width):.6f}",
                        f"{float(detection.height):.6f}",
                        f"{float(detection.confidence):.6f}",
                        feature.tobytes().hex(),
                    ]
                )
                digest = sha256(fingerprint.encode("utf-8")).hexdigest()[:20]
                stable_source = f"observation-{digest}"
            observation_id = f"frame-{int(source_frame_index):06d}:{stable_source}"
            if observation_id in generated_ids:
                raise ReconstructionError(
                    "Detector produced indistinguishable duplicate person observations; "
                    "identity evidence was rejected instead of assigning order-based IDs"
                )
            detection.observation_id = observation_id
        if str(detection.observation_id) in generated_ids:
            raise ReconstructionError(
                f"Duplicate person observationId: {detection.observation_id}"
            )
        generated_ids.add(str(detection.observation_id))


def _identity_embedding_requests(
    frames: list[tuple[Path, float]],
    person_frames: list[tuple[list[Detection], float]],
) -> list[tuple[int, Path, list[dict]]]:
    requests: list[tuple[int, Path, list[dict]]] = []
    for (path, _), (people, _) in zip(frames, person_frames):
        observations = []
        for person in people:
            if not person.observation_id:
                continue
            image_x = person.image_x if person.image_x is not None else person.x
            image_y = person.image_y if person.image_y is not None else person.y
            observations.append(
                {
                    "observationId": person.observation_id,
                    "bbox": {
                        "x": float(image_x) - person.width / 2,
                        "y": float(image_y) - person.height,
                        "width": person.width,
                        "height": person.height,
                    },
                }
            )
        if observations:
            requests.append((_source_frame_index(path), path, observations))
    return requests


def _attach_identity_embeddings(
    person_frames: list[tuple[list[Detection], float]],
    results: dict[str, dict],
) -> dict:
    requested = usable = rejected = 0
    provider = model_version = None
    role_observations = 0
    crop_diagnostics: list[dict] = []
    for observation_id, item in sorted(results.items()):
        is_usable = item.get("usable") is True
        crop_diagnostics.append(
            {
                "observationId": str(observation_id),
                "frameIndex": item.get("frameIndex"),
                "status": "usable" if is_usable else "rejected",
                "usable": is_usable,
                "quality": deepcopy(item.get("quality") or {}),
                "rejectionReasons": list(item.get("rejectionReasons") or []),
                "evidenceFingerprint": item.get("evidenceFingerprint"),
                "provider": item.get("provider"),
                "modelVersion": item.get("modelVersion"),
                "role": item.get("role") if is_usable else None,
                "roleConfidence": item.get("roleConfidence") if is_usable else None,
            }
        )
    for people, _ in person_frames:
        for person in people:
            item = results.get(str(person.observation_id or ""))
            if item is None:
                continue
            requested += 1
            provider = item.get("provider") or provider
            model_version = item.get("modelVersion") or model_version
            person.reid_quality = deepcopy(item.get("quality") or {})
            person.reid_evidence_fingerprint = str(item.get("evidenceFingerprint") or "") or None
            if item.get("usable") is not True:
                rejected += 1
                continue
            vector = np.asarray(item.get("embedding"), dtype=np.float32)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                rejected += 1
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                rejected += 1
                continue
            person.reid_feature = vector / norm
            person.reid_role = item.get("role")
            person.reid_role_confidence = item.get("roleConfidence")
            role_observations += int(person.reid_role is not None)
            usable += 1
    worker_diagnostics = deepcopy(getattr(results, "diagnostics", {}) or {})
    model_contract = worker_diagnostics.get("modelContract")
    if isinstance(model_contract, dict):
        provider = model_contract.get("backend") or provider
        model_version = model_contract.get("modelVersion") or model_version
    return {
        "status": "ready" if requested else "no-observations",
        "provider": provider,
        "modelVersion": model_version,
        **(
            {"modelContract": deepcopy(model_contract)}
            if isinstance(model_contract, dict)
            else {}
        ),
        "requestedObservationCount": requested,
        "usableObservationCount": usable,
        "rejectedObservationCount": rejected,
        "roleObservationCount": role_observations,
        "usableCropRatio": round(usable / max(1, requested), 3),
        "cacheHitCount": int(worker_diagnostics.get("cacheHitCount") or 0),
        "cacheMissCount": int(worker_diagnostics.get("cacheMissCount") or 0),
        "deduplicatedObservationCount": int(
            worker_diagnostics.get("deduplicatedObservationCount") or 0
        ),
        "uniqueEvidenceFingerprintCount": int(
            worker_diagnostics.get("uniqueEvidenceFingerprintCount") or 0
        ),
        "duplicateEvidenceFingerprintCount": int(
            worker_diagnostics.get("duplicateEvidenceFingerprintCount") or 0
        ),
        "concurrentDeduplicatedCount": int(
            worker_diagnostics.get("concurrentDeduplicatedCount") or 0
        ),
        "providerInferenceCount": int(
            worker_diagnostics.get("providerInferenceCount") or 0
        ),
        "crops": crop_diagnostics,
        "corruptCacheMissCount": int(
            worker_diagnostics.get("corruptCacheMissCount") or 0
        ),
        "expiredCacheMissCount": int(
            worker_diagnostics.get("expiredCacheMissCount") or 0
        ),
        **({"cache": worker_diagnostics["cache"]} if "cache" in worker_diagnostics else {}),
    }


JERSEY_OCR_PRE_RESOLVER_MAX_SELECTED_FRAMES = 5
JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION = 5
JERSEY_OCR_MIN_CROP_GAP_SECONDS = 0.45
JERSEY_OCR_FUSION_CONFIG = JerseyFusionConfig(
    # The worker already rejects unreadable crops. Keep a low-confidence top
    # candidate visible as provisional evidence, while publication still
    # requires >=2 samples and the unchanged 0.80 reliable threshold.
    min_ocr_confidence=0.01,
    min_frame_quality=0.0,
    min_back_visibility=0.0,
    min_effective_score=0.01,
)
JERSEY_OCR_PRE_RESOLVER_FUSION_CONFIG = replace(
    JERSEY_OCR_FUSION_CONFIG,
    max_selected_frames=JERSEY_OCR_PRE_RESOLVER_MAX_SELECTED_FRAMES,
)


def _jersey_crop_point_quality(point: dict) -> float:
    """Cheap pre-OCR quality prior for choosing bounded player crops."""

    bbox = point.get("bbox") or {}
    width = max(0.0, float(bbox.get("width") or 0.0))
    height = max(0.0, float(bbox.get("height") or 0.0))
    confidence = max(0.0, min(1.0, float(point.get("confidence") or 0.0)))
    height_score = min(1.0, height / 96.0)
    width_score = min(1.0, width / 40.0)
    aspect = width / max(1.0, height)
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.42) / 0.42)
    return max(
        0.0,
        min(
            1.0,
            0.50 * confidence
            + 0.25 * height_score
            + 0.15 * width_score
            + 0.10 * aspect_score,
        ),
    )


def _select_jersey_crop_points(
    track: TrackState,
    available_frame_indices: set[int],
    prospective_split_ranges: tuple[tuple[float, float], ...] = (),
) -> tuple[list[tuple[dict, float]], int, int]:
    """Select a bounded crop reservoir with coverage for every known partition.

    A manual split is applied after the global identity resolver.  Selecting a
    global top-N here can therefore discard every readable shirt view from a
    later split partition.  Each persisted split range and its remainder get
    their own quality-ranked, temporally diverse reservoir.  This preserves
    partition-local evidence without sending every 10 FPS player crop to the
    OCR worker.  Final temporal sampling/fusion still runs after ownership.
    """

    role = _annotation_role(track.manual_kind) or track.role
    if role in {"referee", "other"}:
        return [], 0, 0
    candidates: list[tuple[dict, float]] = []
    for point in track.points:
        frame_index = point.get("frameIndex")
        bbox = point.get("bbox") or {}
        bbox_values = tuple(
            float(bbox.get(key) or 0.0)
            for key in ("x", "y", "width", "height")
        )
        if (
            frame_index is None
            or int(frame_index) not in available_frame_indices
            or not all(isfinite(value) for value in bbox_values)
            or bbox_values[2] <= 0.0
            or bbox_values[3] <= 0.0
        ):
            continue
        candidates.append((point, _jersey_crop_point_quality(point)))
    partitions: dict[tuple[int, ...], list[tuple[dict, float]]] = {}
    for candidate in candidates:
        timestamp = float(candidate[0].get("t") or 0.0)
        membership = tuple(
            index
            for index, (start, end) in enumerate(prospective_split_ranges)
            if start <= timestamp < end
        )
        partitions.setdefault(membership, []).append(candidate)

    selected: list[tuple[dict, float]] = []
    for partition in partitions.values():
        partition.sort(
            key=lambda item: (
                -item[1],
                -float(item[0].get("confidence") or 0.0),
                float(item[0].get("t") or 0.0),
                int(item[0].get("frameIndex") or 0),
                str(item[0].get("observationId") or ""),
            )
        )
        partition_selected: list[tuple[dict, float]] = []
        for candidate in partition:
            candidate_time = float(candidate[0].get("t") or 0.0)
            if any(
                abs(candidate_time - float(item[0].get("t") or 0.0))
                < JERSEY_OCR_MIN_CROP_GAP_SECONDS
                for item in partition_selected
            ):
                continue
            partition_selected.append(candidate)
            if (
                len(partition_selected)
                >= JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION
            ):
                break
        selected.extend(partition_selected)

    return (
        sorted(
            selected,
            key=lambda item: (
                float(item[0].get("t") or 0.0),
                int(item[0].get("frameIndex") or 0),
                str(item[0].get("observationId") or ""),
            ),
        ),
        len(candidates),
        len(partitions),
    )


def _prospective_jersey_split_ranges(
    track: TrackState,
    scene: dict | None,
) -> tuple[tuple[float, float], ...]:
    """Return persisted split ranges that can own observations of ``track``."""

    if not scene:
        return ()
    observation_ids = {
        str(point.get("observationId"))
        for point in track.points
        if point.get("observationId")
    }
    source_tracklet_ids = set(track.source_tracklet_ids or {track.local_tracklet_id})
    source_tracklet_ids.add(track.local_tracklet_id)
    previous_by_id = {
        str(person.get("canonicalPersonId") or person.get("id")): person
        for person in _previous_canonical_people(scene)
        if person.get("canonicalPersonId") or person.get("id")
    }
    ranges: set[tuple[float, float]] = set()
    for annotation in _identity_annotations(scene):
        time_range = _split_range(annotation)
        if time_range is None:
            continue
        target_id = str(annotation.get("targetObservationId") or "")
        source_id = _annotation_source_identity(annotation)
        relevant = target_id in observation_ids
        snapshot = annotation.get("targetObservation") or {}
        snapshot_bbox = snapshot.get("bbox") or {}
        if (
            not relevant
            and snapshot.get("frameIndex") is not None
            and all(
                snapshot_bbox.get(key) is not None
                for key in ("x", "y", "width", "height")
            )
        ):
            frame_index = int(snapshot["frameIndex"])
            target_box = (
                float(snapshot_bbox["x"]),
                float(snapshot_bbox["y"]),
                float(snapshot_bbox["x"]) + float(snapshot_bbox["width"]),
                float(snapshot_bbox["y"]) + float(snapshot_bbox["height"]),
            )
            for point in track.points:
                bbox = point.get("bbox") or {}
                if int(point.get("frameIndex", -1)) != frame_index or not bbox:
                    continue
                box = (
                    float(bbox["x"]),
                    float(bbox["y"]),
                    float(bbox["x"]) + float(bbox["width"]),
                    float(bbox["y"]) + float(bbox["height"]),
                )
                scale = max(
                    1.0,
                    min(float(snapshot_bbox["height"]), float(bbox["height"])),
                )
                normalized_center = hypot(
                    (box[0] + box[2] - target_box[0] - target_box[2]) / 2.0,
                    (box[1] + box[3] - target_box[1] - target_box[3]) / 2.0,
                ) / scale
                if _iou(target_box, box) >= 0.50 and normalized_center <= 0.50:
                    # Mirror the later fail-closed split remap.  If more than
                    # one track is geometrically viable they each get a small
                    # bounded reservoir; the split stage will still reject the
                    # ambiguity rather than choose silently.
                    relevant = True
                    break
        if source_id and source_id == track.canonical_person_id:
            relevant = True
        previous = previous_by_id.get(str(source_id or ""))
        if previous is not None:
            previous_observation_ids = {
                str(item.get("observationId") or item.get("id"))
                for item in previous.get("observations") or []
                if item.get("observationId") or item.get("id")
            }
            previous_tracklet_ids = {
                str(item)
                for item in (
                    previous.get("sourceTrackletIds")
                    or previous.get("memberTrackletIds")
                    or []
                )
            }
            relevant = relevant or bool(observation_ids & previous_observation_ids)
            relevant = relevant or bool(source_tracklet_ids & previous_tracklet_ids)
        if relevant:
            ranges.add(time_range)
    return tuple(sorted(ranges))


def _low_confidence_jersey_candidate(item: dict) -> tuple[str | None, float]:
    candidates = [
        candidate
        for candidate in item.get("candidates") or []
        if candidate.get("number") is not None
        and candidate.get("confidence") is not None
    ]
    if not candidates:
        return None, 0.0
    best = max(
        candidates,
        key=lambda candidate: (
            float(candidate["confidence"]),
            str(candidate["number"]),
        ),
    )
    return str(best["number"]), float(best["confidence"])


def _run_jersey_ocr_for_tracklets(
    tracks: list[TrackState],
    frames: list[tuple[Path, float]],
    on_progress: Callable[[int, int, int], None] | None = None,
    *,
    scene: dict | None = None,
) -> tuple[dict[str, JerseyEvidenceSummary], dict, list[str]]:
    """Extract optional OCR evidence without making reconstruction depend on it."""

    readiness = jersey_ocr_worker_readiness(timeout=2.0)
    diagnostics = {
        "schemaVersion": 1,
        **deepcopy(readiness),
        "requestedTrackletCount": len(tracks),
        "eligibleTrackletCount": 0,
        "candidateCropCount": 0,
        "selectedCropCount": 0,
        "selectionPartitionCount": 0,
        "prospectiveSplitRangeCount": 0,
        "submittedCropCount": 0,
        "cropReadFailureCount": 0,
        "recognizedCropCount": 0,
        "lowConfidenceCropCount": 0,
        "ambiguousCropCount": 0,
        "rejectedCropCount": 0,
        "backVisibilityAvailable": False,
        "cropCandidatePolicy": "bounded-per-prospective-partition-v2",
        "maxCropsPerProspectivePartition": (
            JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION
        ),
        "preResolverMaxSelectedFrames": (
            JERSEY_OCR_PRE_RESOLVER_MAX_SELECTED_FRAMES
        ),
        "trackletEvidence": {},
        "crops": [],
    }
    if readiness.get("status") != "ready":
        warnings = []
        if readiness.get("status") not in {"disabled", "no-observations"}:
            warnings.append(
                "Jersey OCR is unavailable; reconstruction continued without shirt-number identity evidence."
            )
        return {}, diagnostics, warnings

    frame_by_index = {
        _source_frame_index(Path(path)): Path(path)
        for path, _ in frames
    }
    selected: list[tuple[str, dict, float]] = []
    for track in tracks:
        tracklet_id = track.local_tracklet_id
        prospective_ranges = _prospective_jersey_split_ranges(track, scene)
        points, candidate_count, partition_count = _select_jersey_crop_points(
            track,
            set(frame_by_index),
            prospective_ranges,
        )
        diagnostics["candidateCropCount"] += candidate_count
        diagnostics["selectionPartitionCount"] += partition_count
        diagnostics["prospectiveSplitRangeCount"] += len(prospective_ranges)
        if candidate_count:
            diagnostics["eligibleTrackletCount"] += 1
        selected.extend((tracklet_id, point, quality) for point, quality in points)
    diagnostics["selectedCropCount"] = len(selected)
    if not selected:
        diagnostics["status"] = "no-crops"
        return {}, diagnostics, []

    requests: list[JerseyCropRequest] = []
    request_metadata: dict[str, dict] = {}
    image_cache: dict[int, np.ndarray | None] = {}
    with TemporaryDirectory(prefix="replay-jersey-ocr-") as directory:
        crop_root = Path(directory)
        for request_index, (tracklet_id, point, selection_quality) in enumerate(selected):
            frame_index = int(point["frameIndex"])
            if frame_index not in image_cache:
                image_cache[frame_index] = cv2.imread(str(frame_by_index[frame_index]))
            image = image_cache[frame_index]
            bbox = point["bbox"]
            if image is None:
                diagnostics["cropReadFailureCount"] += 1
                continue
            image_height, image_width = image.shape[:2]
            x1 = max(0, min(image_width, int(np.floor(float(bbox["x"])))))
            y1 = max(0, min(image_height, int(np.floor(float(bbox["y"])))))
            x2 = max(0, min(image_width, int(np.ceil(float(bbox["x"]) + float(bbox["width"])))))
            y2 = max(0, min(image_height, int(np.ceil(float(bbox["y"]) + float(bbox["height"])))))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                diagnostics["cropReadFailureCount"] += 1
                continue
            observation_id = str(
                point.get("observationId") or f"{tracklet_id}:{frame_index}"
            )
            crop_id = (
                "jersey-"
                + sha256(
                    f"{tracklet_id}:{observation_id}:{frame_index}".encode("utf-8")
                ).hexdigest()[:16]
            )
            crop_path = crop_root / f"crop-{request_index:04d}.jpg"
            if not cv2.imwrite(str(crop_path), crop):
                diagnostics["cropReadFailureCount"] += 1
                continue
            request = JerseyCropRequest(
                crop_id=crop_id,
                path=crop_path,
                observation_id=observation_id,
                tracklet_id=tracklet_id,
                frame_index=frame_index,
                timestamp=float(point.get("t") or 0.0),
            )
            requests.append(request)
            request_metadata[crop_id] = {
                "selectionQuality": round(float(selection_quality), 6),
                "clippedCropRatio": round(
                    (max(0, x2 - x1) * max(0, y2 - y1))
                    / max(1.0, float(bbox["width"]) * float(bbox["height"])),
                    6,
                ),
            }
        diagnostics["submittedCropCount"] = len(requests)
        if not requests:
            diagnostics["status"] = "no-readable-crops"
            return {}, diagnostics, []
        try:
            worker_results = analyze_jersey_crops(requests, on_progress)
        except JerseyOcrWorkerError as exc:
            diagnostics.update({"status": "failed", "detail": str(exc)})
            return (
                {},
                diagnostics,
                [
                    "Jersey OCR failed during crop analysis; reconstruction continued without shirt-number identity evidence."
                ],
            )

    worker_cache_diagnostics = deepcopy(
        getattr(worker_results, "diagnostics", {}) or {}
    )
    worker_model_contract = worker_cache_diagnostics.get("modelContract")

    observations: list[JerseyOcrObservation] = []
    requests_by_id = {request.crop_id: request for request in requests}
    status_counts: dict[str, int] = {}
    for crop_id, item in sorted(worker_results.items()):
        request = requests_by_id[crop_id]
        status = str(item.get("status") or "rejected")
        status_counts[status] = status_counts.get(status, 0) + 1
        raw_number: str | None = None
        ocr_confidence = 0.0
        if status == "recognized":
            raw_number = str(item["number"])
            ocr_confidence = float(item["confidence"])
        elif status == "low-confidence":
            raw_number, ocr_confidence = _low_confidence_jersey_candidate(item)
        frame_quality = 1.0 if item.get("usable") is not False else 0.0
        back_visibility = 1.0
        source = str(item.get("provider") or "jersey-ocr-worker")
        observations.append(
            JerseyOcrObservation(
                id=crop_id,
                tracklet_id=str(request.tracklet_id),
                timestamp_seconds=float(request.timestamp or 0.0),
                raw_number=raw_number,
                ocr_confidence=ocr_confidence,
                # Worker-side crop QA is the authoritative readability gate.
                frame_quality=frame_quality,
                # No reliable front/back classifier exists in v1. Keep this
                # neutral and make the missing signal explicit in diagnostics.
                back_visibility=back_visibility,
                frame_index=request.frame_index,
                source=source,
                evidence_fingerprint=str(item.get("evidenceFingerprint") or "") or None,
            )
        )
        diagnostics["crops"].append(
            {
                "cropId": crop_id,
                "observationId": request.observation_id,
                "trackletId": request.tracklet_id,
                "frameIndex": request.frame_index,
                "timestamp": request.timestamp,
                "status": status,
                # Normalized raw evidence is deliberately retained even when
                # the pre-resolver top-N does not select this crop.  A manual
                # split can later give it a different final owner.
                "rawNumber": raw_number,
                "ocrConfidence": round(float(ocr_confidence), 6),
                "frameQuality": frame_quality,
                "backVisibility": back_visibility,
                "source": source,
                "evidenceFingerprint": item.get("evidenceFingerprint"),
                "number": item.get("number"),
                "confidence": item.get("confidence"),
                "candidates": deepcopy(item.get("candidates") or []),
                "quality": deepcopy(item.get("quality") or {}),
                "rejectionReasons": list(item.get("rejectionReasons") or []),
                "decisionReasons": list(item.get("decisionReasons") or []),
                **request_metadata[crop_id],
            }
        )

    summaries = aggregate_tracklets(
        observations,
        config=JERSEY_OCR_PRE_RESOLVER_FUSION_CONFIG,
    )
    diagnostics.update(
        {
            "status": "ready",
            "provider": (
                worker_model_contract.get("backend")
                if isinstance(worker_model_contract, dict)
                else next(
                    (
                        item.get("provider")
                        for item in worker_results.values()
                        if item.get("provider")
                    ),
                    readiness.get("backend"),
                )
            ),
            "modelVersion": (
                worker_model_contract.get("modelVersion")
                if isinstance(worker_model_contract, dict)
                else next(
                    (
                        item.get("modelVersion")
                        for item in worker_results.values()
                        if item.get("modelVersion")
                    ),
                    readiness.get("modelVersion"),
                )
            ),
            **(
                {"modelContract": deepcopy(worker_model_contract)}
                if isinstance(worker_model_contract, dict)
                else {}
            ),
            "recognizedCropCount": status_counts.get("recognized", 0),
            "lowConfidenceCropCount": status_counts.get("low-confidence", 0),
            "ambiguousCropCount": status_counts.get("ambiguous", 0),
            "rejectedCropCount": status_counts.get("rejected", 0),
            "noNumberCropCount": status_counts.get("no-number", 0),
            "rawObservationCount": len(observations),
            "rawUsableObservationCount": sum(
                observation.raw_number is not None
                and observation.frame_quality > 0.0
                for observation in observations
            ),
            "preResolverSelectedCropCount": sum(
                summary.selected_sample_count for summary in summaries.values()
            ),
            "reliableTrackletCount": sum(
                summary.status == "reliable" for summary in summaries.values()
            ),
            "provisionalTrackletCount": sum(
                summary.status == "provisional" for summary in summaries.values()
            ),
            "conflictingTrackletCount": sum(
                summary.status == "conflict" for summary in summaries.values()
            ),
            "trackletEvidence": {
                tracklet_id: summary.to_payload()
                for tracklet_id, summary in sorted(summaries.items())
            },
            "cacheHitCount": int(worker_cache_diagnostics.get("cacheHitCount") or 0),
            "providerInferenceCropCount": int(
                worker_cache_diagnostics.get("providerInferenceCropCount") or 0
            ),
            "requestDeduplicatedCount": int(
                worker_cache_diagnostics.get("requestDeduplicatedCount") or 0
            ),
            "uniqueEvidenceFingerprintCount": int(
                worker_cache_diagnostics.get("uniqueEvidenceFingerprintCount") or 0
            ),
            "duplicateEvidenceFingerprintCount": int(
                worker_cache_diagnostics.get("duplicateEvidenceFingerprintCount") or 0
            ),
            "cacheEnabled": worker_cache_diagnostics.get("cacheEnabled"),
        }
    )
    return summaries, diagnostics, []


def _aggregate_jersey_evidence_for_final_tracks(
    tracks: list[TrackState],
    tracklet_evidence: Mapping[str, JerseyEvidenceSummary],
    diagnostics: dict,
) -> tuple[dict[str, JerseyEvidenceSummary], dict]:
    """Reassign raw OCR evidence to final manual-split/merged identities.

    OCR is executed before the global resolver, while a manual split is
    intentionally applied afterwards as a cannot-link barrier.  Aggregating by
    the old source tracklet would leak a shirt reading from the split range
    back into the remaining person.  Reusing only the pre-resolver summary is
    also unsafe: its top-N may contain no crop owned by the new partition.  The
    immutable video observation ID is the safe bridge between these stages,
    and temporal sampling/fusion therefore runs again after ownership.
    """

    crop_to_observation = {
        str(item.get("cropId")): str(item.get("observationId"))
        for item in diagnostics.get("crops") or []
        if item.get("cropId") and item.get("observationId")
    }
    owners: dict[str, set[str]] = {}
    tracklet_owners: dict[str, set[str]] = {}
    for track in tracks:
        canonical_id = str(track.canonical_person_id or "")
        if not canonical_id:
            continue
        source_tracklet_ids = set(track.source_tracklet_ids)
        for point in track.points:
            observation_id = str(point.get("observationId") or "")
            if observation_id:
                owners.setdefault(observation_id, set()).add(canonical_id)
            source_tracklet_id = str(point.get("sourceTrackletId") or "").strip()
            if source_tracklet_id:
                source_tracklet_ids.add(source_tracklet_id)
        if not source_tracklet_ids:
            source_tracklet_ids.add(track.local_tracklet_id)
        for source_tracklet_id in source_tracklet_ids:
            tracklet_owners.setdefault(source_tracklet_id, set()).add(canonical_id)

    raw_rows = [
        item
        for item in diagnostics.get("crops") or []
        if "ocrConfidence" in item and item.get("cropId") and item.get("trackletId")
    ]
    source_observations: list[JerseyOcrObservation] = []
    invalid_raw_crop_ids: list[str] = []
    for item in raw_rows:
        crop_id = str(item["cropId"])
        try:
            source_observations.append(
                JerseyOcrObservation(
                    id=crop_id,
                    tracklet_id=str(item["trackletId"]),
                    timestamp_seconds=float(item.get("timestamp") or 0.0),
                    raw_number=item.get("rawNumber"),
                    ocr_confidence=float(item.get("ocrConfidence") or 0.0),
                    frame_quality=float(item.get("frameQuality") or 0.0),
                    back_visibility=float(item.get("backVisibility") or 0.0),
                    frame_index=(
                        int(item["frameIndex"])
                        if item.get("frameIndex") is not None
                        else None
                    ),
                    source=str(item.get("source") or "jersey-ocr-worker"),
                    evidence_fingerprint=(
                        str(item.get("evidenceFingerprint"))
                        if item.get("evidenceFingerprint")
                        else None
                    ),
                )
            )
        except (TypeError, ValueError):
            invalid_raw_crop_ids.append(crop_id)

    evidence_source = "raw-crop-results"
    if not raw_rows:
        # Compatibility for an in-memory caller created before diagnostics v2.
        # New reconstructions always retain raw crop results above.
        evidence_source = "legacy-pre-resolver-selection"
        source_observations = [
            observation
            for summary in tracklet_evidence.values()
            for observation in summary.selected_observations
        ]

    reassigned: list[JerseyOcrObservation] = []
    unmapped_crop_ids: list[str] = []
    ambiguous_crop_ids: list[str] = []
    for observation in source_observations:
        if evidence_source == "legacy-pre-resolver-selection":
            # Old in-memory callers did not retain crop->observation metadata.
            # Their selected evidence can still be recovered when the source
            # tracklet has exactly one final owner.  A manual split makes that
            # relation one-to-many, so it deliberately remains ambiguous
            # instead of leaking the reading into both partitions.
            canonical_owners = tracklet_owners.get(observation.tracklet_id) or set()
        else:
            video_observation_id = crop_to_observation.get(observation.id)
            if video_observation_id is None:
                unmapped_crop_ids.append(observation.id)
                continue
            canonical_owners = owners.get(video_observation_id) or set()
        if len(canonical_owners) != 1:
            (ambiguous_crop_ids if canonical_owners else unmapped_crop_ids).append(
                observation.id
            )
            continue
        canonical_id = next(iter(canonical_owners))
        reassigned.append(
            replace(
                observation,
                tracklet_id=f"final:{canonical_id}",
            )
        )

    final_tracklets = aggregate_tracklets(
        reassigned,
        config=JERSEY_OCR_FUSION_CONFIG,
    )
    final_mapping = {
        tracklet_id: tracklet_id.removeprefix("final:")
        for tracklet_id in final_tracklets
    }
    canonical = (
        aggregate_canonical_people(
            final_tracklets,
            final_mapping,
            config=JERSEY_OCR_FUSION_CONFIG,
        )
        if final_tracklets
        else {}
    )
    return canonical, {
        "evidenceSource": evidence_source,
        "rawCandidateCropCount": len(source_observations),
        "mappedRawCropCount": len(reassigned),
        "invalidRawCropIds": sorted(set(invalid_raw_crop_ids)),
        "finalSelectedCropCount": sum(
            summary.selected_sample_count for summary in final_tracklets.values()
        ),
        # Compatibility name retained for existing diagnostics consumers.  It
        # now counts the mapped source pool, not just the pre-resolver top-N.
        "mappedSelectedCropCount": len(reassigned),
        "unmappedRawCropIds": sorted(set(unmapped_crop_ids)),
        "ambiguousRawCropIds": sorted(set(ambiguous_crop_ids)),
        "unmappedSelectedCropIds": sorted(set(unmapped_crop_ids)),
        "ambiguousSelectedCropIds": sorted(set(ambiguous_crop_ids)),
        "finalTrackletCount": len(final_tracklets),
    }


def _partition_local_jersey_evidence_for_resolver(
    tracks: list[TrackState],
    tracklet_evidence: Mapping[str, JerseyEvidenceSummary],
    diagnostics: dict,
) -> tuple[dict[str, JerseyEvidenceSummary], dict]:
    """Re-key raw OCR evidence to each pre-resolver split partition.

    Persisted splits are applied before global identity stitching.  Temporary
    unique canonical keys let the existing immutable-observation mapper fuse
    OCR independently for every partition without leaking a number from the
    selected range into the remaining source tracklet.
    """

    original_ids = {track.id: track.canonical_person_id for track in tracks}
    temporary_ids = {
        track.id: f"resolver-partition:{track.local_tracklet_id}" for track in tracks
    }
    try:
        for track in tracks:
            track.canonical_person_id = temporary_ids[track.id]
        by_temporary_id, mapping_diagnostics = (
            _aggregate_jersey_evidence_for_final_tracks(
                tracks,
                tracklet_evidence,
                diagnostics,
            )
        )
    finally:
        for track in tracks:
            track.canonical_person_id = original_ids[track.id]
    return (
        {
            track.local_tracklet_id: by_temporary_id[temporary_ids[track.id]]
            for track in tracks
            if temporary_ids[track.id] in by_temporary_id
        },
        mapping_diagnostics,
    )


def _predicted_track_point(track: TrackState, time: float) -> tuple[float, float]:
    last = track.points[-1]
    predicted_x, predicted_y = float(last["px"]), float(last["py"])
    if len(track.points) < 2:
        return predicted_x, predicted_y
    previous = track.points[-2]
    sample_elapsed = float(last["t"]) - float(previous["t"])
    prediction_elapsed = time - float(last["t"])
    if sample_elapsed <= 1e-4 or prediction_elapsed <= 0.0:
        return predicted_x, predicted_y
    # Do not extrapolate a noisy two-point velocity indefinitely through an
    # occlusion. The association gate still grows with elapsed time.
    horizon = min(prediction_elapsed, 0.35)
    predicted_x += (float(last["px"]) - float(previous["px"])) / sample_elapsed * horizon
    predicted_y += (float(last["py"]) - float(previous["py"])) / sample_elapsed * horizon
    return predicted_x, predicted_y


def _association_cost(track: TrackState, detection: Detection, time: float) -> float:
    last = track.points[-1]
    elapsed = time - float(last["t"])
    if elapsed <= 0.0 or elapsed > 0.65:
        return float("inf")

    track_team = _annotation_team(track.manual_kind)
    detection_team = _annotation_team(detection.annotation_kind)
    if track_team and detection_team and track_team != detection_team:
        return float("inf")
    if (
        track.manual_identity_owner_ids
        and detection.manual_identity_owner_ids
        and track.manual_identity_owner_ids.isdisjoint(
            detection.manual_identity_owner_ids
        )
    ):
        return float("inf")
    if (
        track.roster_binding_state is not None
        and detection.roster_binding_state is not None
        and (
            track.roster_binding_state != detection.roster_binding_state
            or track.manual_external_player_id != detection.external_player_id
        )
    ):
        return float("inf")
    if (
        track.manual_external_player_id
        and detection.external_player_id
        and track.manual_external_player_id != detection.external_player_id
        and (
            track.roster_binding_state is None
            and detection.roster_binding_state is None
            or track.roster_binding_state is not None
            and detection.roster_binding_state is not None
        )
    ):
        return float("inf")

    predicted_x, predicted_y = _predicted_track_point(track, time)
    pixel_distance = hypot(detection.x - predicted_x, detection.y - predicted_y)
    elapsed_scale = 1.0 + min(2.2, max(0.0, elapsed / 0.1 - 1.0) * 0.45)
    pixel_gate = max(48.0, detection.height * 2.4, track.last_height * 2.4) * elapsed_scale
    if pixel_distance > pixel_gate:
        return float("inf")
    pixel_cost = pixel_distance / max(1.0, pixel_gate)

    appearance_distance = float(np.linalg.norm(detection.feature - track.feature))
    appearance_cost = min(2.0, appearance_distance / 0.9)
    reid_cost: float | None = None
    track_reid = track.reid_feature
    if detection.reid_feature is not None and track_reid is not None:
        detection_reid = np.asarray(detection.reid_feature, dtype=np.float32)
        detection_norm = float(np.linalg.norm(detection_reid))
        if detection_norm > 1e-8 and np.isfinite(detection_reid).all():
            detection_reid = detection_reid / detection_norm
            cosine_distance = max(0.0, min(2.0, 1.0 - float(np.dot(track_reid, detection_reid))))
            # This gate only affects the short-horizon tracker.  Long-gap
            # identity decisions are made later by the audited resolver.
            if cosine_distance > 0.72 and track.reid_feature_count >= 2:
                return float("inf")
            reid_cost = min(2.0, cosine_distance / 0.38)
    height_ratio = max(detection.height, track.last_height) / max(
        1.0, min(detection.height, track.last_height)
    )
    size_cost = min(1.0, abs(float(np.log(height_ratio))) / 0.7)

    pitch_cost: float | None = None
    if (
        last.get("pitchX") is not None
        and last.get("pitchZ") is not None
        and detection.pitch_x is not None
        and detection.pitch_z is not None
    ):
        predicted_pitch_x = float(last["pitchX"])
        predicted_pitch_z = float(last["pitchZ"])
        if len(track.points) > 1:
            previous = track.points[-2]
            pitch_elapsed = float(last["t"]) - float(previous["t"])
            if (
                pitch_elapsed > 1e-4
                and previous.get("pitchX") is not None
                and previous.get("pitchZ") is not None
            ):
                horizon = min(elapsed, 0.35)
                predicted_pitch_x += (
                    float(last["pitchX"]) - float(previous["pitchX"])
                ) / pitch_elapsed * horizon
                predicted_pitch_z += (
                    float(last["pitchZ"]) - float(previous["pitchZ"])
                ) / pitch_elapsed * horizon
        pitch_distance = hypot(
            float(detection.pitch_x) - predicted_pitch_x,
            float(detection.pitch_z) - predicted_pitch_z,
        )
        uncertainty = float(detection.position_uncertainty_metres or 0.0) + float(
            last.get("positionUncertaintyMetres") or 0.0
        )
        pitch_gate = 2.2 + 16.0 * elapsed + min(5.0, uncertainty)
        if pitch_distance > pitch_gate:
            return float("inf")
        pitch_cost = pitch_distance / max(0.5, pitch_gate)

    if pitch_cost is None and reid_cost is None:
        cost = pixel_cost * 0.58 + appearance_cost * 0.34 + size_cost * 0.08
    elif pitch_cost is None:
        cost = (
            pixel_cost * 0.51
            + appearance_cost * 0.18
            + float(reid_cost) * 0.25
            + size_cost * 0.06
        )
    elif reid_cost is None:
        cost = (
            pixel_cost * 0.30
            + appearance_cost * 0.24
            + pitch_cost * 0.40
            + size_cost * 0.06
        )
    else:
        cost = (
            pixel_cost * 0.25
            + appearance_cost * 0.12
            + float(reid_cost) * 0.20
            + pitch_cost * 0.38
            + size_cost * 0.05
        )
    if detection.annotation_id and detection.annotation_id in track.annotation_ids:
        cost *= 0.2
    elif detection.external_player_id and (
        detection.external_player_id == track.manual_external_player_id
    ):
        cost *= 0.35
    return float(cost)


def _track_people(frames: list[tuple[list[Detection], float]]) -> list[TrackState]:
    tracks: list[TrackState] = []
    next_id = 1
    for frame_index, (detections, time) in enumerate(frames):
        active = [
            track
            for track in tracks
            if track.points and time - float(track.points[-1]["t"]) <= 0.65
        ]
        assigned_track_ids: set[int] = set()
        assigned_detections: set[int] = set()
        primary_detection_indices = [
            index
            for index, detection in enumerate(detections)
            if detection.confidence >= NEW_TRACK_CONFIDENCE or detection.annotation_id
        ]
        secondary_detection_indices = [
            index
            for index in range(len(detections))
            if index not in primary_detection_indices
        ]

        def assign(
            candidate_tracks: list[TrackState],
            detection_indices: list[int],
            maximum_cost: float,
        ) -> None:
            if not candidate_tracks or not detection_indices:
                return
            costs = np.full(
                (len(candidate_tracks), len(detection_indices)),
                np.inf,
                dtype=np.float64,
            )
            for track_index, track in enumerate(candidate_tracks):
                for column, detection_index in enumerate(detection_indices):
                    costs[track_index, column] = _association_cost(
                        track, detections[detection_index], time
                    )
            finite = np.isfinite(costs)
            assignment_costs = np.where(finite, costs, 1e6)
            rows, columns = linear_sum_assignment(assignment_costs)
            for track_index, column in zip(rows.tolist(), columns.tolist()):
                cost = float(costs[track_index, column])
                if not np.isfinite(cost) or cost > maximum_cost:
                    continue
                detection_index = detection_indices[column]
                alternatives = [
                    float(value)
                    for index, value in enumerate(costs[track_index])
                    if index != column and np.isfinite(value)
                ]
                margin = max(0.0, min(alternatives) - cost) if alternatives else None
                detection = detections[detection_index]
                detection.association_cost = cost
                detection.association_margin = margin
                track = candidate_tracks[track_index]
                track.append(detection, frame_index, time)
                assigned_track_ids.add(track.id)
                assigned_detections.add(detection_index)

        # ByteTrack-style two-stage association: reliable observations claim
        # identities first; low-confidence detections may continue an existing
        # track but never create a new ghost track by themselves.
        assign(active, primary_detection_indices, 1.05)
        remaining_tracks = [track for track in active if track.id not in assigned_track_ids]
        assign(remaining_tracks, secondary_detection_indices, 0.92)

        for detection_index in primary_detection_indices:
            detection = detections[detection_index]
            if detection_index in assigned_detections:
                continue
            track = TrackState(id=next_id)
            track.append(detection, frame_index, time)
            tracks.append(track)
            next_id += 1
    return tracks


def _tracklet_endpoint_pitch(point: dict) -> tuple[float, float] | None:
    if point.get("pitchX") is None or point.get("pitchZ") is None:
        return None
    return float(point["pitchX"]), float(point["pitchZ"])


def _resolve_canonical_track_states(
    tracks: list[TrackState],
    preliminary_mapping: dict[int, str],
    jersey_evidence: Mapping[str, JerseyEvidenceSummary] | None = None,
) -> tuple[list[TrackState], dict]:
    """Run the conservative offline tracklet-to-identity resolver.

    Screen/pitch proximity only rejects impossible transitions.  It can never
    create an identity link without ReID, reliable jersey OCR, an external
    roster ID, or an explicit manual decision.
    """

    inputs: list[IdentityTracklet] = []
    by_tracklet: dict[str, TrackState] = {}
    for track in tracks:
        if not track.points:
            continue
        points = sorted(track.points, key=lambda item: (float(item["t"]), item["frameIndex"]))
        tracklet_id = track.local_tracklet_id
        by_tracklet[tracklet_id] = track
        external_id = track.manual_external_player_id
        positive_annotation_ids = track.positive_annotation_ids
        jersey_fields = (
            jersey_evidence[tracklet_id].identity_resolver_fields()
            if jersey_evidence is not None and tracklet_id in jersey_evidence
            else {
                "jersey_number": None,
                "jersey_confidence": 0.0,
                "jersey_sample_count": 0,
            }
        )
        inputs.append(
            IdentityTracklet(
                id=tracklet_id,
                start_time=float(points[0]["t"]),
                end_time=float(points[-1]["t"]),
                team_id=preliminary_mapping.get(track.id)
                or _annotation_team(track.manual_kind),
                role=_annotation_role(track.manual_kind) or track.role,
                external_player_id=external_id,
                jersey_number=jersey_fields["jersey_number"],
                jersey_confidence=float(jersey_fields["jersey_confidence"] or 0.0),
                jersey_sample_count=int(jersey_fields["jersey_sample_count"] or 0),
                mean_reid_embedding=(
                    tuple(float(value) for value in track.reid_feature)
                    if track.reid_feature is not None
                    else None
                ),
                reid_embeddings=tuple(
                    tuple(float(value) for value in sample)
                    for sample in track.reid_samples
                ),
                start_pitch=_tracklet_endpoint_pitch(points[0]),
                end_pitch=_tracklet_endpoint_pitch(points[-1]),
                start_uncertainty_metres=points[0].get("positionUncertaintyMetres"),
                end_uncertainty_metres=points[-1].get("positionUncertaintyMetres"),
                observation_count=len(points),
                manual_confirmed=bool(positive_annotation_ids or external_id),
                manual_identity_id=(
                    f"canonical:{next(iter(track.manual_identity_owner_ids))}"
                    if len(track.manual_identity_owner_ids) == 1
                    else f"external:{external_id}"
                    if external_id
                    else None
                ),
                manual_team=bool(
                    positive_annotation_ids
                    and _annotation_team(track.manual_kind) is not None
                ),
                manual_role=bool(
                    positive_annotation_ids
                    and _annotation_role(track.manual_kind) is not None
                ),
            )
        )

    resolution = resolve_global_identities(inputs)
    result: list[TrackState] = []
    review_by_tracklet: dict[str, list] = {}
    for edge in resolution.review_edges:
        review_by_tracklet.setdefault(edge.predecessor_id, []).append(edge)
        review_by_tracklet.setdefault(edge.successor_id, []).append(edge)

    for group in resolution.groups:
        members = [by_tracklet[tracklet_id] for tracklet_id in group.tracklet_ids]
        target = min(
            members,
            key=lambda item: (
                0
                if item.positive_annotation_ids or item.manual_external_player_id
                else 1,
                float(item.points[0]["t"]),
                item.id,
            ),
        )
        for source in members:
            if source is target:
                continue
            _merge_raw_track_states(target, source)
        target.identity_group_id = group.id
        target.identity_status = group.status
        target.identity_confidence = float(group.confidence)
        if group.external_player_id and target.roster_binding_state is None:
            target.manual_external_player_id = group.external_player_id

        group_tracklets = set(group.tracklet_ids)
        for edge in resolution.accepted_edges:
            if (
                edge.predecessor_id not in group_tracklets
                or edge.successor_id not in group_tracklets
            ):
                continue
            reasons = set(edge.reasons)
            kind = (
                "manual"
                if edge.source == "manual"
                else "jersey-ocr"
                if "reliable-jersey-match" in reasons
                else "reid"
            )
            target.identity_evidence.append(
                {
                    "id": f"{group.id}:{edge.predecessor_id}:{edge.successor_id}",
                    "kind": kind,
                    "label": (
                        "Manual identity merge"
                        if edge.source == "manual"
                        else "Offline tracklet stitch"
                    ),
                    "value": ", ".join(edge.reasons),
                    "confidence": edge.score,
                    "source": edge.source,
                    "model": "global-tracklet-resolver-v1",
                    "manual": edge.source == "manual",
                }
            )
        seen_review_edges: set[tuple[str, str]] = set()
        for tracklet_id in group.tracklet_ids:
            for edge in review_by_tracklet.get(tracklet_id, []):
                edge_key = (edge.predecessor_id, edge.successor_id)
                if edge_key in seen_review_edges:
                    continue
                seen_review_edges.add(edge_key)
                target.identity_conflicts.append(
                    {
                        "id": f"review:{edge.predecessor_id}:{edge.successor_id}",
                        "code": "identity-association-review",
                        "message": (
                            f"Possible link {edge.predecessor_id} → {edge.successor_id} "
                            f"was not accepted ({', '.join(edge.reasons)})."
                        ),
                        "severity": "review",
                        "relatedTrackletIds": [
                            edge.predecessor_id,
                            edge.successor_id,
                        ],
                    }
                )
        result.append(target)

    diagnostics = deepcopy(resolution.diagnostics)
    diagnostics.update(
        {
            "schemaVersion": 1,
            "provider": "global-tracklet-resolver-v1",
            "identityEvidencePolicy": "strong-reid-or-reliable-jersey-or-manual",
            "acceptedEdges": [asdict(edge) for edge in resolution.accepted_edges],
            "reviewEdges": [asdict(edge) for edge in resolution.review_edges],
            "jerseyReliableTrackletCount": sum(
                summary.status == "reliable"
                for summary in (jersey_evidence or {}).values()
            ),
            "jerseyProvisionalTrackletCount": sum(
                summary.status == "provisional"
                for summary in (jersey_evidence or {}).values()
            ),
            "jerseyConflictingTrackletCount": sum(
                summary.status == "conflict"
                for summary in (jersey_evidence or {}).values()
            ),
        }
    )
    return sorted(result, key=lambda item: (float(item.points[0]["t"]), item.id)), diagnostics


def _include_goalkeeper_candidates(
    tracks: list[TrackState],
    mapping: dict[int, str],
    frame_width: int,
) -> dict[int, str]:
    """Recover long-lived keepers whose distinct kit forms a third color cluster."""
    if not tracks or not mapping or frame_width <= 0:
        return mapping
    longest = max(len(track.points) for track in tracks)
    minimum = max(5, round(longest * 0.70))
    result = dict(mapping)

    def center_x(track: TrackState) -> float:
        return float(np.mean([point["px"] for point in track.points]))

    for side in ("left", "right"):
        candidates = [
            track
            for track in tracks
            if track.id not in result
            and len(track.points) >= minimum
            and -frame_width * 0.05 <= center_x(track) <= frame_width * 1.05
            and (
                center_x(track) <= frame_width * 0.12
                if side == "left"
                else center_x(track) >= frame_width * 0.88
            )
        ]
        if not candidates:
            continue
        candidate = max(candidates, key=lambda track: (len(track.points), track.feature_count))
        nearby = [
            track
            for track in tracks
            if track.id in result
            and (
                center_x(track) <= frame_width * 0.45
                if side == "left"
                else center_x(track) >= frame_width * 0.55
            )
        ]
        if not nearby:
            nearby = [track for track in tracks if track.id in result]
        support = {
            team: sum(len(track.points) for track in nearby if result[track.id] == team)
            for team in ("home", "away")
        }
        team = max(("home", "away"), key=lambda item: (support[item], item == "away"))
        if sum(value == team for value in result.values()) >= 11:
            continue
        candidate.role = "goalkeeper"
        result[candidate.id] = team
    return result


def _team_clusters(
    tracks: list[TrackState],
    frame_width: int | None = None,
) -> tuple[dict[int, str], dict[str, str]]:
    if len(tracks) < 2:
        return {}, {"home": "#e74a3b", "away": "#e8edf2"}
    features = np.float32([track.feature for track in tracks])
    cluster_count = 3 if len(tracks) >= 6 else 2
    cv2.setRNGSeed(7)
    _, labels, centers = cv2.kmeans(
        features,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01),
        8,
        cv2.KMEANS_PP_CENTERS,
    )
    weights = {
        cluster: sum(len(track.points) for track, label in zip(tracks, labels.ravel()) if int(label) == cluster)
        for cluster in range(cluster_count)
    }
    selected = sorted(weights, key=weights.get, reverse=True)[:2]
    if len(selected) < 2:
        return {}, {"home": "#e74a3b", "away": "#e8edf2"}

    first, second = selected
    if centers[first][8] > centers[second][8]:
        first, second = second, first
    team_by_cluster = {first: "home", second: "away"}
    mapping = {
        track.id: team_by_cluster[int(label)]
        for track, label in zip(tracks, labels.ravel())
        if int(label) in team_by_cluster
    }
    if frame_width is not None:
        mapping = _include_goalkeeper_candidates(tracks, mapping, frame_width)
    colors = {
        "home": _cluster_color(centers[first]),
        "away": _cluster_color(centers[second]),
    }
    return mapping, colors


def _cluster_color(center: np.ndarray) -> str:
    white_ratio, dark_ratio = float(center[8]), float(center[9])
    if white_ratio > 0.28:
        return "#e8edf2"
    if dark_ratio > 0.58:
        return "#30363d"
    hue_bin = int(np.argmax(center[:8]))
    hue = 2 if hue_bin == 0 else int(hue_bin * 22.5 + 11.25)
    pixel = np.uint8([[[hue, 205, 225]]])
    blue, green, red = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"


def _smooth(values: list[float]) -> list[float]:
    if len(values) < 3:
        return values
    return [
        values[index] if index in {0, len(values) - 1}
        else (values[index - 1] + values[index] * 2 + values[index + 1]) / 4
        for index in range(len(values))
    ]


def _presence_keyframe(
    keyframe: dict,
    time: float,
    x: float,
    z: float,
    state: str,
    uncertainty: float,
) -> dict:
    return {
        **keyframe,
        "t": round(time, 3),
        "x": round(x, 2),
        "z": round(z, 2),
        # Presence confidence is deliberately separate from detector evidence.
        # The renderer keeps actors alive for the whole scene, while QA excludes
        # these inferred samples via ``observed=False``.
        "confidence": 0.18,
        "observed": False,
        "presenceState": state,
        "projectionSource": "presence-inferred",
        "calibrationFrameIndex": None,
        "positionUncertaintyMetres": round(uncertainty, 2),
        "projection": {
            "source": "presence-inferred",
            "calibrationFrameIndex": None,
            "uncertaintyMetres": round(uncertainty, 2),
        },
    }


def _bounded_pitch_position(x: float, z: float, pitch: dict) -> tuple[float, float]:
    margin = 0.25
    half_length = max(margin, float(pitch["length"]) / 2 - margin)
    half_width = max(margin, float(pitch["width"]) / 2 - margin)
    return (
        max(-half_length, min(half_length, x)),
        max(-half_width, min(half_width, z)),
    )


def _roaming_presence_position(
    anchor: dict,
    elapsed: float,
    seed: int,
    pitch: dict,
    *,
    reverse: bool = False,
) -> tuple[float, float]:
    """Return a deterministic, deliberately small latent-position movement.

    An off-camera player has an unknown position rather than no position.  We
    keep the estimate close to the last/next observation and expose uncertainty
    in metadata instead of fabricating a high-confidence trajectory.
    """

    direction = -1.0 if reverse else 1.0
    phase = (seed % 17) * 0.37
    amplitude = min(0.65, max(0.0, elapsed) * 0.10)
    x = float(anchor["x"]) + direction * amplitude * sin(elapsed * 0.83 + phase)
    z = float(anchor["z"]) + amplitude * 0.72 * cos(elapsed * 0.67 + phase)
    return _bounded_pitch_position(x, z, pitch)


def _continuous_track_keyframes(
    keyframes: list[dict],
    duration: float,
    pitch: dict,
    seed: int,
) -> tuple[list[dict], dict]:
    """Extend an actor's latent presence from 0% through 100% of a scene.

    Detector-backed points remain the only observed evidence.  Long internal
    gaps are sampled explicitly as low-confidence interpolation, while time
    before the first and after the last observation uses a bounded, deterministic
    roam near the nearest known position.
    """

    if not keyframes:
        return [], {
            "policy": "continuous-latent",
            "coverage": 0.0,
            "observationCount": 0,
            "inferredKeyframeCount": 0,
        }

    duration = max(0.0, float(duration))
    observed = sorted(
        (
            {
                **keyframe,
                "observed": True,
                "presenceState": "observed",
            }
            for keyframe in keyframes
        ),
        key=lambda item: float(item["t"]),
    )
    positive_deltas = [
        float(right["t"]) - float(left["t"])
        for left, right in zip(observed, observed[1:])
        if 1e-6 < float(right["t"]) - float(left["t"]) <= 1.0
    ]
    cadence = float(np.median(positive_deltas)) if positive_deltas else 0.2
    fill_step = max(0.25, min(0.75, cadence * 2.0))
    gap_threshold = max(0.6, cadence * 2.5)
    inferred: list[dict] = []

    first = observed[0]
    first_time = max(0.0, float(first["t"]))
    if first_time > 1e-6:
        times = [0.0]
        cursor = fill_step
        while cursor < first_time - 1e-6:
            times.append(cursor)
            cursor += fill_step
        base_uncertainty = float(first.get("positionUncertaintyMetres") or 1.0)
        for time in times:
            elapsed = first_time - time
            x, z = _roaming_presence_position(
                first,
                elapsed,
                seed,
                pitch,
                reverse=True,
            )
            inferred.append(
                _presence_keyframe(
                    first,
                    time,
                    x,
                    z,
                    "inferred-before-first",
                    min(18.0, base_uncertainty + 1.5 + elapsed * 1.8),
                )
            )

    for left, right in zip(observed, observed[1:]):
        left_time = float(left["t"])
        right_time = float(right["t"])
        gap = right_time - left_time
        if gap <= gap_threshold:
            continue
        cursor = left_time + fill_step
        base_uncertainty = max(
            float(left.get("positionUncertaintyMetres") or 1.0),
            float(right.get("positionUncertaintyMetres") or 1.0),
        )
        while cursor < right_time - 1e-6:
            mix = (cursor - left_time) / gap
            x = float(left["x"]) + (float(right["x"]) - float(left["x"])) * mix
            z = float(left["z"]) + (float(right["z"]) - float(left["z"])) * mix
            x, z = _bounded_pitch_position(x, z, pitch)
            inferred.append(
                _presence_keyframe(
                    left,
                    cursor,
                    x,
                    z,
                    "inferred-gap",
                    min(18.0, base_uncertainty + 1.0 + gap * 0.8),
                )
            )
            cursor += fill_step

    last = observed[-1]
    last_time = min(duration, float(last["t"]))
    if last_time < duration - 1e-6:
        times: list[float] = []
        cursor = last_time + fill_step
        while cursor < duration - 1e-6:
            times.append(cursor)
            cursor += fill_step
        times.append(duration)
        base_uncertainty = float(last.get("positionUncertaintyMetres") or 1.0)
        for time in times:
            elapsed = time - last_time
            x, z = _roaming_presence_position(last, elapsed, seed, pitch)
            inferred.append(
                _presence_keyframe(
                    last,
                    time,
                    x,
                    z,
                    "inferred-after-last",
                    min(18.0, base_uncertainty + 1.5 + elapsed * 1.8),
                )
            )

    combined = sorted([*observed, *inferred], key=lambda item: float(item["t"]))
    # Prefer observed evidence when floating-point rounding produces the same
    # timestamp as an inferred sample.
    deduplicated: list[dict] = []
    for keyframe in combined:
        if deduplicated and abs(float(deduplicated[-1]["t"]) - float(keyframe["t"])) < 1e-6:
            if keyframe.get("observed"):
                deduplicated[-1] = keyframe
            continue
        deduplicated.append(keyframe)

    observed_start = float(observed[0]["t"])
    observed_end = float(observed[-1]["t"])
    inferred_count = sum(item.get("observed") is False for item in deduplicated)
    coverage = (
        1.0
        if deduplicated
        and float(deduplicated[0]["t"]) <= 1e-6
        and float(deduplicated[-1]["t"]) >= duration - 1e-6
        else 0.0
    )
    return deduplicated, {
        "policy": "continuous-latent",
        "coverage": coverage,
        "observationCount": len(observed),
        "inferredKeyframeCount": inferred_count,
        "observedStart": round(observed_start, 3),
        "observedEnd": round(observed_end, 3),
        "observedSpanRatio": (
            round(max(0.0, observed_end - observed_start) / duration, 3)
            if duration > 1e-6
            else 1.0
        ),
        "sampleCadenceSeconds": round(cadence, 3),
    }


def _project_unclamped(
    point_x: float,
    point_y: float,
    width: int,
    height: int,
    pitch: dict,
    calibration: PitchCalibration | None = None,
) -> tuple[float, float]:
    if calibration is not None and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD:
        projected = calibration.image_to_pitch @ np.array([point_x, point_y, 1.0])
        if abs(float(projected[2])) > 1e-8:
            x, z = float(projected[0] / projected[2]), float(projected[1] / projected[2])
        else:
            x, z = 0.0, 0.0
    elif calibration is not None and calibration.rectangle in {
        "penalty-area-left",
        "penalty-area-right",
    }:
        progress = point_x / max(1.0, width)
        half_length = float(pitch["length"]) / 2
        x = progress * half_length if calibration.rectangle.endswith("right") else -half_length + progress * half_length
        z = (point_y / height - 0.5) * float(pitch["width"]) * 1.05
    else:
        x = (point_x / width - 0.5) * float(pitch["length"]) * 0.96
        z = (point_y / height - 0.5) * float(pitch["width"]) * 1.05
    return x, z


def _project(
    point_x: float,
    point_y: float,
    width: int,
    height: int,
    pitch: dict,
    calibration: PitchCalibration | None = None,
) -> tuple[float, float]:
    if calibration is not None and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD:
        metric = _project_metric_point(point_x, point_y, calibration, pitch)
        if metric is not None:
            return metric
        # A trusted matrix can still be invalid for a distant frame after a cut,
        # zoom, or failed optical-flow transform. Do not pile every observation
        # onto a pitch corner by clamping an arbitrarily large projection.
        x, z = _project_unclamped(point_x, point_y, width, height, pitch, None)
        return (
            max(-float(pitch["length"]) / 2, min(float(pitch["length"]) / 2, x)),
            max(-float(pitch["width"]) / 2, min(float(pitch["width"]) / 2, z)),
        )
    x, z = _project_unclamped(point_x, point_y, width, height, pitch, calibration)
    return (
        max(-float(pitch["length"]) / 2, min(float(pitch["length"]) / 2, x)),
        max(-float(pitch["width"]) / 2, min(float(pitch["width"]) / 2, z)),
    )


def _observation_priority(observation: dict) -> tuple[int, float]:
    return (
        1 if observation.get("annotationId") else 0,
        float(observation.get("confidence") or 0.0),
    )


def _merge_track_observations(*collections: list[dict]) -> list[dict]:
    """Return at most one authoritative video observation per source frame."""

    by_frame: dict[int, dict] = {}
    for observation in (item for collection in collections for item in collection):
        if observation.get("frameIndex") is None or not observation.get("bbox"):
            continue
        frame_index = int(observation["frameIndex"])
        previous = by_frame.get(frame_index)
        if previous is None or _observation_priority(observation) > _observation_priority(previous):
            by_frame[frame_index] = deepcopy(observation)
    return [
        by_frame[frame_index]
        for frame_index in sorted(by_frame)
    ]


def _track_state_observations(
    track: TrackState,
    *,
    canonical_person_id: str | None = None,
    source_start: float = 0.0,
) -> list[dict]:
    """Publish image evidence independently from 3D trajectory acceptance."""

    rows: list[dict] = []
    for point in track.points:
        if point.get("frameIndex") is None or not point.get("bbox"):
            continue
        frame_index = int(point["frameIndex"])
        source_tracklet_id = str(
            point.get("sourceTrackletId") or track.local_tracklet_id
        )
        observation_id = str(
            point.get("observationId") or f"{source_tracklet_id}:{frame_index}"
        )
        row = {
            "id": observation_id,
            "observationId": observation_id,
            "frameIndex": frame_index,
            "sourceFrameIndex": frame_index,
            "sceneTime": round(float(point["t"]), 3),
            "sourceTime": round(source_start + float(point["t"]), 3),
            "bbox": {
                "x": round(float(point["bbox"]["x"]), 2),
                "y": round(float(point["bbox"]["y"]), 2),
                "width": round(float(point["bbox"]["width"]), 2),
                "height": round(float(point["bbox"]["height"]), 2),
            },
            "confidence": round(float(point.get("confidence") or 0.0), 3),
            "annotationId": point.get("annotationId"),
            "sourceTrackletId": source_tracklet_id,
            "canonicalPersonId": canonical_person_id,
        }
        if point.get("pitchX") is not None and point.get("pitchZ") is not None:
            row.update(
                {
                    "metricStatus": "accepted",
                    "metricReason": None,
                    "pitch": {
                        "x": round(float(point["pitchX"]), 2),
                        "z": round(float(point["pitchZ"]), 2),
                    },
                    "positionSource": "observation",
                }
            )
        else:
            row.update(
                {
                    "metricStatus": "unprojected",
                    "metricReason": "metric-projection-unavailable",
                    "positionSource": "track-inferred",
                }
            )
        if point.get("projectionSource"):
            row["projectionSource"] = str(point["projectionSource"])
        if point.get("calibrationFrameIndex") is not None:
            row["calibrationFrameIndex"] = int(point["calibrationFrameIndex"])
        if point.get("positionUncertaintyMetres") is not None:
            row["positionUncertaintyMetres"] = round(
                float(point["positionUncertaintyMetres"]), 3
            )
        rows.append(row)
    return _merge_track_observations(rows)


def _previous_canonical_people(scene: dict) -> list[dict]:
    payload = scene.get("payload", {})
    canonical = payload.get("canonicalPeople") or []
    if canonical:
        return [deepcopy(item) for item in canonical]
    # Migration bridge: a pre-identity-layer render track was also the identity.
    return [
        {
            **deepcopy(track),
            "canonicalPersonId": track.get("canonicalPersonId") or track.get("id"),
        }
        for track in payload.get("tracks") or []
        if track.get("id")
    ]


def _canonical_match_score(
    track: TrackState,
    previous: dict,
    team_id: str | None = None,
) -> float:
    """Score only evidence strong enough to preserve a canonical ID.

    Exact manual/roster evidence is authoritative. Automatic image remapping
    requires several shared observations over time; one crowded-frame IoU is
    deliberately worth zero.
    """

    previous_annotations = set(previous.get("annotationIds") or [])
    annotation_overlap = len(track.annotation_ids & previous_annotations)
    previous_external_id = previous.get("externalPlayerId")
    if (
        track.manual_external_player_id
        and previous_external_id
        and track.manual_external_player_id != previous_external_id
    ):
        return 0.0
    resolved_team = team_id or _annotation_team(track.manual_kind)
    previous_team = previous.get("teamId")
    if resolved_team and previous_team and resolved_team != previous_team:
        return 0.0
    resolved_role = _annotation_role(track.manual_kind) or track.role
    previous_role = previous.get("role")
    if resolved_role and previous_role and resolved_role != previous_role:
        return 0.0
    score = annotation_overlap * 100.0
    if track.manual_external_player_id and track.manual_external_player_id == previous_external_id:
        score += 80.0
    previous_by_frame = {
        int(item["frameIndex"]): item
        for item in previous.get("observations") or []
        if item.get("frameIndex") is not None and item.get("bbox")
    }
    overlaps: list[float] = []
    normalized_center_residuals: list[float] = []
    matched_times: list[float] = []
    for point in track.points:
        frame_index = point.get("frameIndex")
        bbox = point.get("bbox")
        if frame_index is None or not bbox or int(frame_index) not in previous_by_frame:
            continue
        old_bbox = previous_by_frame[int(frame_index)]["bbox"]
        overlap = _iou(
            (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["x"]) + float(bbox["width"]),
                float(bbox["y"]) + float(bbox["height"]),
            ),
            (
                float(old_bbox["x"]),
                float(old_bbox["y"]),
                float(old_bbox["x"]) + float(old_bbox["width"]),
                float(old_bbox["y"]) + float(old_bbox["height"]),
            ),
        )
        new_center_x = float(bbox["x"]) + float(bbox["width"]) / 2.0
        new_center_y = float(bbox["y"]) + float(bbox["height"]) / 2.0
        old_center_x = float(old_bbox["x"]) + float(old_bbox["width"]) / 2.0
        old_center_y = float(old_bbox["y"]) + float(old_bbox["height"]) / 2.0
        scale = max(
            1.0,
            min(float(bbox["height"]), float(old_bbox["height"])),
        )
        overlaps.append(float(overlap))
        normalized_center_residuals.append(
            hypot(new_center_x - old_center_x, new_center_y - old_center_y) / scale
        )
        matched_times.append(float(point.get("t") or 0.0))

    if score >= 80.0:
        return score + sum(overlap >= 0.25 for overlap in overlaps)
    if len(overlaps) < 3:
        return 0.0
    time_span = max(matched_times) - min(matched_times)
    median_iou = float(np.median(overlaps))
    residual_p90 = float(np.percentile(normalized_center_residuals, 90))
    if time_span < 0.4 or median_iou < 0.25 or residual_p90 > 1.5:
        return 0.0
    observation_denominator = max(
        1,
        min(len(track.points), len(previous_by_frame)),
    )
    coverage = len(overlaps) / observation_denominator
    return round(
        10.0
        + len(overlaps)
        + median_iou * 2.0
        + min(1.0, coverage)
        + min(2.0, time_span),
        6,
    )


def _new_canonical_person_id(track: TrackState) -> str:
    if track.annotation_ids:
        seed = "annotation:" + ",".join(sorted(track.annotation_ids))
    else:
        first = min(track.points, key=lambda point: (float(point["t"]), int(point["frameIndex"])))
        bbox = first.get("bbox") or {}
        seed = ":".join(
            [
                str(first.get("frameIndex")),
                str(round(float(bbox.get("x") or 0.0))),
                str(round(float(bbox.get("y") or 0.0))),
                str(round(float(bbox.get("width") or 0.0))),
                str(round(float(bbox.get("height") or 0.0))),
            ]
        )
    return f"canonical-{sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def _assign_persistent_canonical_person_ids(
    tracks: list[TrackState],
    scene: dict,
    mapping: dict[int, str] | None = None,
) -> None:
    """Keep identity IDs stable across rebuilds when image evidence overlaps."""

    previous = _previous_canonical_people(scene)
    previous_by_identifier: dict[str, list[str]] = {}
    for item in previous:
        canonical_id = str(
            item.get("canonicalPersonId") or item.get("id") or ""
        ).strip()
        if not canonical_id:
            continue
        for identifier in {canonical_id, str(item.get("id") or "").strip()}:
            if identifier:
                previous_by_identifier.setdefault(identifier, []).append(canonical_id)

    manual_claims: dict[str, list[TrackState]] = {}
    for track in tracks:
        if len(track.manual_identity_owner_ids) != 1:
            continue
        owner_id = next(iter(track.manual_identity_owner_ids))
        matches = sorted(set(previous_by_identifier.get(owner_id, [])))
        if len(matches) > 1:
            raise ReconstructionError(
                f"Explicit canonical owner {owner_id} resolves to multiple saved identities"
            )
        if not matches:
            continue
        canonical_id = matches[0]
        manual_claims.setdefault(canonical_id, []).append(track)
    duplicate_manual_claims = {
        canonical_id: claimants
        for canonical_id, claimants in manual_claims.items()
        if len(claimants) > 1
    }
    if duplicate_manual_claims:
        canonical_id = sorted(duplicate_manual_claims)[0]
        raise ReconstructionError(
            f"Explicit canonical owner {canonical_id} reached multiple unresolved tracks"
        )
    for canonical_id, claimants in manual_claims.items():
        claimants[0].canonical_person_id = canonical_id

    preassigned: dict[str, list[TrackState]] = {}
    for track in tracks:
        if track.canonical_person_id:
            preassigned.setdefault(str(track.canonical_person_id), []).append(track)
    duplicate_preassigned = {
        canonical_id: claimants
        for canonical_id, claimants in preassigned.items()
        if len(claimants) > 1
    }
    if duplicate_preassigned:
        canonical_id = sorted(duplicate_preassigned)[0]
        raise ReconstructionError(
            f"Canonical identity {canonical_id} is claimed by multiple resolved tracks"
        )

    claimed_previous_ids = set(manual_claims)
    claimed_previous_ids.update(
        str(track.canonical_person_id)
        for track in tracks
        if track.canonical_person_id
    )
    if tracks and previous:
        scores = np.zeros((len(tracks), len(previous)), dtype=np.float64)
        for track_index, track in enumerate(tracks):
            if track.canonical_person_id:
                continue
            for previous_index, item in enumerate(previous):
                previous_id = str(
                    item.get("canonicalPersonId") or item.get("id") or ""
                )
                if previous_id in claimed_previous_ids:
                    continue
                scores[track_index, previous_index] = _canonical_match_score(
                    track,
                    item,
                    (mapping or {}).get(track.id),
                )
        rows, columns = linear_sum_assignment(-scores)
        for track_index, previous_index in zip(rows.tolist(), columns.tolist()):
            if tracks[track_index].canonical_person_id:
                continue
            score = float(scores[track_index, previous_index])
            row_alternatives = sorted(
                float(value)
                for index, value in enumerate(scores[track_index])
                if index != previous_index
            )
            column_alternatives = sorted(
                float(value)
                for index, value in enumerate(scores[:, previous_index])
                if index != track_index
            )
            ambiguous = (
                (row_alternatives and score - row_alternatives[-1] < 0.35)
                or (column_alternatives and score - column_alternatives[-1] < 0.35)
            )
            if score <= 0.0 or ambiguous:
                if score > 0.0 and ambiguous:
                    tracks[track_index].identity_conflicts.append(
                        {
                            "id": f"canonical-remap:{tracks[track_index].local_tracklet_id}",
                            "code": "canonical-id-remap-ambiguous",
                            "message": "Previous canonical identity was not reused because another candidate had similar image evidence.",
                            "severity": "review",
                            "relatedTrackletIds": [tracks[track_index].local_tracklet_id],
                        }
                    )
                continue
            previous_id = previous[previous_index].get("canonicalPersonId") or previous[
                previous_index
            ].get("id")
            if previous_id:
                tracks[track_index].canonical_person_id = str(previous_id)

    # Previous IDs that failed or were ambiguous remain reserved. Otherwise a
    # deterministic bbox-derived ID could silently recreate the very mapping
    # that the evidence gate rejected.
    used = {
        str(track.canonical_person_id)
        for track in tracks
        if track.canonical_person_id
    }
    used.update(
        str(item.get("canonicalPersonId") or item.get("id"))
        for item in previous
        if item.get("canonicalPersonId") or item.get("id")
    )
    for track in sorted(tracks, key=lambda item: (float(item.points[0]["t"]), item.id)):
        if track.canonical_person_id:
            continue
        base = _new_canonical_person_id(track)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        track.canonical_person_id = candidate
        used.add(candidate)


def _match_binding_roster(scene: dict) -> tuple[list[RosterPlayer], dict]:
    binding = scene.get("payload", {}).get("matchBinding") or {}
    roster_quality = (
        binding.get("rosterQuality")
        if isinstance(binding.get("rosterQuality"), dict)
        else {}
    )
    raw_players = binding.get("players") or []
    players: list[RosterPlayer] = []
    invalid_count = 0
    identifiers: set[str] = set()
    duplicate_identifiers: set[str] = set()
    for item in raw_players:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        identifier = str(item.get("id") or "").strip()
        if not identifier:
            invalid_count += 1
            continue
        if identifier in identifiers:
            duplicate_identifiers.add(identifier)
            continue
        identifiers.add(identifier)
        try:
            players.append(
                RosterPlayer(
                    external_player_id=identifier,
                    display_name=str(item.get("name") or identifier),
                    jersey_number=item.get("number"),
                    team_id=item.get("team_id") or item.get("teamId"),
                    role=item.get("position") or item.get("role"),
                )
            )
        except ValueError:
            invalid_count += 1
    if duplicate_identifiers:
        # Duplicate external IDs make any candidate confirmation ambiguous.
        players = []
    return players, {
        "availablePlayerCount": len(raw_players),
        "usablePlayerCount": len(players),
        "invalidPlayerCount": invalid_count,
        "duplicateExternalPlayerIds": sorted(duplicate_identifiers),
        "automaticIdentityEligible": bool(
            roster_quality.get("automaticIdentityEligible")
        ),
        "manualIdentityEligible": bool(
            roster_quality.get("manualIdentityEligible", bool(players))
        ),
        "qualityStatus": roster_quality.get("status") or "legacy",
        "qualityReasons": list(roster_quality.get("reasons") or []),
        "status": (
            "invalid-duplicate-ids"
            if duplicate_identifiers
            else "ready"
            if players
            else "unavailable"
        ),
    }


def _external_roster_team_id(scene: dict, local_team_id: str | None) -> str | None:
    if local_team_id is None:
        return None
    binding = scene.get("payload", {}).get("matchBinding") or {}
    teams = binding.get("teams") or {}
    team = teams.get(local_team_id)
    if isinstance(team, dict) and team.get("id"):
        return str(team["id"])
    return str(local_team_id)


def _apply_closed_set_roster_resolution(
    documents: list[dict],
    scene: dict,
    roster: list[RosterPlayer],
    roster_diagnostics: dict,
    jersey_evidence: Mapping[str, JerseyEvidenceSummary] | None,
) -> dict:
    """Publish review-only, globally unique roster hypotheses.

    The persisted match snapshot is the only closed set. Incomplete snapshots
    remain usable for explicit manual binding but never constrain automatic
    hypotheses. The resolver itself cannot write ``externalPlayerId``.
    """

    for document in documents:
        document["rosterCandidates"] = []

    base_diagnostics = {
        "status": "unavailable",
        "schemaVersion": 1,
        "automaticBindingCount": 0,
        "requiresManualConfirmation": True,
        "matchClockAligned": False,
        "reasons": [],
    }
    if not roster_diagnostics.get("automaticIdentityEligible"):
        base_diagnostics["status"] = "disabled-incomplete-roster"
        base_diagnostics["reasons"] = list(
            roster_diagnostics.get("qualityReasons")
            or ["persisted-roster-not-eligible-for-automatic-identity"]
        )
        for document in documents:
            document["rosterResolution"] = {
                "status": "abstain",
                "suggestedExternalPlayerId": None,
                "requiresManualConfirmation": False,
                "reasons": list(base_diagnostics["reasons"]),
                "conflicts": [],
            }
        return base_diagnostics
    if not roster:
        base_diagnostics["reasons"] = ["persisted-roster-empty-or-invalid"]
        return base_diagnostics

    persisted_players = [
        PersistedRosterPlayer(
            external_player_id=player.external_player_id,
            display_name=player.display_name,
            team_id=player.team_id,
            jersey_number=player.jersey_number,
            role=player.role,
            # Video source time is not match-clock time. Availability windows
            # are deliberately omitted until clock alignment is explicit.
            active_intervals=(),
        )
        for player in roster
    ]
    resolver_people: list[RosterCanonicalPersonEvidence] = []
    skipped_ids: set[str] = set()
    for document in documents:
        canonical_id = str(document["canonicalPersonId"])
        if document.get("identityStatus") == "excluded":
            skipped_ids.add(canonical_id)
            document["rosterResolution"] = {
                "status": "abstain",
                "suggestedExternalPlayerId": None,
                "requiresManualConfirmation": False,
                "reasons": ["canonical-identity-excluded"],
                "conflicts": [],
            }
            continue
        manual = document.get("provenance") == "manual"
        local_team_id = document.get("teamId")
        external_team_id = _external_roster_team_id(scene, local_team_id)
        team_evidence = (
            RosterAttributeEvidence(
                value=external_team_id,
                confidence=0.96 if manual else 0.76,
                source="manual-team-label" if manual else "team-clustering",
                confirmed=manual,
            )
            if external_team_id
            else None
        )
        role = document.get("role")
        role_evidence = (
            RosterAttributeEvidence(
                value=str(role),
                confidence=0.96 if manual else 0.72,
                source="manual-role-label" if manual else "role-classifier",
                confirmed=manual,
            )
            if role
            else None
        )
        jersey_summary = (jersey_evidence or {}).get(canonical_id)
        jersey_value = (
            jersey_summary.jersey_number or jersey_summary.candidate_number
            if jersey_summary is not None
            else None
        )
        jersey_attribute = (
            RosterAttributeEvidence(
                value=jersey_value,
                confidence=float(jersey_summary.confidence),
                source="jersey-ocr-worker",
                support_count=max(1, int(jersey_summary.support_count)),
                confirmed=False,
            )
            if jersey_summary is not None and jersey_value is not None
            else None
        )
        resolver_people.append(
            RosterCanonicalPersonEvidence(
                canonical_person_id=canonical_id,
                visible_intervals=(),
                team=team_evidence,
                role=role_evidence,
                jersey_number=jersey_attribute,
                confirmed_external_player_id=document.get("externalPlayerId"),
                excluded_external_player_ids=tuple(
                    sorted(rejected_roster_candidate_ids(scene, canonical_id))
                ),
            )
        )

    result = resolve_closed_set_roster(resolver_people, persisted_players)
    resolutions = {
        resolution.canonical_person_id: resolution
        for resolution in result.people
    }
    for document in documents:
        canonical_id = str(document["canonicalPersonId"])
        if canonical_id in skipped_ids:
            continue
        resolution = resolutions[canonical_id]
        resolution_payload = resolution.to_payload()
        document["rosterResolution"] = {
            key: value
            for key, value in resolution_payload.items()
            if key != "candidates"
        }
        if resolution.status == "suggested":
            published_candidates = []
            for candidate in resolution.candidates:
                if not candidate.eligible or candidate.identity_signal_score <= 0.0:
                    continue
                payload = candidate.to_payload()
                # Backward-compatible UI confidence remains a hypothesis score,
                # never a probability or accepted roster binding.
                payload["confidence"] = payload["score"]
                published_candidates.append(payload)
            document["rosterCandidates"] = published_candidates
        for code in resolution.conflicts:
            conflict_id = f"{canonical_id}:roster-resolution:{code}"
            if any(item.get("id") == conflict_id for item in document["conflicts"]):
                continue
            document["conflicts"].append(
                {
                    "id": conflict_id,
                    "code": code,
                    "message": (
                        "The closed-set roster resolver found contradictory identity evidence; "
                        "the existing manual decision was retained."
                    ),
                    "severity": "review",
                }
            )

    return {
        **result.to_payload()["diagnostics"],
        "status": "ready",
        "schemaVersion": 1,
        "matchClockAligned": False,
        "skippedExcludedIdentityCount": len(skipped_ids),
    }


def _canonical_people_documents(
    tracks: list[TrackState],
    mapping: dict[int, str],
    rendered_tracks: list[dict],
    scene: dict,
    resolver_diagnostics: dict | None = None,
    jersey_evidence: Mapping[str, JerseyEvidenceSummary] | None = None,
) -> tuple[list[dict], dict]:
    source_start = float(
        scene.get("payload", {}).get("videoAsset", {}).get("sourceStart") or 0.0
    )
    rendered_by_identity = {
        str(item.get("canonicalPersonId")): item
        for item in rendered_tracks
        if item.get("canonicalPersonId")
    }
    documents: list[dict] = []
    all_source_tracklets: set[str] = set()
    total_observations = 0
    total_reid_observations = 0
    roster, roster_diagnostics = _match_binding_roster(scene)
    roster_by_external_id = {
        player.external_player_id: player for player in roster
    }
    for track in tracks:
        canonical_id = str(track.canonical_person_id or _new_canonical_person_id(track))
        positive_annotation_ids = track.positive_annotation_ids
        jersey_summary = (jersey_evidence or {}).get(canonical_id)
        all_source_tracklets.update(track.source_tracklet_ids or {track.local_tracklet_id})
        rendered = rendered_by_identity.get(canonical_id)
        raw_observations = _track_state_observations(
            track,
            canonical_person_id=canonical_id,
            source_start=source_start,
        )
        observations = []
        for observation in (
            rendered.get("observations") if rendered is not None else raw_observations
        ) or []:
            enriched = {
                **deepcopy(observation),
                "id": observation.get("id") or observation.get("observationId"),
                "observationId": observation.get("observationId") or observation.get("id"),
                "canonicalPersonId": canonical_id,
            }
            if not enriched.get("sourceTrackletId"):
                matching = next(
                    (
                        item
                        for item in raw_observations
                        if int(item["frameIndex"]) == int(enriched["frameIndex"])
                    ),
                    None,
                )
                enriched["sourceTrackletId"] = (
                    matching.get("sourceTrackletId") if matching else track.local_tracklet_id
                )
            observations.append(enriched)
        total_observations += len(observations)
        total_reid_observations += track.reid_observation_count

        team = mapping.get(track.id) or _annotation_team(track.manual_kind)
        role = _annotation_role(track.manual_kind) or track.role or "player"
        conflicts = deepcopy(track.identity_conflicts)
        bound_external_player_id = str(track.manual_external_player_id or "")
        bound_roster_player = roster_by_external_id.get(bound_external_player_id)
        if bound_external_player_id and bound_roster_player is None:
            conflicts.append(
                {
                    "id": f"{canonical_id}:manual-roster-player-missing",
                    "code": "manual-roster-player-missing",
                    "message": (
                        "The confirmed roster player is absent or ambiguous in the current "
                        "persisted match roster; the manual binding was retained for review."
                    ),
                    "severity": "review",
                    "externalPlayerId": bound_external_player_id,
                    "bindingAnnotationIds": sorted(
                        track.roster_binding_annotation_ids
                    ),
                    "rosterStatus": roster_diagnostics["status"],
                }
            )
        resolver_status = track.identity_status
        identity_status = (
            "resolved"
            if positive_annotation_ids
            or track.manual_external_player_id
            or resolver_status == "resolved"
            else "excluded"
            if resolver_status == "excluded"
            else "provisional"
        )
        confidence = track.identity_confidence
        if positive_annotation_ids or track.manual_external_player_id:
            confidence = 1.0
        if confidence is not None:
            confidence = round(max(0.0, min(1.0, float(confidence))), 3)

        evidence = deepcopy(track.identity_evidence)
        if jersey_summary is not None:
            evidence.append(
                {
                    "id": f"{canonical_id}:jersey-ocr",
                    "kind": "jersey-ocr",
                    "label": "Jersey number OCR",
                    "value": jersey_summary.jersey_number
                    or jersey_summary.candidate_number,
                    "confidence": round(float(jersey_summary.confidence), 6),
                    "supportCount": jersey_summary.support_count,
                    "sampleCount": jersey_summary.selected_sample_count,
                    "source": "jersey-ocr-worker",
                    "model": (
                        (resolver_diagnostics or {})
                        .get("jerseyOcr", {})
                        .get("modelVersion")
                    ),
                    "frameIndices": [
                        int(item.frame_index)
                        for item in jersey_summary.selected_observations
                        if item.frame_index is not None
                    ],
                    "status": jersey_summary.status,
                    "votes": [
                        {
                            "number": vote.number,
                            "supportCount": vote.support_count,
                            "weightShare": round(vote.weight_share, 6),
                        }
                        for vote in jersey_summary.votes
                    ],
                }
            )
            if jersey_summary.status == "conflict":
                conflicts.append(
                    {
                        "id": f"{canonical_id}:jersey-ocr-conflict",
                        "code": "jersey-ocr-conflict",
                        "message": (
                            "Independent jersey OCR readings disagree; no shirt number "
                            "or roster identity was accepted."
                        ),
                        "severity": "review",
                        "relatedTrackletIds": list(jersey_summary.tracklet_ids),
                        "reasons": list(jersey_summary.conflict_reasons),
                    }
                )
            observed_number = normalize_jersey_number(
                jersey_summary.jersey_number
            )
            expected_number = normalize_jersey_number(
                bound_roster_player.jersey_number
                if bound_roster_player is not None
                else None
            )
            if (
                jersey_summary.status == "reliable"
                and bound_roster_player is not None
                and observed_number is not None
                and expected_number is not None
                and observed_number != expected_number
            ):
                conflicts.append(
                    {
                        "id": f"{canonical_id}:manual-roster-jersey-conflict",
                        "code": "manual-roster-jersey-conflict",
                        "message": (
                            "The confirmed roster player has a different shirt number "
                            "from repeated reliable OCR; the manual binding was retained for review."
                        ),
                        "severity": "review",
                        "externalPlayerId": bound_roster_player.external_player_id,
                        "expectedNumber": expected_number,
                        "observedNumber": observed_number,
                        "bindingAnnotationIds": sorted(
                            track.roster_binding_annotation_ids
                        ),
                        "relatedTrackletIds": list(jersey_summary.tracklet_ids),
                    }
                )
        if positive_annotation_ids:
            evidence.append(
                {
                    "id": f"{canonical_id}:manual",
                    "kind": "manual",
                    "label": "Confirmed by frame annotation",
                    "supportCount": len(positive_annotation_ids),
                    "manual": True,
                }
            )
        if track.reid_feature_count:
            evidence.append(
                {
                    "id": f"{canonical_id}:reid",
                    "kind": "reid",
                    "label": "Soccer player appearance embedding",
                    "supportCount": track.reid_observation_count,
                    "sampleCount": track.reid_feature_count,
                    "uniqueEvidenceFingerprintCount": track.reid_observation_count,
                    "duplicateEvidenceFingerprintCount": (
                        track.reid_duplicate_evidence_count
                    ),
                    "source": "identity-worker",
                    "selectionPolicy": "pixel-deduplicated-quality-ranked-temporally-separated-v2",
                    "selectedFrameIndices": [
                        int(item["frameIndex"])
                        for item in track.reid_selected_metadata
                    ],
                    "selectedQualities": [
                        round(float(item["quality"]), 4)
                        for item in track.reid_selected_metadata
                    ],
                    "selectedEvidenceFingerprints": [
                        item.get("evidenceFingerprint")
                        for item in track.reid_selected_metadata
                        if item.get("evidenceFingerprint")
                    ],
                }
            )
        evidence.append(
            {
                "id": f"{canonical_id}:trajectory",
                "kind": "trajectory",
                "label": "Continuous local tracklet observations",
                "supportCount": len(observations),
            }
        )
        default_label = track.manual_label or (
            rendered.get("label") if rendered is not None else None
        )
        if not default_label:
            default_label = f"{str(team).title()} person" if team else "Unassigned person"
        member_tracklets = sorted(track.source_tracklet_ids or {track.local_tracklet_id})
        identity_source = (
            "manual"
            if positive_annotation_ids or track.manual_external_player_id
            else "reid+trajectory"
            if track.reid_feature_count
            else "jersey-ocr+trajectory"
            if jersey_summary is not None and jersey_summary.status == "reliable"
            else "tracker+trajectory"
        )
        documents.append(
            {
                "id": canonical_id,
                "canonicalPersonId": canonical_id,
                "displayName": default_label,
                "identityStatus": identity_status,
                "identityConfidence": confidence,
                "identitySource": identity_source,
                "teamId": team,
                "role": role,
                "jerseyNumber": (
                    jersey_summary.jersey_number if jersey_summary is not None else None
                ),
                "candidateNumber": (
                    jersey_summary.candidate_number if jersey_summary is not None else None
                ),
                "externalPlayerId": track.manual_external_player_id,
                "annotationIds": sorted(track.annotation_ids),
                "sourceTrackletIds": member_tracklets,
                "memberTrackletIds": member_tracklets,
                "observationCount": len(observations),
                "observations": observations,
                "renderTrackId": rendered.get("id") if rendered is not None else None,
                "evidence": evidence,
                "rosterCandidates": [],
                "conflicts": conflicts,
                "provenance": (
                    "manual"
                    if positive_annotation_ids
                    else "mixed"
                    if len(member_tracklets) > 1
                    else "automatic"
                ),
            }
        )
    closed_set_diagnostics = _apply_closed_set_roster_resolution(
        documents,
        scene,
        roster,
        roster_diagnostics,
        jersey_evidence,
    )
    documents.sort(
        key=lambda item: (
            item.get("teamId") or "unknown",
            item.get("displayLabel") or item["id"],
            item["id"],
        )
    )
    diagnostics = {
        **(deepcopy(resolver_diagnostics) if resolver_diagnostics else {}),
        "sourceTrackletCount": len(all_source_tracklets),
        "canonicalPersonCount": len(documents),
        "resolvedPersonCount": sum(
            item["identityStatus"] == "resolved" for item in documents
        ),
        "provisionalPersonCount": sum(
            item["identityStatus"] == "provisional" for item in documents
        ),
        "excludedPersonCount": sum(
            item["identityStatus"] == "excluded" for item in documents
        ),
        "conflictPersonCount": sum(bool(item["conflicts"]) for item in documents),
        "manualRosterJerseyConflictCount": sum(
            any(
                conflict.get("code") == "manual-roster-jersey-conflict"
                for conflict in item["conflicts"]
            )
            for item in documents
        ),
        "manualRosterMissingConflictCount": sum(
            any(
                conflict.get("code") == "manual-roster-player-missing"
                for conflict in item["conflicts"]
            )
            for item in documents
        ),
        "manualDecisionCount": sum(
            item["identitySource"] == "manual" for item in documents
        ),
        "identityObservationCount": total_observations,
        "reidUsableObservationCount": total_reid_observations,
        "reidSelectedIndependentSampleCount": sum(
            track.reid_feature_count for track in tracks
        ),
        "reidCropCoverage": round(
            total_reid_observations / max(1, total_observations), 3
        ),
        "jerseyReadablePersonCount": sum(
            summary.selected_sample_count > 0
            for summary in (jersey_evidence or {}).values()
        ),
        "jerseyReliablePersonCount": sum(
            summary.status == "reliable"
            for summary in (jersey_evidence or {}).values()
        ),
        "jerseyProvisionalPersonCount": sum(
            summary.status == "provisional"
            for summary in (jersey_evidence or {}).values()
        ),
        "jerseyConflictPersonCount": sum(
            summary.status == "conflict"
            for summary in (jersey_evidence or {}).values()
        ),
        "jerseyReadableCoverage": round(
            sum(
                summary.selected_sample_count > 0
                for summary in (jersey_evidence or {}).values()
            )
            / max(1, len(documents)),
            3,
        ),
        "rosterCandidateCount": sum(
            len(item.get("rosterCandidates") or []) for item in documents
        ),
        "rosterPrior": roster_diagnostics,
        "closedSetRoster": closed_set_diagnostics,
    }
    return documents, diagnostics


def _scene_tracks(
    tracks: list[TrackState],
    mapping: dict[int, str],
    colors: dict[str, str],
    frame_size: tuple[int, int],
    scene: dict,
    calibration: PitchCalibration | None = None,
    coordinate_mode: str | None = None,
    diagnostics: dict | None = None,
) -> list[dict]:
    width, height = frame_size
    minimum = max(5, round(len(_frame_paths(scene)) * 0.24))
    accepted = [
        track
        for track in tracks
        if track.identity_status != "excluded"
        and track.id in mapping
        and (len(track.points) >= minimum or track.positive_annotation_ids)
    ]
    accepted.sort(
        key=lambda track: (
            mapping[track.id],
            0 if track.positive_annotation_ids else 1,
            0 if track.role == "goalkeeper" else 1,
            -len(track.points),
        )
    )
    counts = {"home": 0, "away": 0, "officials": 0, "unknown": 0}
    result = []
    trajectory_diagnostics = {
        "rawProjectedObservationCount": 0,
        "retainedProjectedObservationCount": 0,
        "discardedProjectedObservationCount": 0,
        "preFilterSpeedSampleCount": 0,
        "preFilterSpeedViolationCount": 0,
        "preFilterMaximumSpeedMetresPerSecond": None,
        "splitTrajectoryCount": 0,
        "discardedTrajectoryFragmentCount": 0,
        "acceptedIdentityImageObservationCount": 0,
        "publishedIdentityObservationCount": 0,
        "metricAcceptedIdentityObservationCount": 0,
        "metricRejectedIdentityObservationCount": 0,
        "metricUnprojectedIdentityObservationCount": 0,
    }
    resolved_coordinate_mode = coordinate_mode or (
        "metric"
        if calibration is not None and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD
        else "approximate"
    )
    for track in accepted:
        team = mapping[track.id]
        projected_points: list[tuple[dict, tuple[float, float], str, int | None, float | None]] = []
        for point in track.points:
            if resolved_coordinate_mode == "metric":
                if point.get("pitchX") is None or point.get("pitchZ") is None:
                    # Missing metric observations remain missing. Never hide them
                    # behind screen coordinates in a metric run.
                    continue
                position = (float(point["pitchX"]), float(point["pitchZ"]))
                source = str(point.get("projectionSource") or "direct")
                calibration_frame_index = point.get("calibrationFrameIndex")
                uncertainty = point.get("positionUncertaintyMetres")
            else:
                position = _project(
                    point["px"],
                    point["py"],
                    width,
                    height,
                    scene["payload"]["pitch"],
                    None,
                )
                source = "screen-approximate"
                calibration_frame_index = None
                uncertainty = 12.0
            projected_points.append(
                (point, position, source, calibration_frame_index, uncertainty)
            )
        if not projected_points:
            continue
        trajectory_diagnostics["rawProjectedObservationCount"] += len(projected_points)
        segments: list[tuple[list[dict], list[tuple[float, float]]]] = [([], [])]
        segment_meta: list[list[tuple[str, int | None, float | None]]] = [[]]
        raw_speeds: list[float] = []
        for point, position, source, calibration_frame_index, uncertainty in projected_points:
            points, positions = segments[-1]
            if points:
                elapsed = max(0.001, point["t"] - points[-1]["t"])
                speed = hypot(position[0] - positions[-1][0], position[1] - positions[-1][1]) / elapsed
                raw_speeds.append(speed)
                if speed > 14.0:
                    segments.append(([], []))
                    segment_meta.append([])
                    points, positions = segments[-1]
            points.append(point)
            positions.append(position)
            segment_meta[-1].append((source, calibration_frame_index, uncertainty))
        selected_index = max(range(len(segments)), key=lambda index: len(segments[index][0]))
        selected_points, selected_positions = segments[selected_index]
        selected_meta = segment_meta[selected_index]
        non_empty_segments = [segment for segment, _ in segments if segment]
        impossible_speed_count = sum(speed > 14.0 for speed in raw_speeds)
        discarded_observations = len(projected_points) - len(selected_points)
        trajectory_qa = {
            "rawObservationCount": len(projected_points),
            "retainedObservationCount": len(selected_points),
            "discardedObservationCount": discarded_observations,
            "fragmentCount": len(non_empty_segments),
            "discardedFragmentCount": max(0, len(non_empty_segments) - 1),
            "rawSpeedSampleCount": len(raw_speeds),
            "impossibleSpeedSegmentCount": impossible_speed_count,
            "maximumRawSpeedMetresPerSecond": (
                round(max(raw_speeds), 3) if raw_speeds else None
            ),
        }
        trajectory_diagnostics["retainedProjectedObservationCount"] += len(selected_points)
        trajectory_diagnostics["discardedProjectedObservationCount"] += discarded_observations
        trajectory_diagnostics["preFilterSpeedSampleCount"] += len(raw_speeds)
        trajectory_diagnostics["preFilterSpeedViolationCount"] += impossible_speed_count
        if len(non_empty_segments) > 1:
            trajectory_diagnostics["splitTrajectoryCount"] += 1
            trajectory_diagnostics["discardedTrajectoryFragmentCount"] += len(non_empty_segments) - 1
        if raw_speeds:
            current_maximum = trajectory_diagnostics["preFilterMaximumSpeedMetresPerSecond"]
            trajectory_diagnostics["preFilterMaximumSpeedMetresPerSecond"] = max(
                float(current_maximum or 0.0),
                max(raw_speeds),
            )
        is_manual = bool(track.positive_annotation_ids)
        maximum = 11 if team in {"home", "away"} else 6
        if (len(selected_points) < minimum and not is_manual) or (
            counts[team] >= maximum and not is_manual
        ):
            continue
        counts[team] += 1
        xs = _smooth([point[0] for point in selected_positions])
        zs = _smooth([point[1] for point in selected_positions])
        keyframes = [
            {
                "t": round(point["t"], 3),
                "x": round(x, 2),
                "z": round(z, 2),
                "confidence": round(0.35 + min(1.0, point["confidence"]) * 0.62, 3),
                "observed": True,
                "presenceState": "observed",
                "projectionSource": source,
                "calibrationFrameIndex": calibration_frame_index,
                "positionUncertaintyMetres": uncertainty,
                "projection": {
                    "source": source,
                    "calibrationFrameIndex": calibration_frame_index,
                    "uncertaintyMetres": uncertainty,
                },
            }
            for point, x, z, (source, calibration_frame_index, uncertainty) in zip(
                selected_points,
                xs,
                zs,
                selected_meta,
            )
        ]
        projected_by_point = {
            id(point): (position, source, calibration_frame_index, uncertainty)
            for point, position, source, calibration_frame_index, uncertainty in projected_points
        }
        selected_by_point = {
            id(point): (x, z, source, calibration_frame_index, uncertainty)
            for point, x, z, (source, calibration_frame_index, uncertainty) in zip(
                selected_points,
                xs,
                zs,
                selected_meta,
            )
        }
        observation_rows: list[dict] = []
        for point in track.points:
            if point.get("frameIndex") is None or not point.get("bbox"):
                continue
            observation = {
                "frameIndex": int(point["frameIndex"]),
                "sceneTime": round(float(point["t"]), 3),
                "bbox": {
                    "x": round(float(point["bbox"]["x"]), 2),
                    "y": round(float(point["bbox"]["y"]), 2),
                    "width": round(float(point["bbox"]["width"]), 2),
                    "height": round(float(point["bbox"]["height"]), 2),
                },
                "confidence": round(float(point.get("confidence") or 0.0), 3),
                "annotationId": point.get("annotationId"),
            }
            if track.canonical_person_id:
                observation.update(
                    {
                        "id": point.get("observationId"),
                        "observationId": point.get("observationId"),
                        "sourceFrameIndex": int(point["frameIndex"]),
                        "sourceTime": round(
                            float(
                                scene.get("payload", {})
                                .get("videoAsset", {})
                                .get("sourceStart")
                                or 0.0
                            )
                            + float(point["t"]),
                            3,
                        ),
                        "sourceTrackletId": point.get("sourceTrackletId")
                        or track.local_tracklet_id,
                        "canonicalPersonId": track.canonical_person_id,
                    }
                )
            selected = selected_by_point.get(id(point))
            projected = projected_by_point.get(id(point))
            if selected is not None:
                x, z, source, calibration_frame_index, uncertainty = selected
                observation.update(
                    {
                        "metricStatus": "accepted",
                        "metricReason": None,
                        "pitch": {"x": round(float(x), 2), "z": round(float(z), 2)},
                    }
                )
            elif projected is not None:
                raw_position, source, calibration_frame_index, uncertainty = projected
                observation.update(
                    {
                        "metricStatus": "rejected",
                        "metricReason": "trajectory-fragment-rejected",
                        "rawPitch": {
                            "x": round(float(raw_position[0]), 2),
                            "z": round(float(raw_position[1]), 2),
                        },
                    }
                )
            else:
                source = point.get("projectionSource")
                calibration_frame_index = point.get("calibrationFrameIndex")
                uncertainty = point.get("positionUncertaintyMetres")
                observation.update(
                    {
                        "metricStatus": "unprojected",
                        "metricReason": "metric-projection-unavailable",
                    }
                )
            if source:
                observation["projectionSource"] = str(source)
            if calibration_frame_index is not None:
                observation["calibrationFrameIndex"] = int(calibration_frame_index)
            if uncertainty is not None:
                observation["positionUncertaintyMetres"] = round(float(uncertainty), 3)
            observation_rows.append(observation)
        observations = _merge_track_observations(observation_rows)
        observation_status_counts = {
            status: sum(item.get("metricStatus") == status for item in observations)
            for status in ("accepted", "rejected", "unprojected")
        }
        source_observation_count = sum(
            point.get("frameIndex") is not None and bool(point.get("bbox"))
            for point in track.points
        )
        trajectory_qa.update(
            {
                "imageObservationCount": source_observation_count,
                "publishedIdentityObservationCount": len(observations),
                "metricAcceptedObservationCount": observation_status_counts["accepted"],
                "metricRejectedObservationCount": observation_status_counts["rejected"],
                "metricUnprojectedObservationCount": observation_status_counts["unprojected"],
                "identityObservationCoverage": round(
                    len(observations) / max(1, source_observation_count), 3
                ),
                "metricObservationCoverage": round(
                    observation_status_counts["accepted"] / max(1, len(observations)), 3
                ),
            }
        )
        trajectory_diagnostics["acceptedIdentityImageObservationCount"] += source_observation_count
        trajectory_diagnostics["publishedIdentityObservationCount"] += len(observations)
        trajectory_diagnostics["metricAcceptedIdentityObservationCount"] += observation_status_counts[
            "accepted"
        ]
        trajectory_diagnostics["metricRejectedIdentityObservationCount"] += observation_status_counts[
            "rejected"
        ]
        trajectory_diagnostics["metricUnprojectedIdentityObservationCount"] += observation_status_counts[
            "unprojected"
        ]
        keyframes, presence = _continuous_track_keyframes(
            keyframes,
            float(scene["duration"]),
            scene["payload"]["pitch"],
            track.id,
        )
        role = _annotation_role(track.manual_kind) or track.role
        is_goalkeeper = role == "goalkeeper"
        default_label = (
            f"{team.title()} goalkeeper"
            if is_goalkeeper
            else "Referee"
            if role == "referee"
            else "Other person"
            if role == "other"
            else f"{team.title()} track {counts[team]:02d}"
        )
        color = (
            "#f1c84c"
            if role == "referee"
            else "#a78bfa"
            if role == "other"
            else _cluster_color(track.feature)
            if is_goalkeeper
            else colors.get(team, "#d7dce8")
        )
        result.append(
            {
                "id": f"auto-{team}-{counts[team]:02d}",
                "label": track.manual_label or default_label,
                "teamId": team,
                "color": color,
                "number": counts[team] if team in {"home", "away"} else 0,
                "externalPlayerId": track.manual_external_player_id,
                "source": "manual-anchor" if is_manual else "automatic",
                "coordinateMode": resolved_coordinate_mode,
                **({"role": role} if role else {}),
                **({"annotationIds": sorted(track.annotation_ids)} if track.annotation_ids else {}),
                "trajectoryQa": trajectory_qa,
                "presence": presence,
                "observations": observations,
                "keyframes": keyframes,
                **(
                    {
                        "canonicalPersonId": track.canonical_person_id,
                        "sourceTrackletIds": sorted(
                            track.source_tracklet_ids or {track.local_tracklet_id}
                        ),
                    }
                    if track.canonical_person_id
                    else {}
                ),
                **(
                    {"identitySplitPartitions": dict(track.identity_split_partitions)}
                    if track.identity_split_partitions
                    else {}
                ),
            }
        )
    if diagnostics is not None:
        maximum = trajectory_diagnostics["preFilterMaximumSpeedMetresPerSecond"]
        if maximum is not None:
            trajectory_diagnostics["preFilterMaximumSpeedMetresPerSecond"] = round(
                float(maximum), 3
            )
        accepted_identity_observations = int(
            trajectory_diagnostics["acceptedIdentityImageObservationCount"]
        )
        published_identity_observations = int(
            trajectory_diagnostics["publishedIdentityObservationCount"]
        )
        metric_accepted_observations = int(
            trajectory_diagnostics["metricAcceptedIdentityObservationCount"]
        )
        trajectory_diagnostics["identityObservationCoverage"] = round(
            published_identity_observations / max(1, accepted_identity_observations), 3
        )
        trajectory_diagnostics["metricObservationCoverage"] = round(
            metric_accepted_observations / max(1, published_identity_observations), 3
        )
        diagnostics.update(trajectory_diagnostics)
    return result


def _ball_keyframes(
    ball_frames: list[tuple[list[dict], float]],
    frame_size: tuple[int, int],
    scene: dict,
    calibration: PitchCalibration | None = None,
    coordinate_mode: str | None = None,
) -> list[dict]:
    candidates: list[list[dict]] = []
    for frame_index, (detections, time) in enumerate(ball_frames):
        for detection in detections:
            best = None
            for track in candidates:
                gap = frame_index - track[-1]["frame"]
                if gap < 1 or gap > 2:
                    continue
                distance = hypot(detection["x"] - track[-1]["x"], detection["y"] - track[-1]["y"])
                if distance <= 72 * gap and (best is None or distance < best[0]):
                    best = (distance, track)
            point = {**detection, "frame": frame_index, "t": time}
            if best is None:
                candidates.append([point])
            else:
                best[1].append(point)
    stable = [track for track in candidates if len(track) >= 3]
    if not stable:
        return []
    selected = max(
        stable,
        key=lambda track: (
            max(point["confidence"] for point in track) * 2.0
            + float(np.mean([point["confidence"] for point in track]))
            + min(12, len(track)) * 0.01
        ),
    )
    peak_confidence = max(point["confidence"] for point in selected)
    mean_confidence = float(np.mean([point["confidence"] for point in selected]))
    if peak_confidence < 0.25 or mean_confidence < 0.09:
        return []
    width, height = frame_size
    resolved_coordinate_mode = coordinate_mode or (
        "metric"
        if calibration is not None and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD
        else "approximate"
    )
    keyframes: list[dict] = []
    for point in selected:
        if resolved_coordinate_mode == "metric":
            if point.get("pitchX") is None or point.get("pitchZ") is None:
                continue
            position = (float(point["pitchX"]), float(point["pitchZ"]))
            source = str(point.get("projectionSource") or "direct")
            calibration_frame_index = point.get("calibrationFrameIndex")
            uncertainty = point.get("positionUncertaintyMetres")
        else:
            position = _project(
                point["x"],
                point["y"],
                width,
                height,
                scene["payload"]["pitch"],
                None,
            )
            source = "screen-approximate"
            calibration_frame_index = None
            uncertainty = 14.0
        keyframes.append(
            {
                "t": round(point["t"], 3),
                "x": round(position[0], 2),
                # Height is explicitly unknown for a single broadcast camera.
                # Keep the legacy rendering value but mark the estimate as such.
                "y": 0.22,
                "heightSource": "rendering-placeholder",
                "z": round(position[1], 2),
                "confidence": round(0.25 + point["confidence"] * 0.7, 3),
                "projectionSource": source,
                "calibrationFrameIndex": calibration_frame_index,
                "positionUncertaintyMetres": uncertainty,
                "projection": {
                    "source": source,
                    "calibrationFrameIndex": calibration_frame_index,
                    "uncertaintyMetres": uncertainty,
                },
            }
        )
    return keyframes


def _detect_ball_frames(
    scene: dict,
    detector: BallDetector,
    fallback_detector: BallDetector | None,
    sampled_frames: list[tuple[Path, float]],
    legacy_ball_frames: list[tuple[list[dict], float]],
    on_progress: Callable[[int, int, str], None] | None = None,
    *,
    failure_policy: str | None = None,
    detector_input: dict | None = None,
) -> tuple[list[tuple[list[dict], float]], dict, list[dict], list[str]]:
    """Run the independent dense ball detector without risking player output.

    Dense source decoding is accuracy-critical but deliberately non-fatal: a
    missing source/cache or unavailable challenger falls back to the sampled
    frames and, as the final honest fallback, the already-computed COCO
    candidates. Every fallback is returned in metadata and warnings.
    """

    warnings: list[str] = []
    dense_cache_key: str | None = None
    detection_cache_asset_directory: Path | None = None
    try:
        dense = dense_ball_frame_paths(scene)
        source_frames = list(dense.frames)
        dense_cache_key = dense.cache_key
        if isinstance(detector_input, dict):
            detection_cache_asset_directory = (
                Path(get_settings().media_root).resolve()
                / str(scene["payload"]["videoAsset"]["id"])
            )
        source_metadata = {
            "source": "dense-source-cache",
            "frameRate": round(dense.frame_rate, 3),
            "frameCount": len(source_frames),
            "cacheKey": dense.cache_key,
            "cacheHit": dense.cache_hit,
            "sourceStart": dense.source_start,
            "sourceEnd": dense.source_end,
        }
    except (DenseBallFramesError, KeyError, OSError, ValueError) as exc:
        source_frames = sampled_frames
        source_metadata = {
            "source": "sampled-frame-fallback",
            "frameRate": float(
                scene.get("payload", {})
                .get("videoAsset", {})
                .get("analysisFps")
                or get_settings().reconstruction_frame_rate
            ),
            "frameCount": len(source_frames),
            "cacheHit": False,
            "fallbackReason": str(exc),
        }
        warnings.append(
            f"Dense ball frames were unavailable; sampled frames were used: {exc}"
        )

    if (
        dense_cache_key
        and detection_cache_asset_directory is not None
        and isinstance(detector_input, dict)
    ):
        cached_entry = load_ball_detection_cache(
            detection_cache_asset_directory,
            dense_cache_key=dense_cache_key,
            detector_input=detector_input,
        )
        if cached_entry is not None and cached_entry.primary_backend == detector.backend_name:
            cached_resolved, cached_batches = cached_entry.as_pipeline_data()
            timestamps_match = len(cached_resolved) == len(source_frames) and all(
                abs(float(cached_time) - float(source_time)) <= 1e-6
                for (_, cached_time), (_, source_time) in zip(
                    cached_resolved, source_frames, strict=True
                )
            )
            if timestamps_match and len(cached_batches) == len(source_frames):
                backend_names = sorted(
                    {
                        str(item.get("backend") or "unknown")
                        for item in cached_batches
                    }
                )
                source_metadata.update(
                    {
                        "detectionCacheHit": True,
                        "detectionCacheKey": cached_entry.cache_key,
                        "failedFrameCount": 0,
                        "fallbackFrameCount": 0,
                        "circuitBreaker": {"opened": False, "reason": None},
                        "backendCounts": {
                            backend: sum(
                                item.get("backend") == backend
                                for item in cached_batches
                            )
                            for backend in backend_names
                        },
                    }
                )
                if on_progress is not None:
                    on_progress(
                        len(source_frames),
                        len(source_frames),
                        f"{detector.backend_name} · cached detections",
                    )
                return cached_resolved, source_metadata, cached_batches, warnings
            source_metadata["detectionCacheInvalidReason"] = (
                "cached frame count or timestamps do not match dense frames"
            )
    source_metadata["detectionCacheHit"] = False

    resolved: list[tuple[list[dict], float]] = []
    batches: list[dict] = []
    per_frame_failures = 0
    fallback_frame_count = 0
    settings = get_settings()
    failure_policy = str(failure_policy or settings.ball_detection_failure_policy)
    if failure_policy not in {"raise", "fallback"}:
        raise ReconstructionError(
            "BALL_DETECTION_FAILURE_POLICY must be raise or fallback"
        )
    primary_circuit_reason: str | None = None
    source_paths = [Path(path) for path, _ in source_frames]
    legacy_match_tolerance = 0.51 / max(
        1.0, float(source_metadata.get("frameRate") or 1.0)
    )
    # Map every low-rate legacy observation to at most one dense frame.  A
    # nearest lookup inside the loop used to duplicate the same COCO ball over
    # several dense frames (or, with sparse inputs, over an entire clip), which
    # falsely looked like high-coverage observed evidence.
    legacy_by_dense_frame: dict[int, list[dict]] = {}
    legacy_distance_by_dense_frame: dict[int, float] = {}
    if source_frames:
        dense_times = [float(item[1]) for item in source_frames]
        for legacy_items, legacy_time in legacy_ball_frames:
            if not legacy_items:
                continue
            dense_index = min(
                range(len(dense_times)),
                key=lambda index: abs(dense_times[index] - float(legacy_time)),
            )
            distance = abs(dense_times[dense_index] - float(legacy_time))
            if distance > legacy_match_tolerance:
                continue
            legacy_by_dense_frame.setdefault(dense_index, []).extend(legacy_items)
            legacy_distance_by_dense_frame[dense_index] = min(
                distance,
                legacy_distance_by_dense_frame.get(dense_index, distance),
            )
    for frame_index, (path, time) in enumerate(source_frames):
        batch = None
        failure_detail: str | None = None
        previous_path = source_paths[max(0, frame_index - 1)]
        following_path = source_paths[min(len(source_paths) - 1, frame_index + 1)]
        context_paths = (previous_path, following_path)

        if primary_circuit_reason is not None and fallback_detector is not None:
            failure_detail = f"circuit-open after {primary_circuit_reason}"
            try:
                batch = fallback_detector.detect(
                    path,
                    frame_index=frame_index,
                    timestamp=float(time),
                    context_frames=context_paths,
                )
            except Exception as fallback_exc:
                failure_detail += (
                    f"; fallback {type(fallback_exc).__name__}: {fallback_exc}"
                )
        else:
            try:
                batch = detector.detect(
                    path,
                    frame_index=frame_index,
                    timestamp=float(time),
                    context_frames=context_paths,
                )
            except Exception as exc:  # fail or cross the explicit fallback boundary
                failure_detail = f"{type(exc).__name__}: {exc}"
                if failure_policy == "raise":
                    raise ReconstructionError(
                        f"Ball detector {detector.backend_name} failed on dense frame "
                        f"{frame_index + 1}/{len(source_frames)}: {failure_detail}"
                    ) from exc
                if fallback_detector is not None:
                    try:
                        batch = fallback_detector.detect(
                            path,
                            frame_index=frame_index,
                            timestamp=float(time),
                            context_frames=context_paths,
                        )
                    except Exception as fallback_exc:
                        failure_detail += (
                            f"; fallback {type(fallback_exc).__name__}: {fallback_exc}"
                        )
                        primary_circuit_reason = failure_detail
                    else:
                        # A model/worker failure is unlikely to recover within
                        # one offline shot.  Open the circuit after the first
                        # failed request so a 60-second clip cannot accumulate
                        # 1,500 identical network timeouts.
                        primary_circuit_reason = failure_detail

        if batch is not None:
            adapter_fallback_reason = batch.metadata.get("fallbackReason")
            if adapter_fallback_reason and failure_detail is None:
                # The WASB adapter can successfully return the explicitly
                # configured dedicated detector.  That is a usable batch, but
                # it is still degraded evidence and must not be reported as a
                # successful WASB frame.
                failure_detail = str(adapter_fallback_reason)
        if failure_detail is not None:
            fallback_frame_count += 1

        if batch is None:
            per_frame_failures += 1
            legacy_items = legacy_by_dense_frame.get(frame_index, [])
            legacy_distance = legacy_distance_by_dense_frame.get(frame_index)
            detections = [
                {
                    **deepcopy(item),
                    "detectorBackend": "legacy-coco-fallback",
                    "candidateId": f"ball-f{frame_index:05d}-legacy-{rank:02d}",
                    "provenance": {
                        "backend": "legacy-coco-fallback",
                        "failureReason": failure_detail,
                    },
                }
                for rank, item in enumerate(legacy_items, start=1)
            ]
            backend = "legacy-coco-fallback"
            metadata = {
                "fallbackReason": failure_detail,
                "legacyMatchDistanceSeconds": (
                    round(legacy_distance, 5) if legacy_distance is not None else None
                ),
                "legacyMatchToleranceSeconds": round(legacy_match_tolerance, 5),
                "legacyCandidateAccepted": bool(legacy_items),
            }
            image_size = None
        else:
            detections = batch.as_reconstruction_detections()
            backend = batch.backend
            metadata = dict(batch.metadata)
            image_size = batch.image_size
            for rank, item in enumerate(detections, start=1):
                candidate_id = f"ball-f{frame_index:05d}-c{rank:02d}"
                item["candidateId"] = candidate_id
                item["sourceFrameIndex"] = frame_index
                item["imageWidth"] = int(batch.image_size[0])
                item["imageHeight"] = int(batch.image_size[1])
                item["provenance"] = {
                    "backend": item.get("detectorBackend") or backend,
                    "candidateId": candidate_id,
                    "detectorMetadata": deepcopy(item.get("detectorMetadata") or {}),
                    "batchMetadata": deepcopy(metadata),
                }
        resolved.append((detections, float(time)))
        batches.append(
            {
                "frameIndex": frame_index,
                "t": round(float(time), 4),
                "backend": backend,
                "candidateCount": len(detections),
                "imageSize": list(image_size) if image_size else None,
                "fallbackReason": failure_detail,
                "metadata": metadata,
            }
        )
        if on_progress is not None:
            on_progress(
                frame_index + 1,
                len(source_frames),
                f"{backend} · {len(detections)} candidate(s)",
            )

    if per_frame_failures:
        warnings.append(
            f"Ball detector failed on {per_frame_failures}/{len(source_frames)} frames; explicit legacy candidates were retained where available."
        )
    if fallback_frame_count:
        warnings.append(
            f"The requested ball detector used an explicit fallback on {fallback_frame_count}/{len(source_frames)} frames; inspect backendCounts and per-frame fallbackReason."
        )
    source_metadata["failedFrameCount"] = per_frame_failures
    source_metadata["fallbackFrameCount"] = fallback_frame_count
    source_metadata["circuitBreaker"] = {
        "opened": primary_circuit_reason is not None,
        "reason": primary_circuit_reason,
    }
    source_metadata["backendCounts"] = {
        backend: sum(item["backend"] == backend for item in batches)
        for backend in sorted({str(item["backend"]) for item in batches})
    }
    if (
        dense_cache_key
        and detection_cache_asset_directory is not None
        and isinstance(detector_input, dict)
    ):
        try:
            stored_entry = store_clean_ball_detection_cache(
                detection_cache_asset_directory,
                dense_cache_key=dense_cache_key,
                detector_input=detector_input,
                primary_backend=detector.backend_name,
                resolved_frames=resolved,
                batches=batches,
                failed_frame_count=per_frame_failures,
                fallback_frame_count=fallback_frame_count,
            )
        except (BallDetectionCacheError, OSError) as exc:
            # Cache publication is an optimization boundary. The detector
            # evidence remains valid and must not turn into a failed or review
            # reconstruction merely because local cache storage is unavailable.
            source_metadata["detectionCacheWriteError"] = str(exc)
        else:
            if stored_entry is not None:
                source_metadata["detectionCacheKey"] = stored_entry.cache_key
                source_metadata["detectionCacheStored"] = True
            else:
                source_metadata["detectionCacheStored"] = False
    return resolved, source_metadata, batches, warnings


def _ball_world_projection_status(coordinate_mode: str, keyframes: list[dict]) -> str:
    if coordinate_mode == "unavailable":
        return "calibration-rejected"
    return "published" if keyframes else "no-stable-trajectory"


def reconstruct_scene(
    scene: dict,
    progress_listener: Callable[[dict], None] | None = None,
    expected_run_id: str | None = None,
    expected_input_fingerprint: str | None = None,
    expected_lease_owner_id: str | None = None,
) -> dict:
    previous_tracks = deepcopy(scene.get("payload", {}).get("tracks") or [])
    previous_canonical_people = deepcopy(
        scene.get("payload", {}).get("canonicalPeople") or []
    )
    previous_ball = deepcopy(scene.get("payload", {}).get("ball") or {"keyframes": []})
    previous_team_colors = {
        str(team.get("id")): team.get("color")
        for team in scene.get("payload", {}).get("teams") or []
    }
    previous_processing_state = (
        scene.get("payload", {}).get("videoAsset", {}).get("processingState") or "frames-ready"
    )
    model_name = str(
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("model")
        or get_settings().reconstruction_model
    )
    reconstruction_request = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        or {}
    )
    ball_backend = str(
        reconstruction_request.get("ballBackend")
        or get_settings().ball_detection_backend
    )
    queued_ball_detection_input = reconstruction_request.get("ballDetectionInput")
    ball_detection_input = (
        deepcopy(queued_ball_detection_input)
        if isinstance(queued_ball_detection_input, dict)
        else _ball_detection_input(ball_backend)
    )
    progress = ReconstructionProgress(
        scene,
        progress_listener,
        expected_run_id,
        expected_input_fingerprint,
        expected_lease_owner_id,
    )
    set_reconstruction_status(
        scene,
        "processing",
        expected_run_id=expected_run_id,
        expected_input_fingerprint=expected_input_fingerprint,
        expected_lease_owner_id=expected_lease_owner_id,
        processingStatus="processing",
        qualityVerdict="pending",
        model=model_name,
        ballBackend=ball_backend,
        ballDetectionInput=ball_detection_input,
        startedAt=datetime.now(UTC).isoformat(),
        error=None,
    )
    try:
        progress.update(
            "preparing",
            1,
            "Preparing sampled frames",
            "Reading the scene range and checking extracted images.",
            0,
            4,
            completed=0,
            total=1,
        )
        frames = _frame_paths(scene)
        if not frames:
            raise ReconstructionError("No sampled frames are available for this moment")
        progress.update(
            "preparing",
            1,
            "Inputs ready",
            f"Found {len(frames)} sampled frames for analysis.",
            0,
            4,
            completed=1,
            total=1,
            eta_padding=max(8.0, len(frames) * 0.35),
        )
        reconstruction_input = reconstruction_request
        manual_overrides = _manual_pitch_calibration_overrides(reconstruction_input)
        # Legacy metadata consumers still receive one representative override;
        # direct reconstruction below uses every item in the collection.
        manual_override = (
            reconstruction_input.get("pitchCalibrationOverride")
            or (manual_overrides[-1] if manual_overrides else {})
        )
        frame_calibrations: dict[int, PitchCalibration] = {}
        calibration_warnings: list[str] = []
        identity_warnings: list[str] = []
        identity_worker_diagnostics: dict = {
            "status": "pending",
            "provider": "prtreid-bpbreid-soccernet",
        }
        def calibration_progress(
            backend: str,
            completed: int,
            total: int,
            fraction: float,
            calibrated: int,
        ) -> None:
            backend_label = (
                "PnLCalib points + lines"
                if backend == "pnlcalib"
                else "Local semantic-keypoint fallback"
            )
            manual_detail = (
                f" · {len(manual_overrides)} manual frame anchor(s) will override matching samples"
                if manual_overrides
                else ""
            )
            progress.update(
                "calibration",
                2,
                "Calibrating the pitch",
                f"{backend_label} · {completed}/{total} frames · {calibrated} valid homographies{manual_detail}.",
                4,
                62,
                completed=completed,
                total=total,
                fraction=fraction,
                eta_padding=max(6.0, len(frames) * 0.25),
            )

        frame_calibrations, calibration_warnings = _automatic_frame_calibrations(
            frames,
            calibration_progress,
        )
        # Side canonicalization happens later with the source frame width.
        # Attack direction is independent match semantics and never flips H.
        progress.update(
            "detection",
            3,
            "Loading object detectors",
            (
                f"Preparing {model_name} for people and {ball_backend} for "
                "dense ball inference."
            ),
            62,
            84,
            completed=0,
            total=len(frames),
        )
        model = _load_model(model_name)
        person_detection_input = _person_detection_input(model_name, model)
        person_detection_cache_diagnostics = _base_detection_cache_diagnostics(
            len(frames),
            person_detection_input,
        )
        person_detection_cache_directory = (
            Path(get_settings().media_root).resolve()
            / str(scene["payload"]["videoAsset"]["id"])
        )
        ball_detector, ball_fallback_detector = _configured_ball_detectors(
            model,
            ball_backend,
            ball_detection_input,
        )
        person_frames: list[tuple[list[Detection], float]] = []
        legacy_ball_frames: list[tuple[list[dict], float]] = []
        ball_frames: list[tuple[list[dict], float]] = []
        ball_detection_batches: list[dict] = []
        ball_detection_warnings: list[str] = []
        ball_dense_frame_metadata: dict = {}
        person_counts: list[int] = []
        ball_counts: list[int] = []
        frame_size = (960, 540)
        previous_image: np.ndarray | None = None
        camera_transform = np.eye(3, dtype=np.float64)
        camera_transforms: dict[int, np.ndarray] = {}
        camera_motion_edges: dict[int, CameraMotionEstimate] = {}
        frame_sizes: dict[int, tuple[int, int]] = {}
        accepted_frame_calibrations: dict[int, PitchCalibration] = {}
        accepted_automatic_direct_by_sample: dict[int, PitchCalibration] = {}
        accepted_manual_direct_by_sample: dict[int, PitchCalibration] = {}
        resolved_calibrations_by_sample: dict[int, PitchCalibration] = {}
        calibration_anchor_by_sample: dict[int, int] = {}
        calibration_uncertainty_by_sample: dict[int, float] = {}
        frame_evidence: list[dict] = []
        rejected_calibration_frames = 0
        calibration: PitchCalibration | None = None
        metric_person_samples = 0
        metric_ball_samples = 0
        manual_stabilized_by_sample: dict[int, PitchCalibration] = {}
        manual_override_by_sample: dict[int, dict] = {}
        for stored_override in manual_overrides:
            requested_source_frame = stored_override.get("sourceFrameIndex")
            if requested_source_frame is not None:
                manual_sample_index = min(
                    range(len(frames)),
                    key=lambda index: abs(
                        _source_frame_index(frames[index][0])
                        - int(requested_source_frame)
                    ),
                )
            else:
                requested_scene_time = float(stored_override.get("sceneTime") or 0.0)
                manual_sample_index = min(
                    range(len(frames)),
                    key=lambda index: abs(float(frames[index][1]) - requested_scene_time),
                )
            anchor_source_frame_index = _source_frame_index(
                frames[manual_sample_index][0]
            )
            manual_alignment_error = stored_override.get("alignmentError")
            manual_stabilized_by_sample[manual_sample_index] = PitchCalibration(
                image_to_pitch=np.asarray(stored_override["imageToPitch"], dtype=np.float64),
                confidence=float(stored_override.get("confidence") or 0.0),
                supported_lines=int(stored_override.get("supportedLines") or 4),
                mean_line_score=float(stored_override.get("meanLineScore") or 0.0),
                rectangle=str(stored_override.get("preset") or "manual"),
                matched_curves=int(stored_override.get("matchedCurves") or 0),
                method="manual-pitch-anchors",
                keypoint_count=int(stored_override.get("supportedLines") or 4),
                inlier_count=int(stored_override.get("supportedLines") or 4),
                reprojection_error=(
                    float(manual_alignment_error)
                    if manual_alignment_error is not None
                    else None
                ),
                frame_index=anchor_source_frame_index,
                confidence_kind="manual-alignment-quality-score",
            )
            manual_override_by_sample[manual_sample_index] = stored_override
        for frame_index, (path, time) in enumerate(frames):
            image, people, balls = _cached_base_frame_detections(
                model,
                path,
                person_detection_cache_directory,
                person_detection_input,
                person_detection_cache_diagnostics,
            )
            frame_size = (image.shape[1], image.shape[0])
            frame_sizes[frame_index] = frame_size
            source_frame_index = _source_frame_index(path)
            people = _apply_person_annotations(
                image,
                people,
                _frame_annotations(scene, source_frame_index),
            )
            _capture_detection_observations(people, source_frame_index)
            person_counts.append(len(people))
            ball_counts.append(len(balls))
            camera_motion_payload: dict = {
                "status": "first-frame",
                "model": "projective-homography",
                "confidence": 1.0,
                "currentToPrevious": _matrix_payload(np.eye(3, dtype=np.float64)),
                "metrics": {},
                "rejectionReasons": [],
            }
            if previous_image is not None:
                motion = _camera_motion_estimate(previous_image, image)
                camera_motion_edges[frame_index] = motion
                camera_motion_payload = motion.as_dict()
                if motion.reliable:
                    camera_transform = camera_transform @ motion.matrix
                else:
                    # A new shot (or an unestimated edge) starts a new image
                    # stabilization coordinate system. Never compose the next
                    # shot onto a stale transform from the previous camera.
                    camera_transform = np.eye(3, dtype=np.float64)
            camera_transforms[source_frame_index] = camera_transform.copy()
            automatic_calibration = frame_calibrations.get(source_frame_index)
            if automatic_calibration is not None:
                automatic_calibration = canonicalize_penalty_side(
                    automatic_calibration,
                    frame_size[0],
                )
            automatic_evidence = _frame_calibration_evidence(
                scene,
                frame_index,
                time,
                image,
                automatic_calibration,
                projection_source="direct",
                people=people,
                pitch=scene["payload"]["pitch"],
                source_frame_index=source_frame_index,
            )
            manual_stabilized = manual_stabilized_by_sample.get(frame_index)
            frame_calibration = automatic_calibration
            evidence = automatic_evidence
            selected_is_manual = False
            if manual_stabilized is not None:
                current_to_pitch = manual_stabilized.image_to_pitch @ camera_transform
                current_to_pitch /= current_to_pitch[2, 2]
                manual_calibration = replace(
                    manual_stabilized,
                    image_to_pitch=current_to_pitch,
                    frame_index=source_frame_index,
                )
                manual_evidence = _frame_calibration_evidence(
                    scene,
                    frame_index,
                    time,
                    image,
                    manual_calibration,
                    projection_source="manual-direct",
                    people=people,
                    pitch=scene["payload"]["pitch"],
                    source_frame_index=source_frame_index,
                    manual=True,
                )
                # An accepted manual observation is authoritative at its exact
                # sample. If it is rejected, a separately accepted automatic
                # observation remains eligible instead of leaving a false gap.
                if manual_evidence["status"] == "accepted":
                    frame_calibration = manual_calibration
                    evidence = manual_evidence
                    selected_is_manual = True
                elif automatic_evidence["status"] != "accepted":
                    frame_calibration = manual_calibration
                    evidence = manual_evidence
                    selected_is_manual = True
                evidence["manualObservation"] = {
                    "kind": "manual",
                    **_calibration_attempt_payload(manual_evidence),
                }
                evidence["automaticObservation"] = {
                    "kind": "automatic",
                    **_calibration_attempt_payload(automatic_evidence),
                }
                evidence["observations"] = [
                    evidence["manualObservation"],
                    evidence["automaticObservation"],
                ]
                # Compatibility/diagnostic alias: both values describe raw
                # candidate observations before temporal resolution.
                evidence["directAttempts"] = deepcopy(evidence["observations"])
            evidence["cameraMotion"] = {
                **camera_motion_payload,
                "currentToReference": _matrix_payload(camera_transform),
            }
            accepted_calibration = evidence["status"] == "accepted"
            if accepted_calibration:
                assert frame_calibration is not None
                accepted_frame_calibrations[source_frame_index] = frame_calibration
                if selected_is_manual:
                    accepted_manual_direct_by_sample[frame_index] = frame_calibration
                else:
                    accepted_automatic_direct_by_sample[frame_index] = frame_calibration
            elif frame_calibration is not None:
                rejected_calibration_frames += 1
            frame_evidence.append(evidence)
            person_frames.append((people, time))
            legacy_ball_frames.append((balls, time))
            previous_image = image
            progress.update(
                "detection",
                3,
                "Detecting people and camera evidence",
                (
                    f"Sample {frame_index + 1}/{len(frames)} · {len(people)} people · "
                    f"{len(balls)} generic ball fallback candidate(s)."
                ),
                62,
                84,
                completed=frame_index + 1,
                total=len(frames),
                eta_padding=5.0,
            )

        person_detection_cache_diagnostics.update(
            {
                "status": (
                    "degraded"
                    if person_detection_cache_diagnostics["errors"]
                    else "ready"
                ),
                "hitRatio": round(
                    person_detection_cache_diagnostics["hits"] / max(1, len(frames)),
                    4,
                ),
                "baseBoundary": (
                    "pre-annotation/pre-calibration/pre-tracking/pre-reid/pre-ocr"
                ),
            }
        )

        identity_requests = _identity_embedding_requests(frames, person_frames)
        if identity_requests:
            worker_status = identity_worker_readiness(timeout=2.0)
            identity_worker_diagnostics = deepcopy(worker_status)
            if worker_status.get("status") == "ready":
                progress.update(
                    "detection",
                    3,
                    "Extracting player identity evidence",
                    f"PRTReID is evaluating {sum(len(item[2]) for item in identity_requests)} player crops.",
                    82,
                    84,
                    completed=0,
                    total=len(identity_requests),
                )

                def identity_progress(completed: int, total: int, usable: int) -> None:
                    progress.update(
                        "detection",
                        3,
                        "Extracting player identity evidence",
                        f"PRTReID frames {completed}/{total} · {usable} usable player crops.",
                        82,
                        84,
                        completed=completed,
                        total=total,
                        eta_padding=2.0,
                    )

                try:
                    identity_results = embed_identity_frames(
                        identity_requests,
                        identity_progress,
                    )
                    identity_worker_diagnostics = {
                        **worker_status,
                        **_attach_identity_embeddings(person_frames, identity_results),
                    }
                except IdentityWorkerError as exc:
                    identity_worker_diagnostics = {
                        **worker_status,
                        "status": "failed",
                        "detail": str(exc),
                    }
                    identity_warnings.append(
                        "PRTReID identity extraction failed; automatic cross-gap identity merging was disabled."
                    )
            else:
                identity_warnings.append(
                    "PRTReID identity worker is unavailable; local tracklets remain provisional and are not auto-merged across gaps."
                )
        else:
            identity_worker_diagnostics = {
                "status": "no-observations",
                "provider": "prtreid-bpbreid-soccernet",
                "requestedObservationCount": 0,
                "usableObservationCount": 0,
                "rejectedObservationCount": 0,
                "usableCropRatio": 0.0,
                "crops": [],
            }

        accepted_direct_by_sample = _merge_direct_calibration_anchors(
            accepted_automatic_direct_by_sample,
            accepted_manual_direct_by_sample,
        )
        temporal_recovered_frames = 0
        progress.update(
            "detection",
            3,
            "Resolving camera hypotheses",
            "Running forward and backward camera inference; later strong frames may recover earlier partial views.",
            62,
            84,
            completed=len(frames),
            total=len(frames),
            eta_padding=3.0,
        )
        (
            resolved_calibrations_by_sample,
            calibration_anchor_by_sample,
            calibration_uncertainty_by_sample,
            temporal_recovered_frames,
        ) = _resolve_temporal_frame_calibrations(
            frames,
            frame_sizes,
            accepted_direct_by_sample,
            camera_motion_edges,
            frame_evidence,
            person_frames,
            scene["payload"]["pitch"],
            max_gap_seconds=(
                max(2.0, float(scene["duration"]))
                if manual_stabilized_by_sample
                else 2.0
            ),
        )

        for sample_index, (people, _) in enumerate(person_frames):
            selected_calibration = resolved_calibrations_by_sample.get(sample_index)
            evidence = frame_evidence[sample_index]
            _attach_metric_positions(
                people,
                [],
                selected_calibration,
                scene["payload"]["pitch"],
                projection_source=str(evidence.get("projectionSource") or "none"),
                calibration_frame_index=calibration_anchor_by_sample.get(sample_index),
                position_uncertainty_metres=calibration_uncertainty_by_sample.get(
                    sample_index
                ),
            )
            metric_person_samples += sum(person.pitch_x is not None for person in people)
            source_frame_index = int(evidence["sourceFrameIndex"])
            _stabilize_detections(
                people,
                [],
                camera_transforms.get(source_frame_index, np.eye(3, dtype=np.float64)),
            )

        progress.update(
            "detection",
            3,
            "Preparing dense ball analysis",
            (
                f"Decoding up to {get_settings().ball_analysis_frame_rate:g} FPS "
                f"for {ball_backend}; player/calibration samples stay unchanged."
            ),
            62,
            84,
            completed=0,
            total=max(1, round(float(scene["duration"]) * get_settings().ball_analysis_frame_rate)),
            eta_padding=5.0,
        )

        def ball_progress(completed: int, total: int, detail: str) -> None:
            progress.update(
                "detection",
                3,
                "Detecting and scoring ball hypotheses",
                f"Dense ball frame {completed}/{total} · {detail}.",
                62,
                84,
                completed=completed,
                total=total,
                eta_padding=3.0,
            )

        (
            ball_frames,
            ball_dense_frame_metadata,
            ball_detection_batches,
            ball_detection_warnings,
        ) = _detect_ball_frames(
            scene,
            ball_detector,
            ball_fallback_detector,
            frames,
            legacy_ball_frames,
            ball_progress,
            failure_policy=str(
                ball_detection_input.get("failurePolicy")
                or get_settings().ball_detection_failure_policy
            ),
            detector_input=ball_detection_input,
        )
        ball_counts = [len(detections) for detections, _ in ball_frames]

        # Dense frames keep native source resolution for tiny-object recall.
        # Interior frames use a bounded interpolation only when both adjacent
        # calibration samples passed QA and their camera edge is reliable.
        # Every unsafe case is retained as an explicit nearest-sample fallback.
        sampled_times = [float(time) for _, time in frames]
        for ball_frame_index, (balls, ball_time) in enumerate(ball_frames):
            if not frames:
                continue
            projection_context = _dense_ball_projection_context(
                float(ball_time),
                sampled_times,
                frame_sizes,
                resolved_calibrations_by_sample,
                calibration_anchor_by_sample,
                calibration_uncertainty_by_sample,
                frame_evidence,
                camera_transforms,
            )
            metric_ball_samples += _apply_dense_ball_projection(
                balls,
                projection_context,
                scene["payload"]["pitch"],
                ball_frame_index,
            )

        if rejected_calibration_frames:
            calibration_warnings.append(
                f"Rejected {rejected_calibration_frames} frame calibrations that failed geometric QA."
            )
        if temporal_recovered_frames:
            calibration_warnings.append(
                f"Recovered {temporal_recovered_frames} frame calibrations from forward/backward camera hypotheses."
            )
        representative = _best_pitch_calibration(accepted_frame_calibrations)
        representative_manual_sample: int | None = None
        if accepted_manual_direct_by_sample:
            representative_manual_sample = min(
                accepted_manual_direct_by_sample,
                key=lambda index: abs(
                    float(frames[index][1])
                    - float(manual_override.get("sceneTime") or 0.0)
                ),
            )
        if representative is not None and representative.frame_index in camera_transforms:
            try:
                image_to_stabilized_pitch = (
                    representative.image_to_pitch
                    @ np.linalg.inv(camera_transforms[int(representative.frame_index)])
                )
                image_to_stabilized_pitch /= image_to_stabilized_pitch[2, 2]
                calibration = replace(
                    representative,
                    image_to_pitch=image_to_stabilized_pitch,
                )
            except np.linalg.LinAlgError:
                calibration_warnings.append("The representative frame transform could not be inverted.")
        if representative_manual_sample is not None:
            calibration = manual_stabilized_by_sample[representative_manual_sample]

        if calibration is not None:
            calibration = canonicalize_penalty_side(calibration, frame_size[0])

        calibration_quality = _evaluate_calibration_quality(frame_evidence)
        metric_calibration = calibration_quality["verdict"] == "pass"
        coordinate_mode = (
            "metric"
            if calibration_quality["verdict"] in {"pass", "review"}
            else "unavailable"
        )

        progress.update(
            "tracking",
            4,
            "Linking observations into tracks",
            f"Associating detections across {len(frames)} frames.",
            84,
            91,
            completed=0,
            total=4,
        )
        local_tracks = _apply_track_identity_corrections(_track_people(person_frames), scene)
        progress.update(
            "tracking",
            4,
            "Building local tracklets",
            f"Built {len(local_tracks)} local tracks; preparing team and role constraints.",
            84,
            91,
            completed=1,
            total=4,
        )
        minimum = max(5, round(len(frames) * 0.24))
        preliminary_stable_tracks = [
            track
            for track in local_tracks
            if len(track.points) >= minimum or track.positive_annotation_ids
        ]
        preliminary_cluster_tracks = [
            track for track in preliminary_stable_tracks if len(track.points) >= minimum
        ]
        preliminary_mapping, _ = _team_clusters(
            preliminary_cluster_tracks,
            frame_size[0],
        )
        for track in preliminary_stable_tracks:
            manual_team = _annotation_team(track.manual_kind)
            if manual_team:
                preliminary_mapping[track.id] = manual_team

        def jersey_ocr_progress(completed: int, total: int, recognized: int) -> None:
            progress.update(
                "tracking",
                4,
                "Reading jersey numbers",
                (
                    f"OCR crops {completed}/{total} · {recognized} readable "
                    "shirt-number observations."
                ),
                84,
                91,
                completed=1,
                total=4,
                eta_padding=2.0,
            )

        (
            jersey_tracklet_evidence,
            jersey_ocr_diagnostics,
            jersey_ocr_warnings,
        ) = _run_jersey_ocr_for_tracklets(
            local_tracks,
            frames,
            jersey_ocr_progress,
            scene=scene,
        )
        identity_warnings.extend(jersey_ocr_warnings)
        partitioned_tracks, split_identity_diagnostics = _apply_canonical_split_corrections(
            local_tracks,
            scene,
        )
        resolver_jersey_evidence = jersey_tracklet_evidence
        if split_identity_diagnostics["appliedCount"]:
            (
                resolver_jersey_evidence,
                split_jersey_mapping_diagnostics,
            ) = _partition_local_jersey_evidence_for_resolver(
                partitioned_tracks,
                jersey_tracklet_evidence,
                jersey_ocr_diagnostics,
            )
            jersey_ocr_diagnostics["preResolverSplitObservationMapping"] = (
                split_jersey_mapping_diagnostics
            )
        partitioned_mapping = {
            track.id: (
                _annotation_team(track.manual_kind)
                or preliminary_mapping.get(track.id)
            )
            for track in partitioned_tracks
            if _annotation_team(track.manual_kind)
            or preliminary_mapping.get(track.id)
        }
        canonical_tracks, identity_resolution_diagnostics = _resolve_canonical_track_states(
            partitioned_tracks,
            partitioned_mapping,
            resolver_jersey_evidence,
        )
        identity_resolution_diagnostics["manualSplits"] = split_identity_diagnostics
        identity_resolution_diagnostics["reid"] = deepcopy(identity_worker_diagnostics)
        identity_resolution_diagnostics["jerseyOcr"] = deepcopy(
            jersey_ocr_diagnostics
        )
        progress.update(
            "tracking",
            4,
            "Resolving canonical people",
            (
                f"Resolved {len(local_tracks)} local tracklets into "
                f"{len(canonical_tracks)} canonical people; ambiguous links remain provisional."
            ),
            84,
            91,
            completed=2,
            total=4,
        )
        stable_tracks = [
            track
            for track in canonical_tracks
            if len(track.points) >= minimum or track.positive_annotation_ids
        ]
        cluster_tracks = [track for track in stable_tracks if len(track.points) >= minimum]
        mapping, colors = _team_clusters(cluster_tracks, frame_size[0])
        for track in stable_tracks:
            manual_team = _annotation_team(track.manual_kind)
            if manual_team:
                mapping[track.id] = manual_team
        _assign_persistent_canonical_person_ids(canonical_tracks, scene, mapping)
        try:
            (
                canonical_jersey_evidence,
                final_jersey_mapping_diagnostics,
            ) = _aggregate_jersey_evidence_for_final_tracks(
                canonical_tracks,
                jersey_tracklet_evidence,
                jersey_ocr_diagnostics,
            )
            jersey_ocr_diagnostics["canonicalAggregationStatus"] = "ready"
            jersey_ocr_diagnostics["finalObservationMapping"] = (
                final_jersey_mapping_diagnostics
            )
        except ValueError as exc:
            # Jersey OCR is optional identity evidence. A bad mapping remains
            # visible, but cannot make an otherwise valid reconstruction fail.
            canonical_jersey_evidence = {}
            jersey_ocr_diagnostics.update(
                {
                    "canonicalAggregationStatus": "failed",
                    "canonicalAggregationDetail": str(exc),
                }
            )
            identity_warnings.append(
                "Jersey OCR canonical aggregation failed; shirt numbers were omitted from this reconstruction."
            )
        jersey_ocr_diagnostics.update(
            {
                "canonicalPersonEvidence": {
                    canonical_id: summary.to_payload()
                    for canonical_id, summary in sorted(
                        canonical_jersey_evidence.items()
                    )
                },
                "reliableCanonicalPersonCount": sum(
                    summary.status == "reliable"
                    for summary in canonical_jersey_evidence.values()
                ),
                "provisionalCanonicalPersonCount": sum(
                    summary.status == "provisional"
                    for summary in canonical_jersey_evidence.values()
                ),
                "conflictingCanonicalPersonCount": sum(
                    summary.status == "conflict"
                    for summary in canonical_jersey_evidence.values()
                ),
            }
        )
        identity_resolution_diagnostics["jerseyOcr"] = deepcopy(
            jersey_ocr_diagnostics
        )
        progress.update(
            "tracking",
            4,
            "Assigning teams and roles",
            (
                f"Kept {len(stable_tracks)} renderable identities and preserved "
                f"{len(canonical_tracks)} video identities."
            ),
            84,
            91,
            completed=4,
            total=4,
            eta_padding=3.0,
        )
        progress.update(
            "projection",
            5,
            "Building metric 3D trajectories",
            "Projecting foot points and the ball onto the pitch, rejecting geometric outliers.",
            91,
            97,
            completed=0,
            total=2,
        )
        track_projection_diagnostics: dict = {}
        tracks = (
            _scene_tracks(
                canonical_tracks,
                mapping,
                colors,
                frame_size,
                scene,
                calibration,
                coordinate_mode=coordinate_mode,
                diagnostics=track_projection_diagnostics,
            )
            if coordinate_mode != "unavailable"
            else []
        )
        tracks = _apply_scene_track_identity_corrections(tracks, scene)
        canonical_people, canonical_identity_diagnostics = _canonical_people_documents(
            canonical_tracks,
            mapping,
            tracks,
            scene,
            identity_resolution_diagnostics,
            canonical_jersey_evidence,
        )
        resolver_frames = (
            [
                (
                    [
                        candidate
                        for candidate in candidates
                        if candidate.get("pitchX") is not None
                        and candidate.get("pitchZ") is not None
                    ],
                    time,
                )
                for candidates, time in ball_frames
            ]
            if coordinate_mode != "unavailable"
            else ball_frames
        )
        ball_resolution = resolve_ball_trajectory(
            resolver_frames,
            frame_size,
            scene["payload"]["pitch"],
            config=BallTrackingConfig(
                top_k_per_frame=min(8, get_settings().ball_detection_max_candidates),
                max_interpolation_gap_seconds=0.8,
                max_ball_speed_metres_per_second=55.0,
            ),
        )
        ball = (
            ball_resolution.keyframes
            if coordinate_mode != "unavailable"
            else []
        )
        ball_tracking_diagnostics = {
            **ball_resolution.diagnostics,
            "worldProjectionStatus": _ball_world_projection_status(
                coordinate_mode,
                ball,
            ),
            "detectorFrameCount": len(ball_frames),
            "detectorCandidateFrameCount": sum(bool(items) for items, _ in ball_frames),
        }
        progress.update(
            "projection",
            5,
            "3D trajectories ready",
            f"Accepted {len(tracks)} player tracks and {len(ball)} ball samples.",
            91,
            97,
            completed=2,
            total=2,
            eta_padding=2.0,
        )

        scene["payload"]["tracks"] = tracks
        scene["payload"]["canonicalPeople"] = canonical_people
        ball_payload = _publish_automatic_ball_trajectory(
            scene,
            ball,
            ball_tracking_diagnostics,
        )
        active_ball_keyframes = ball_payload["keyframes"]
        runtime_model_versions = sorted(
            {
                str(worker["modelVersion"])
                for frame_batch in ball_detection_batches
                for worker in [
                    (frame_batch.get("metadata") or {}).get("worker") or {}
                ]
                if worker.get("modelVersion")
            }
        )
        ball_detection_metadata = {
            "schemaVersion": 1,
            "status": (
                "degraded"
                if ball_dense_frame_metadata.get("failedFrameCount")
                or ball_dense_frame_metadata.get("fallbackFrameCount")
                or ball_dense_frame_metadata.get("source") == "sampled-frame-fallback"
                else "ready"
            ),
            "requestedBackend": ball_backend,
            "runtimeModelVersions": runtime_model_versions,
            "input": deepcopy(ball_detection_input),
            "frameSource": deepcopy(ball_dense_frame_metadata),
            "frameCount": len(ball_frames),
            "candidateCount": sum(len(items) for items, _ in ball_frames),
            "framesWithCandidates": sum(bool(items) for items, _ in ball_frames),
            "fallbackFrameCount": int(
                ball_dense_frame_metadata.get("fallbackFrameCount") or 0
            ),
            "failedFrameCount": int(
                ball_dense_frame_metadata.get("failedFrameCount") or 0
            ),
            "backendCounts": deepcopy(
                ball_dense_frame_metadata.get("backendCounts") or {}
            ),
            "observedFrameCount": ball_tracking_diagnostics.get(
                "observedFrameCount", 0
            ),
            "inferredFrameCount": ball_tracking_diagnostics.get(
                "inferredFrameCount", 0
            ),
            "occludedFrameCount": ball_tracking_diagnostics.get(
                "occludedFrameCount", 0
            ),
            "observedCoverage": ball_tracking_diagnostics.get("observedCoverage"),
            "publishedCoverage": ball_tracking_diagnostics.get("publishedCoverage"),
            "tracking": deepcopy(ball_tracking_diagnostics),
            "frames": ball_detection_batches,
        }
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
        if calibration is not None:
            calibration_metadata = {
                **calibration.as_dict(),
                "status": (
                    "ready"
                    if calibration_quality["verdict"] == "pass"
                    else "review"
                    if calibration_quality["verdict"] == "review"
                    else "rejected"
                ),
                "reason": (
                    None
                    if calibration_quality["verdict"] == "pass"
                    else "Calibration QA gates did not permit metric coordinates."
                ),
            }
            if representative_manual_sample is not None:
                representative_manual_override = manual_override_by_sample[
                    representative_manual_sample
                ]
                calibration_metadata.update(
                    {
                        "method": "manual-pitch-anchors",
                        "preset": representative_manual_override.get("preset"),
                        "sceneTime": representative_manual_override.get("sceneTime"),
                        "frameIndex": representative_manual_override.get("frameIndex"),
                        "sourceFrameIndex": representative_manual_override.get("sourceFrameIndex"),
                        "alignmentError": representative_manual_override.get("alignmentError"),
                        "alignmentMetrics": representative_manual_override.get("alignmentMetrics"),
                        "anchors": representative_manual_override.get("anchors") or [],
                        "manualFrameAnchorCount": len(accepted_manual_direct_by_sample),
                    }
                )
        else:
            calibration_metadata = {
                "status": "rejected",
                "method": None,
                "pitchSide": None,
                "reason": "No frame produced a calibration that passed geometric QA.",
            }
        existing_calibration_contract = (
            scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
            .get("calibration")
            or {}
        )
        calibration_contract = {
            **{
                key: deepcopy(value)
                for key, value in existing_calibration_contract.items()
                if key in {"framePreviews", "lastFramePreview"}
            },
            "schemaVersion": 2,
            "summary": calibration_quality["summary"],
            "frameEvidence": frame_evidence,
            "manualFrameAnchors": [
                {
                    "id": item.get("id"),
                    "sampleIndex": sample_index,
                    "sourceFrameIndex": item.get("sourceFrameIndex"),
                    "sceneTime": item.get("sceneTime"),
                    "status": (
                        "accepted"
                        if sample_index in accepted_manual_direct_by_sample
                        else "rejected"
                    ),
                }
                for sample_index, item in sorted(manual_override_by_sample.items())
            ],
        }
        existing_orientation = (video.get("reconstruction") or {}).get("pitchOrientation") or {}
        detected_side = calibration_quality["summary"].get("visiblePitchSide")
        attacking_goal = existing_orientation.get("attackingGoal")
        if attacking_goal not in {"left", "right"}:
            attacking_goal = "unknown"
        pitch_orientation = {
            "visiblePitchSide": detected_side or "unknown",
            "visiblePitchSideSource": "calibration" if detected_side else "unknown",
            "visiblePitchSideAgreement": calibration_quality["summary"].get("sideAgreement"),
            "attackingGoal": attacking_goal,
            "attackingGoalSource": existing_orientation.get("attackingGoalSource")
            or ("manual" if attacking_goal != "unknown" else "unknown"),
            # Compatibility for older clients; this describes attack semantics only.
            "source": existing_orientation.get("source")
            or ("manual" if attacking_goal != "unknown" else "unknown"),
            "updatedAt": datetime.now(UTC).isoformat(),
        }
        working_reconstruction = video.get("reconstruction") or {}
        working_reconstruction["diagnostics"] = {
            **(working_reconstruction.get("diagnostics") or {}),
            **track_projection_diagnostics,
            "identity": canonical_identity_diagnostics,
            "identityResolver": canonical_identity_diagnostics,
            "jerseyOcr": jersey_ocr_diagnostics,
            "personDetectionCache": deepcopy(person_detection_cache_diagnostics),
            "ballTracking": ball_tracking_diagnostics,
            "ballTrajectoryMode": ball_payload["mode"],
        }
        working_reconstruction["ballDetection"] = ball_detection_metadata
        video["reconstruction"] = working_reconstruction
        quality = evaluate_reconstruction_quality(scene, frame_evidence)
        quality["processingStatus"] = "completed"
        quality["calibrationQuality"] = calibration_quality
        verdict_rank = {"pass": 0, "review": 1, "reject": 2}
        quality["verdict"] = max(
            (quality["verdict"], calibration_quality["verdict"]),
            key=lambda value: verdict_rank[value],
        )
        progress.update(
            "finalizing",
            6,
            "Saving reconstruction",
            "Writing tracks, calibration diagnostics, and orientation metadata.",
            97,
            100,
            completed=0,
            total=1,
        )
        completed_progress = progress.complete(len(tracks), len(active_ball_keyframes))
        set_reconstruction_status(
            scene,
            "ready",
            expected_run_id=expected_run_id,
            expected_input_fingerprint=expected_input_fingerprint,
            expected_lease_owner_id=expected_lease_owner_id,
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
            ballDetection=ball_detection_metadata,
            # v3 makes canonical video identity authoritative even when no
            # renderable metric 3D trajectory exists. v1/v2 remain readable.
            trackObservationSchemaVersion=3,
            coordinateSpace=(
                "pitch-metric-mixed-direct-anchors"
                if metric_calibration
                and accepted_manual_direct_by_sample
                and accepted_automatic_direct_by_sample
                else "pitch-metric-manual-anchors"
                if metric_calibration and accepted_manual_direct_by_sample
                else "pitch-metric-temporal-hypotheses"
                if metric_calibration and temporal_recovered_frames
                else "pitch-metric-per-frame"
                if metric_calibration
                else "pitch-metric-temporal-partial-review"
                if calibration_quality["verdict"] == "review"
                and temporal_recovered_frames
                else "pitch-metric-partial-review"
                if calibration_quality["verdict"] == "review"
                else "unavailable-calibration-rejected"
            ),
            pitchCalibration=calibration_metadata,
            calibration=calibration_contract,
            calibrationFrames=frame_evidence,
            pitchOrientation=pitch_orientation,
            cameraMotionCompensated=any(
                (item.get("cameraMotion") or {}).get("status") == "estimated"
                for item in frame_evidence
            ),
            progress=completed_progress,
            warnings=[
                *(
                    [
                        f"Metric positions combine {len(accepted_manual_direct_by_sample)} manual frame anchor(s) with {len(accepted_automatic_direct_by_sample)} accepted automatic direct observation(s); manual wins only at the same sample."
                    ]
                    if metric_calibration
                    and accepted_manual_direct_by_sample
                    and accepted_automatic_direct_by_sample
                    else [
                        f"Metric positions use {len(accepted_manual_direct_by_sample)} accepted manual frame anchor(s) and QA-gated temporal propagation."
                    ]
                    if metric_calibration and accepted_manual_direct_by_sample
                    else [
                        "Metric positions combine direct pitch observations with QA-gated forward/backward camera hypotheses."
                    ]
                    if metric_calibration and temporal_recovered_frames
                    else [
                        "Metric positions use semantic per-frame homographies; PnLCalib frames include point-and-line refinement."
                    ]
                    if metric_calibration
                    else [
                        "Calibration requires review; only accepted metric observations were published and gaps remain missing."
                    ]
                    if calibration_quality["verdict"] == "review"
                    else [
                        "Calibration QA rejected this run; no new world-space tracks or ball trajectory were published."
                    ]
                ),
                *calibration_warnings,
                *identity_warnings,
                *ball_detection_warnings,
                *(
                    [
                        f"Metric calibration remains unresolved on {len(frames) - len(resolved_calibrations_by_sample)} of {len(frames)} sampled frames; no representative homography was used to hide those gaps."
                    ]
                    if resolved_calibrations_by_sample
                    and len(resolved_calibrations_by_sample) < len(frames)
                    else []
                ),
                *(
                    []
                    if ball
                    else [
                        "No stable automatic ball trajectory was found; active manual keypoints were preserved."
                        if ball_payload["mode"] == "manual"
                        else "No stable ball trajectory was found."
                    ]
                ),
            ],
            inputRange={
                "sourceStart": float(video.get("sourceStart") or 0.0),
                "sourceEnd": float(video.get("sourceEnd") or scene["duration"]),
                "firstFrameTime": round(float(frames[0][1]), 3),
                "lastFrameTime": round(float(frames[-1][1]), 3),
            },
            diagnostics={
                **track_projection_diagnostics,
                "meanPersonDetections": round(float(np.mean(person_counts)), 2),
                "framesWithBall": sum(count > 0 for count in ball_counts),
                "ballCandidateCount": sum(ball_counts),
                "ballObservedFrameCount": ball_tracking_diagnostics.get(
                    "observedFrameCount", 0
                ),
                "ballInferredFrameCount": ball_tracking_diagnostics.get(
                    "inferredFrameCount", 0
                ),
                "ballOccludedFrameCount": ball_tracking_diagnostics.get(
                    "occludedFrameCount", 0
                ),
                "ballObservedCoverage": ball_tracking_diagnostics.get(
                    "observedCoverage"
                ),
                "ballPublishedCoverage": ball_tracking_diagnostics.get(
                    "publishedCoverage"
                ),
                "ballTracking": ball_tracking_diagnostics,
                "ballTrajectoryMode": ball_payload["mode"],
                "rawTrackCount": len(local_tracks),
                "canonicalPersonCount": len(canonical_people),
                "stableTrackCount": len(stable_tracks),
                "acceptedTrackCount": len(tracks),
                "identity": canonical_identity_diagnostics,
                "identityResolver": canonical_identity_diagnostics,
                "jerseyOcr": jersey_ocr_diagnostics,
                "personDetectionCache": deepcopy(
                    person_detection_cache_diagnostics
                ),
                "calibrationBackend": calibration.method if calibration is not None else None,
                "calibrationBackendCounts": {
                    method: sum(
                        item.method == method for item in accepted_frame_calibrations.values()
                    )
                    for method in sorted(
                        {item.method for item in accepted_frame_calibrations.values()}
                    )
                },
                "calibratedFrameCount": len(resolved_calibrations_by_sample),
                "directCalibratedFrameCount": len(accepted_frame_calibrations),
                "temporalRecoveredFrameCount": temporal_recovered_frames,
                "temporalAmbiguousFrameCount": sum(
                    item.get("solutionStatus") == "ambiguous" for item in frame_evidence
                ),
                "cameraMotionCutCount": sum(
                    (item.get("cameraMotion") or {}).get("status") == "cut"
                    for item in frame_evidence
                ),
                "cameraMotionUnreliableCount": sum(
                    (item.get("cameraMotion") or {}).get("status") == "unreliable"
                    for item in frame_evidence
                ),
                "calibrationFrameCoverage": calibration_quality["summary"]["usableCoverage"],
                "calibrationDirectCoverage": calibration_quality["summary"]["directCoverage"],
                "calibrationMaxGapSeconds": calibration_quality["summary"]["maxGapSeconds"],
                "calibrationReprojectionP95": calibration_quality["summary"]["reprojectionP95"],
                "calibrationSideAgreement": calibration_quality["summary"]["sideAgreement"],
                "rejectedCalibrationFrames": rejected_calibration_frames,
                "screenApproximateSamples": (
                    sum(
                        keyframe.get("projectionSource") == "screen-approximate"
                        for track in tracks
                        for keyframe in track.get("keyframes") or []
                    )
                    + sum(
                        keyframe.get("projectionSource") == "screen-approximate"
                        for keyframe in ball
                    )
                ),
                "metricPersonSamples": metric_person_samples,
                "metricBallSamples": metric_ball_samples,
                "calibrationReprojectionError": (
                    round(calibration.reprojection_error, 3)
                    if calibration is not None and calibration.reprojection_error is not None
                    else None
                ),
            },
        )
        return scene
    except StaleReconstructionRun:
        # The current database document belongs to a newer run or newer manual
        # inputs. Never restore snapshots or mark that newer state as failed.
        raise
    except Exception as exc:
        scene["payload"]["tracks"] = previous_tracks
        scene["payload"]["canonicalPeople"] = previous_canonical_people
        scene["payload"]["ball"] = previous_ball
        for team in scene.get("payload", {}).get("teams") or []:
            if str(team.get("id")) in previous_team_colors:
                team["color"] = previous_team_colors[str(team.get("id"))]
        scene["payload"]["videoAsset"]["processingState"] = previous_processing_state
        identity_correction_diagnostics = (
            [deepcopy(exc.diagnostic)]
            if isinstance(exc, IdentityCorrectionError)
            else []
        )
        failure_values: dict = {}
        failed_progress = progress.failed(str(exc))
        if identity_correction_diagnostics:
            reconstruction = scene["payload"]["videoAsset"].get("reconstruction") or {}
            diagnostics = {
                **(reconstruction.get("diagnostics") or {}),
                "identityCorrections": identity_correction_diagnostics,
            }
            failed_progress["identityCorrections"] = identity_correction_diagnostics
            failure_values = {
                "identityCorrectionDiagnostics": identity_correction_diagnostics,
                "diagnostics": diagnostics,
            }
        set_reconstruction_status(
            scene,
            "failed",
            expected_run_id=expected_run_id,
            expected_input_fingerprint=expected_input_fingerprint,
            expected_lease_owner_id=expected_lease_owner_id,
            processingStatus="failed",
            qualityVerdict="reject",
            error=str(exc),
            completedAt=datetime.now(UTC).isoformat(),
            progress=failed_progress,
            **failure_values,
        )
        if isinstance(exc, ReconstructionError):
            raise
        raise ReconstructionError(str(exc)) from exc


def _interpolate_scene_keyframes(keyframes: list[dict], time: float) -> dict | None:
    if not keyframes:
        return None
    if time <= float(keyframes[0]["t"]):
        return keyframes[0]
    if time >= float(keyframes[-1]["t"]):
        return keyframes[-1]
    for index in range(1, len(keyframes)):
        right = keyframes[index]
        if float(right["t"]) < time:
            continue
        left = keyframes[index - 1]
        span = max(0.0001, float(right["t"]) - float(left["t"]))
        progress = (time - float(left["t"])) / span
        return {
            "t": time,
            "x": float(left["x"]) + (float(right["x"]) - float(left["x"])) * progress,
            "z": float(left["z"]) + (float(right["z"]) - float(left["z"])) * progress,
            "confidence": float(left.get("confidence") or 0.0)
            + (float(right.get("confidence") or 0.0) - float(left.get("confidence") or 0.0))
            * progress,
        }
    return keyframes[-1]


def _saved_pitch_calibration(scene: dict) -> PitchCalibration | None:
    metadata = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("pitchCalibration")
        or {}
    )
    if metadata.get("status") not in {"ready", "review", "approximate"}:
        return None
    matrix = metadata.get("imageToPitch") or np.eye(3, dtype=np.float64)
    return PitchCalibration(
        image_to_pitch=np.asarray(matrix, dtype=np.float64),
        confidence=float(metadata.get("confidence") or 0.0),
        supported_lines=int(metadata.get("supportedLines") or 0),
        mean_line_score=float(metadata.get("meanLineScore") or 0.0),
        rectangle=str(metadata.get("rectangle") or ""),
        matched_curves=int(metadata.get("matchedCurves") or 0),
    )


def _identity_annotations(scene: dict) -> list[dict]:
    return list(
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("frameAnnotations")
        or []
    )


def _annotation_source_identity(annotation: dict | None) -> str | None:
    if not annotation:
        return None
    value = annotation.get("canonicalPersonId") or annotation.get("sourceTrackId")
    return str(value).strip() or None if value is not None else None


def _split_range(annotation: dict) -> tuple[float, float] | None:
    if _annotation_action(annotation) != "split":
        return None
    try:
        start = float(annotation["rangeStart"])
        end = float(annotation["rangeEnd"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([start, end]).all() or end <= start:
        return None
    return start, end


def _ordered_split_corrections(annotations: list[dict]) -> list[dict]:
    """Topologically order nested splits by canonical lineage.

    Range ordering alone is incorrect when a child range starts at the same
    time as its parent: the child does not exist until the parent has produced
    its ``splitCanonicalPersonId``.  Independent splits keep deterministic
    range/id ordering.
    """

    splits = [
        annotation
        for annotation in annotations
        if _annotation_action(annotation) == "split" and annotation.get("id")
    ]
    by_id = {str(annotation["id"]): annotation for annotation in splits}
    producers: dict[str, str] = {}
    for annotation in splits:
        correction_id = str(annotation["id"])
        produced_id = str(annotation.get("splitCanonicalPersonId") or "").strip()
        if not produced_id:
            continue
        previous = producers.get(produced_id)
        if previous is not None and previous != correction_id:
            raise IdentityCorrectionError(
                f"Split corrections {previous} and {correction_id} produce the same canonical identity",
                correction_id=correction_id,
                action="split",
                status="conflict",
                reason="duplicate-split-identity-producer",
                source_track_id=_annotation_source_identity(annotation),
                target_id=produced_id,
            )
        producers[produced_id] = correction_id

    dependencies: dict[str, set[str]] = {correction_id: set() for correction_id in by_id}
    children: dict[str, set[str]] = {correction_id: set() for correction_id in by_id}
    for correction_id, annotation in by_id.items():
        parent_id = producers.get(str(_annotation_source_identity(annotation) or ""))
        if parent_id is None:
            continue
        dependencies[correction_id].add(parent_id)
        children[parent_id].add(correction_id)

    def sort_key(correction_id: str) -> tuple[float, float, str]:
        time_range = _split_range(by_id[correction_id])
        start, end = time_range if time_range is not None else (float("inf"), float("inf"))
        return start, end, correction_id

    ready = sorted(
        (correction_id for correction_id, parents in dependencies.items() if not parents),
        key=sort_key,
    )
    ordered: list[dict] = []
    while ready:
        correction_id = ready.pop(0)
        ordered.append(by_id[correction_id])
        for child_id in sorted(children[correction_id], key=sort_key):
            dependencies[child_id].discard(correction_id)
            if not dependencies[child_id] and child_id not in {
                str(item["id"]) for item in ordered
            } and child_id not in ready:
                ready.append(child_id)
        ready.sort(key=sort_key)
    if len(ordered) != len(splits):
        cyclic_ids = sorted(
            correction_id for correction_id, parents in dependencies.items() if parents
        )
        correction_id = cyclic_ids[0]
        annotation = by_id[correction_id]
        raise IdentityCorrectionError(
            "Split correction lineage contains a cycle",
            correction_id=correction_id,
            action="split",
            status="conflict",
            reason="split-lineage-cycle",
            source_track_id=_annotation_source_identity(annotation),
            target_id=str(annotation.get("splitCanonicalPersonId") or "") or None,
            candidates=[{"correctionId": value} for value in cyclic_ids],
        )
    return ordered


def _observation_identifier(observation: dict) -> str | None:
    value = observation.get("observationId") or observation.get("id")
    return str(value).strip() or None if value is not None else None


def _split_target_snapshot(
    scene: dict,
    canonical_person_id: str,
    target_observation_id: str,
) -> tuple[dict, dict]:
    """Resolve a user-selected published observation exactly once.

    The snapshot, rather than a detector list position, becomes the immutable
    correction input. A later rebuild may remap its bbox conservatively, but an
    ambiguous observation ID is rejected here rather than silently choosing a
    neighbour.
    """

    subjects = [
        subject
        for subject in _canonical_analysis_subjects(scene)
        if str(subject.get("canonicalPersonId") or "") == canonical_person_id
    ]
    if len(subjects) != 1:
        raise ReconstructionError("The canonical person no longer exists or is ambiguous")
    subject = subjects[0]
    matches = [
        observation
        for observation in subject.get("observations") or []
        if _observation_identifier(observation) == target_observation_id
    ]
    if len(matches) != 1:
        raise ReconstructionError(
            "Split requires one immutable tracked observation; rebuild or select another frame"
        )
    observation = matches[0]
    if observation.get("frameIndex") is None or not observation.get("bbox"):
        raise ReconstructionError("The selected split observation has no detector-backed bbox")
    scene_time = observation.get("sceneTime")
    if scene_time is None:
        raise ReconstructionError("The selected split observation has no scene timestamp")
    bbox = observation["bbox"]
    snapshot = {
        "observationId": target_observation_id,
        "frameIndex": int(observation["frameIndex"]),
        "sceneTime": round(float(scene_time), 3),
        "bbox": {
            "x": round(float(bbox["x"]), 2),
            "y": round(float(bbox["y"]), 2),
            "width": round(float(bbox["width"]), 2),
            "height": round(float(bbox["height"]), 2),
        },
        "canonicalPersonId": canonical_person_id,
    }
    return subject, snapshot


def _terminal_identity_target(target_id: str, annotations: dict[str, dict]) -> str:
    current = target_id
    visited: set[str] = set()
    while current in annotations and _annotation_action(annotations[current]) == "merge":
        if current in visited:
            raise ReconstructionError("Identity merge graph contains a cycle")
        visited.add(current)
        next_target = annotations[current].get("mergeTargetId")
        if not next_target:
            raise ReconstructionError("A merge correction is missing its target")
        current = str(next_target)
    return current


def _canonical_correction_identity_key(
    scene: dict,
    annotation_by_id: dict[str, dict],
    identifier: str | None,
) -> str | None:
    current = str(identifier or "").strip()
    if not current:
        return None
    if current in annotation_by_id:
        annotation = annotation_by_id[current]
        if _annotation_action(annotation) == "merge":
            current = _terminal_identity_target(
                str(annotation.get("mergeTargetId") or ""), annotation_by_id
            )
        else:
            current = str(_annotation_source_identity(annotation) or current)
    subject = next(
        (
            item
            for item in _canonical_analysis_subjects(scene)
            if current
            in {
                str(item.get("id") or ""),
                str(item.get("canonicalPersonId") or ""),
            }
        ),
        None,
    )
    return str(
        (subject or {}).get("canonicalPersonId")
        or (subject or {}).get("id")
        or current
    )


def _dedicated_roster_corrections_by_owner(
    scene: dict,
    annotations: list[dict],
) -> dict[str, list[dict]]:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    result: dict[str, list[dict]] = {}
    for annotation in annotations:
        if (
            annotation.get("correctionKind")
            != CANONICAL_ROSTER_BINDING_CORRECTION
            or annotation.get("rosterBindingState") not in {"bound", "unbound"}
        ):
            continue
        owners = _roster_binding_correction_owner_ids(scene, annotation)
        if not owners:
            persisted_owner = _annotation_source_identity(annotation)
            owners = {persisted_owner} if persisted_owner else set()
        for owner in owners:
            canonical_owner = _canonical_correction_identity_key(
                scene, annotation_by_id, owner
            )
            if canonical_owner:
                result.setdefault(canonical_owner, []).append(annotation)
    return result


def _roster_correction_decision(annotation: dict) -> tuple[str, str | None]:
    state = str(annotation.get("rosterBindingState") or "")
    external_id = str(annotation.get("externalPlayerId") or "").strip() or None
    return state, external_id if state == "bound" else None


def _consolidate_compatible_merge_roster_corrections(
    scene: dict,
    annotations: list[dict],
    merge_annotation: dict,
) -> tuple[list[dict], list[dict]]:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    source_key = _canonical_correction_identity_key(
        scene,
        annotation_by_id,
        _annotation_source_identity(merge_annotation),
    )
    terminal_id = _terminal_identity_target(
        str(merge_annotation.get("mergeTargetId") or ""), annotation_by_id
    )
    terminal_annotation = annotation_by_id.get(terminal_id)
    target_key = _canonical_correction_identity_key(
        scene,
        annotation_by_id,
        _annotation_source_identity(terminal_annotation)
        if terminal_annotation is not None
        else terminal_id,
    )
    if not source_key or not target_key or source_key == target_key:
        return annotations, []
    by_owner = _dedicated_roster_corrections_by_owner(scene, annotations)
    source_rows = by_owner.get(source_key, [])
    target_rows = by_owner.get(target_key, [])
    if not source_rows or not target_rows:
        return annotations, []
    decisions = {
        _roster_correction_decision(item) for item in [*source_rows, *target_rows]
    }
    if len(decisions) != 1:
        raise ReconstructionError(
            "Cannot merge identities with different dedicated Bind / Unbind decisions"
        )
    decision = next(iter(decisions))
    if decision[0] == "bound":
        raise ReconstructionError(
            "Cannot merge two identities that both carry a dedicated roster binding; unbind one duplicate first"
        )
    keep = min(target_rows, key=lambda item: str(item.get("id") or ""))
    removed_ids = sorted(
        {
            str(item.get("id"))
            for item in [*source_rows, *target_rows]
            if item is not keep and item.get("id")
        }
    )
    if not removed_ids:
        return annotations, []
    removed = [
        deepcopy(item)
        for item in [*source_rows, *target_rows]
        if str(item.get("id") or "") in removed_ids
    ]
    return (
        [item for item in annotations if str(item.get("id") or "") not in removed_ids],
        removed,
    )


def _remove_annotation_references(scene: dict, annotation_ids: set[str]) -> None:
    if not annotation_ids:
        return
    payload = scene.get("payload", {})
    for subject in [
        *(payload.get("canonicalPeople") or []),
        *(payload.get("tracks") or []),
    ]:
        retained = [
            str(value)
            for value in subject.get("annotationIds") or []
            if str(value) not in annotation_ids
        ]
        if retained:
            subject["annotationIds"] = sorted(set(retained))
        else:
            subject.pop("annotationIds", None)
        for observation in subject.get("observations") or []:
            if str(observation.get("annotationId") or "") in annotation_ids:
                observation["annotationId"] = None
            observation_annotation_ids = [
                str(value)
                for value in observation.get("annotationIds") or []
                if str(value) not in annotation_ids
            ]
            if observation_annotation_ids:
                observation["annotationIds"] = sorted(
                    set(observation_annotation_ids)
                )
            else:
                observation.pop("annotationIds", None)


def _roster_undo_snapshot_owner_id(snapshot: dict) -> str | None:
    """Return one internally consistent persisted owner for an undo snapshot."""

    owner_ids = {
        str(value).strip()
        for value in (
            snapshot.get("canonicalPersonId"),
            (
                snapshot.get("targetObservation") or {}
            ).get("canonicalPersonId")
            if isinstance(snapshot.get("targetObservation"), dict)
            else None,
        )
        if str(value or "").strip()
    }
    return next(iter(owner_ids)) if len(owner_ids) == 1 else None


def _correction_endpoint_ids(
    scene: dict,
    correction: dict,
    annotations: list[dict],
) -> tuple[set[str], set[str]]:
    """Return valid source and target lineage ids for split/merge undo data."""

    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }

    def expanded(*values: object) -> set[str]:
        result = {
            str(value).strip()
            for value in values
            if str(value or "").strip()
        }
        for value in list(result):
            canonical = _canonical_correction_identity_key(
                scene,
                annotation_by_id,
                value,
            )
            if canonical:
                result.add(str(canonical))
        return result

    source_id = str(_annotation_source_identity(correction) or "").strip()
    source_ids = expanded(source_id)
    if _annotation_action(correction) == "split":
        return source_ids, expanded(correction.get("splitCanonicalPersonId"))
    if _annotation_action(correction) != "merge":
        return source_ids, set()
    target_id = str(correction.get("mergeTargetId") or "").strip()
    terminal_id = _terminal_identity_target(target_id, annotation_by_id)
    terminal_annotation = annotation_by_id.get(terminal_id)
    return source_ids, expanded(
        target_id,
        terminal_id,
        _annotation_source_identity(terminal_annotation),
    )


def _clear_unbound_roster_correction(
    scene: dict,
    annotation: dict,
    *,
    active_merge_ids: set[str] | None = None,
    active_split_ids: set[str] | None = None,
    persist: bool,
) -> dict:
    """Remove one authorized Unbind and every active undo resurrection path."""

    if (
        annotation.get("correctionKind")
        != CANONICAL_ROSTER_BINDING_CORRECTION
        or annotation.get("rosterBindingState") != "unbound"
        or not annotation.get("id")
    ):
        raise ReconstructionError(
            "Unbind the roster player before clearing its roster decision"
        )
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    annotations = list(reconstruction.get("frameAnnotations") or [])
    annotation_id = str(annotation["id"])
    cleared_annotation_ids = {annotation_id}
    cleared_origin_ids = {_roster_decision_origin_id(annotation)}
    active_merge_ids = {
        str(value).strip()
        for value in active_merge_ids or set()
        if str(value or "").strip()
    }
    active_split_ids = {
        str(value).strip()
        for value in active_split_ids or set()
        if str(value or "").strip()
    }

    # A merge of two compatible Unbind decisions keeps one live correction and
    # stores the others solely for Undo Merge.  From the merged identity those
    # rows are one semantic negative decision, so Clear must remove the complete
    # active lineage.  Validate the snapshots before changing the scene so
    # malformed metadata fails closed instead of being partially discarded.
    for item in annotations:
        item_id = str(item.get("id") or "")
        if item_id not in active_merge_ids:
            continue
        has_rows = "consolidatedRosterCorrections" in item
        has_ids = "consolidatedRosterCorrectionIds" in item
        if has_ids and not has_rows:
            raise ReconstructionError(
                "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
            )
        if not has_rows:
            continue
        merge_source_ids, merge_target_ids = _correction_endpoint_ids(
            scene,
            item,
            annotations,
        )
        merge_owner_ids = merge_source_ids | merge_target_ids
        if not merge_source_ids or not merge_target_ids:
            raise ReconstructionError(
                "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
            )
        stored_rows = item.get("consolidatedRosterCorrections")
        if not isinstance(stored_rows, list):
            raise ReconstructionError(
                "The merge has invalid roster undo metadata"
            )
        stored_ids: list[str] = []
        for stored in stored_rows:
            if not isinstance(stored, dict) or (
                stored.get("correctionKind")
                != CANONICAL_ROSTER_BINDING_CORRECTION
                or stored.get("rosterBindingState") != "unbound"
                or not stored.get("id")
            ):
                raise ReconstructionError(
                    "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
                )
            stored_owner_id = _roster_undo_snapshot_owner_id(stored)
            if not stored_owner_id or stored_owner_id not in merge_owner_ids:
                raise ReconstructionError(
                    "The merge roster undo metadata belongs to another identity; rebuild before clearing the roster decision"
                )
            stored_id = str(stored["id"])
            stored_ids.append(stored_id)
            cleared_annotation_ids.add(stored_id)
            cleared_origin_ids.add(_roster_decision_origin_id(stored))
        if has_ids:
            metadata_ids = item.get("consolidatedRosterCorrectionIds")
            if not isinstance(metadata_ids, list) or any(
                not str(value or "").strip() for value in metadata_ids
            ):
                raise ReconstructionError(
                    "The merge has unsafe roster undo metadata; rebuild before clearing the roster decision"
                )
            if sorted(set(stored_ids)) != sorted(
                {str(value).strip() for value in metadata_ids}
            ):
                raise ReconstructionError(
                    "The merge has inconsistent roster undo metadata; rebuild before clearing the roster decision"
                )

    remaining = []
    for item in annotations:
        if str(item.get("id") or "") == annotation_id:
            continue
        item = deepcopy(item)
        if (
            str(item.get("id") or "") in active_split_ids
            and _annotation_action(item) == "split"
            and SPLIT_ROSTER_UNDO_FIELD in item
        ):
            stored_rows = item.get(SPLIT_ROSTER_UNDO_FIELD)
            if not isinstance(stored_rows, list):
                raise ReconstructionError(
                    "The split has invalid roster undo metadata"
                )
            split_source_ids, _split_target_ids = _correction_endpoint_ids(
                scene,
                item,
                annotations,
            )
            if not split_source_ids:
                raise ReconstructionError(
                    "The split has unsafe roster undo metadata; rebuild before clearing the roster decision"
                )
            for stored in stored_rows:
                if not isinstance(stored, dict) or (
                    stored.get("correctionKind")
                    != CANONICAL_ROSTER_BINDING_CORRECTION
                    or stored.get("rosterBindingState") != "unbound"
                    or not stored.get("id")
                ):
                    raise ReconstructionError(
                        "The split has unsafe roster undo metadata; rebuild before clearing the roster decision"
                    )
                stored_owner_id = _roster_undo_snapshot_owner_id(stored)
                if not stored_owner_id or stored_owner_id not in split_source_ids:
                    raise ReconstructionError(
                        "The split roster undo metadata belongs to another identity; rebuild before clearing the roster decision"
                    )
                if _roster_decision_origin_id(stored) in cleared_origin_ids:
                    cleared_annotation_ids.add(str(stored["id"]))
            item[SPLIT_ROSTER_UNDO_FIELD] = [
                stored
                for stored in stored_rows
                if _roster_decision_origin_id(stored)
                not in cleared_origin_ids
            ]
        if (
            str(item.get("id") or "") in active_merge_ids
            and _annotation_action(item) == "merge"
        ):
            item.pop("consolidatedRosterCorrectionIds", None)
            item.pop("consolidatedRosterCorrections", None)
        remaining.append(item)
    _validate_identity_corrections(scene, remaining)
    reconstruction["frameAnnotations"] = sorted(
        remaining,
        key=lambda item: (
            int(item.get("frameIndex") or 0),
            str(item.get("id") or ""),
        ),
    )
    video["reconstruction"] = reconstruction
    _remove_annotation_references(scene, cleared_annotation_ids)
    if persist:
        scene_store.put(scene)
    return annotation


def _validate_identity_corrections(scene: dict, annotations: list[dict]) -> None:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    identity_subjects = _canonical_analysis_subjects(scene)
    subjects_by_id: dict[str, dict] = {}
    for subject in identity_subjects:
        for identifier in (subject.get("id"), subject.get("canonicalPersonId")):
            if identifier:
                subjects_by_id[str(identifier)] = subject
    track_ids = {
        str(identifier)
        for track in identity_subjects
        for identifier in (track.get("id"), track.get("canonicalPersonId"))
        if identifier
    }
    excluded_track_ids = {
        str(_annotation_source_identity(annotation))
        for annotation in annotations
        if _annotation_action(annotation) == "exclude"
        and _annotation_scope(annotation) == "identity"
        and _annotation_source_identity(annotation)
    }
    split_ranges: dict[str, list[tuple[float, float, str]]] = {}
    split_target_ids: dict[str, str] = {}
    dedicated_by_owner = _dedicated_roster_corrections_by_owner(
        scene, annotations
    )
    ordered_splits = _ordered_split_corrections(annotations)
    produced_split_ids = {
        str(annotation.get("splitCanonicalPersonId") or "").strip()
        for annotation in ordered_splits
        if str(annotation.get("splitCanonicalPersonId") or "").strip()
    }
    for annotation in annotations:
        referenced_ids = {
            str(value).strip()
            for value in (
                _annotation_source_identity(annotation),
                annotation.get("mergeTargetId")
                if _annotation_action(annotation) == "merge"
                else None,
            )
            if str(value or "").strip()
        }
        orphaned = sorted(
            value
            for value in referenced_ids
            if value.startswith("canonical-split-")
            and value not in produced_split_ids
            and value not in annotation_by_id
        )
        if orphaned:
            raise ReconstructionError(
                "Identity correction references a split identity whose parent correction is missing: "
                + ", ".join(orphaned)
            )
    for annotation_id, annotation in annotation_by_id.items():
        action = _annotation_action(annotation)
        if (
            action == "exclude"
            and _annotation_scope(annotation) == "identity"
            and not _annotation_source_identity(annotation)
        ):
            raise ReconstructionError(
                "Choose the tracked identity before excluding the whole trajectory"
            )
        if action == "split":
            source_id = _annotation_source_identity(annotation)
            target_observation_id = str(annotation.get("targetObservationId") or "").strip()
            target_snapshot = annotation.get("targetObservation")
            time_range = _split_range(annotation)
            if not source_id:
                raise ReconstructionError("Choose the canonical identity before splitting it")
            if not target_observation_id or not isinstance(target_snapshot, dict):
                raise ReconstructionError(
                    "Split requires one immutable tracked observation; rebuild or select another frame"
                )
            if _observation_identifier(target_snapshot) != target_observation_id:
                raise ReconstructionError("The split observation snapshot does not match its target")
            snapshot_identity = str(target_snapshot.get("canonicalPersonId") or "").strip()
            if snapshot_identity and snapshot_identity != source_id:
                raise ReconstructionError("The split observation belongs to another canonical identity")
            prior_target = split_target_ids.get(target_observation_id)
            if prior_target and prior_target != annotation_id:
                raise ReconstructionError("The same observation cannot anchor two split corrections")
            split_target_ids[target_observation_id] = annotation_id
            if target_snapshot.get("frameIndex") is None or not target_snapshot.get("bbox"):
                raise ReconstructionError("The split observation snapshot is incomplete")
            if time_range is None:
                raise ReconstructionError("Split range must have a valid start before its end")
            start, end = time_range
            if start < 0.0 or end > float(scene.get("duration") or 0.0) + 1e-6:
                raise ReconstructionError("Split range is outside this scene")
            target_time = float(target_snapshot.get("sceneTime") or 0.0)
            if not start <= target_time < end:
                raise ReconstructionError("The target observation must be inside the split range")
            split_identity_id = str(annotation.get("splitCanonicalPersonId") or "").strip()
            if not split_identity_id or split_identity_id == source_id:
                raise ReconstructionError("The split identity key is missing or invalid")
            prior_ranges = split_ranges.setdefault(source_id, [])
            if any(max(start, old_start) < min(end, old_end) - 1e-6 for old_start, old_end, _ in prior_ranges):
                raise ReconstructionError("Split ranges for the same identity cannot overlap")
            prior_ranges.append((start, end, annotation_id))
            source_key = _canonical_correction_identity_key(
                scene, annotation_by_id, source_id
            )
            for roster_correction in dedicated_by_owner.get(source_key or "", []):
                if roster_correction.get("rosterBindingState") != "bound":
                    continue
                roster_snapshot = roster_correction.get("targetObservation")
                roster_time = (
                    roster_snapshot.get("sceneTime")
                    if isinstance(roster_snapshot, dict)
                    else roster_correction.get("sceneTime")
                )
                try:
                    roster_time = float(roster_time)
                except (TypeError, ValueError):
                    raise ReconstructionError(
                        "The bound roster correction has no usable split anchor; rebuild before splitting"
                    ) from None
                if not isfinite(roster_time):
                    raise ReconstructionError(
                        "The bound roster correction has no usable split anchor; rebuild before splitting"
                    )
                if start <= roster_time < end and not _bound_roster_semantics_compatible(
                    str(annotation.get("kind") or ""),
                    {
                        annotation_id,
                        str(roster_correction.get("id") or ""),
                    },
                    annotation_by_id,
                ):
                    raise ReconstructionError(
                        "Unbind the roster player before splitting its anchored partition into another team or non-player role"
                    )
            continue
        if action != "merge":
            continue
        target_id = str(annotation.get("mergeTargetId") or "")
        if not target_id:
            raise ReconstructionError("Choose an existing track or labeled person to merge into")
        if target_id == annotation_id:
            raise ReconstructionError("A person cannot be merged into itself")
        if target_id == str(_annotation_source_identity(annotation) or ""):
            raise ReconstructionError("The selected detection already belongs to that track")
        if target_id not in track_ids and target_id not in annotation_by_id:
            raise ReconstructionError("The merge target no longer exists")
        target_annotation = annotation_by_id.get(target_id)
        if target_annotation is not None and _annotation_action(target_annotation) == "exclude":
            raise ReconstructionError("An excluded person cannot be an identity merge target")
        terminal_id = _terminal_identity_target(annotation_id, annotation_by_id)
        if terminal_id in excluded_track_ids:
            raise ReconstructionError("An excluded track cannot be an identity merge target")
        source_subject = subjects_by_id.get(
            str(_annotation_source_identity(annotation) or "")
        )
        terminal_annotation = annotation_by_id.get(terminal_id)
        target_subject = subjects_by_id.get(terminal_id) or subjects_by_id.get(
            str(_annotation_source_identity(terminal_annotation or {}) or "")
        )
        source_key = _canonical_correction_identity_key(
            scene,
            annotation_by_id,
            _annotation_source_identity(annotation),
        )
        target_key = _canonical_correction_identity_key(
            scene,
            annotation_by_id,
            (
                _annotation_source_identity(terminal_annotation)
                if terminal_annotation is not None
                else terminal_id
            ),
        )
        source_roster_corrections = dedicated_by_owner.get(source_key or "", [])
        target_roster_corrections = dedicated_by_owner.get(target_key or "", [])
        if source_roster_corrections and target_roster_corrections:
            source_decisions = {
                _roster_correction_decision(item)
                for item in source_roster_corrections
            }
            target_decisions = {
                _roster_correction_decision(item)
                for item in target_roster_corrections
            }
            if source_decisions != target_decisions:
                raise ReconstructionError(
                    "Cannot merge identities with different dedicated Bind / Unbind decisions"
                )
            raise ReconstructionError(
                "Compatible dedicated roster corrections must be consolidated before merging identities"
            )
        source_external_ids = {
            str(value).strip()
            for value in ((source_subject or {}).get("externalPlayerId"),)
            if str(value or "").strip()
        }
        if not source_roster_corrections and not target_roster_corrections:
            legacy_annotation_external_id = str(
                annotation.get("externalPlayerId") or ""
            ).strip()
            if legacy_annotation_external_id:
                source_external_ids.add(legacy_annotation_external_id)
        target_external_ids = {
            str(value).strip()
            for value in (
                (terminal_annotation or {}).get("externalPlayerId"),
                (target_subject or {}).get("externalPlayerId"),
            )
            if str(value or "").strip()
        }
        confirmed_external_ids = sorted(source_external_ids | target_external_ids)
        if len(confirmed_external_ids) > 1:
            raise ReconstructionError(
                "Cannot merge identities with different confirmed roster players: "
                + " and ".join(confirmed_external_ids)
            )


def _track_annotation_kind(track: dict) -> str:
    team = track.get("teamId")
    role = track.get("role")
    if role == "referee" or team == "officials":
        return "referee"
    if role == "other" or team == "unknown":
        return "other"
    if team == "away":
        return "away-goalkeeper" if role == "goalkeeper" else "away-player"
    return "home-goalkeeper" if role == "goalkeeper" else "home-player"


def _identity_target_defaults(
    scene: dict,
    annotations: list[dict],
    target_id: str,
) -> tuple[str, str | None, str | None]:
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    terminal_id = _terminal_identity_target(target_id, annotation_by_id)
    target_annotation = annotation_by_id.get(terminal_id)
    if target_annotation is not None:
        return (
            str(target_annotation.get("kind") or "other"),
            target_annotation.get("label"),
            target_annotation.get("externalPlayerId"),
        )
    target_track = next(
        (
            track
            for track in _canonical_analysis_subjects(scene)
            if str(track.get("id") or "") == terminal_id
            or str(track.get("canonicalPersonId") or "") == terminal_id
        ),
        None,
    )
    if target_track is None:
        raise ReconstructionError("The merge target no longer exists")
    return (
        _track_annotation_kind(target_track),
        target_track.get("displayName") or target_track.get("label"),
        target_track.get("externalPlayerId"),
    )


def _identity_annotation_response(annotation: dict) -> dict:
    action = _annotation_action(annotation)
    return {
        **annotation,
        "action": action,
        "scope": _annotation_scope(annotation),
        "mergeTargetId": annotation.get("mergeTargetId") if action == "merge" else None,
        "sourceTrackId": annotation.get("sourceTrackId"),
        "canonicalPersonId": annotation.get("canonicalPersonId"),
        "targetObservationId": annotation.get("targetObservationId") if action == "split" else None,
        "targetObservation": deepcopy(annotation.get("targetObservation")) if action == "split" else None,
        "rangeStart": annotation.get("rangeStart") if action == "split" else None,
        "rangeEnd": annotation.get("rangeEnd") if action == "split" else None,
        "splitCanonicalPersonId": annotation.get("splitCanonicalPersonId") if action == "split" else None,
        "affectedPreview": deepcopy(annotation.get("affectedPreview")) if action == "split" else None,
        "previewState": {
            "confirm": "confirmed",
            "exclude": "excluded",
            "merge": "merged",
            "split": "split",
        }[action],
    }


def _raw_track_match_score(track: TrackState, target: dict) -> dict | None:
    target_observations = {
        int(item["frameIndex"]): item
        for item in target.get("observations") or []
        if item.get("frameIndex") is not None and item.get("bbox")
    }
    image_costs: list[float] = []
    image_times: list[float] = []
    for point in track.points:
        frame_index = point.get("frameIndex")
        bbox = point.get("bbox")
        if frame_index is None or not bbox or int(frame_index) not in target_observations:
            continue
        target_bbox = target_observations[int(frame_index)]["bbox"]
        overlap = _iou(
            (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["x"]) + float(bbox["width"]),
                float(bbox["y"]) + float(bbox["height"]),
            ),
            (
                float(target_bbox["x"]),
                float(target_bbox["y"]),
                float(target_bbox["x"]) + float(target_bbox["width"]),
                float(target_bbox["y"]) + float(target_bbox["height"]),
            ),
        )
        if overlap >= 0.25:
            image_costs.append((1.0 - overlap) * 2.0)
            image_times.append(float(point["t"]))
    if image_costs:
        return {
            "median": float(np.median(image_costs)),
            "p90": float(np.percentile(image_costs, 90)),
            "normalizedMedian": float(np.median(image_costs)) / 2.0,
            "overlap": len(image_costs),
            "span": max(image_times) - min(image_times),
            "source": "image-observation-overlap",
        }

    keyframes = [
        keyframe
        for keyframe in target.get("keyframes") or []
        if keyframe.get("observed") is not False
    ]
    distances: list[float] = []
    normalized_distances: list[float] = []
    shared_times: list[float] = []
    for point in track.points:
        if point.get("pitchX") is None or point.get("pitchZ") is None or not keyframes:
            continue
        nearest = min(keyframes, key=lambda item: abs(float(item["t"]) - float(point["t"])))
        if abs(float(nearest["t"]) - float(point["t"])) > 0.16:
            continue
        distance = hypot(
            float(nearest["x"]) - float(point["pitchX"]),
            float(nearest["z"]) - float(point["pitchZ"]),
        )
        uncertainty = max(
            0.5,
            float(nearest.get("positionUncertaintyMetres") or 0.0)
            + float(point.get("positionUncertaintyMetres") or 0.0),
        )
        distances.append(distance)
        normalized_distances.append(distance / uncertainty)
        shared_times.append(float(point["t"]))
    if not distances:
        return None
    return {
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
        "normalizedMedian": float(np.median(normalized_distances)),
        "overlap": len(distances),
        "span": max(shared_times) - min(shared_times),
    }


def _raw_track_matches_identity_metadata(track: TrackState, target: dict) -> bool:
    target_team = str(target.get("teamId") or "") or None
    target_role = str(target.get("role") or "") or None
    track_team = _annotation_team(track.manual_kind)
    track_role = _annotation_role(track.manual_kind)
    if target_team and track_team and target_team != track_team:
        return False
    if target_role and track_role and target_role != track_role:
        return False
    target_player = str(target.get("externalPlayerId") or "") or None
    track_player = str(track.manual_external_player_id or "") or None
    return not (target_player and track_player and target_player != track_player)


def _resolve_previous_identity_track(
    tracks: list[TrackState],
    target: dict,
    *,
    correction_id: str,
    action: str,
    source_track_id: str | None = None,
    target_id: str | None = None,
    exclude: TrackState | None = None,
) -> TrackState:
    candidates = [
        track
        for track in tracks
        if track is not exclude and _raw_track_matches_identity_metadata(track, target)
    ]
    target_player = str(target.get("externalPlayerId") or "") or None
    if target_player:
        exact_roster = [
            track
            for track in candidates
            if track.manual_external_player_id == target_player
        ]
        if len(exact_roster) == 1:
            return exact_roster[0]
        if len(exact_roster) > 1:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} is ambiguous across roster-matched tracks",
                correction_id=correction_id,
                action=action,
                status="ambiguous",
                reason="multiple-roster-matches",
                source_track_id=source_track_id,
                target_id=target_id,
                candidates=[{"rawTrackId": track.id} for track in exact_roster],
            )

    scored: list[tuple[dict, TrackState]] = []
    for track in candidates:
        score = _raw_track_match_score(track, target)
        if score is None or score["overlap"] < 3 or score["span"] < 0.4:
            continue
        scored.append((score, track))
    scored.sort(
        key=lambda item: (
            item[0]["median"],
            item[0]["p90"],
            -item[0]["overlap"],
        )
    )
    candidate_diagnostics = [
        {
            "rawTrackId": track.id,
            "medianDistanceMetres": round(float(score["median"]), 3),
            "p90DistanceMetres": round(float(score["p90"]), 3),
            "normalizedMedian": round(float(score["normalizedMedian"]), 3),
            "overlapSamples": int(score["overlap"]),
            "overlapSpanSeconds": round(float(score["span"]), 3),
        }
        for score, track in scored
    ]
    if not scored:
        raise IdentityCorrectionError(
            f"Identity correction {correction_id} could not resolve its previous trajectory",
            correction_id=correction_id,
            action=action,
            status="unresolved",
            reason="insufficient-observation-overlap",
            source_track_id=source_track_id,
            target_id=target_id,
            candidates=[{"rawTrackId": track.id} for track in candidates],
        )
    best_score, best_track = scored[0]
    if (
        best_score["median"] > 4.0
        or best_score["p90"] > 6.0
        or best_score["normalizedMedian"] > 2.0
    ):
        raise IdentityCorrectionError(
            f"Identity correction {correction_id} no longer matches the rebuilt trajectory",
            correction_id=correction_id,
            action=action,
            status="unresolved",
            reason="trajectory-outside-remap-threshold",
            source_track_id=source_track_id,
            target_id=target_id,
            candidates=candidate_diagnostics,
        )
    if len(scored) > 1:
        runner_up = scored[1][0]
        absolute_margin = runner_up["median"] - best_score["median"]
        relative_margin = runner_up["median"] / max(0.25, best_score["median"])
        if absolute_margin < 2.0 and relative_margin < 1.5:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} is ambiguous between nearby trajectories",
                correction_id=correction_id,
                action=action,
                status="ambiguous",
                reason="nearby-trajectories",
                source_track_id=source_track_id,
                target_id=target_id,
                candidates=candidate_diagnostics,
            )
    return best_track


def _confirmed_external_player_conflict(
    target_external_player_id: object,
    source_external_player_id: object,
) -> tuple[str, str] | None:
    target_id = str(target_external_player_id or "").strip()
    source_id = str(source_external_player_id or "").strip()
    if target_id and source_id and target_id != source_id:
        return target_id, source_id
    return None


def _dedicated_roster_binding_conflict(
    target: TrackState,
    source: TrackState,
) -> tuple[str, str] | None:
    if target.roster_binding_state is None or source.roster_binding_state is None:
        return None
    target_decision = (
        str(target.manual_external_player_id)
        if target.roster_binding_state == "bound"
        else "<unbound>"
    )
    source_decision = (
        str(source.manual_external_player_id)
        if source.roster_binding_state == "bound"
        else "<unbound>"
    )
    if target_decision != source_decision:
        return target_decision, source_decision
    return None


def _raise_manual_merge_external_player_conflict(
    target: TrackState,
    source: TrackState,
    annotation: dict,
) -> None:
    dedicated_conflict = _dedicated_roster_binding_conflict(target, source)
    if dedicated_conflict is not None:
        conflict = dedicated_conflict
    elif target.roster_binding_state is not None or source.roster_binding_state is not None:
        # A dedicated Bind/Unbind decision supersedes a legacy generic roster
        # value on the other fragment.
        return
    else:
        conflict = _confirmed_external_player_conflict(
            target.manual_external_player_id,
            source.manual_external_player_id,
        )
    if conflict is None:
        return
    target_external_id, source_external_id = conflict
    correction_id = str(annotation.get("id") or "identity-merge")
    raise IdentityCorrectionError(
        (
            f"Identity correction {correction_id} cannot merge confirmed roster "
            f"players {source_external_id} and {target_external_id}"
        ),
        correction_id=correction_id,
        action="merge",
        status="conflict",
        reason="conflicting-confirmed-external-player-ids",
        source_track_id=_annotation_source_identity(annotation),
        target_id=str(annotation.get("mergeTargetId") or "") or None,
        candidates=[
            {"rawTrackId": source.id, "externalPlayerId": source_external_id},
            {"rawTrackId": target.id, "externalPlayerId": target_external_id},
        ],
    )


def _merge_raw_track_states(
    target: TrackState,
    source: TrackState,
    *,
    allow_manual_owner_merge: bool = False,
    manual_target_owner_id: str | None = None,
) -> None:
    owner_conflict = bool(
        target.manual_identity_owner_ids
        and source.manual_identity_owner_ids
        and target.manual_identity_owner_ids.isdisjoint(
            source.manual_identity_owner_ids
        )
    )
    if owner_conflict and not allow_manual_owner_merge:
        raise ReconstructionError(
            "Cannot automatically merge different explicitly confirmed canonical identities"
        )
    dedicated_conflict = _dedicated_roster_binding_conflict(target, source)
    if dedicated_conflict is not None:
        conflict = dedicated_conflict
    elif target.roster_binding_state is not None or source.roster_binding_state is not None:
        conflict = None
    else:
        conflict = _confirmed_external_player_conflict(
            target.manual_external_player_id,
            source.manual_external_player_id,
        )
    if conflict is not None:
        target_external_id, source_external_id = conflict
        raise ReconstructionError(
            "Cannot merge identities with different confirmed roster players: "
            f"{source_external_id} and {target_external_id}"
        )
    points_by_time: dict[float, dict] = {}
    for point in [*target.points, *source.points]:
        key = round(float(point["t"]), 4)
        previous = points_by_time.get(key)
        point_priority = (
            1 if point.get("annotationId") else 0,
            float(point.get("confidence") or 0.0),
        )
        previous_priority = (
            1 if previous and previous.get("annotationId") else 0,
            float(previous.get("confidence") or 0.0) if previous else 0.0,
        )
        if previous is None or point_priority >= previous_priority:
            points_by_time[key] = point
    target.points = [points_by_time[key] for key in sorted(points_by_time)]
    if source.feature_sum is not None:
        if target.feature_sum is None:
            target.feature_sum = source.feature_sum.copy()
        else:
            target.feature_sum += source.feature_sum
    target.feature_count += source.feature_count
    target.last_frame = max(target.last_frame, source.last_frame)
    target.last_height = max(target.last_height, source.last_height)
    target.role = target.role or source.role
    target.annotation_ids.update(source.annotation_ids)
    target.identity_tombstone_ids.update(source.identity_tombstone_ids)
    target.identity_tombstone_ids.intersection_update(target.annotation_ids)
    if (
        source.manual_semantic_key is not None
        and (
            target.manual_semantic_key is None
            or source.manual_semantic_key >= target.manual_semantic_key
        )
    ):
        target.manual_kind = source.manual_kind
        target.manual_label = source.manual_label
        target.manual_semantic_key = source.manual_semantic_key
    elif source.manual_semantic_key is None and target.manual_semantic_key is None:
        # Compatibility for synthetic/legacy TrackState values that predate
        # persisted authoring timestamps.
        target.manual_kind = source.manual_kind or target.manual_kind
        target.manual_label = source.manual_label or target.manual_label
    if source.roster_binding_state is not None:
        target.roster_binding_state = source.roster_binding_state
        target.roster_binding_annotation_ids.update(
            source.roster_binding_annotation_ids
        )
        target.manual_external_player_id = source.manual_external_player_id
    elif target.roster_binding_state is None:
        target.manual_external_player_id = (
            source.manual_external_player_id or target.manual_external_player_id
        )
    if allow_manual_owner_merge and manual_target_owner_id:
        # The user selected this canonical target as the survivor.  The source
        # owner becomes an alias authorized by the merge; it must not replace
        # the target merely because the target raw fragment had no frame-level
        # confirmation of its own.
        target.manual_identity_owner_ids = {manual_target_owner_id}
    elif owner_conflict:
        # An explicit Merge correction chooses the target identity as the
        # survivor. Do not keep both cannot-link owner labels on one raw track.
        target.manual_identity_owner_ids = set(
            target.manual_identity_owner_ids
            or source.manual_identity_owner_ids
        )
    else:
        target.manual_identity_owner_ids.update(source.manual_identity_owner_ids)
    target.source_tracklet_ids.update(source.source_tracklet_ids or {source.local_tracklet_id})
    if target.reid_sample_candidates or source.reid_sample_candidates:
        target.reid_observation_ids.update(source.reid_observation_ids)
        overlapping_fingerprints = target.reid_evidence_fingerprints.intersection(
            source.reid_evidence_fingerprints
        )
        target.reid_duplicate_evidence_count += (
            source.reid_duplicate_evidence_count + len(overlapping_fingerprints)
        )
        target.reid_evidence_fingerprints.update(source.reid_evidence_fingerprints)
        target.reid_observation_count = len(target.reid_evidence_fingerprints)
        target.reid_sample_candidates.extend(
            {
                **item,
                "vector": item["vector"].copy(),
            }
            for item in source.reid_sample_candidates
        )
        # IDs and timestamps are metadata, not independent evidence. Collapse
        # identical decoded crops before the temporal reservoir is rebuilt.
        best_by_fingerprint: dict[str, dict] = {}
        for item in target.reid_sample_candidates:
            fingerprint = str(
                item.get("evidenceFingerprint")
                or "observation:" + str(item.get("observationId") or "")
            )
            previous = best_by_fingerprint.get(fingerprint)
            if previous is None or (
                float(item["quality"]),
                -int(item["frameIndex"]),
            ) > (
                float(previous["quality"]),
                -int(previous["frameIndex"]),
            ):
                best_by_fingerprint[fingerprint] = item
        target.reid_sample_candidates = list(best_by_fingerprint.values())
        # Deduplicate temporal bins after a manual merge/split and recompute
        # the representative mean from independent quality-ranked views.
        best_by_bin: dict[int, dict] = {}
        for item in target.reid_sample_candidates:
            temporal_bin = int(item["temporalBin"])
            previous = best_by_bin.get(temporal_bin)
            if previous is None or (
                float(item["quality"]),
                -int(item["frameIndex"]),
            ) > (
                float(previous["quality"]),
                -int(previous["frameIndex"]),
            ):
                best_by_bin[temporal_bin] = item
        target.reid_sample_candidates = sorted(
            best_by_bin.values(),
            key=lambda item: (-float(item["quality"]), int(item["frameIndex"])),
        )[:64]
        target._select_reid_samples()
    elif source.reid_feature_sum is not None:
        # Compatibility for test/legacy TrackState objects created before the
        # quality-ranked candidate metadata existed.
        if target.reid_feature_sum is None:
            target.reid_feature_sum = source.reid_feature_sum.copy()
        else:
            target.reid_feature_sum += source.reid_feature_sum
        target.reid_feature_count += source.reid_feature_count
        for sample in source.reid_samples:
            if len(target.reid_samples) >= 12:
                break
            target.reid_samples.append(sample.copy())
    role_rows = [
        point
        for point in target.points
        if point.get("_reidRole") in {"player", "goalkeeper", "referee", "other"}
        and point.get("_reidRoleConfidence") is not None
    ]
    if role_rows:
        target.reid_role_votes = {}
        seen_role_fingerprints: set[str] = set()
        for point in role_rows:
            fingerprint = str(
                point.get("_reidEvidenceFingerprint")
                or "observation:" + str(point.get("observationId") or "")
            )
            if fingerprint in seen_role_fingerprints:
                continue
            seen_role_fingerprints.add(fingerprint)
            role = str(point["_reidRole"])
            confidence = float(point["_reidRoleConfidence"])
            target.reid_role_votes[role] = (
                target.reid_role_votes.get(role, 0.0) + confidence
            )
    else:
        for role, weight in source.reid_role_votes.items():
            target.reid_role_votes[role] = (
                target.reid_role_votes.get(role, 0.0) + weight
            )
    if target.reid_role_votes and not target.manual_kind:
        target.role = max(
            target.reid_role_votes,
            key=lambda value: (target.reid_role_votes[value], value),
        )
    target.identity_evidence.extend(deepcopy(source.identity_evidence))
    target.identity_conflicts.extend(deepcopy(source.identity_conflicts))


def _resolve_split_target_point(
    tracks: list[TrackState],
    annotation: dict,
    *,
    require_source_identity: bool = False,
) -> tuple[TrackState, dict]:
    """Remap a snapshotted observation without trusting detector ordering."""

    correction_id = str(annotation.get("id") or "split")
    snapshot = annotation.get("targetObservation") or {}
    snapshot_bbox = snapshot.get("bbox")
    if snapshot.get("frameIndex") is None or not snapshot_bbox:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} has no immutable target snapshot",
            correction_id=correction_id,
            action="split",
            status="unresolved",
            reason="missing-target-observation-snapshot",
            source_track_id=_annotation_source_identity(annotation),
        )
    target_id = str(annotation.get("targetObservationId") or "")
    expected_source_id = str(_annotation_source_identity(annotation) or "")
    lineage_tracks = [
        track
        for track in tracks
        if str(track.canonical_person_id or "") == expected_source_id
    ]
    if require_source_identity and not lineage_tracks:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} cannot find its produced parent identity",
            correction_id=correction_id,
            action="split",
            status="unresolved",
            reason="split-source-lineage-not-found",
            source_track_id=expected_source_id or None,
            target_id=target_id or None,
        )
    candidate_tracks = lineage_tracks or tracks
    frame_index = int(snapshot["frameIndex"])
    target_box = (
        float(snapshot_bbox["x"]),
        float(snapshot_bbox["y"]),
        float(snapshot_bbox["x"]) + float(snapshot_bbox["width"]),
        float(snapshot_bbox["y"]) + float(snapshot_bbox["height"]),
    )
    candidates: list[tuple[int, float, float, TrackState, dict]] = []
    rejected_exact: list[dict] = []
    for track in candidate_tracks:
        for point in track.points:
            if int(point.get("frameIndex", -1)) != frame_index or not point.get("bbox"):
                continue
            bbox = point["bbox"]
            box = (
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["x"]) + float(bbox["width"]),
                float(bbox["y"]) + float(bbox["height"]),
            )
            overlap = float(_iou(target_box, box))
            target_center = (
                (target_box[0] + target_box[2]) / 2.0,
                (target_box[1] + target_box[3]) / 2.0,
            )
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            scale = max(1.0, min(float(snapshot_bbox["height"]), float(bbox["height"])))
            normalized_center = hypot(
                center[0] - target_center[0],
                center[1] - target_center[1],
            ) / scale
            exact_id = int(str(point.get("observationId") or "") == target_id)
            diagnostic = {
                "rawTrackId": track.id,
                "observationId": point.get("observationId"),
                "frameIndex": frame_index,
                "bboxIou": round(overlap, 4),
                "normalizedCenterDistance": round(normalized_center, 4),
                "exactObservationId": bool(exact_id),
            }
            if overlap >= 0.50 and normalized_center <= 0.50:
                candidates.append((exact_id, overlap, normalized_center, track, point))
            elif exact_id:
                # Detector-index IDs from an older scene may now name a nearby
                # person. Keep the evidence for diagnostics, but never trust it.
                rejected_exact.append(diagnostic)
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].id))
    candidate_diagnostics = [
        {
            "rawTrackId": track.id,
            "observationId": point.get("observationId"),
            "frameIndex": frame_index,
            "bboxIou": round(overlap, 4),
            "normalizedCenterDistance": round(center_distance, 4),
            "exactObservationId": bool(exact_id),
        }
        for exact_id, overlap, center_distance, track, point in candidates
    ]
    if not candidates:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} could not remap its target observation",
            correction_id=correction_id,
            action="split",
            status="unresolved",
            reason="target-observation-not-found",
            source_track_id=_annotation_source_identity(annotation),
            target_id=target_id or None,
            candidates=rejected_exact,
        )
    # Any second geometrically viable row is unsafe. This deliberately prefers
    # a failed rebuild over splitting the person standing next to the target.
    if len(candidates) != 1:
        raise IdentityCorrectionError(
            f"Split correction {correction_id} is ambiguous at the target frame",
            correction_id=correction_id,
            action="split",
            status="ambiguous",
            reason="multiple-target-observation-matches",
            source_track_id=_annotation_source_identity(annotation),
            target_id=target_id or None,
            candidates=candidate_diagnostics,
        )
    return candidates[0][3], candidates[0][4]


def _split_annotation_partition(
    annotation_id: str,
    annotation: dict | None,
    inside: list[dict],
    outside: list[dict],
    start: float,
    end: float,
) -> str:
    """Locate a semantic correction on the side containing its anchor.

    Point annotation ids are authoritative.  The persisted target observation
    and time are fallbacks for detector reorder/outage rebuilds where the new
    observation id cannot equal the old snapshot id.
    """

    def point_has_annotation(point: dict) -> bool:
        return annotation_id == str(point.get("annotationId") or "") or annotation_id in {
            str(value) for value in point.get("annotationIds") or []
        }

    in_point = any(point_has_annotation(point) for point in inside)
    out_point = any(point_has_annotation(point) for point in outside)
    if in_point != out_point:
        return "range" if in_point else "remaining"
    if in_point and out_point:
        return "ambiguous"
    if annotation is None:
        return "unknown"

    target_observation_id = str(annotation.get("targetObservationId") or "").strip()
    if target_observation_id:
        in_observation = any(
            str(point.get("observationId") or "") == target_observation_id
            for point in inside
        )
        out_observation = any(
            str(point.get("observationId") or "") == target_observation_id
            for point in outside
        )
        if in_observation != out_observation:
            return "range" if in_observation else "remaining"
        if in_observation and out_observation:
            return "ambiguous"

    snapshot = annotation.get("targetObservation")
    anchor_time = snapshot.get("sceneTime") if isinstance(snapshot, dict) else None
    if anchor_time is None:
        anchor_time = annotation.get("sceneTime")
    try:
        anchor_time = float(anchor_time)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(anchor_time):
        return "unknown"
    return "range" if start <= anchor_time < end else "remaining"


def _partition_external_player_ids(
    source: TrackState,
    source_annotation_ids: set[str],
    split_annotation_ids: set[str],
    annotations_by_id: dict[str, dict],
    *,
    correction_id: str,
    source_identity_id: str,
    split_identity_id: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Assign confirmed roster semantics only to their anchored partition."""

    positive_bindings: dict[str, str] = {}
    dedicated_decisions: dict[str, tuple[str, str | None]] = {}
    for annotation_id in source.annotation_ids:
        annotation = annotations_by_id.get(annotation_id)
        if annotation is None:
            continue
        if (
            annotation.get("correctionKind")
            == CANONICAL_ROSTER_BINDING_CORRECTION
            and annotation.get("rosterBindingState") in {"bound", "unbound"}
        ):
            dedicated_decisions[annotation_id] = (
                str(annotation["rosterBindingState"]),
                str(annotation.get("externalPlayerId") or "").strip() or None,
            )
            continue
        if _is_identity_unbind_tombstone(annotation):
            continue
        external_id = str(annotation.get("externalPlayerId") or "").strip()
        if external_id:
            positive_bindings[annotation_id] = external_id

    if not positive_bindings and not dedicated_decisions:
        # Legacy/resolver-derived external semantics have no durable correction
        # anchor. Preserve them on the original identity and never guess that
        # the newly-created range inherited them.
        return (
            source.manual_external_player_id,
            None,
            source.roster_binding_state,
            None,
        )

    def value_for(
        annotation_ids: set[str], canonical_person_id: str
    ) -> tuple[str | None, str | None]:
        dedicated = {
            decision
            for annotation_id, decision in dedicated_decisions.items()
            if annotation_id in annotation_ids
        }
        if len(dedicated) > 1:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} assigned conflicting roster edits to one partition",
                correction_id=correction_id,
                action="split",
                status="conflict",
                reason="conflicting-dedicated-roster-decisions",
                source_track_id=source_identity_id,
                target_id=canonical_person_id,
                candidates=[
                    {"rosterBindingState": state, "externalPlayerId": external_id}
                    for state, external_id in sorted(
                        dedicated,
                        key=lambda item: (item[0], item[1] or ""),
                    )
                ],
            )
        if dedicated:
            state, external_id = next(iter(dedicated))
            return (external_id if state == "bound" else None), state
        values = {
            external_id
            for annotation_id, external_id in positive_bindings.items()
            if annotation_id in annotation_ids
        }
        if len(values) > 1:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} assigned two roster players to one partition",
                correction_id=correction_id,
                action="split",
                status="conflict",
                reason="conflicting-confirmed-external-player-ids",
                source_track_id=source_identity_id,
                target_id=canonical_person_id,
                candidates=[{"externalPlayerId": value} for value in sorted(values)],
            )
        return next(iter(values), None), None

    source_external_id, source_roster_state = value_for(
        source_annotation_ids, source_identity_id
    )
    split_external_id, split_roster_state = value_for(
        split_annotation_ids, split_identity_id
    )
    return (
        source_external_id,
        split_external_id,
        source_roster_state,
        split_roster_state,
    )


def _partition_manual_semantics(
    annotation_ids: set[str],
    annotations_by_id: dict[str, dict],
) -> tuple[str | None, str | None, bool]:
    """Rebuild manual role/label only from positive local corrections."""

    rows = [
        annotations_by_id[annotation_id]
        for annotation_id in annotation_ids
        if annotation_id in annotations_by_id
        and _annotation_action(annotations_by_id[annotation_id])
        in {"confirm", "merge", "split"}
        and not _is_identity_unbind_tombstone(annotations_by_id[annotation_id])
        and annotations_by_id[annotation_id].get("kind") != "ignore"
    ]
    rows.sort(
        key=lambda item: (
            1 if _annotation_action(item) == "split" else 0,
            str(item.get("updatedAt") or ""),
            float(item.get("sceneTime") or 0.0),
            int(item.get("frameIndex") or 0),
            str(item.get("id") or ""),
        )
    )
    if not rows:
        return None, None, False
    kind = next(
        (str(item["kind"]) for item in reversed(rows) if item.get("kind")),
        None,
    )
    label = next(
        (
            str(item["label"]).strip()
            for item in reversed(rows)
            if str(item.get("label") or "").strip()
        ),
        None,
    )
    return kind, label, True


def _bound_roster_semantics_compatible(
    manual_kind: str | None,
    annotation_ids: set[str],
    annotations_by_id: dict[str, dict],
) -> bool:
    """Check the local role/team against its durable bound roster anchor.

    A split may intentionally change the semantics of the newly-created
    partition.  It may not, however, carry a bound home/away player into a
    partition labelled as the other team, referee, or unknown.  The durable
    correction's kind is used as the team anchor so this also works while the
    canonical output is being rebuilt.
    """

    bound_rows = [
        annotations_by_id[annotation_id]
        for annotation_id in annotation_ids
        if annotation_id in annotations_by_id
        and annotations_by_id[annotation_id].get("correctionKind")
        == CANONICAL_ROSTER_BINDING_CORRECTION
        and annotations_by_id[annotation_id].get("rosterBindingState") == "bound"
    ]
    if not bound_rows:
        return True
    expected_teams = {
        team
        for row in bound_rows
        if (team := _annotation_team(str(row.get("kind") or "")))
        in {"home", "away"}
    }
    requested_team = _annotation_team(manual_kind)
    requested_role = _annotation_role(manual_kind)
    return (
        len(expected_teams) == 1
        and requested_team in expected_teams
        and requested_role in {"player", "goalkeeper"}
    )


def _refresh_split_track_state(track: TrackState) -> None:
    track.points.sort(key=lambda point: (float(point["t"]), int(point.get("frameIndex") or 0)))
    if not track.points:
        return
    track.last_frame = max(int(point.get("frameIndex") or 0) for point in track.points)
    track.last_height = float((track.points[-1].get("bbox") or {}).get("height") or track.last_height)
    appearance_features = [
        np.asarray(point["_appearanceFeature"], dtype=np.float32)
        for point in track.points
        if point.get("_appearanceFeature") is not None
    ]
    if appearance_features:
        track.feature_sum = np.sum(np.stack(appearance_features), axis=0)
        track.feature_count = len(appearance_features)
    else:
        # A split partition must never retain the source identity's aggregate
        # when none of its own observations carries appearance evidence (for
        # example after importing a legacy scene without per-point metadata).
        track.feature_sum = None
        track.feature_count = 0
    role_votes: dict[str, float] = {}
    for point in track.points:
        role = str(point.get("_reidRole") or "")
        if role not in {"player", "goalkeeper", "referee", "other"}:
            continue
        try:
            confidence = float(point.get("_reidRoleConfidence"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(confidence) and confidence >= 0.60:
            role_votes[role] = role_votes.get(role, 0.0) + confidence
    track.reid_role_votes = role_votes
    if not track.manual_kind:
        track.role = (
            max(role_votes, key=lambda value: (role_votes[value], value))
            if role_votes
            else None
        )
    retained_observation_ids = {
        str(point.get("observationId"))
        for point in track.points
        if point.get("observationId")
    }
    track.reid_observation_ids.intersection_update(retained_observation_ids)
    if track.reid_sample_candidates:
        track.reid_sample_candidates = [
            item
            for item in track.reid_sample_candidates
            if str(item.get("observationId") or "") in retained_observation_ids
        ]
        track._select_reid_samples()
    evidence_rows = [
        point
        for point in track.points
        if point.get("_hasReidEvidence") and point.get("observationId")
    ]
    if evidence_rows:
        fingerprints = [
            str(
                point.get("_reidEvidenceFingerprint")
                or "observation:" + str(point["observationId"])
            )
            for point in evidence_rows
        ]
        track.reid_observation_ids = {
            str(point["observationId"]) for point in evidence_rows
        }
        track.reid_evidence_fingerprints = set(fingerprints)
        track.reid_observation_count = len(track.reid_evidence_fingerprints)
        track.reid_duplicate_evidence_count = len(fingerprints) - len(
            track.reid_evidence_fingerprints
        )
    elif track.reid_sample_candidates:
        track.reid_evidence_fingerprints = {
            str(
                item.get("evidenceFingerprint")
                or "observation:" + str(item.get("observationId") or "")
            )
            for item in track.reid_sample_candidates
        }
        track.reid_observation_count = len(track.reid_evidence_fingerprints)
        track.reid_duplicate_evidence_count = 0
    else:
        track.reid_evidence_fingerprints.clear()
        track.reid_observation_count = 0
        track.reid_duplicate_evidence_count = 0
    track.identity_tombstone_ids.intersection_update(track.annotation_ids)


def _apply_canonical_split_corrections(
    tracks: list[TrackState],
    scene: dict,
) -> tuple[list[TrackState], dict]:
    """Partition resolved identities by persisted [start, end) manual ranges.

    Splits run after automatic identity resolution, which makes each persisted
    range a real cannot-link barrier rather than another hint the resolver may
    immediately undo.
    """

    result = list(tracks)
    annotations_by_id = {
        str(item.get("id")): item
        for item in _identity_annotations(scene)
        if item.get("id")
    }
    splits = _ordered_split_corrections(_identity_annotations(scene))
    produced_split_ids = {
        str(annotation.get("splitCanonicalPersonId") or "").strip()
        for annotation in splits
        if str(annotation.get("splitCanonicalPersonId") or "").strip()
    }
    next_track_id = max((track.id for track in result), default=0) + 1
    applied: list[dict] = []
    for annotation in splits:
        correction_id = str(annotation["id"])
        time_range = _split_range(annotation)
        if time_range is None:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} has an invalid range",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="invalid-split-range",
                source_track_id=_annotation_source_identity(annotation),
            )
        start, end = time_range
        source_identity_id_hint = str(_annotation_source_identity(annotation) or "")
        source, target_point = _resolve_split_target_point(
            result,
            annotation,
            require_source_identity=source_identity_id_hint in produced_split_ids,
        )
        inside = [point for point in source.points if start <= float(point["t"]) < end]
        outside = [point for point in source.points if not start <= float(point["t"]) < end]
        if target_point not in inside:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} target is outside its range",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="target-outside-split-range",
                source_track_id=_annotation_source_identity(annotation),
                target_id=str(annotation.get("targetObservationId") or "") or None,
            )
        if not inside or not outside:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} would consume the complete identity",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="empty-split-partition",
                source_track_id=_annotation_source_identity(annotation),
                target_id=str(annotation.get("targetObservationId") or "") or None,
            )

        original_annotation_ids = set(source.annotation_ids)
        source_annotation_ids: set[str] = set()
        split_annotation_ids: set[str] = {correction_id}
        for annotation_id in sorted(original_annotation_ids):
            semantic_annotation = annotations_by_id.get(annotation_id)
            partition = _split_annotation_partition(
                annotation_id,
                semantic_annotation,
                inside,
                outside,
                start,
                end,
            )
            if partition == "range":
                split_annotation_ids.add(annotation_id)
                continue
            if partition == "ambiguous" and (
                _is_identity_unbind_tombstone(semantic_annotation)
                or str((semantic_annotation or {}).get("externalPlayerId") or "").strip()
            ):
                raise IdentityCorrectionError(
                    f"Split correction {correction_id} cannot localize a roster correction",
                    correction_id=correction_id,
                    action="split",
                    status="ambiguous",
                    reason="ambiguous-roster-correction-partition",
                    source_track_id=_annotation_source_identity(annotation),
                    target_id=annotation_id,
                )
            # Unknown legacy semantics remain with the original identity.  A
            # split must never guess that a new identity inherited them.
            source_annotation_ids.add(annotation_id)

        split_track = deepcopy(source)
        split_track.id = next_track_id
        next_track_id += 1
        split_track.points = [deepcopy(point) for point in inside]
        source.points = [deepcopy(point) for point in outside]

        source_identity_id = str(
            annotation.get("canonicalPersonId")
            or source.canonical_person_id
            or _new_canonical_person_id(source)
        )
        split_identity_id = str(annotation.get("splitCanonicalPersonId") or "")
        if not split_identity_id or split_identity_id == source_identity_id:
            raise IdentityCorrectionError(
                f"Split correction {correction_id} has no distinct identity key",
                correction_id=correction_id,
                action="split",
                status="unresolved",
                reason="invalid-split-identity",
                source_track_id=source_identity_id,
            )
        (
            source_external_player_id,
            split_external_player_id,
            source_roster_binding_state,
            split_roster_binding_state,
        ) = (
            _partition_external_player_ids(
                source,
                source_annotation_ids,
                split_annotation_ids,
                annotations_by_id,
                correction_id=correction_id,
                source_identity_id=source_identity_id,
                split_identity_id=split_identity_id,
            )
        )
        source.annotation_ids = source_annotation_ids
        split_track.annotation_ids = split_annotation_ids
        source.identity_tombstone_ids = {
            annotation_id
            for annotation_id in source_annotation_ids
            if annotation_id in source.identity_tombstone_ids
            or _is_identity_unbind_tombstone(annotations_by_id.get(annotation_id))
        }
        split_track.identity_tombstone_ids = {
            annotation_id
            for annotation_id in split_annotation_ids
            if annotation_id in split_track.identity_tombstone_ids
            or _is_identity_unbind_tombstone(annotations_by_id.get(annotation_id))
        }
        source.manual_external_player_id = source_external_player_id
        split_track.manual_external_player_id = split_external_player_id
        source.roster_binding_state = source_roster_binding_state
        split_track.roster_binding_state = split_roster_binding_state
        source.roster_binding_annotation_ids.intersection_update(
            source_annotation_ids
        )
        split_track.roster_binding_annotation_ids.intersection_update(
            split_annotation_ids
        )
        source_kind, source_label, source_has_positive_semantics = (
            _partition_manual_semantics(source_annotation_ids, annotations_by_id)
        )
        split_kind, split_label, _ = _partition_manual_semantics(
            split_annotation_ids, annotations_by_id
        )
        original_has_known_semantics = any(
            annotation_id in annotations_by_id
            and _annotation_action(annotations_by_id[annotation_id])
            in {"confirm", "merge", "split"}
            for annotation_id in original_annotation_ids
        )
        if source_has_positive_semantics:
            source.manual_kind = source_kind
            source.manual_label = source_label
        elif original_has_known_semantics:
            source.manual_kind = None
            source.manual_label = None
        split_track.manual_kind = split_kind or str(
            annotation.get("kind") or "other"
        )
        split_track.manual_label = split_label
        for partition_name, partition_track in (
            ("remaining", source),
            ("range", split_track),
        ):
            if (
                partition_track.roster_binding_state == "bound"
                and not _bound_roster_semantics_compatible(
                    partition_track.manual_kind,
                    partition_track.annotation_ids,
                    annotations_by_id,
                )
            ):
                raise IdentityCorrectionError(
                    f"Split correction {correction_id} gives its bound roster partition incompatible team or role semantics",
                    correction_id=correction_id,
                    action="split",
                    status="conflict",
                    reason="bound-roster-partition-semantics-conflict",
                    source_track_id=source_identity_id,
                    target_id=(
                        source_identity_id
                        if partition_name == "remaining"
                        else split_identity_id
                    ),
                    candidates=[
                        {
                            "partition": partition_name,
                            "kind": partition_track.manual_kind,
                            "externalPlayerId": partition_track.manual_external_player_id,
                        }
                    ],
                )
        source.canonical_person_id = source_identity_id
        split_track.canonical_person_id = split_identity_id
        source.manual_identity_owner_ids = {source_identity_id}
        split_track.manual_identity_owner_ids = {split_identity_id}
        source.identity_split_partitions[correction_id] = "remaining"
        split_track.identity_split_partitions[correction_id] = "range"
        split_track.identity_group_id = split_identity_id
        split_track.identity_status = "resolved"
        split_track.identity_confidence = 1.0
        # The selected range is a new manual identity partition, not another
        # vote for the source tracklet's jersey/roster prior.
        split_track.source_tracklet_ids = {split_track.local_tracklet_id}
        split_track.identity_evidence = []
        split_track.identity_conflicts = []
        evidence = {
            "id": f"{correction_id}:manual-split",
            "kind": "manual",
            "label": "Manual identity split",
            "value": f"[{start:.3f}, {end:.3f})",
            "supportCount": len(inside),
            "source": "identity-correction",
            "manual": True,
        }
        source.identity_evidence.append({**evidence, "partition": "remaining"})
        split_track.identity_evidence.append({**evidence, "partition": "range"})
        _refresh_split_track_state(source)
        _refresh_split_track_state(split_track)
        result.append(split_track)
        applied.append(
            {
                "correctionId": correction_id,
                "sourceCanonicalPersonId": source_identity_id,
                "splitCanonicalPersonId": split_identity_id,
                "rangeStart": round(start, 3),
                "rangeEnd": round(end, 3),
                "affectedObservationCount": len(inside),
                "remainingObservationCount": len(outside),
                "targetObservationId": annotation.get("targetObservationId"),
            }
        )
    return (
        sorted(result, key=lambda track: (float(track.points[0]["t"]), track.id)),
        {"appliedCount": len(applied), "applied": applied},
    )


def _apply_track_identity_corrections(tracks: list[TrackState], scene: dict) -> list[TrackState]:
    """Resolve explicit identity merges after online association and before QA."""

    result = list(tracks)
    annotations = _identity_annotations(scene)
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    scene_tracks: dict[str, dict] = {}
    for track in _canonical_analysis_subjects(scene):
        for identifier in (track.get("id"), track.get("canonicalPersonId")):
            if identifier:
                scene_tracks[str(identifier)] = track
    excluded_track_ids = {
        str(_annotation_source_identity(annotation))
        for annotation in annotations
        if _annotation_action(annotation) == "exclude"
        and _annotation_scope(annotation) == "identity"
        and _annotation_source_identity(annotation)
    }
    for track_id in excluded_track_ids:
        correction = next(
            annotation
            for annotation in annotations
            if _annotation_action(annotation) == "exclude"
            and _annotation_scope(annotation) == "identity"
            and str(_annotation_source_identity(annotation) or "") == track_id
        )
        correction_id = str(correction.get("id") or track_id)
        exact = [
            track for track in result if correction_id in track.annotation_ids
        ]
        if len(exact) == 1:
            result.remove(exact[0])
            continue
        if len(exact) > 1:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} matched multiple raw tracks",
                correction_id=correction_id,
                action="exclude",
                status="ambiguous",
                reason="multiple-exact-source-anchors",
                source_track_id=track_id,
                candidates=[{"rawTrackId": track.id} for track in exact],
            )
        previous = scene_tracks.get(track_id)
        if previous is None:
            raise IdentityCorrectionError(
                f"Identity correction {correction_id} references a missing source track",
                correction_id=correction_id,
                action="exclude",
                status="unresolved",
                reason="missing-source-track",
                source_track_id=track_id,
            )
        result.remove(
            _resolve_previous_identity_track(
                result,
                previous,
                correction_id=correction_id,
                action="exclude",
                source_track_id=track_id,
                target_id=track_id,
            )
        )
    for annotation in annotations:
        if _annotation_action(annotation) != "merge" or not annotation.get("id"):
            continue
        source = next(
            (
                track
                for track in result
                if str(annotation["id"]) in track.annotation_ids
            ),
            None,
        )
        if source is None:
            raise IdentityCorrectionError(
                f"Identity correction {annotation['id']} did not attach to a raw source track",
                correction_id=str(annotation["id"]),
                action="merge",
                status="unresolved",
                reason="missing-source-anchor",
                source_track_id=_annotation_source_identity(annotation),
                target_id=str(annotation.get("mergeTargetId") or "") or None,
            )
        terminal_id = _terminal_identity_target(
            str(annotation.get("mergeTargetId") or ""), annotation_by_id
        )
        exact_targets = [
            track for track in result if terminal_id in track.annotation_ids
        ]
        if len(exact_targets) > 1:
            raise IdentityCorrectionError(
                f"Identity correction {annotation['id']} matched multiple merge targets",
                correction_id=str(annotation["id"]),
                action="merge",
                status="ambiguous",
                reason="multiple-exact-merge-targets",
                source_track_id=_annotation_source_identity(annotation),
                target_id=terminal_id,
                candidates=[{"rawTrackId": track.id} for track in exact_targets],
            )
        target = exact_targets[0] if exact_targets else None
        if target is None and terminal_id in scene_tracks:
            target = _resolve_previous_identity_track(
                result,
                scene_tracks[terminal_id],
                correction_id=str(annotation["id"]),
                action="merge",
                source_track_id=_annotation_source_identity(annotation),
                target_id=terminal_id,
                exclude=source,
            )
        if target is None:
            raise IdentityCorrectionError(
                f"Identity correction {annotation['id']} could not resolve its merge target",
                correction_id=str(annotation["id"]),
                action="merge",
                status="unresolved",
                reason="missing-merge-target",
                source_track_id=_annotation_source_identity(annotation),
                target_id=terminal_id,
            )
        if target is source:
            continue
        _raise_manual_merge_external_player_conflict(target, source, annotation)
        terminal_annotation = annotation_by_id.get(terminal_id)
        terminal_subject = scene_tracks.get(terminal_id) or (
            scene_tracks.get(str(_annotation_source_identity(terminal_annotation) or ""))
            if terminal_annotation is not None
            else None
        )
        target_owner_id = str(
            (terminal_subject or {}).get("canonicalPersonId")
            or _annotation_source_identity(terminal_annotation)
            or terminal_id
        ).strip()
        _merge_raw_track_states(
            target,
            source,
            allow_manual_owner_merge=True,
            manual_target_owner_id=target_owner_id or None,
        )
        result.remove(source)
    return result


def _merge_scene_track_documents(
    target: dict,
    source: dict,
    annotation: dict,
    scene: dict,
) -> dict:
    conflict = _confirmed_external_player_conflict(
        target.get("externalPlayerId"), source.get("externalPlayerId")
    )
    if conflict is not None:
        target_external_id, source_external_id = conflict
        raise ReconstructionError(
            "Cannot merge identities with different confirmed roster players: "
            f"{source_external_id} and {target_external_id}"
        )
    keyframes_by_time: dict[float, dict] = {}
    observed_keyframes = [
        keyframe
        for keyframe in [*(target.get("keyframes") or []), *(source.get("keyframes") or [])]
        if keyframe.get("observed") is not False
    ]
    for keyframe in observed_keyframes:
        key = round(float(keyframe["t"]), 4)
        previous = keyframes_by_time.get(key)
        if previous is None or float(keyframe.get("confidence") or 0.0) >= float(
            previous.get("confidence") or 0.0
        ):
            keyframes_by_time[key] = keyframe
    merged_from = set(
        (target.get("identityCorrection") or {}).get("mergedTrackIds") or []
    )
    if source.get("id") and source.get("id") != target.get("id"):
        merged_from.add(str(source["id"]))
    correction_annotations = set(
        (target.get("identityCorrection") or {}).get("annotationIds") or []
    )
    correction_annotations.add(str(annotation["id"]))
    merged_keyframes, presence = _continuous_track_keyframes(
        [keyframes_by_time[key] for key in sorted(keyframes_by_time)],
        float(scene.get("duration") or 0.0),
        scene.get("payload", {}).get("pitch") or {"length": 105, "width": 68},
        sum((index + 1) * ord(character) for index, character in enumerate(str(target["id"]))),
    )
    merged_observations = _merge_track_observations(
        list(target.get("observations") or []),
        list(source.get("observations") or []),
    )
    return {
        **target,
        "annotationIds": sorted(
            {
                *(target.get("annotationIds") or []),
                *(source.get("annotationIds") or []),
            }
        ),
        "identityCorrection": {
            "status": "merged",
            "targetId": str(target["id"]),
            "annotationIds": sorted(correction_annotations),
            "mergedTrackIds": sorted(merged_from),
        },
        "presence": presence,
        "observations": merged_observations,
        "keyframes": merged_keyframes,
    }


def _apply_scene_track_identity_corrections(tracks: list[dict], scene: dict) -> list[dict]:
    """Publish one stable scene identity for every explicit merge directive."""

    result = [deepcopy(track) for track in tracks]
    annotations = _identity_annotations(scene)
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    previous_tracks: dict[str, dict] = {}
    for track in scene.get("payload", {}).get("tracks") or []:
        for identifier in (track.get("id"), track.get("canonicalPersonId")):
            if identifier:
                previous_tracks[str(identifier)] = track
    def by_annotation(annotation_id: str) -> dict | None:
        return next(
            (
                track
                for track in result
                if annotation_id in (track.get("annotationIds") or [])
            ),
            None,
        )

    def crosses_manual_split_barrier(left: dict, right: dict) -> bool:
        left_partitions = left.get("identitySplitPartitions") or {}
        right_partitions = right.get("identitySplitPartitions") or {}
        return any(
            correction_id in right_partitions
            and right_partitions[correction_id] != partition
            for correction_id, partition in left_partitions.items()
        )

    for annotation in annotations:
        if _annotation_action(annotation) != "merge" or not annotation.get("id"):
            continue
        source = by_annotation(str(annotation["id"]))
        if source is None:
            continue
        terminal_id = _terminal_identity_target(
            str(annotation.get("mergeTargetId") or ""), annotation_by_id
        )
        target = (
            by_annotation(terminal_id)
            if terminal_id in annotation_by_id
            else next(
                (
                    track
                    for track in result
                    if str(track.get("id") or "") == terminal_id
                    or str(track.get("canonicalPersonId") or "") == terminal_id
                ),
                None,
            )
        )
        if target is source:
            source["identityCorrection"] = {
                "status": "merged",
                "targetId": str(source["id"]),
                "annotationIds": sorted(
                    {
                        *((source.get("identityCorrection") or {}).get("annotationIds") or []),
                        str(annotation["id"]),
                    }
                ),
                "mergedTrackIds": sorted(
                    set((source.get("identityCorrection") or {}).get("mergedTrackIds") or [])
                ),
            }
            continue
        if target is None and terminal_id in previous_tracks:
            previous = previous_tracks[terminal_id]
            target = {
                **source,
                **{
                    key: previous[key]
                    for key in (
                        "id",
                        "canonicalPersonId",
                        "label",
                        "teamId",
                        "color",
                        "number",
                        "role",
                        "externalPlayerId",
                    )
                    if key in previous
                },
            }
            result[result.index(source)] = target
        if target is None:
            continue
        if crosses_manual_split_barrier(source, target):
            # The post-resolver split is an explicit cannot-link. Older merge
            # corrections may still exist for audit/undo, but cannot reconnect
            # the two partitions in the published scene document.
            continue
        conflict = _confirmed_external_player_conflict(
            target.get("externalPlayerId"), source.get("externalPlayerId")
        )
        if conflict is not None:
            target_external_id, source_external_id = conflict
            correction_id = str(annotation["id"])
            raise IdentityCorrectionError(
                (
                    f"Identity correction {correction_id} cannot merge confirmed roster "
                    f"players {source_external_id} and {target_external_id}"
                ),
                correction_id=correction_id,
                action="merge",
                status="conflict",
                reason="conflicting-confirmed-external-player-ids",
                source_track_id=_annotation_source_identity(annotation),
                target_id=terminal_id,
                candidates=[
                    {
                        "trackId": source.get("id"),
                        "externalPlayerId": source_external_id,
                    },
                    {
                        "trackId": target.get("id"),
                        "externalPlayerId": target_external_id,
                    },
                ],
            )
        merged = _merge_scene_track_documents(target, source, annotation, scene)
        target_index = result.index(target)
        result[target_index] = merged
        if source in result and source is not target:
            result.remove(source)
    return result


def upsert_frame_person_annotation(
    scene: dict,
    values: dict,
    *,
    persist: bool = True,
) -> dict:
    requested_annotation_id = str(values.get("annotation_id") or "").strip() or None
    if requested_annotation_id is not None:
        existing_annotation = next(
            (
                item
                for item in (
                    scene.get("payload", {})
                    .get("videoAsset", {})
                    .get("reconstruction", {})
                    .get("frameAnnotations", [])
                )
                if str(item.get("id") or "") == requested_annotation_id
            ),
            None,
        )
        if (
            existing_annotation is not None
            and existing_annotation.get("correctionKind")
            == CANONICAL_ROSTER_BINDING_CORRECTION
        ):
            raise ReconstructionError(
                "Canonical roster corrections can only be changed through Bind / Unbind"
            )
    frames = _frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    scene_time = float(values["scene_time"])
    target_path, frame_time = min(frames, key=lambda item: abs(item[1] - scene_time))
    image = cv2.imread(str(target_path))
    if image is None:
        raise ReconstructionError("The sampled frame could not be read")
    frame_height, frame_width = image.shape[:2]
    requested_bbox = values["bbox"]
    x = min(max(0.0, float(requested_bbox["x"])), frame_width - 4.0)
    y = min(max(0.0, float(requested_bbox["y"])), frame_height - 4.0)
    width = min(float(requested_bbox["width"]), frame_width - x)
    height = min(float(requested_bbox["height"]), frame_height - y)
    if width < 4 or height < 4:
        raise ReconstructionError("The person box is outside the video frame")
    annotation_id = requested_annotation_id or f"annotation-{uuid4().hex[:12]}"
    action = _annotation_action(values)
    if values.get("external_player_id") is not None:
        raise ReconstructionError(
            "Roster identity must be changed through the canonical Bind / Unbind endpoint"
        )
    if action != "exclude" and values.get("kind") == "ignore":
        raise ReconstructionError(
            "Choose a person role when confirming or merging an excluded detection"
        )
    explicit_action = values.get("action") is not None
    requested_scope = str(values.get("scope") or "").strip().lower()
    scope = (
        "range"
        if action == "split"
        else requested_scope
        if explicit_action and requested_scope in {"observation", "identity"}
        else "identity"
        if explicit_action
        else "observation"
    )
    merge_target_id = (
        str(values.get("merge_target_id") or "").strip() or None
        if action == "merge"
        else None
    )
    source_track_id = str(values.get("source_track_id") or "").strip() or None
    canonical_person_id = (
        str(values.get("canonical_person_id") or "").strip() or None
    )
    canonical_subject: dict | None = None
    if canonical_person_id is not None:
        canonical_subject = next(
            (
                subject
                for subject in _canonical_analysis_subjects(scene)
                if str(subject.get("canonicalPersonId") or "")
                == canonical_person_id
            ),
            None,
        )
        if canonical_subject is None:
            raise ReconstructionError("The canonical person no longer exists")
    if (
        action == "confirm"
        and canonical_subject is not None
        and canonical_subject.get("externalPlayerId")
    ):
        expected_team = str(canonical_subject.get("teamId") or "").strip()
        requested_team = _annotation_team(str(values.get("kind") or ""))
        requested_role = _annotation_role(str(values.get("kind") or ""))
        if (
            requested_team != expected_team
            or requested_role in {"referee", "other", None}
        ):
            raise ReconstructionError(
                "Unbind the roster player before changing this person to another team or non-player role"
            )
    target_observation_id = (
        str(values.get("target_observation_id") or "").strip() or None
        if action == "split"
        else None
    )
    target_observation: dict | None = None
    split_canonical_person_id: str | None = None
    range_start: float | None = None
    range_end: float | None = None
    affected_preview: dict | None = None
    if action == "split":
        if canonical_person_id is None:
            raise ReconstructionError("Choose the canonical identity before splitting it")
        if target_observation_id is None:
            raise ReconstructionError(
                "Split requires one immutable tracked observation; rebuild or select another frame"
            )
        subject, target_observation = _split_target_snapshot(
            scene,
            canonical_person_id,
            target_observation_id,
        )
        if int(target_observation["frameIndex"]) != int(target_path.stem.split("_")[-1]):
            raise ReconstructionError("The split target does not belong to the selected frame")
        range_start = float(
            values.get("range_start")
            if values.get("range_start") is not None
            else target_observation["sceneTime"]
        )
        range_end = float(
            values.get("range_end")
            if values.get("range_end") is not None
            else scene.get("duration")
        )
        if (
            not np.isfinite([range_start, range_end]).all()
            or range_start < 0.0
            or range_end > float(scene.get("duration") or 0.0) + 1e-6
            or range_end <= range_start
        ):
            raise ReconstructionError("Split range must be inside the scene and have a valid end")
        target_time = float(target_observation["sceneTime"])
        if not range_start <= target_time < range_end:
            raise ReconstructionError("The target observation must be inside the split range")
        subject_observations = [
            observation
            for observation in subject.get("observations") or []
            if observation.get("sceneTime") is not None
        ]
        affected_count = sum(
            range_start <= float(observation["sceneTime"]) < range_end
            for observation in subject_observations
        )
        remaining_count = len(subject_observations) - affected_count
        if affected_count <= 0 or remaining_count <= 0:
            raise ReconstructionError(
                "Split must leave at least one detector observation on both identities"
            )
        split_seed = f"{annotation_id}:{target_observation_id}"
        split_canonical_person_id = (
            f"canonical-split-{sha256(split_seed.encode('utf-8')).hexdigest()[:12]}"
        )
        affected_preview = {
            "canonicalPersonId": canonical_person_id,
            "splitCanonicalPersonId": split_canonical_person_id,
            "rangeStart": round(range_start, 3),
            "rangeEnd": round(range_end, 3),
            "affectedObservationCount": affected_count,
            "remainingObservationCount": remaining_count,
        }
    if (
        action == "exclude"
        and scope == "identity"
        and canonical_person_id is None
        and source_track_id is None
    ):
        raise ReconstructionError(
            "Choose the tracked identity before excluding the whole trajectory"
        )
    annotation = {
        "id": annotation_id,
        "sceneTime": round(float(frame_time), 3),
        "sourceTime": round(
            float(scene.get("payload", {}).get("videoAsset", {}).get("sourceStart") or 0.0)
            + float(frame_time),
            3,
        ),
        "frameIndex": int(target_path.stem.split("_")[-1]),
        "bbox": {
            "x": round(x, 2),
            "y": round(y, 2),
            "width": round(width, 2),
            "height": round(height, 2),
        },
        "kind": "ignore" if action == "exclude" else values["kind"],
        "label": (
            None
            if action in {"exclude", "split"}
            else (values.get("label") or "").strip() or None
        ),
        "externalPlayerId": None,
        "action": action,
        "scope": scope,
        "mergeTargetId": merge_target_id,
        "sourceTrackId": source_track_id,
        "canonicalPersonId": canonical_person_id,
        "targetObservationId": target_observation_id,
        "targetObservation": target_observation,
        "rangeStart": round(range_start, 3) if range_start is not None else None,
        "rangeEnd": round(range_end, 3) if range_end is not None else None,
        "splitCanonicalPersonId": split_canonical_person_id,
        "affectedPreview": affected_preview,
        "previewState": {
            "confirm": "confirmed",
            "exclude": "excluded",
            "merge": "merged",
            "split": "split",
        }[action],
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    annotations = list(reconstruction.get("frameAnnotations") or [])
    if action == "split":
        # Snapshot only decisions that existed before this split.  Undo can
        # then restore their original owner/id instead of silently deleting a
        # durable Unbind after the correction has moved to the range child.
        pre_split_unbinds = [
            deepcopy(item)
            for item in _dedicated_roster_corrections_by_owner(
                scene, annotations
            ).get(str(canonical_person_id or ""), [])
            if item.get("rosterBindingState") == "unbound"
        ]
        for item in pre_split_unbinds:
            item[ROSTER_DECISION_ORIGIN_FIELD] = _roster_decision_origin_id(item)
        annotation[SPLIT_ROSTER_UNDO_FIELD] = pre_split_unbinds
    existing_index = next(
        (index for index, item in enumerate(annotations) if item.get("id") == annotation_id),
        None,
    )
    if existing_index is None:
        annotations.append(annotation)
    else:
        annotations[existing_index] = annotation
    consolidated_roster_corrections: list[dict] = []
    if action == "merge":
        annotations, consolidated_roster_corrections = (
            _consolidate_compatible_merge_roster_corrections(
                scene, annotations, annotation
            )
        )
        if consolidated_roster_corrections:
            consolidated_roster_ids = sorted(
                str(item["id"])
                for item in consolidated_roster_corrections
                if item.get("id")
            )
            annotation["consolidatedRosterCorrectionIds"] = consolidated_roster_ids
            annotation["consolidatedRosterCorrections"] = deepcopy(
                consolidated_roster_corrections
            )
    consolidated_roster_ids = {
        str(item["id"])
        for item in consolidated_roster_corrections
        if item.get("id")
    }
    _validate_identity_corrections(scene, annotations)
    if action == "merge" and merge_target_id is not None:
        kind, label, external_player_id = _identity_target_defaults(
            scene, annotations, merge_target_id
        )
        annotation.update(
            {
                "kind": kind,
                "label": label,
                # A merge points at the live target identity. Persisting its
                # roster ID here creates a stale independent claim after a
                # later Bind/Unbind/Rebind.
                "externalPlayerId": None,
            }
        )
    reconstruction["frameAnnotations"] = sorted(
        annotations,
        key=lambda item: (int(item.get("frameIndex") or 0), str(item.get("id") or "")),
    )
    video["reconstruction"] = reconstruction
    _remove_annotation_references(scene, consolidated_roster_ids)
    if persist:
        scene_store.put(scene)
    return annotation


def delete_frame_person_annotation(
    scene: dict,
    annotation_id: str,
    *,
    persist: bool = True,
) -> dict:
    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    annotations = list(reconstruction.get("frameAnnotations") or [])
    annotation = next((item for item in annotations if item.get("id") == annotation_id), None)
    if annotation is None:
        raise ReconstructionError("Frame annotation was not found")
    if annotation.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION:
        raise ReconstructionError(
            "Canonical roster corrections can only be changed through Bind / Unbind / Clear"
        )
    dependent_unbound_ids: set[str] = set()
    restored_roster_corrections: list[dict] = []
    if _annotation_action(annotation) in {"split", "merge"}:
        if _annotation_action(annotation) == "merge":
            post_merge_roster_corrections = [
                item
                for item in annotations
                if item.get("correctionKind")
                == CANONICAL_ROSTER_BINDING_CORRECTION
                and str(annotation_id)
                in {
                    str(value)
                    for value in item.get(ROSTER_IDENTITY_DEPENDENCIES_FIELD)
                    or []
                }
            ]
            if post_merge_roster_corrections:
                correction_ids = ", ".join(
                    sorted(
                        str(item.get("id") or "unknown")
                        for item in post_merge_roster_corrections
                    )
                )
                raise ReconstructionError(
                    "A roster decision was created or changed after this merge "
                    "and cannot be distributed safely. Unbind any bound player, "
                    "then delete the roster correction before undoing the merge: "
                    + correction_ids
                )
        dependency_keys = {str(annotation_id)}
        if _annotation_action(annotation) == "split":
            produced_identity = str(
                annotation.get("splitCanonicalPersonId") or ""
            ).strip()
            if produced_identity:
                dependency_keys.add(produced_identity)
        direct_dependents = []
        for item in annotations:
            if item is annotation:
                continue
            references = {
                str(value).strip()
                for value in (
                    _annotation_source_identity(item),
                    item.get("mergeTargetId")
                    if _annotation_action(item) == "merge"
                    else None,
                )
                if str(value or "").strip()
            }
            if references & dependency_keys:
                direct_dependents.append(item)
        blocking_dependents = [
            item
            for item in direct_dependents
            if not (
                item.get("correctionKind")
                == CANONICAL_ROSTER_BINDING_CORRECTION
                and item.get("rosterBindingState") == "unbound"
            )
        ]
        if blocking_dependents:
            dependent_ids = ", ".join(
                sorted(str(item.get("id") or "unknown") for item in blocking_dependents)
            )
            raise ReconstructionError(
                "Delete dependent identity corrections before undoing this split or merge: "
                + dependent_ids
            )
        annotation_by_id = {
            str(item.get("id")): item for item in annotations if item.get("id")
        }
        affected_identity_ids = {
            value
            for value in (
                _canonical_correction_identity_key(
                    scene,
                    annotation_by_id,
                    _annotation_source_identity(annotation),
                ),
                _canonical_correction_identity_key(
                    scene,
                    annotation_by_id,
                    str(annotation.get("splitCanonicalPersonId") or "")
                    if _annotation_action(annotation) == "split"
                    else _terminal_identity_target(
                        str(annotation.get("mergeTargetId") or ""),
                        annotation_by_id,
                    ),
                ),
            )
            if value
        }
        dependent = [
            item
            for owner, rows in _dedicated_roster_corrections_by_owner(
                scene, annotations
            ).items()
            if owner in affected_identity_ids
            for item in rows
        ]
        if any(item.get("rosterBindingState") == "bound" for item in dependent):
            raise ReconstructionError(
                "Unbind roster players on the affected identities before undoing this split or merge"
            )
        if _annotation_action(annotation) == "split":
            dependent_unbound_ids = {
                str(item.get("id")) for item in dependent if item.get("id")
            }
            stored_rows = annotation.get(SPLIT_ROSTER_UNDO_FIELD) or []
            if not isinstance(stored_rows, list):
                raise ReconstructionError(
                    "The split has invalid roster undo metadata"
                )
            for stored in stored_rows:
                if not isinstance(stored, dict) or (
                    stored.get("correctionKind")
                    != CANONICAL_ROSTER_BINDING_CORRECTION
                    or stored.get("rosterBindingState") != "unbound"
                    or not stored.get("id")
                ):
                    raise ReconstructionError(
                        "The split has unsafe roster undo metadata; rebuild before deleting it"
                    )
            candidates = [deepcopy(item) for item in stored_rows] or [
                deepcopy(item)
                for item in dependent
                if item.get("rosterBindingState") == "unbound"
            ]
            if candidates:
                # All explicit Unbind rows carry the same compatible negative
                # decision. Prefer the exact pre-split snapshot when available,
                # otherwise retain one deterministic current row and migrate it
                # to the recombined source identity instead of deleting it.
                retained = min(
                    candidates,
                    key=lambda item: (
                        _roster_decision_origin_id(item),
                        str(item.get("id") or ""),
                    ),
                )
                source_owner = str(
                    _canonical_correction_identity_key(
                        scene,
                        annotation_by_id,
                        _annotation_source_identity(annotation),
                    )
                    or _annotation_source_identity(annotation)
                    or ""
                ).strip()
                if not source_owner:
                    raise ReconstructionError(
                        "The split source identity is missing from roster undo metadata"
                    )
                retained_id = _canonical_roster_binding_annotation_id(source_owner)
                retained["id"] = retained_id
                retained["canonicalPersonId"] = source_owner
                retained["sourceTrackId"] = None
                retained[ROSTER_DECISION_ORIGIN_FIELD] = (
                    _roster_decision_origin_id(retained) or retained_id
                )
                retained[ROSTER_IDENTITY_DEPENDENCIES_FIELD] = sorted(
                    {
                        str(value)
                        for value in retained.get(
                            ROSTER_IDENTITY_DEPENDENCIES_FIELD
                        )
                        or []
                        if str(value) != str(annotation_id)
                    }
                )
                target_observation = retained.get("targetObservation")
                if isinstance(target_observation, dict):
                    target_observation = deepcopy(target_observation)
                    target_observation["canonicalPersonId"] = source_owner
                    target_observation["annotationId"] = retained_id
                    target_observation["annotationIds"] = sorted(
                        {
                            str(value)
                            for value in target_observation.get("annotationIds")
                            or []
                            if str(value) not in dependent_unbound_ids
                        }
                        | {retained_id}
                    )
                    retained["targetObservation"] = target_observation
                restored_roster_corrections.append(retained)
        else:
            stored_rows = annotation.get("consolidatedRosterCorrections") or []
            if not isinstance(stored_rows, list):
                raise ReconstructionError(
                    "The merge has invalid roster undo metadata"
                )
            for stored in stored_rows:
                if not isinstance(stored, dict) or (
                    stored.get("correctionKind")
                    != CANONICAL_ROSTER_BINDING_CORRECTION
                    or stored.get("rosterBindingState") != "unbound"
                    or not stored.get("id")
                ):
                    raise ReconstructionError(
                        "The merge has unsafe roster undo metadata; rebuild before deleting it"
                    )
                restored_roster_corrections.append(deepcopy(stored))
    remaining = [
        item
        for item in annotations
        if item.get("id") != annotation_id
        and str(item.get("id") or "") not in dependent_unbound_ids
    ]
    existing_ids = {str(item.get("id") or "") for item in remaining}
    conflicting_restore_ids = sorted(
        str(item["id"])
        for item in restored_roster_corrections
        if str(item["id"]) in existing_ids
    )
    if conflicting_restore_ids:
        raise ReconstructionError(
            "Roster undo metadata conflicts with current corrections: "
            + ", ".join(conflicting_restore_ids)
        )
    remaining.extend(restored_roster_corrections)
    remaining.sort(
        key=lambda item: (int(item.get("frameIndex") or 0), str(item.get("id") or ""))
    )
    _validate_identity_corrections(scene, remaining)
    reconstruction["frameAnnotations"] = remaining
    video["reconstruction"] = reconstruction
    restored_ids = {
        str(item.get("id"))
        for item in restored_roster_corrections
        if item.get("id")
    }
    _remove_annotation_references(
        scene, dependent_unbound_ids - restored_ids
    )
    if persist:
        scene_store.put(scene)
    return annotation


def _bbox_payload_box(bbox: dict) -> tuple[float, float, float, float]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    return x, y, x + float(bbox["width"]), y + float(bbox["height"])


def _raw_person_bbox(raw: dict) -> dict:
    return {
        "x": round(float(raw["x"]) - float(raw["width"]) / 2, 2),
        "y": round(float(raw["y"]) - float(raw["height"]), 2),
        "width": round(float(raw["width"]), 2),
        "height": round(float(raw["height"]), 2),
    }


def _track_observation_schema_version(scene: dict) -> int | None:
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    try:
        version = int(reconstruction.get("trackObservationSchemaVersion") or 0)
        if version >= 1:
            return version
    except (TypeError, ValueError):
        pass
    return 1 if any(
        "observations" in track
        for track in [
            *(scene.get("payload", {}).get("tracks") or []),
            *(scene.get("payload", {}).get("canonicalPeople") or []),
        ]
    ) else None


def _has_track_observation_schema(scene: dict) -> bool:
    return _track_observation_schema_version(scene) is not None


def _canonical_analysis_subjects(scene: dict) -> list[dict]:
    payload = scene.get("payload", {})
    render_tracks = payload.get("tracks") or []
    canonical_people = payload.get("canonicalPeople") or []
    if not canonical_people:
        return [deepcopy(track) for track in render_tracks]
    by_id = {str(track.get("id")): track for track in render_tracks if track.get("id")}
    by_canonical = {
        str(track.get("canonicalPersonId")): track
        for track in render_tracks
        if track.get("canonicalPersonId")
    }
    result = []
    for person in canonical_people:
        canonical_id = str(person.get("canonicalPersonId") or person.get("id") or "")
        render = by_id.get(str(person.get("renderTrackId") or "")) or by_canonical.get(
            canonical_id
        )
        result.append(
            {
                **(deepcopy(render) if render is not None else {}),
                "id": render.get("id") if render is not None else None,
                "canonicalPersonId": canonical_id,
                "label": person.get("displayName")
                or (render.get("label") if render is not None else canonical_id),
                "displayName": person.get("displayName"),
                "identityStatus": person.get("identityStatus"),
                "identityConfidence": person.get("identityConfidence"),
                "identitySource": person.get("identitySource"),
                "jerseyNumber": person.get("jerseyNumber"),
                "teamId": person.get("teamId")
                or (render.get("teamId") if render is not None else None),
                "role": person.get("role")
                or (render.get("role") if render is not None else None),
                "externalPlayerId": person.get("externalPlayerId"),
                "annotationIds": person.get("annotationIds")
                or (render.get("annotationIds") if render is not None else []),
                "observations": deepcopy(person.get("observations") or []),
                "keyframes": deepcopy(render.get("keyframes") or [])
                if render is not None
                else [],
                "renderTrackId": render.get("id") if render is not None else None,
            }
        )
    return result


CANONICAL_ROSTER_BINDING_CORRECTION = "canonical-roster-binding-v1"
ROSTER_DECISION_ORIGIN_FIELD = "rosterDecisionOriginId"
ROSTER_IDENTITY_DEPENDENCIES_FIELD = "identityCorrectionDependencies"
SPLIT_ROSTER_UNDO_FIELD = "preSplitRosterCorrections"


def _canonical_roster_binding_annotation_id(canonical_person_id: str) -> str:
    digest = sha256(canonical_person_id.encode("utf-8")).hexdigest()[:16]
    return f"roster-binding-{digest}"


def _roster_decision_origin_id(annotation: dict) -> str:
    """Return the stable lineage key retained while a decision is re-keyed."""

    return str(
        annotation.get(ROSTER_DECISION_ORIGIN_FIELD)
        or annotation.get("id")
        or ""
    ).strip()


def _active_merge_dependencies(
    scene: dict, person: dict, annotations: list[dict]
) -> set[str]:
    """Return merge corrections currently represented by a published person.

    A roster decision authored after a merge cannot later be assigned to either
    pre-merge identity merely from its persisted canonical owner.  Recording the
    active merge ids at authoring time gives undo a durable fail-closed gate.
    """

    person_annotation_ids = {
        str(value).strip()
        for value in person.get("annotationIds") or []
        if str(value or "").strip()
    }
    canonical_person_id = str(
        person.get("canonicalPersonId") or person.get("id") or ""
    ).strip()
    annotation_by_id = {
        str(item.get("id")): item for item in annotations if item.get("id")
    }
    dependencies: set[str] = set()
    for annotation in annotations:
        if not annotation.get("id") or _annotation_action(annotation) != "merge":
            continue
        correction_id = str(annotation["id"])
        if correction_id in person_annotation_ids:
            dependencies.add(correction_id)
            continue
        terminal_id = _terminal_identity_target(
            str(annotation.get("mergeTargetId") or ""), annotation_by_id
        )
        terminal_annotation = annotation_by_id.get(terminal_id)
        target_owner = _canonical_correction_identity_key(
            scene,
            annotation_by_id,
            _annotation_source_identity(terminal_annotation)
            if terminal_annotation is not None
            else terminal_id,
        )
        if canonical_person_id and target_owner == canonical_person_id:
            dependencies.add(correction_id)
    return dependencies


def _active_split_dependencies(
    scene: dict, person: dict, annotations: list[dict]
) -> set[str]:
    """Return direct split branches and the selected partition's ancestors."""

    person_annotation_ids = {
        str(value).strip()
        for value in person.get("annotationIds") or []
        if str(value or "").strip()
    }
    canonical_person_id = str(
        person.get("canonicalPersonId") or person.get("id") or ""
    ).strip()
    split_endpoints: dict[str, tuple[set[str], set[str]]] = {}
    for correction in annotations:
        if not correction.get("id") or _annotation_action(correction) != "split":
            continue
        correction_id = str(correction["id"])
        split_endpoints[correction_id] = _correction_endpoint_ids(
            scene,
            correction,
            annotations,
        )

    # An annotation id is authoritative evidence that the published person
    # represents that split.  The selected source partition also owns every
    # direct outgoing split snapshot because Undo Split can restore its prior
    # roster decision.
    dependencies = set(person_annotation_ids) & set(split_endpoints)
    dependencies.update(
        correction_id
        for correction_id, (source_ids, target_ids) in split_endpoints.items()
        if canonical_person_id in source_ids | target_ids
    )

    # Walk only toward ancestors.  An undirected closure would incorrectly
    # absorb sibling partitions and let Clear on C erase a distinct D branch.
    ancestor_identity_ids = {canonical_person_id}
    for correction_id in dependencies:
        ancestor_identity_ids.update(split_endpoints[correction_id][0])
    changed = True
    while changed:
        changed = False
        for correction_id, (source_ids, target_ids) in split_endpoints.items():
            if correction_id in dependencies or not (
                target_ids & ancestor_identity_ids
            ):
                continue
            dependencies.add(correction_id)
            ancestor_identity_ids.update(source_ids)
            changed = True
    return dependencies


def _canonical_person_for_binding(scene: dict, canonical_person_id: str) -> dict:
    people = scene.get("payload", {}).get("canonicalPeople") or []
    person = next(
        (
            item
            for item in people
            if str(item.get("canonicalPersonId") or item.get("id") or "")
            == canonical_person_id
        ),
        None,
    )
    if person is None:
        raise ReconstructionError("The canonical person no longer exists")
    return person


def _roster_binding_correction_owner_ids(scene: dict, correction: dict) -> set[str]:
    """Find canonical people that currently own a roster correction anchor.

    A post-resolver split intentionally changes that ownership without
    rewriting reconstruction input.  The next roster edit can therefore rekey
    the old correction transactionally instead of leaving a second positive
    binding behind.
    """

    correction_id = str(correction.get("id") or "").strip()
    target_observation_id = str(correction.get("targetObservationId") or "").strip()
    people = scene.get("payload", {}).get("canonicalPeople") or []
    strong_owners: set[str] = set()
    for person in people:
        canonical_id = str(
            person.get("canonicalPersonId") or person.get("id") or ""
        ).strip()
        if not canonical_id:
            continue
        observations = [
            item for item in person.get("observations") or [] if isinstance(item, dict)
        ]
        observation_ids = {
            str(item.get("observationId") or item.get("id") or "").strip()
            for item in observations
        }
        observation_annotation_ids = {
            str(item.get("annotationId") or "").strip() for item in observations
        }
        if (
            correction_id in set(person.get("annotationIds") or [])
            or correction_id in observation_annotation_ids
            or target_observation_id
            and target_observation_id in observation_ids
        ):
            strong_owners.add(canonical_id)
    if strong_owners:
        return strong_owners

    snapshot = correction.get("targetObservation")
    if not isinstance(snapshot, dict) or snapshot.get("frameIndex") is None:
        return set()
    snapshot_bbox = snapshot.get("bbox")
    if not isinstance(snapshot_bbox, dict):
        return set()
    try:
        frame_index = int(snapshot["frameIndex"])
        snapshot_time = float(snapshot.get("sceneTime"))
        target_box = _bbox_payload_box(snapshot_bbox)
    except (KeyError, TypeError, ValueError):
        return set()
    geometric_owners: set[str] = set()
    for person in people:
        canonical_id = str(
            person.get("canonicalPersonId") or person.get("id") or ""
        ).strip()
        if not canonical_id:
            continue
        for observation in person.get("observations") or []:
            if not isinstance(observation, dict) or not observation.get("bbox"):
                continue
            try:
                if int(observation["frameIndex"]) != frame_index:
                    continue
                if abs(float(observation.get("sceneTime")) - snapshot_time) > 0.08:
                    continue
                overlap = _iou(_bbox_payload_box(observation["bbox"]), target_box)
            except (KeyError, TypeError, ValueError):
                continue
            if overlap >= 0.75:
                geometric_owners.add(canonical_id)
                break
    return geometric_owners


def _replace_roster_annotation_references(
    scene: dict,
    canonical_person_id: str,
    old_annotation_id: str,
    new_annotation_id: str,
) -> None:
    """Move optimistic published references when a split correction is rekeyed."""

    payload = scene.get("payload", {})
    for person in payload.get("canonicalPeople") or []:
        person_id = str(person.get("canonicalPersonId") or person.get("id") or "")
        annotation_ids = {
            str(value)
            for value in person.get("annotationIds") or []
            if str(value) != old_annotation_id
        }
        if person_id == canonical_person_id:
            annotation_ids.add(new_annotation_id)
            for observation in person.get("observations") or []:
                if str(observation.get("annotationId") or "") == old_annotation_id:
                    observation["annotationId"] = new_annotation_id
                observation_ids = {
                    str(value)
                    for value in observation.get("annotationIds") or []
                    if str(value) != old_annotation_id
                }
                if old_annotation_id in {
                    str(value)
                    for value in observation.get("annotationIds") or []
                }:
                    observation_ids.add(new_annotation_id)
                if observation_ids:
                    observation["annotationIds"] = sorted(observation_ids)
                else:
                    observation.pop("annotationIds", None)
        person["annotationIds"] = sorted(annotation_ids)
    for track in payload.get("tracks") or []:
        is_owner = str(track.get("canonicalPersonId") or "") == canonical_person_id
        annotation_ids = {
            str(value)
            for value in track.get("annotationIds") or []
            if str(value) != old_annotation_id
        }
        if is_owner:
            annotation_ids.add(new_annotation_id)
            for observation in track.get("observations") or []:
                if str(observation.get("annotationId") or "") == old_annotation_id:
                    observation["annotationId"] = new_annotation_id
                observation_ids = {
                    str(value)
                    for value in observation.get("annotationIds") or []
                    if str(value) != old_annotation_id
                }
                if old_annotation_id in {
                    str(value)
                    for value in observation.get("annotationIds") or []
                }:
                    observation_ids.add(new_annotation_id)
                if observation_ids:
                    observation["annotationIds"] = sorted(observation_ids)
                else:
                    observation.pop("annotationIds", None)
        if annotation_ids:
            track["annotationIds"] = sorted(annotation_ids)
        else:
            track.pop("annotationIds", None)


def _saved_detector_observation_for_binding(
    person: dict,
    existing_annotation: dict | None,
    scene_duration: float,
    *,
    preserve_existing: bool = False,
) -> dict:
    """Choose a durable image-space anchor without consulting the live frame.

    Production observations carry an immutable observation id, source
    tracklet, frame/time and detector bbox.  Legacy observations may lack the
    source-tracklet field, so the immutable id remains the minimum provenance
    requirement.  An existing roster correction keeps the original detector
    snapshot available after its anchor becomes annotation-backed on rebuild.
    """

    raw_candidates = (
        [] if preserve_existing else list(person.get("observations") or [])
    )
    if isinstance((existing_annotation or {}).get("targetObservation"), dict):
        raw_candidates.append(existing_annotation["targetObservation"])

    candidates: list[dict] = []
    for observation in raw_candidates:
        if not isinstance(observation, dict):
            continue
        observation_id = str(
            observation.get("observationId") or observation.get("id") or ""
        ).strip()
        bbox = observation.get("bbox")
        try:
            frame_index = int(observation["frameIndex"])
            scene_time = float(observation["sceneTime"])
            values = [
                float(bbox["x"]),
                float(bbox["y"]),
                float(bbox["width"]),
                float(bbox["height"]),
                scene_time,
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not observation_id
            or frame_index < 0
            or not np.isfinite(values).all()
            or scene_time < 0.0
            or scene_time > scene_duration + 1e-6
            or values[0] < 0.0
            or values[1] < 0.0
            or values[2] < 4.0
            or values[3] < 4.0
        ):
            continue
        candidates.append(
            {
                **deepcopy(observation),
                "id": observation_id,
                "observationId": observation_id,
                "frameIndex": frame_index,
                "sceneTime": scene_time,
                "bbox": {
                    "x": values[0],
                    "y": values[1],
                    "width": values[2],
                    "height": values[3],
                },
            }
        )
    if not candidates:
        raise ReconstructionError(
            "This canonical person has no saved detector observation to anchor the roster binding"
        )

    return max(
        candidates,
        key=lambda item: (
            1 if item.get("sourceTrackletId") else 0,
            1 if not item.get("annotationId") else 0,
            1 if item.get("metricStatus") == "accepted" else 0,
            float(item.get("confidence") or 0.0),
            float(item["bbox"]["width"]) * float(item["bbox"]["height"]),
            -int(item["frameIndex"]),
            str(item["observationId"]),
        ),
    )


def _match_binding_player(scene: dict, external_player_id: str) -> dict:
    binding = scene.get("payload", {}).get("matchBinding") or {}
    players = [
        item
        for item in binding.get("players") or []
        if isinstance(item, dict)
        and str(item.get("id") or "").strip() == external_player_id
    ]
    if len(players) != 1:
        if players:
            raise ReconstructionError(
                "The bound match roster contains a duplicate external player id"
            )
        raise ReconstructionError("The selected player is not present in the bound match roster")
    return players[0]


def _validate_canonical_roster_team(scene: dict, person: dict, player: dict) -> None:
    local_team_id = str(person.get("teamId") or "").strip()
    role = str(person.get("role") or "").strip()
    if local_team_id not in {"home", "away"} or role in {"referee", "other"}:
        raise ReconstructionError(
            "Only a canonical home or away player can be bound to the match roster"
        )
    bound_team = (
        (scene.get("payload", {}).get("matchBinding") or {})
        .get("teams", {})
        .get(local_team_id)
    )
    expected_team_id = (
        str(bound_team.get("id") or "").strip()
        if isinstance(bound_team, dict)
        else str(bound_team or "").strip()
    )
    roster_team_id = str(player.get("team_id") or player.get("teamId") or "").strip()
    if expected_team_id and roster_team_id and roster_team_id != expected_team_id:
        raise ReconstructionError(
            f"The selected roster player belongs to the other team ({local_team_id} expected)"
        )


def _partition_local_identity_baseline(
    person: dict,
    annotations: list[dict],
) -> tuple[str, str, float | None, str]:
    """Rebuild the non-roster baseline after correction ownership changes."""

    local_annotation_ids = {str(value) for value in person.get("annotationIds") or []}
    positive = [
        annotation
        for annotation in annotations
        if str(annotation.get("id") or "") in local_annotation_ids
        and annotation.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION
        and _annotation_action(annotation) in {"confirm", "merge", "split"}
        and annotation.get("kind") != "ignore"
    ]
    positive.sort(
        key=lambda item: (
            str(item.get("updatedAt") or ""),
            float(item.get("sceneTime") or 0.0),
            str(item.get("id") or ""),
        )
    )
    display_name = next(
        (
            str(item.get("label") or "").strip()
            for item in reversed(positive)
            if str(item.get("label") or "").strip()
        ),
        "",
    )
    if not display_name:
        team = str(person.get("teamId") or "").strip()
        role = str(person.get("role") or "player").strip()
        display_name = (
            "Referee"
            if role == "referee"
            else "Other person"
            if role == "other"
            else f"{team.title()} goalkeeper"
            if team in {"home", "away"} and role == "goalkeeper"
            else f"{team.title()} person"
            if team in {"home", "away"}
            else "Unassigned person"
        )
    if positive:
        return display_name, "resolved", 1.0, "manual"
    if person.get("identitySource") != "manual":
        return (
            display_name,
            str(person.get("identityStatus") or "provisional"),
            person.get("identityConfidence"),
            str(person.get("identitySource") or "tracker+trajectory"),
        )
    return display_name, "provisional", None, "tracker+trajectory"


def set_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
    external_player_id: str | None,
    *,
    persist: bool = True,
) -> dict:
    """Store a canonical roster decision as a stable identity correction.

    The operation intentionally does not read sampled frames and does not run a
    detector.  It snapshots an already-published detector observation and
    updates the canonical output optimistically; the caller can then persist
    both correction and queued reconstruction with ``queue_reconstruction``'s
    compare-and-swap write.
    """

    canonical_person_id = str(canonical_person_id or "").strip()
    if not canonical_person_id:
        raise ReconstructionError("The canonical person no longer exists")
    normalized_external_id = (
        str(external_player_id).strip() if external_player_id is not None else None
    )
    if normalized_external_id == "":
        raise ReconstructionError("The external player id cannot be empty")

    person = _canonical_person_for_binding(scene, canonical_person_id)
    role = str(person.get("role") or "").strip()
    team_id = str(person.get("teamId") or "").strip()
    if normalized_external_id is not None and (
        team_id not in {"home", "away"} or role in {"referee", "other"}
    ):
        raise ReconstructionError(
            "Only a canonical home or away player can be bound to the match roster"
        )

    video = scene.get("payload", {}).get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    annotations = deepcopy(list(reconstruction.get("frameAnnotations") or []))
    annotation_id = _canonical_roster_binding_annotation_id(canonical_person_id)
    dedicated_owner_ids: set[str] = set()
    for item in annotations:
        if item.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION:
            continue
        owners = _roster_binding_correction_owner_ids(scene, item)
        if owners:
            dedicated_owner_ids.update(owners)
        else:
            persisted_owner = str(item.get("canonicalPersonId") or "").strip()
            if persisted_owner:
                dedicated_owner_ids.add(persisted_owner)
    # This operation is about to create (or replace) the durable decision for
    # the selected owner.  Treat that owner as dedicated immediately so a
    # legacy generic confirmation cannot survive the first Bind and later
    # resurrect an obsolete roster value.
    dedicated_owner_ids.add(canonical_person_id)
    for item in annotations:
        if item.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION:
            continue
        owners = _roster_binding_correction_owner_ids(scene, item)
        if not owners:
            persisted_owner = str(item.get("canonicalPersonId") or "").strip()
            owners = {persisted_owner} if persisted_owner else set()
        if (
            item.get("externalPlayerId") is not None
            and (
                _annotation_action(item) == "merge"
                or bool(owners & dedicated_owner_ids)
            )
        ):
            item["externalPlayerId"] = None
            item["rosterValueSupersededByDedicatedCorrection"] = True
    player: dict | None = None
    if normalized_external_id is not None:
        player = _match_binding_player(scene, normalized_external_id)
        _validate_canonical_roster_team(scene, person, player)
        conflicting_person = next(
            (
                item
                for item in scene.get("payload", {}).get("canonicalPeople") or []
                if str(item.get("canonicalPersonId") or item.get("id") or "")
                != canonical_person_id
                and str(item.get("externalPlayerId") or "") == normalized_external_id
            ),
            None,
        )
        if conflicting_person is not None:
            raise ReconstructionError(
                "The selected roster player is already bound to another canonical person"
            )
        conflicting_correction = next(
            (
                item
                for item in (
                    annotations
                )
                if _annotation_action(item) == "confirm"
                and _annotation_scope(item) == "identity"
                and str(item.get("externalPlayerId") or "") == normalized_external_id
                and str(item.get("canonicalPersonId") or "") != canonical_person_id
                and _roster_binding_correction_owner_ids(scene, item)
                != {canonical_person_id}
            ),
            None,
        )
        if conflicting_correction is not None:
            raise ReconstructionError(
                "The selected roster player is already bound to another canonical person"
            )

    owned_correction_indices: list[int] = []
    for index, item in enumerate(annotations):
        if item.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION:
            continue
        owners = _roster_binding_correction_owner_ids(scene, item)
        persisted_owner = str(item.get("canonicalPersonId") or "").strip()
        if canonical_person_id in owners and owners != {canonical_person_id}:
            raise ReconstructionError(
                "The roster correction anchor is owned by multiple canonical people; rebuild before editing"
            )
        if owners == {canonical_person_id} or (
            not owners and persisted_owner == canonical_person_id
        ):
            owned_correction_indices.append(index)
    if len(owned_correction_indices) > 1:
        raise ReconstructionError(
            "This canonical person has multiple durable roster corrections; rebuild before editing"
        )
    desired_id_index = next(
        (
            index
            for index, item in enumerate(annotations)
            if str(item.get("id") or "") == annotation_id
        ),
        None,
    )
    existing_index = (
        owned_correction_indices[0] if owned_correction_indices else desired_id_index
    )
    if existing_index is not None and existing_index not in owned_correction_indices:
        existing_owners = _roster_binding_correction_owner_ids(
            scene, annotations[existing_index]
        )
        persisted_owner = str(
            annotations[existing_index].get("canonicalPersonId") or ""
        )
        if (
            existing_owners
            and existing_owners != {canonical_person_id}
            or persisted_owner != canonical_person_id
        ):
            raise ReconstructionError(
                "The roster correction id is owned by another canonical person; edit that identity first"
            )
    if (
        owned_correction_indices
        and desired_id_index is not None
        and desired_id_index != owned_correction_indices[0]
    ):
        raise ReconstructionError(
            "The roster correction cannot be rekeyed because its target id already exists"
        )
    existing = annotations[existing_index] if existing_index is not None else None
    previous_annotation_id = str((existing or {}).get("id") or "") or None
    same_decision = (
        existing is not None
        and existing.get("externalPlayerId") == normalized_external_id
    )
    owner_changed = bool(
        existing is not None
        and str(existing.get("canonicalPersonId") or "") != canonical_person_id
    )
    existing_dependencies = (existing or {}).get(
        ROSTER_IDENTITY_DEPENDENCIES_FIELD
    ) or []
    if not isinstance(existing_dependencies, list):
        raise ReconstructionError(
            "The roster correction has invalid identity provenance"
        )
    identity_dependencies = {
        str(value).strip()
        for value in existing_dependencies
        if str(value or "").strip()
    }
    if existing is None or not same_decision or owner_changed:
        identity_dependencies.update(
            _active_merge_dependencies(scene, person, annotations)
        )
    observation = _saved_detector_observation_for_binding(
        person,
        existing,
        float(scene.get("duration") or 0.0),
        preserve_existing=same_decision,
    )
    observation["canonicalPersonId"] = canonical_person_id
    observation["annotationId"] = annotation_id
    bbox = observation["bbox"]
    source_start = float(video.get("sourceStart") or 0.0)
    scene_time = float(observation["sceneTime"])
    source_time = observation.get("sourceTime")
    try:
        source_time = float(source_time)
    except (TypeError, ValueError):
        source_time = source_start + scene_time
    if not isfinite(source_time):
        source_time = source_start + scene_time

    base_display_name = str(
        (existing or {}).get("baseDisplayName")
        or person.get("displayName")
        or person.get("label")
        or canonical_person_id
    )
    existing_is_bound = bool(
        existing
        and existing.get("correctionKind") == CANONICAL_ROSTER_BINDING_CORRECTION
        and existing.get("rosterBindingState") == "bound"
    )
    local_annotation_ids = {str(value) for value in person.get("annotationIds") or []}
    has_non_roster_manual_semantics = any(
        str(item.get("id") or "") in local_annotation_ids
        and item.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION
        and _annotation_action(item) in {"confirm", "merge", "split"}
        and item.get("kind") != "ignore"
        for item in annotations
    )
    if owner_changed or (
        normalized_external_id is None and has_non_roster_manual_semantics
    ):
        (
            base_display_name,
            base_identity_status,
            base_identity_confidence,
            base_identity_source,
        ) = _partition_local_identity_baseline(person, annotations)
    elif existing is not None and "baseIdentityStatus" in existing:
        base_identity_status = existing.get("baseIdentityStatus")
        base_identity_confidence = existing.get("baseIdentityConfidence")
        base_identity_source = existing.get("baseIdentitySource")
    elif existing_is_bound:
        # Migration fallback for an early binding record that predates the
        # baseline snapshot.  Never preserve the optimistic manual 1.0 state
        # as the result of an unbind.
        base_identity_status = "provisional"
        base_identity_confidence = None
        base_identity_source = "tracker+trajectory"
    else:
        base_identity_status = person.get("identityStatus") or "provisional"
        base_identity_confidence = person.get("identityConfidence")
        base_identity_source = person.get("identitySource") or "tracker+trajectory"
    display_name = (
        str((player or {}).get("name") or normalized_external_id)
        if normalized_external_id is not None
        else base_display_name
    )
    annotation = {
        "id": annotation_id,
        "sceneTime": round(scene_time, 3),
        "sourceTime": round(source_time, 3),
        "frameIndex": int(observation["frameIndex"]),
        "bbox": {
            "x": round(float(bbox["x"]), 2),
            "y": round(float(bbox["y"]), 2),
            "width": round(float(bbox["width"]), 2),
            "height": round(float(bbox["height"]), 2),
        },
        "kind": _track_annotation_kind(person),
        "label": display_name,
        "externalPlayerId": normalized_external_id,
        "action": "confirm",
        "scope": "identity",
        "mergeTargetId": None,
        "sourceTrackId": person.get("renderTrackId"),
        "canonicalPersonId": canonical_person_id,
        "targetObservationId": observation["observationId"],
        "targetObservation": observation,
        "rangeStart": None,
        "rangeEnd": None,
        "splitCanonicalPersonId": None,
        "affectedPreview": None,
        "previewState": (
            "confirmed" if normalized_external_id is not None else "unbound"
        ),
        "correctionKind": CANONICAL_ROSTER_BINDING_CORRECTION,
        "rosterBindingState": (
            "bound" if normalized_external_id is not None else "unbound"
        ),
        ROSTER_DECISION_ORIGIN_FIELD: (
            _roster_decision_origin_id(existing or {}) or annotation_id
        ),
        ROSTER_IDENTITY_DEPENDENCIES_FIELD: sorted(identity_dependencies),
        "baseDisplayName": base_display_name,
        "baseIdentityStatus": base_identity_status,
        "baseIdentityConfidence": base_identity_confidence,
        "baseIdentitySource": base_identity_source,
        "updatedAt": (
            existing.get("updatedAt")
            if same_decision and existing is not None and existing.get("updatedAt")
            else datetime.now(UTC).isoformat()
        ),
    }
    if existing_index is None:
        annotations.append(annotation)
    else:
        annotations[existing_index] = annotation
    _validate_identity_corrections(scene, annotations)
    reconstruction["frameAnnotations"] = sorted(
        annotations,
        key=lambda item: (int(item.get("frameIndex") or 0), str(item.get("id") or "")),
    )
    video["reconstruction"] = reconstruction
    if previous_annotation_id and previous_annotation_id != annotation_id:
        _replace_roster_annotation_references(
            scene,
            canonical_person_id,
            previous_annotation_id,
            annotation_id,
        )

    person["externalPlayerId"] = normalized_external_id
    person["displayName"] = display_name
    if normalized_external_id is not None:
        person["identityStatus"] = "resolved"
        person["identityConfidence"] = 1.0
        person["identitySource"] = "manual"
    else:
        person["identityStatus"] = base_identity_status
        person["identityConfidence"] = base_identity_confidence
        person["identitySource"] = base_identity_source
    person["annotationIds"] = sorted(
        {*list(person.get("annotationIds") or []), annotation_id}
    )
    for track in scene.get("payload", {}).get("tracks") or []:
        if (
            str(track.get("canonicalPersonId") or "") == canonical_person_id
            or person.get("renderTrackId")
            and str(track.get("id") or "") == str(person.get("renderTrackId"))
        ):
            track["externalPlayerId"] = normalized_external_id
            track["label"] = display_name
            track["annotationIds"] = sorted(
                {*list(track.get("annotationIds") or []), annotation_id}
            )
    if persist:
        scene_store.put(scene)
    return annotation


def clear_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
    *,
    persist: bool = True,
) -> dict:
    """Clear one explicit Unbind decision without resurrecting split metadata.

    Clearing is deliberately distinct from ``Unbind``: PUT with a null player
    creates a durable negative decision, while this operation removes that
    decision. A positive binding must first be converted to Unbind so the
    destructive intent is explicit and reviewable.
    """

    canonical_person_id = str(canonical_person_id or "").strip()
    person = _canonical_person_for_binding(scene, canonical_person_id)
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction")
        or {}
    )
    annotations = list(reconstruction.get("frameAnnotations") or [])
    owned: list[dict] = []
    for item in annotations:
        if item.get("correctionKind") != CANONICAL_ROSTER_BINDING_CORRECTION:
            continue
        owners = _roster_binding_correction_owner_ids(scene, item)
        persisted_owner = str(item.get("canonicalPersonId") or "").strip()
        if owners == {canonical_person_id} or (
            not owners and persisted_owner == canonical_person_id
        ):
            owned.append(item)
    if not owned:
        raise ReconstructionError(
            "This canonical person has no roster decision to clear"
        )
    if len(owned) > 1:
        raise ReconstructionError(
            "This canonical person has multiple durable roster corrections; rebuild before clearing"
        )
    correction = owned[0]
    if correction.get("rosterBindingState") != "unbound":
        raise ReconstructionError(
            "Unbind the roster player before clearing its roster decision"
        )
    # Keep the selected person lookup above: it is the ownership authorization
    # for clearing the opaque correction id through this dedicated lifecycle.
    assert person is not None
    return _clear_unbound_roster_correction(
        scene,
        correction,
        active_merge_ids=_active_merge_dependencies(
            scene,
            person,
            annotations,
        ),
        active_split_ids=_active_split_dependencies(
            scene,
            person,
            annotations,
        ),
        persist=persist,
    )


def _frame_track_observations(scene: dict, frame_index: int) -> list[tuple[dict, dict]]:
    return [
        (subject, observation)
        for subject in _canonical_analysis_subjects(scene)
        for observation in subject.get("observations") or []
        if observation.get("frameIndex") is not None
        and int(observation["frameIndex"]) == frame_index
        and observation.get("bbox")
    ]


def _pair_detections_to_stored_observations(
    detection_boxes: list[dict],
    observations: list[tuple[dict, dict]],
) -> tuple[dict[int, int], set[int]]:
    """Pair fresh detector rows only to recover metadata, never identity."""

    pairs = sorted(
        (
            (
                _iou(_bbox_payload_box(box), _bbox_payload_box(observation["bbox"])),
                observation_index,
                detection_index,
            )
            for observation_index, (_, observation) in enumerate(observations)
            for detection_index, box in enumerate(detection_boxes)
        ),
        reverse=True,
    )
    by_observation: dict[int, int] = {}
    used_detections: set[int] = set()
    for overlap, observation_index, detection_index in pairs:
        if overlap < 0.20:
            break
        if observation_index in by_observation or detection_index in used_detections:
            continue
        by_observation[observation_index] = detection_index
        used_detections.add(detection_index)
    return by_observation, used_detections


def _accepted_frame_image_to_pitch(scene: dict, frame_index: int) -> np.ndarray | None:
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    calibration = reconstruction.get("calibration") or {}
    candidates = [
        item
        for item in [
            *(calibration.get("frameEvidence") or []),
            *(reconstruction.get("calibrationFrames") or []),
        ]
        if item.get("sourceFrameIndex") is not None
        and int(item["sourceFrameIndex"]) == frame_index
        and item.get("status") == "accepted"
        and item.get("imageToPitch") is not None
    ]
    if not candidates:
        return None
    evidence = max(
        candidates,
        key=lambda item: (
            1 if item.get("solutionStatus") == "direct-accepted" else 0,
            float(item.get("confidence") or 0.0),
        ),
    )
    matrix = np.asarray(evidence["imageToPitch"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    return matrix


def _legacy_observed_frame_matches(
    scene: dict,
    frame_index: int,
    frame_time: float,
    raw_people: list[dict],
) -> dict[int, tuple[dict, float]]:
    """Conservative bridge for pre-observation-schema scenes.

    Only detector evidence and an observed 3D keyframe from this exact sampled
    frame participate.  Ambiguous assignments deliberately remain unmatched.
    """

    image_to_pitch = _accepted_frame_image_to_pitch(scene, frame_index)
    if image_to_pitch is None or not raw_people:
        return {}
    projected: list[tuple[float, float] | None] = []
    for raw in raw_people:
        value = image_to_pitch @ np.array(
            [float(raw["x"]), float(raw["y"]), 1.0], dtype=np.float64
        )
        if abs(float(value[2])) < 1e-8:
            projected.append(None)
            continue
        position = (float(value[0] / value[2]), float(value[1] / value[2]))
        projected.append(position if np.isfinite(position).all() else None)

    observed_tracks: list[tuple[dict, dict]] = []
    for track in scene.get("payload", {}).get("tracks") or []:
        candidates = [
            keyframe
            for keyframe in track.get("keyframes") or []
            if keyframe.get("observed") is not False
            and abs(float(keyframe.get("t") or 0.0) - frame_time) <= 0.021
        ]
        if candidates:
            observed_tracks.append(
                (track, min(candidates, key=lambda item: abs(float(item["t"]) - frame_time)))
            )
    if not observed_tracks:
        return {}

    costs = np.full((len(raw_people), len(observed_tracks)), np.inf, dtype=np.float64)
    for detection_index, position in enumerate(projected):
        if position is None:
            continue
        for track_index, (_, keyframe) in enumerate(observed_tracks):
            costs[detection_index, track_index] = hypot(
                position[0] - float(keyframe["x"]),
                position[1] - float(keyframe["z"]),
            )
    finite = np.isfinite(costs)
    if not finite.any():
        return {}
    rows, columns = linear_sum_assignment(np.where(finite, costs, 1e6))
    matches: dict[int, tuple[dict, float]] = {}
    for detection_index, track_index in zip(rows.tolist(), columns.tolist()):
        distance = float(costs[detection_index, track_index])
        if not np.isfinite(distance) or distance > 2.5:
            continue
        detection_alternatives = sorted(
            float(value)
            for index, value in enumerate(costs[detection_index])
            if index != track_index and np.isfinite(value)
        )
        track_alternatives = sorted(
            float(value)
            for index, value in enumerate(costs[:, track_index])
            if index != detection_index and np.isfinite(value)
        )
        if detection_alternatives and detection_alternatives[0] - distance < 1.0:
            continue
        if track_alternatives and track_alternatives[0] - distance < 1.0:
            continue
        matches[detection_index] = (observed_tracks[track_index][0], distance)
    return matches


def analyze_scene_frame(scene: dict, scene_time: float) -> dict:
    frames = _frame_paths(scene)
    if not frames:
        raise ReconstructionError("No sampled frames are available for this moment")
    target_index = min(range(len(frames)), key=lambda index: abs(frames[index][1] - scene_time))
    target_path, frame_time = frames[target_index]
    model_name = str(
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("model")
        or get_settings().reconstruction_model
    )
    model = _load_model(model_name)
    result = _predict_frame(model, target_path)
    frame_width, frame_height = result.orig_img.shape[1], result.orig_img.shape[0]
    people, legacy_balls = _person_detections(result)
    reconstruction_metadata = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    ball_backend = str(
        reconstruction_metadata.get("ballBackend")
        or get_settings().ball_detection_backend
    )
    ball_detector, ball_fallback = _configured_ball_detectors(
        model,
        ball_backend,
        reconstruction_metadata.get("ballDetectionInput"),
    )
    ball_detection_warning: str | None = None
    ball_target_path = Path(target_path)
    ball_frame_time = float(frame_time)
    ball_frame_index = target_index
    sampled_paths = [Path(path) for path, _ in frames]
    ball_context_paths = (
        sampled_paths[max(0, target_index - 1)],
        sampled_paths[min(len(sampled_paths) - 1, target_index + 1)],
    )

    def scaled_ball_detections(batch) -> list[dict]:
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

    try:
        ball_batch = ball_detector.detect(
            ball_target_path,
            frame_index=ball_frame_index,
            timestamp=ball_frame_time,
            context_frames=ball_context_paths,
        )
        balls = scaled_ball_detections(ball_batch)
        for rank, item in enumerate(balls, start=1):
            item["candidateId"] = f"frame-ball-{ball_frame_index:05d}-{rank:02d}"
            item["provenance"] = {
                "backend": item.get("detectorBackend") or ball_batch.backend,
                "detectorMetadata": deepcopy(item.get("detectorMetadata") or {}),
                "batchMetadata": deepcopy(dict(ball_batch.metadata)),
            }
    except Exception as exc:
        detector_detail = f"{type(exc).__name__}: {exc}"
        ball_detection_warning = (
            f"{ball_detection_warning}; {detector_detail}"
            if ball_detection_warning
            else detector_detail
        )
        balls = legacy_balls
        if ball_fallback is not None:
            try:
                fallback_batch = ball_fallback.detect(
                    ball_target_path,
                    frame_index=ball_frame_index,
                    timestamp=ball_frame_time,
                    context_frames=ball_context_paths,
                )
                balls = scaled_ball_detections(fallback_batch)
            except Exception as fallback_exc:
                ball_detection_warning += (
                    f"; fallback {type(fallback_exc).__name__}: {fallback_exc}"
                )
    frame_index = int(target_path.stem.split("_")[-1])
    frame_annotations = _frame_annotations(scene, frame_index)
    response_annotations = [
        _identity_annotation_response(annotation) for annotation in frame_annotations
    ]
    annotation_by_id = {
        str(annotation["id"]): annotation for annotation in response_annotations
    }
    all_identity_annotations = {
        str(annotation["id"]): annotation
        for annotation in _identity_annotations(scene)
        if annotation.get("id")
    }
    people = _apply_person_annotations(result.orig_img, people, frame_annotations)
    raw_people = [
        {"x": item.x, "y": item.y, "width": item.width, "height": item.height}
        for item in people
    ]
    raw_balls = [{**item} for item in balls]

    previous_image: np.ndarray | None = None
    camera_transform = np.eye(3, dtype=np.float64)
    for index, (path, _) in enumerate(frames[: target_index + 1]):
        image = result.orig_img if index == target_index else cv2.imread(str(path))
        if image is None:
            continue
        if previous_image is not None:
            motion = _camera_motion_estimate(previous_image, image)
            camera_transform = (
                camera_transform @ motion.matrix
                if motion.reliable
                else np.eye(3, dtype=np.float64)
            )
        previous_image = image
    _stabilize_detections(people, balls, camera_transform)

    calibration = _saved_pitch_calibration(scene)
    pitch = scene["payload"]["pitch"]
    projected_people = [
        _project(item.x, item.y, frame_width, frame_height, pitch, calibration) for item in people
    ]
    tracks = scene.get("payload", {}).get("tracks") or []
    identity_subjects = _canonical_analysis_subjects(scene)
    observation_schema_version = _track_observation_schema_version(scene)
    has_observation_schema = observation_schema_version is not None
    detection_boxes = [_raw_person_bbox(raw) for raw in raw_people]
    stored_observations = (
        _frame_track_observations(scene, frame_index)
        if has_observation_schema
        else []
    )
    observation_pairs, consumed_detection_indices = (
        _pair_detections_to_stored_observations(detection_boxes, stored_observations)
        if stored_observations
        else ({}, set())
    )
    legacy_matches = (
        {}
        if has_observation_schema
        else _legacy_observed_frame_matches(scene, frame_index, frame_time, raw_people)
    )

    def manually_forced_track(correction: dict | None) -> dict | None:
        if correction is None:
            return None
        action = _annotation_action(correction)
        requested_id: str | None = None
        if action == "merge" and correction.get("mergeTargetId"):
            requested_id = _terminal_identity_target(
                str(correction["mergeTargetId"]), all_identity_annotations
            )
        elif (
            action == "confirm"
            and _annotation_scope(correction) == "identity"
            and _annotation_source_identity(correction)
        ):
            requested_id = str(_annotation_source_identity(correction))
        if not requested_id:
            return None
        return next(
            (
                track
                for track in identity_subjects
                if str(track.get("id") or "") == requested_id
                or str(track.get("canonicalPersonId") or "") == requested_id
                or requested_id in (track.get("annotationIds") or [])
            ),
            None,
        )

    detections = []
    for observation_index, (track, observation) in enumerate(stored_observations):
        detection_index = observation_pairs.get(observation_index)
        item = people[detection_index] if detection_index is not None else None
        annotation_id = (
            item.annotation_id
            if item is not None and item.annotation_id
            else observation.get("annotationId")
        )
        correction = annotation_by_id.get(str(annotation_id or ""))
        if correction is None:
            observation_id = _observation_identifier(observation)
            split_matches = [
                candidate
                for candidate in response_annotations
                if candidate.get("action") == "split"
                and candidate.get("targetObservationId") == observation_id
            ]
            if len(split_matches) == 1:
                correction = split_matches[0]
                annotation_id = correction["id"]
        correction_action = _annotation_action(correction) if correction else None
        merge_target_id = correction.get("mergeTargetId") if correction_action == "merge" else None
        forced_match = manually_forced_track(correction)
        matched = forced_match or track
        observation_pitch = observation.get("pitch")
        metric_status = str(
            observation.get("metricStatus")
            or ("accepted" if observation_pitch else "unprojected")
        )
        metric_reason = observation.get("metricReason")
        if metric_status == "accepted" and not observation_pitch:
            metric_status = "unprojected"
            metric_reason = "metric-projection-unavailable"
        accepted_observation_pitch = (
            observation_pitch if metric_status == "accepted" else None
        )
        position_source = "observation" if accepted_observation_pitch else "track-inferred"
        position = accepted_observation_pitch
        if position is None:
            position = _interpolate_scene_keyframes(track.get("keyframes") or [], frame_time)
        position = position or {"x": 0.0, "z": 0.0}
        annotation_kind = (
            item.annotation_kind
            if item is not None and item.annotation_kind
            else correction.get("kind") if correction else None
        )
        annotation_label = (
            item.annotation_label
            if item is not None and item.annotation_label
            else correction.get("label") if correction else None
        )
        detections.append(
            {
                "id": str(
                    observation.get("observationId")
                    or observation.get("id")
                    or f"observation-{track.get('canonicalPersonId') or track.get('id')}-{frame_index}"
                ),
                "observationId": observation.get("observationId")
                or observation.get("id"),
                "confidence": round(float(observation.get("confidence") or 0.0), 3),
                "bbox": deepcopy(observation["bbox"]),
                "pitch": {
                    "x": round(float(position["x"]), 2),
                    "z": round(float(position["z"]), 2),
                },
                "jerseyColor": (
                    _cluster_color(item.feature)
                    if item is not None
                    else str(matched.get("color") or "#d7dce8")
                ),
                "annotationId": annotation_id,
                "kind": annotation_kind,
                "annotationLabel": annotation_label,
                "source": "manual" if annotation_id else "automatic",
                "matchedTrackId": matched.get("renderTrackId") or matched.get("id"),
                "matchedTrackLabel": annotation_label or matched.get("label"),
                "canonicalPersonId": matched.get("canonicalPersonId"),
                "identityStatus": matched.get("identityStatus"),
                "identityConfidence": matched.get("identityConfidence"),
                "identitySource": matched.get("identitySource"),
                "displayName": matched.get("displayName") or matched.get("label"),
                "jerseyNumber": matched.get("jerseyNumber"),
                "teamId": _annotation_team(annotation_kind) or matched.get("teamId"),
                "matchDistance": None,
                "matchSource": "manual-identity" if forced_match else "persisted-observation",
                "metricStatus": metric_status,
                "metricReason": metric_reason,
                "rawPitch": deepcopy(observation.get("rawPitch")),
                "projectionSource": observation.get("projectionSource"),
                "positionUncertaintyMetres": observation.get("positionUncertaintyMetres"),
                "positionSource": position_source,
                "correctionAction": correction_action,
                "correctionScope": _annotation_scope(correction) if correction else None,
                "mergeTargetId": merge_target_id,
                "sourceTrackId": correction.get("sourceTrackId") if correction else None,
                "targetObservationId": correction.get("targetObservationId") if correction else None,
                "rangeStart": correction.get("rangeStart") if correction else None,
                "rangeEnd": correction.get("rangeEnd") if correction else None,
                "splitCanonicalPersonId": correction.get("splitCanonicalPersonId") if correction else None,
                "affectedPreview": deepcopy(correction.get("affectedPreview")) if correction else None,
                "previewState": correction.get("previewState") if correction else "uncorrected",
            }
        )

    for index, (item, position) in enumerate(zip(people, projected_people)):
        if index in consumed_detection_indices:
            continue
        correction = annotation_by_id.get(str(item.annotation_id or ""))
        correction_action = _annotation_action(correction) if correction else None
        merge_target_id = correction.get("mergeTargetId") if correction_action == "merge" else None
        forced_match = manually_forced_track(correction)
        legacy_match, distance = legacy_matches.get(index, (None, None))
        matched = forced_match or legacy_match
        match_source = (
            "manual-identity"
            if forced_match is not None
            else "legacy-observed-frame"
            if legacy_match is not None
            else None
        )
        fresh_metric_accepted = legacy_match is not None or (
            calibration is not None
            and calibration.confidence >= METRIC_CALIBRATION_THRESHOLD
        )
        detections.append(
            {
                "id": f"person-{index + 1}",
                "observationId": None,
                "confidence": round(float(item.confidence), 3),
                "bbox": detection_boxes[index],
                "pitch": {"x": round(position[0], 2), "z": round(position[1], 2)},
                "jerseyColor": _cluster_color(item.feature),
                "annotationId": item.annotation_id,
                "kind": item.annotation_kind,
                "annotationLabel": item.annotation_label,
                "source": "manual" if item.annotation_id else "automatic",
                "matchedTrackId": matched.get("id") if matched else None,
                "matchedTrackLabel": item.annotation_label or (matched.get("label") if matched else None),
                "canonicalPersonId": matched.get("canonicalPersonId") if matched else None,
                "identityStatus": matched.get("identityStatus") if matched else None,
                "identityConfidence": matched.get("identityConfidence") if matched else None,
                "identitySource": matched.get("identitySource") if matched else None,
                "displayName": matched.get("displayName") if matched else None,
                "jerseyNumber": matched.get("jerseyNumber") if matched else None,
                "teamId": _annotation_team(item.annotation_kind) or (matched.get("teamId") if matched else None),
                "matchDistance": round(distance, 2) if distance is not None else None,
                "matchSource": match_source,
                "metricStatus": "accepted" if fresh_metric_accepted else "unprojected",
                "metricReason": None if fresh_metric_accepted else "metric-projection-unavailable",
                "rawPitch": None,
                "projectionSource": (
                    "legacy-observed-frame"
                    if legacy_match is not None
                    else calibration.method
                    if fresh_metric_accepted and calibration is not None
                    else None
                ),
                "positionUncertaintyMetres": None,
                "positionSource": "observation",
                "correctionAction": correction_action,
                "correctionScope": _annotation_scope(correction) if correction else None,
                "mergeTargetId": merge_target_id,
                "sourceTrackId": correction.get("sourceTrackId") if correction else None,
                "targetObservationId": correction.get("targetObservationId") if correction else None,
                "rangeStart": correction.get("rangeStart") if correction else None,
                "rangeEnd": correction.get("rangeEnd") if correction else None,
                "splitCanonicalPersonId": correction.get("splitCanonicalPersonId") if correction else None,
                "affectedPreview": deepcopy(correction.get("affectedPreview")) if correction else None,
                "previewState": correction.get("previewState") if correction else "uncorrected",
            }
        )

    projected_balls = [
        _project(
            float(item.get("stabilizedX", item["x"])),
            float(item.get("stabilizedY", item["y"])),
            frame_width,
            frame_height,
            pitch,
            calibration,
        )
        for item in balls
    ]
    ball_keyframes = scene.get("payload", {}).get("ball", {}).get("keyframes") or []
    ball_position = (
        _interpolate_scene_keyframes(ball_keyframes, ball_frame_time)
        if ball_keyframes
        and float(ball_keyframes[0]["t"])
        <= ball_frame_time
        <= float(ball_keyframes[-1]["t"])
        else None
    )
    primary_ball = None
    if projected_balls and ball_position is not None and float(ball_position.get("confidence") or 0.0) > 0.12:
        primary_ball = min(
            range(len(projected_balls)),
            key=lambda index: hypot(
                projected_balls[index][0] - float(ball_position["x"]),
                projected_balls[index][1] - float(ball_position["z"]),
            ),
        )
    elif projected_balls:
        strongest = max(range(len(balls)), key=lambda index: balls[index]["confidence"])
        if float(balls[strongest]["confidence"]) >= 0.25:
            primary_ball = strongest
    ball_candidates = [
        {
            "id": f"ball-{index + 1}",
            "confidence": round(float(item["confidence"]), 3),
            "image": {"x": round(raw["x"], 2), "y": round(raw["y"], 2)},
            "pitch": {"x": round(position[0], 2), "z": round(position[1], 2)},
            "primary": index == primary_ball,
            "backend": item.get("detectorBackend") or ball_backend,
        }
        for index, (item, raw, position) in enumerate(zip(balls, raw_balls, projected_balls))
    ]
    reconstruction = scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    calibration_metadata = reconstruction.get("pitchCalibration") or {}
    source_start = float(scene["payload"]["videoAsset"].get("sourceStart") or 0.0)
    return {
        "sceneId": scene["id"],
        "requestedTime": round(float(scene_time), 3),
        "sceneTime": round(float(frame_time), 3),
        "ballSceneTime": round(float(ball_frame_time), 3),
        "ballFrameIndex": int(ball_frame_index),
        "sourceTime": round(source_start + float(frame_time), 3),
        "frameIndex": frame_index,
        "frameWidth": frame_width,
        "frameHeight": frame_height,
        "model": model_name,
        "ballBackend": ball_backend,
        "projectionMode": reconstruction.get("coordinateSpace") or "screen-relative",
        "calibrationStatus": calibration_metadata.get("status") or "fallback",
        "identityLinking": {
            "mode": (
                "canonical-observations"
                if observation_schema_version is not None
                and observation_schema_version >= 3
                else "persisted-observations"
                if has_observation_schema
                else "legacy-observed-frame"
                if legacy_matches
                else "rebuild-required"
            ),
            "schemaVersion": observation_schema_version,
        },
        "people": detections,
        "annotations": response_annotations,
        "correctionSummary": {
            "confirmed": sum(item["action"] == "confirm" for item in response_annotations),
            "excluded": sum(item["action"] == "exclude" for item in response_annotations),
            "merged": sum(item["action"] == "merge" for item in response_annotations),
            "split": sum(item["action"] == "split" for item in response_annotations),
        },
        "ballCandidates": ball_candidates,
        "matchedTracks": sum(item["matchedTrackId"] is not None for item in detections),
        "matchedCanonicalPeople": sum(
            item.get("canonicalPersonId") is not None for item in detections
        ),
        "warnings": [
            *(
                [
                    "This scene predates authoritative video-track observations. Only unambiguous observed-track matches on this exact calibrated frame are linked; rebuild tracks for reliable video ↔ 3D selection."
                ]
                if not has_observation_schema
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
                [f"Specialized ball detector degraded to fallback: {ball_detection_warning}"]
                if ball_detection_warning
                else []
            ),
        ],
    }


class _ReconstructionLeaseHeartbeat:
    """Keep one claimed database lease alive without mutating scene revision."""

    def __init__(
        self,
        scene_id: str,
        run_id: str,
        input_fingerprint: str,
        owner_id: str,
    ) -> None:
        settings = get_settings()
        ttl = max(1.0, float(settings.reconstruction_lease_ttl_seconds))
        configured = max(
            0.05,
            float(settings.reconstruction_lease_heartbeat_seconds),
        )
        self.interval = min(configured, max(0.05, ttl / 3.0))
        self.scene_id = scene_id
        self.run_id = run_id
        self.input_fingerprint = input_fingerprint
        self.owner_id = owner_id
        self._stop = Event()
        self._thread = Thread(
            target=self._run,
            name=f"reconstruction-heartbeat-{scene_id}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                renewed = scene_store.heartbeat_reconstruction_run(
                    self.scene_id,
                    self.run_id,
                    self.input_fingerprint,
                    self.owner_id,
                )
            except Exception:
                # A transient database busy/connection error must not turn a
                # healthy long-running analysis into an abandoned lease. Retry
                # on the next interval; expiry and every publish still fence
                # the worker if renewal never succeeds.
                continue
            if not renewed:
                return

    def __enter__(self) -> "_ReconstructionLeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.1, self.interval * 2.0))


def _mark_owned_reconstruction_crashed(
    scene_id: str,
    run_id: str,
    input_fingerprint: str,
    owner_id: str,
    error: Exception,
) -> None:
    """Best-effort terminal cleanup for failures outside pipeline handling."""

    scene = scene_store.get(scene_id)
    if scene is None:
        return
    reconstruction = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    if (
        reconstruction.get("status") != "processing"
        or str(reconstruction.get("runId") or "") != run_id
        or reconstruction_input_fingerprint(scene) != input_fingerprint
    ):
        return
    now = datetime.now(UTC).isoformat()
    message = f"Reconstruction worker crashed: {error}"
    reconstruction.update(
        {
            "status": "failed",
            "processingStatus": "failed",
            "qualityVerdict": "reject",
            "error": message,
            "completedAt": now,
            "progress": {
                **(reconstruction.get("progress") or _queued_progress(0)),
                "phase": "failed",
                "label": "Analysis failed",
                "detail": message,
                "etaSeconds": 0.0,
                "updatedAt": now,
            },
        }
    )
    scene_store.put_if_reconstruction_run(
        scene,
        run_id,
        input_fingerprint,
        owner_id,
    )


def reconstruct_scene_by_id(
    scene_id: str,
    expected_run_id: str | None = None,
    expected_input_fingerprint: str | None = None,
) -> bool:
    scene = scene_store.get(scene_id)
    if scene is None:
        return False
    reconstruction = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    # Backward compatibility for direct/internal callers: when no token is
    # supplied, adopt the currently queued run. HTTP background tasks always
    # pass both values captured at queue time.
    run_id = expected_run_id or str(reconstruction.get("runId") or "") or None
    input_fingerprint = (
        expected_input_fingerprint
        or str(reconstruction.get("inputFingerprint") or "")
        or None
    )
    if run_id is not None and input_fingerprint is None:
        input_fingerprint = reconstruction_input_fingerprint(scene)
    if expected_run_id is not None and str(reconstruction.get("runId") or "") != expected_run_id:
        return False
    if (
        expected_input_fingerprint is not None
        and reconstruction_input_fingerprint(scene) != expected_input_fingerprint
    ):
        return False
    if reconstruction.get("status") not in {"queued", "processing"}:
        # Delayed duplicate BackgroundTasks must never re-run a terminal job,
        # even if its run id and input fingerprint are still identical.
        return False
    if not run_id or not input_fingerprint:
        return False
    owner_id = f"worker-{uuid4().hex}"
    if not scene_store.claim_reconstruction_run(
        scene_id,
        run_id,
        input_fingerprint,
        owner_id,
    ):
        return False
    claimed_scene = scene_store.get(scene_id)
    if claimed_scene is None:
        return False
    try:
        with _ReconstructionLeaseHeartbeat(
            scene_id,
            run_id,
            input_fingerprint,
            owner_id,
        ):
            reconstruct_scene(
                claimed_scene,
                expected_run_id=run_id,
                expected_input_fingerprint=input_fingerprint,
                expected_lease_owner_id=owner_id,
            )
    except ReconstructionError:
        return True
    except Exception as exc:
        _mark_owned_reconstruction_crashed(
            scene_id,
            run_id,
            input_fingerprint,
            owner_id,
            exc,
        )
        return True
    return True
