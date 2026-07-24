from __future__ import annotations

"""Temporal calibration hypothesis resolution and target-frame validation."""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .pitch_calibration_contract import PitchCalibration, pitch_side
from .pitch_calibration_quality import calibration_alignment_metrics_from_mask
from .pitch_image_evidence import pitch_line_mask
from .pitch_geometry import calibration_horizon
from .reconstruction_calibration_evidence import matrix_payload
from .reconstruction_calibration_policy import TEMPORAL_REVIEW_UNCERTAINTY_METRES
from .reconstruction_frame_calibration_quality import semantic_alignment_passes_review
from .reconstruction_person_detection_contract import Detection
from .reconstruction_inputs import source_frame_index as parse_source_frame_index
from .reconstruction_metric_projection import calibration_person_support
from .camera_motion_contract import CameraMotionEstimate
from .temporal_calibration_contract import TemporalCalibrationFrame
from .temporal_calibration_solver import solve_calibration_sequence


ObservedLineMaskLoader = Callable[[Path], "np.ndarray | None"]


def default_observed_line_mask(path: Path | str) -> np.ndarray | None:
    """Decode the frame and compute its line mask without any caching."""

    image = cv2.imread(str(path))
    if image is None:
        return None
    return pitch_line_mask(image)


def demote_outlier_direct_anchors(
    automatic_direct: dict[int, "PitchCalibration"],
    frame_evidence: list[dict],
    frames: list[tuple[Path, float]],
    *,
    manual_direct: dict[int, "PitchCalibration"],
    max_gap_seconds: float,
    residual_floor_pixels: float,
    best_quartile_ratio: float,
) -> tuple[dict[int, "PitchCalibration"], list[dict]]:
    """Demote automatic direct anchors whose line-residual tail is an outlier.

    A direct solve can fit the visible lines well at the median yet be
    twisted at the extremes (high ``residualP95``); every frame it anchors
    then inherits metres of ground error. Such anchors are demoted to
    rejected observations so their frames re-solve temporally from healthier
    neighbours — but never when a demotion would strip the only anchor some
    frame can reach within ``max_gap_seconds``. Manual anchors are
    authoritative and are never demoted.
    """

    measured = [
        (sample, float((frame_evidence[sample].get("alignmentMetrics") or {}).get("residualP95")))
        for sample in sorted(automatic_direct)
        if sample < len(frame_evidence)
        and isinstance(
            (frame_evidence[sample].get("alignmentMetrics") or {}).get("residualP95"),
            (int, float),
        )
    ]
    if len(measured) < 3 or best_quartile_ratio <= 0:
        return automatic_direct, []
    baseline = float(np.percentile([value for _, value in measured], 25))
    threshold = max(float(residual_floor_pixels), baseline * float(best_quartile_ratio))
    candidates = sorted(
        ((sample, value) for sample, value in measured if value > threshold),
        key=lambda item: -item[1],
    )
    if not candidates:
        return automatic_direct, []

    times = [float(time) for _, time in frames]

    def covered_samples(anchor_samples: set[int]) -> set[int]:
        anchor_times = [times[sample] for sample in anchor_samples if sample < len(times)]
        return {
            index
            for index, time in enumerate(times)
            if any(abs(time - anchor) <= max_gap_seconds + 1e-9 for anchor in anchor_times)
        }

    surviving = dict(automatic_direct)
    baseline_cover = covered_samples(set(surviving) | set(manual_direct))
    demotions: list[dict] = []
    for sample, value in candidates:
        trial = set(surviving) - {sample} | set(manual_direct)
        if not baseline_cover <= covered_samples(trial):
            continue
        surviving.pop(sample, None)
        evidence = frame_evidence[sample]
        evidence["status"] = "rejected"
        evidence["rejectionReasons"] = list(
            dict.fromkeys(
                [
                    *(evidence.get("rejectionReasons") or []),
                    "direct-anchor-residual-p95-outlier",
                ]
            )
        )
        demotions.append(
            {
                "sampleIndex": int(sample),
                "residualP95": round(value, 3),
                "thresholdPixels": round(threshold, 3),
            }
        )
    return surviving, demotions


def merge_direct_calibration_anchors(
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


def resolve_temporal_frame_calibrations(
    frames: list[tuple[Path, float]],
    frame_sizes: dict[int, tuple[int, int]],
    direct_calibrations: dict[int, PitchCalibration],
    motion_edges: dict[int, CameraMotionEstimate],
    frame_evidence: list[dict],
    person_frames: list[tuple[list[Detection], float]],
    pitch: dict,
    *,
    max_gap_seconds: float = 2.0,
    observed_mask_loader: ObservedLineMaskLoader | None = None,
    target_sample_indices: set[int] | None = None,
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
            source_frame_index=parse_source_frame_index(path),
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
        if (
            target_sample_indices is not None
            and sample_index not in target_sample_indices
        ):
            continue
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
        observed_mask = (observed_mask_loader or default_observed_line_mask)(
            frames[sample_index][0]
        )
        if observed_mask is None:
            validation_reasons.append("temporal-target-frame-unreadable")
        else:
            alignment_metrics = calibration_alignment_metrics_from_mask(
                observed_mask, calibration
            )
            alignment = alignment_metrics.as_dict() if alignment_metrics is not None else None
            if alignment_metrics is None:
                target_uncertainty_penalty += 1.25
            if (
                alignment_metrics is not None
                and not semantic_alignment_passes_review(alignment_metrics)
            ):
                validation_reasons.append("temporal-semantic-line-alignment-poor")

        people = person_frames[sample_index][0]
        person_support = None
        if len(people) >= 4:
            supported_people, total_people = calibration_person_support(
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
                "imageToPitch": matrix_payload(calibration.image_to_pitch),
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
