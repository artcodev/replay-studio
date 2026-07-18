from __future__ import annotations

"""Pure quality gates for one directly calibrated broadcast frame."""

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_calibration_quality import calibration_alignment_metrics
from .reconstruction_calibration_policy import (
    CALIBRATION_REVIEW_REPROJECTION_P95,
    METRIC_CALIBRATION_THRESHOLD,
    PARTIAL_VIEW_ALIGNMENT_F1_MINIMUM,
    PARTIAL_VIEW_REPROJECTION_P50_LIMIT,
    PARTIAL_VIEW_REPROJECTION_P95_LIMIT,
)
from .reconstruction_person_detection_contract import Detection
from .reconstruction_metric_projection import calibration_person_support


def semantic_alignment_passes_review(alignment) -> bool:
    """Accept a partial view only when its visible central evidence is strong."""

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


def calibration_quality_gate(
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


def direct_calibration_qa(
    image: np.ndarray,
    calibration: PitchCalibration,
    *,
    people: list[Detection] | None = None,
    pitch: dict | None = None,
    manual: bool = False,
) -> dict:
    """Apply auditable direct-observation gates in previews and rebuilds."""

    rejection_reasons: list[str] = []
    gates: list[dict] = []
    matrix = calibration.image_to_pitch
    finite_matrix = matrix.shape == (3, 3) and bool(np.isfinite(matrix).all())
    gates.append(
        calibration_quality_gate(
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
        calibration_quality_gate(
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
        calibration_quality_gate(
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
        calibration_quality_gate(
            "model-reprojection-p95",
            "Model reprojection p95",
            (
                "not-available"
                if candidate_p95 is None
                else "pass"
                if raw_reprojection_pass
                else "fail"
            ),
            value=(
                round(float(candidate_p95), 3)
                if candidate_p95 is not None
                else None
            ),
            threshold={
                "atMostPixels": CALIBRATION_REVIEW_REPROJECTION_P95,
                "partialViewAtMostPixels": 25.0,
            },
            reason=None if raw_reprojection_pass else "reprojection-error-too-high",
        )
    )

    alignment_metrics = calibration_alignment_metrics(image, calibration)
    alignment = alignment_metrics.as_dict() if alignment_metrics is not None else None
    semantic_pass = semantic_alignment_passes_review(alignment_metrics)
    if alignment_metrics is None:
        rejection_reasons.append("semantic-line-alignment-unscored")
    elif not semantic_pass:
        rejection_reasons.append("semantic-line-alignment-poor")
    gates.append(
        calibration_quality_gate(
            "semantic-line-alignment",
            "Projected markings match observed pitch lines",
            (
                "not-available"
                if alignment_metrics is None
                else "pass"
                if semantic_pass
                else "fail"
            ),
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
            calibration_quality_gate(
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
                calibration_quality_gate(
                    "supported-pitch-lines",
                    "Supported pitch markings",
                    "pass" if line_support_pass else "fail",
                    value=calibration.supported_lines,
                    threshold={"atLeast": 4},
                    reason=(
                        None if line_support_pass else "insufficient-supported-lines"
                    ),
                ),
                calibration_quality_gate(
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
                calibration_quality_gate(
                    "semantic-keypoints",
                    "Detected semantic pitch keypoints",
                    "pass" if keypoint_pass else "fail",
                    value=detected_keypoints,
                    threshold={"atLeast": 6},
                    reason=(
                        None if keypoint_pass else "insufficient-detected-keypoints"
                    ),
                ),
                calibration_quality_gate(
                    "keypoint-inlier-ratio",
                    "Semantic keypoint inlier ratio",
                    "pass" if inlier_pass else "fail",
                    value=(
                        round(float(inlier_ratio), 5)
                        if inlier_ratio is not None
                        else None
                    ),
                    threshold={"atLeast": 0.65},
                    reason=(
                        None if inlier_pass else "insufficient-keypoint-inlier-ratio"
                    ),
                ),
            ]
        )

    person_support = None
    if people is not None and pitch is not None and len(people) >= 4:
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
        person_support_pass = supported_people >= 4 and support_ratio >= 0.55
        if not person_support_pass:
            rejection_reasons.append("insufficient-person-pitch-support")
        gates.append(
            calibration_quality_gate(
                "person-pitch-support",
                "Detected people project inside the pitch",
                "pass" if person_support_pass else "fail",
                value=person_support,
                threshold={"supportedAtLeast": 4, "ratioAtLeast": 0.55},
                reason=(
                    None
                    if person_support_pass
                    else "insufficient-person-pitch-support"
                ),
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
