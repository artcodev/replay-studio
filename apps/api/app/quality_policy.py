from __future__ import annotations

"""Versioned quality thresholds and gate classification policy."""

from dataclasses import dataclass
from typing import Any, Literal


GateStatus = Literal["pass", "review", "reject", "unknown"]


@dataclass(frozen=True)
class QualityThresholds:
    """Engineering gates tuned against the project's labelled gold set."""

    calibration_coverage_pass: float = 0.90
    calibration_coverage_review: float = 0.75
    calibration_gap_pass_seconds: float = 0.60
    calibration_gap_review_seconds: float = 1.20
    temporal_uncertainty_p95_pass_metres: float = 2.50
    temporal_uncertainty_p95_review_metres: float = 5.00
    reprojection_p50_pass_px: float = 4.0
    reprojection_p50_review_px: float = 8.0
    reprojection_p95_pass_px: float = 8.0
    reprojection_p95_review_px: float = 20.0
    visible_side_agreement_pass: float = 0.90
    visible_side_agreement_review: float = 0.80
    semantic_alignment_f1_p10_pass: float = 0.15
    semantic_alignment_f1_p10_review: float = 0.08
    inlier_ratio_p10_pass: float = 0.70
    inlier_ratio_p10_review: float = 0.50
    projection_fallback_pass: float = 0.0
    projection_fallback_review: float = 0.05
    boundary_clamp_pass: float = 0.005
    boundary_clamp_review: float = 0.02
    player_speed_limit_mps: float = 14.0
    player_speed_violation_pass: float = 0.01
    player_speed_violation_review: float = 0.05
    ball_speed_limit_mps: float = 50.0
    ball_speed_violation_pass: float = 0.01
    ball_speed_violation_review: float = 0.05
    ball_observed_coverage_pass: float = 0.65
    ball_observed_coverage_review: float = 0.35
    ball_published_coverage_pass: float = 0.85
    ball_published_coverage_review: float = 0.60
    track_continuity_pass: float = 0.90
    track_continuity_review: float = 0.75
    track_fragmentation_pass: float = 0.10
    track_fragmentation_review: float = 0.30
    identity_idf1_pass: float = 0.75
    identity_idf1_review: float = 0.55


DEFAULT_THRESHOLDS = QualityThresholds()


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def lower_gate(
    gate_id: str,
    label: str,
    value: float | None,
    unit: str,
    pass_at: float,
    review_at: float,
    *,
    required: bool = True,
    evidence: str = "measured",
    note: str | None = None,
) -> dict[str, Any]:
    if value is None:
        status: GateStatus = "unknown"
    elif value <= pass_at:
        status = "pass"
    elif value <= review_at:
        status = "review"
    else:
        status = "reject"
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "required": required,
        "value": _rounded(value),
        "unit": unit,
        "evidence": evidence if value is not None else "missing",
        "thresholds": {"passAtMost": pass_at, "reviewAtMost": review_at},
        **({"note": note} if note else {}),
    }


def higher_gate(
    gate_id: str,
    label: str,
    value: float | None,
    unit: str,
    pass_at: float,
    review_at: float,
    *,
    required: bool = True,
    evidence: str = "measured",
    note: str | None = None,
) -> dict[str, Any]:
    if value is None:
        status: GateStatus = "unknown"
    elif value >= pass_at:
        status = "pass"
    elif value >= review_at:
        status = "review"
    else:
        status = "reject"
    return {
        "id": gate_id,
        "label": label,
        "status": status,
        "required": required,
        "value": _rounded(value),
        "unit": unit,
        "evidence": evidence if value is not None else "missing",
        "thresholds": {"passAtLeast": pass_at, "reviewAtLeast": review_at},
        **({"note": note} if note else {}),
    }
