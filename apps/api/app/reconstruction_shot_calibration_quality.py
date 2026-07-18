from __future__ import annotations

"""Pure shot-level calibration evidence aggregation and acceptance policy."""

import numpy as np

from .reconstruction_calibration_policy import (
    CALIBRATION_PASS_COVERAGE,
    CALIBRATION_PASS_MAX_GAP_SECONDS,
    CALIBRATION_PASS_REPROJECTION_P95,
    CALIBRATION_PASS_SIDE_AGREEMENT,
    CALIBRATION_REVIEW_COVERAGE,
    CALIBRATION_REVIEW_MAX_GAP_SECONDS,
    CALIBRATION_REVIEW_SIDE_AGREEMENT,
    CALIBRATION_SHOT_REVIEW_REPROJECTION_P95,
    TEMPORAL_PASS_UNCERTAINTY_METRES,
    TEMPORAL_REVIEW_UNCERTAINTY_METRES,
)


def calibration_summary(frame_evidence: list[dict]) -> dict:
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
    motion_unreliable = sum(
        item.get("status") == "unreliable" for item in motion_edges
    )
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
        if item.get("reprojectionP95") is not None
        or item.get("reprojectionError") is not None
    )
    alignment_f1 = sorted(
        float((item.get("alignmentMetrics") or {})["f1"])
        for item in accepted
        if (item.get("alignmentMetrics") or {}).get("f1") is not None
    )
    orientation_observations = direct
    if not orientation_observations:
        # Propagated frames are not independent orientation votes. When there
        # is no direct frame, retain only one vote per temporal anchor.
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
            round(float(np.percentile(median_errors, 50)), 3)
            if median_errors
            else None
        ),
        "reprojectionP95": (
            round(float(np.percentile(p95_errors, 95)), 3) if p95_errors else None
        ),
        "alignmentF1P10": (
            round(float(np.percentile(alignment_f1, 10)), 3)
            if alignment_f1
            else None
        ),
        "visiblePitchSide": visible_side,
        "sideAgreement": (
            round(side_agreement, 3) if side_agreement is not None else None
        ),
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


def calibration_gate(
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
        status = (
            "pass"
            if value >= pass_limit
            else "review"
            if value >= review_limit
            else "reject"
        )
    else:
        status = (
            "pass"
            if value <= pass_limit
            else "review"
            if value <= review_limit
            else "reject"
        )
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


def evaluate_calibration_quality(frame_evidence: list[dict]) -> dict:
    summary = calibration_summary(frame_evidence)
    gates = [
        calibration_gate(
            "calibration-coverage",
            "Usable calibrated frames",
            summary["usableCoverage"],
            CALIBRATION_PASS_COVERAGE,
            CALIBRATION_REVIEW_COVERAGE,
            higher_is_better=True,
            unit="ratio",
        ),
        calibration_gate(
            "calibration-gap",
            "Longest gap between calibrated frames",
            summary["maxGapSeconds"],
            CALIBRATION_PASS_MAX_GAP_SECONDS,
            CALIBRATION_REVIEW_MAX_GAP_SECONDS,
            higher_is_better=False,
            unit="seconds",
        ),
        calibration_gate(
            "reprojection-error",
            "Reprojection error p95",
            summary["reprojectionP95"],
            CALIBRATION_PASS_REPROJECTION_P95,
            CALIBRATION_SHOT_REVIEW_REPROJECTION_P95,
            higher_is_better=False,
            unit="pixels",
        ),
        calibration_gate(
            "orientation-stability",
            "Visible-side agreement",
            summary["sideAgreement"],
            CALIBRATION_PASS_SIDE_AGREEMENT,
            CALIBRATION_REVIEW_SIDE_AGREEMENT,
            higher_is_better=True,
            unit="ratio",
            unavailable_status="not-applicable",
        ),
        calibration_gate(
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
            calibration_gate(
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
    failed = [
        gate["id"] for gate in gates if gate["status"] in {"review", "reject"}
    ]
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
