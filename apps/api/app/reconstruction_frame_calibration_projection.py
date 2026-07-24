from __future__ import annotations

"""Resolve and apply the exact published calibration for frame inspection."""

from dataclasses import replace

import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .reconstruction_metric_projection import project_metric_point
from .reconstruction_pitch_projection import (
    project_pitch_point,
    project_pitch_point_unclamped,
)


def published_frame_evidence(
    reconstruction: dict,
    frame_index: int,
) -> dict | None:
    entries = (reconstruction.get("calibration") or {}).get(
        "frameEvidence"
    )
    if not isinstance(entries, list):
        return None
    return next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("sourceFrameIndex") == frame_index
        ),
        None,
    )


def selected_hypothesis_matrix(evidence: dict) -> list | None:
    selected_id = evidence.get("selectedHypothesisId")
    for hypothesis in evidence.get("hypotheses") or []:
        if hypothesis.get("id") == selected_id:
            return hypothesis.get("imageToPitch")
    return evidence.get("imageToPitch")


def resolve_frame_calibration(
    reconstruction: dict,
    frame_index: int,
    representative: PitchCalibration | None,
) -> tuple[PitchCalibration | None, str]:
    """Return the same frame-local matrix consumed by reconstruction."""

    evidence = published_frame_evidence(reconstruction, frame_index)
    if evidence is None:
        return representative, "saved-representative-homography"
    matrix = selected_hypothesis_matrix(evidence)
    if (
        matrix is None
        or str(evidence.get("projectionSource") or "none") == "none"
        or "accepted" not in str(evidence.get("solutionStatus") or "")
    ):
        return None, "published-frame-unresolved"
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        return None, "published-frame-invalid"
    if representative is not None:
        return (
            replace(
                representative,
                image_to_pitch=values,
                confidence=max(
                    float(representative.confidence),
                    float(evidence.get("confidence") or 0.0),
                ),
                method=str(
                    evidence.get("projectionSource")
                    or evidence.get("source")
                    or representative.method
                ),
                frame_index=frame_index,
            ),
            "published-per-frame-homography",
        )
    return (
        PitchCalibration(
            image_to_pitch=values,
            confidence=max(
                0.7,
                float(evidence.get("confidence") or 0.0),
            ),
            supported_lines=int(evidence.get("supportedLines") or 0),
            mean_line_score=float(evidence.get("meanLineScore") or 0.0),
            rectangle=str(
                evidence.get("rectangle") or "published-frame"
            ),
            method=str(
                evidence.get("projectionSource")
                or evidence.get("source")
                or "published-frame"
            ),
            frame_index=frame_index,
        ),
        "published-per-frame-homography",
    )


def project_inspection_people(
    people: list,
    *,
    frame_size: tuple[int, int],
    pitch: dict,
    calibration: PitchCalibration | None,
) -> tuple[
    list[tuple[float, float] | None],
    list[tuple[float, float] | None],
]:
    if calibration is None:
        return (
            [
                project_pitch_point(
                    item.x,
                    item.y,
                    *frame_size,
                    pitch,
                    None,
                )
                for item in people
            ],
            [None] * len(people),
        )
    return (
        [
            project_metric_point(item.x, item.y, calibration, pitch)
            for item in people
        ],
        [
            project_pitch_point_unclamped(
                item.x,
                item.y,
                *frame_size,
                pitch,
                calibration,
            )
            for item in people
        ],
    )


__all__ = (
    "project_inspection_people",
    "published_frame_evidence",
    "resolve_frame_calibration",
    "selected_hypothesis_matrix",
)
