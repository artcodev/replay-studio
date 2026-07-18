from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .camera_motion_contract import CameraMotionEstimate
from .pitch_calibration_contract import PitchCalibration
from .temporal_calibration_consensus import bidirectional_consensus_homography
from .temporal_calibration_contract import (
    CalibrationHypothesis,
    TemporalCalibrationFrame,
    TemporalCalibrationResolution,
)
from .temporal_calibration_hypothesis import (
    anchor_uncertainty,
    make_temporal_hypothesis,
)
from .temporal_homography import homography_disagreement_metres


def solve_calibration_sequence(
    frames: Sequence[TemporalCalibrationFrame],
    direct_calibrations: dict[int, PitchCalibration],
    motion_edges: dict[int, CameraMotionEstimate],
    *,
    max_gap_seconds: float = 2.0,
    minimum_score: float = 0.58,
    maximum_uncertainty_metres: float = 5.0,
    consensus_metres: float = 2.5,
    ambiguity_score_margin: float = 0.10,
    max_anchors_per_direction: int = 2,
) -> dict[int, TemporalCalibrationResolution]:
    """Resolve immutable direct anchors and auditable temporal hypotheses."""

    ordered = sorted(frames, key=lambda item: item.sample_index)
    anchors = [frame for frame in ordered if frame.sample_index in direct_calibrations]
    results: dict[int, TemporalCalibrationResolution] = {}

    for target in ordered:
        direct = direct_calibrations.get(target.sample_index)
        if direct is not None:
            direct_hypothesis = CalibrationHypothesis(
                id=f"direct-s{target.sample_index}",
                target_sample_index=target.sample_index,
                anchor_sample_index=target.sample_index,
                anchor_source_frame_index=target.source_frame_index,
                anchor_scene_time=target.scene_time,
                direction="direct",
                calibration=direct,
                score=float(direct.confidence),
                uncertainty_metres=anchor_uncertainty(direct),
                motion_confidence=1.0,
                temporal_distance_seconds=0.0,
                motion_edge_indices=(),
            )
            results[target.sample_index] = TemporalCalibrationResolution(
                selected=direct_hypothesis,
                hypotheses=(direct_hypothesis,),
                projection_source="direct",
            )
            continue

        before = sorted(
            (
                anchor
                for anchor in anchors
                if anchor.sample_index < target.sample_index
                and target.scene_time - anchor.scene_time <= max_gap_seconds + 1e-9
            ),
            key=lambda item: item.sample_index,
            reverse=True,
        )[:max_anchors_per_direction]
        after = sorted(
            (
                anchor
                for anchor in anchors
                if anchor.sample_index > target.sample_index
                and anchor.scene_time - target.scene_time <= max_gap_seconds + 1e-9
            ),
            key=lambda item: item.sample_index,
        )[:max_anchors_per_direction]
        candidates = [
            hypothesis
            for anchor in (*before, *after)
            if (
                hypothesis := make_temporal_hypothesis(
                    target,
                    anchor,
                    direct_calibrations[anchor.sample_index],
                    motion_edges,
                )
            )
            is not None
        ]
        candidates.sort(
            key=lambda item: (item.score, -item.uncertainty_metres), reverse=True
        )
        if not candidates:
            results[target.sample_index] = TemporalCalibrationResolution(
                selected=None,
                hypotheses=(),
                projection_source="none",
                rejection_reasons=("no-reliable-temporal-path",),
            )
            continue

        top = candidates[0]
        ambiguity_margin = None
        projection_source = f"temporal-{top.direction}"
        rejection_reasons: list[str] = []
        if len(candidates) > 1:
            ambiguity_margin = top.score - candidates[1].score
            comparisons: list[tuple[int, CalibrationHypothesis, float | None]] = []
            for index, candidate in enumerate(candidates[1:], start=1):
                disagreement = homography_disagreement_metres(
                    top.calibration.image_to_pitch,
                    candidate.calibration.image_to_pitch,
                    target.width,
                    target.height,
                )
                candidate = replace(candidate, disagreement_metres=disagreement)
                candidates[index] = candidate
                comparisons.append((index, candidate, disagreement))

            nearby_conflicts = [
                candidate
                for _, candidate, disagreement in comparisons
                if top.score - candidate.score < ambiguity_score_margin
                and (disagreement is None or disagreement > consensus_metres)
            ]
            compatible_opposite = [
                candidate
                for _, candidate, disagreement in comparisons
                if candidate.direction != top.direction
                and disagreement is not None
                and disagreement <= consensus_metres
                and top.score - candidate.score < ambiguity_score_margin
            ]
            finite_disagreements = [
                disagreement
                for _, _, disagreement in comparisons
                if disagreement is not None
            ]
            candidates[0] = replace(
                top,
                disagreement_metres=(
                    min(finite_disagreements) if finite_disagreements else None
                ),
            )
            top = candidates[0]
            if nearby_conflicts:
                rejection_reasons.append("conflicting-temporal-hypotheses")
            elif compatible_opposite:
                consensus_peer = compatible_opposite[0]
                consensus_matrix = bidirectional_consensus_homography(
                    target, top, consensus_peer
                )
                if consensus_matrix is None:
                    rejection_reasons.append(
                        "temporal-bidirectional-consensus-failed"
                    )
                else:
                    projection_source = "temporal-bidirectional"
                    top = replace(
                        top,
                        calibration=replace(
                            top.calibration,
                            image_to_pitch=consensus_matrix,
                            method=projection_source,
                        ),
                        uncertainty_metres=max(
                            0.25,
                            min(
                                top.uncertainty_metres,
                                consensus_peer.uncertainty_metres,
                            )
                            * 0.82,
                        ),
                    )
                    candidates[0] = top

        if top.score < minimum_score:
            rejection_reasons.append("temporal-score-below-threshold")
        if top.uncertainty_metres > maximum_uncertainty_metres:
            rejection_reasons.append("temporal-uncertainty-too-high")
        selected = None if rejection_reasons else top
        results[target.sample_index] = TemporalCalibrationResolution(
            selected=selected,
            hypotheses=tuple(candidates),
            projection_source=projection_source if selected is not None else "none",
            ambiguity_margin=ambiguity_margin,
            rejection_reasons=tuple(rejection_reasons),
        )

    for sample_index in (frame.sample_index for frame in ordered):
        results.setdefault(
            sample_index,
            TemporalCalibrationResolution(
                selected=None,
                hypotheses=(),
                projection_source="none",
                rejection_reasons=("temporal-solver-did-not-return-frame",),
            ),
        )
    return results
