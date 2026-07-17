from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp
from typing import Sequence

import numpy as np

from .pitch_calibration import PitchCalibration, pitch_side


@dataclass(frozen=True)
class CameraMotionEstimate:
    """Projective motion from the current image into the previous image.

    ``matrix`` follows a single invariant throughout the reconstruction code:
    column-vector image coordinates in frame ``t`` are mapped into frame
    ``t - 1``.  A valid near-identity transform is therefore different from an
    unestimated edge.  ``cut`` and ``unreliable`` edges are hard graph barriers.
    """

    matrix: np.ndarray
    status: str
    confidence: float
    tracked_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    residual_p50: float | None = None
    residual_p95: float | None = None
    forward_backward_p95: float | None = None
    coverage_ratio: float = 0.0
    scene_change_score: float | None = None
    reason: str | None = None

    @property
    def reliable(self) -> bool:
        return self.status == "estimated"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "model": "projective-homography",
            "confidence": round(float(self.confidence), 5),
            "currentToPrevious": _matrix_payload(self.matrix),
            "metrics": {
                "trackedCount": int(self.tracked_count),
                "inlierCount": int(self.inlier_count),
                "inlierRatio": round(float(self.inlier_ratio), 5),
                "residualP50Px": _round_optional(self.residual_p50),
                "residualP95Px": _round_optional(self.residual_p95),
                "forwardBackwardP95Px": _round_optional(self.forward_backward_p95),
                "coverageRatio": round(float(self.coverage_ratio), 5),
                "sceneChangeScore": _round_optional(self.scene_change_score),
            },
            "rejectionReasons": [self.reason] if self.reason else [],
        }


@dataclass(frozen=True)
class TemporalCalibrationFrame:
    sample_index: int
    source_frame_index: int
    scene_time: float
    width: int
    height: int


@dataclass(frozen=True)
class CalibrationHypothesis:
    id: str
    target_sample_index: int
    anchor_sample_index: int
    anchor_source_frame_index: int
    anchor_scene_time: float
    direction: str
    calibration: PitchCalibration
    score: float
    uncertainty_metres: float
    motion_confidence: float
    temporal_distance_seconds: float
    motion_edge_indices: tuple[int, ...]
    disagreement_metres: float | None = None
    rejection_reasons: tuple[str, ...] = ()

    def as_dict(self, rank: int, selected: bool = False) -> dict:
        origin = "direct" if self.direction == "direct" else f"temporal-{self.direction}"
        return {
            "id": self.id,
            "rank": rank,
            "selected": selected,
            "origin": origin,
            "score": round(float(self.score), 5),
            "scoreKind": "heuristic-temporal-hypothesis-score",
            "visiblePitchSide": pitch_side(self.calibration.rectangle),
            "anchorFrameIndices": [self.anchor_source_frame_index],
            "anchorSampleIndices": [self.anchor_sample_index],
            "motionEdgeIndices": list(self.motion_edge_indices),
            "temporalDistanceSeconds": round(float(self.temporal_distance_seconds), 4),
            "motionConfidence": round(float(self.motion_confidence), 5),
            "uncertaintyP95Metres": round(float(self.uncertainty_metres), 4),
            "disagreementMetres": _round_optional(self.disagreement_metres, 4),
            "imageToPitch": _matrix_payload(self.calibration.image_to_pitch),
            "rejectionReasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class TemporalCalibrationResolution:
    selected: CalibrationHypothesis | None
    hypotheses: tuple[CalibrationHypothesis, ...]
    projection_source: str
    ambiguity_margin: float | None = None
    rejection_reasons: tuple[str, ...] = ()

    def hypotheses_payload(self) -> list[dict]:
        selected_id = self.selected.id if self.selected is not None else None
        return [
            hypothesis.as_dict(rank, selected=hypothesis.id == selected_id)
            for rank, hypothesis in enumerate(self.hypotheses, start=1)
        ]


@dataclass(frozen=True)
class _MotionPath:
    target_to_anchor: np.ndarray
    confidence: float
    residual_sum: float
    edge_indices: tuple[int, ...]


def _round_optional(value: float | None, digits: int = 3) -> float | None:
    return round(float(value), digits) if value is not None and np.isfinite(value) else None


def _matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("Homography must be a finite 3 x 3 matrix")
    if abs(float(result[2, 2])) < 1e-10:
        norm = float(np.linalg.norm(result))
        if norm < 1e-10:
            raise ValueError("Homography has no finite scale")
        result /= norm
    else:
        result /= result[2, 2]
    return result


def _motion_path(
    motion_edges: dict[int, CameraMotionEstimate],
    target_sample_index: int,
    anchor_sample_index: int,
) -> _MotionPath | None:
    if target_sample_index == anchor_sample_index:
        return _MotionPath(np.eye(3, dtype=np.float64), 1.0, 0.0, ())
    lower = min(target_sample_index, anchor_sample_index)
    upper = max(target_sample_index, anchor_sample_index)
    edge_indices = tuple(range(lower + 1, upper + 1))
    edges = [motion_edges.get(index) for index in edge_indices]
    if any(edge is None or not edge.reliable for edge in edges):
        return None

    # edge[i] maps frame i -> frame i-1.  Compose high -> low first.
    high_to_low = np.eye(3, dtype=np.float64)
    for index in range(upper, lower, -1):
        edge = motion_edges[index]
        high_to_low = normalize_homography(edge.matrix @ high_to_low)
    if target_sample_index == upper:
        target_to_anchor = high_to_low
    else:
        try:
            target_to_anchor = normalize_homography(np.linalg.inv(high_to_low))
        except np.linalg.LinAlgError:
            return None

    confidences = [max(1e-4, min(1.0, edge.confidence)) for edge in edges if edge]
    geometric_mean = float(np.prod(confidences) ** (1.0 / len(confidences)))
    path_confidence = geometric_mean * exp(-0.025 * max(0, len(confidences) - 1))
    residual_sum = sum(
        float(edge.residual_p95 if edge.residual_p95 is not None else 3.0)
        for edge in edges
        if edge
    )
    return _MotionPath(
        target_to_anchor=target_to_anchor,
        confidence=max(0.0, min(1.0, path_confidence)),
        residual_sum=residual_sum,
        edge_indices=edge_indices,
    )


def propagate_homography(
    anchor_image_to_pitch: np.ndarray,
    motion_edges: dict[int, CameraMotionEstimate],
    target_sample_index: int,
    anchor_sample_index: int,
) -> np.ndarray | None:
    path = _motion_path(motion_edges, target_sample_index, anchor_sample_index)
    if path is None:
        return None
    return normalize_homography(anchor_image_to_pitch @ path.target_to_anchor)


def _project_image_points(points: np.ndarray, image_to_pitch: np.ndarray) -> np.ndarray:
    source = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = source @ image_to_pitch.T
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-8
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def homography_disagreement_metres(
    left: np.ndarray,
    right: np.ndarray,
    width: int,
    height: int,
) -> float | None:
    xs = np.linspace(width * 0.18, width * 0.82, 5)
    ys = np.linspace(height * 0.42, height * 0.92, 4)
    points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    left_pitch = _project_image_points(points, left)
    right_pitch = _project_image_points(points, right)
    valid = np.isfinite(left_pitch).all(axis=1) & np.isfinite(right_pitch).all(axis=1)
    if int(valid.sum()) < 5:
        return None
    distances = np.linalg.norm(left_pitch[valid] - right_pitch[valid], axis=1)
    finite = distances[np.isfinite(distances)]
    return float(np.median(finite)) if len(finite) else None


def _point_normalization(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        return None
    if not np.isfinite(points).all():
        return None
    centroid = np.mean(points, axis=0)
    centred = points - centroid
    mean_distance = float(np.mean(np.linalg.norm(centred, axis=1)))
    if not np.isfinite(mean_distance) or mean_distance < 1e-9:
        return None
    scale = np.sqrt(2.0) / mean_distance
    transform = np.asarray(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = _project_image_points(points, transform)
    if not np.isfinite(normalized).all():
        return None
    return normalized, transform


def _fit_image_to_pitch_homography(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
) -> np.ndarray | None:
    """Fit a homography with Hartley-normalized DLT and fail closed."""

    if image_points.shape != pitch_points.shape or len(image_points) < 8:
        return None
    source_normalization = _point_normalization(image_points)
    target_normalization = _point_normalization(pitch_points)
    if source_normalization is None or target_normalization is None:
        return None
    normalized_source, source_transform = source_normalization
    normalized_target, target_transform = target_normalization

    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(normalized_source, normalized_target):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(design).all() or np.linalg.matrix_rank(design) < 8:
        return None
    try:
        _, singular_values, right_vectors = np.linalg.svd(design, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if (
        len(singular_values) < 9
        or not np.isfinite(singular_values).all()
        or singular_values[0] <= 0.0
        or singular_values[-2] <= singular_values[0] * 1e-12
    ):
        return None

    normalized_homography = right_vectors[-1].reshape(3, 3)
    try:
        matrix = normalize_homography(
            np.linalg.inv(target_transform)
            @ normalized_homography
            @ source_transform
        )
    except (np.linalg.LinAlgError, ValueError):
        return None
    determinant = float(np.linalg.det(matrix))
    condition = float(np.linalg.cond(matrix))
    if (
        not np.isfinite(determinant)
        or abs(determinant) < 1e-12
        or not np.isfinite(condition)
        or condition > 1e12
    ):
        return None

    fitted = _project_image_points(image_points, matrix)
    if not np.isfinite(fitted).all():
        return None
    residuals = np.linalg.norm(fitted - pitch_points, axis=1)
    pitch_span = float(np.linalg.norm(np.ptp(pitch_points, axis=0)))
    maximum_residual = max(0.35, min(1.5, pitch_span * 0.03))
    if (
        not np.isfinite(residuals).all()
        or float(np.percentile(residuals, 95)) > maximum_residual
    ):
        return None
    return matrix


def _bidirectional_consensus_homography(
    target: TemporalCalibrationFrame,
    first: CalibrationHypothesis,
    second: CalibrationHypothesis,
) -> np.ndarray | None:
    """Blend compatible bracketing projections, then refit one homography.

    Both hypotheses already map the target image into pitch coordinates.  The
    blend is performed in pitch space rather than directly between matrix
    coefficients, whose arbitrary projective scale makes interpolation unsafe.
    """

    earlier, later = sorted(
        (first, second),
        key=lambda item: (item.anchor_scene_time, item.anchor_sample_index),
    )
    span = float(later.anchor_scene_time) - float(earlier.anchor_scene_time)
    if span <= 1e-9:
        return None
    progress = (float(target.scene_time) - float(earlier.anchor_scene_time)) / span
    if progress < -1e-9 or progress > 1.0 + 1e-9:
        return None
    progress = max(0.0, min(1.0, progress))
    # Smoothstep keeps the consensus pinned to each immutable direct anchor
    # and avoids a derivative discontinuity at the ends of the interval.
    later_weight = progress * progress * (3.0 - 2.0 * progress)

    xs = np.linspace(target.width * 0.14, target.width * 0.86, 6)
    ys = np.linspace(target.height * 0.40, target.height * 0.92, 5)
    image_points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    earlier_pitch = _project_image_points(
        image_points,
        earlier.calibration.image_to_pitch,
    )
    later_pitch = _project_image_points(
        image_points,
        later.calibration.image_to_pitch,
    )
    valid = (
        np.isfinite(earlier_pitch).all(axis=1)
        & np.isfinite(later_pitch).all(axis=1)
        & (np.max(np.abs(earlier_pitch), axis=1) < 1e5)
        & (np.max(np.abs(later_pitch), axis=1) < 1e5)
    )
    if int(valid.sum()) < 12:
        return None
    source = image_points[valid]
    blended_pitch = (
        earlier_pitch[valid] * (1.0 - later_weight)
        + later_pitch[valid] * later_weight
    )
    return _fit_image_to_pitch_homography(source, blended_pitch)


def _anchor_uncertainty(calibration: PitchCalibration) -> float:
    error = calibration.reprojection_p95
    if error is None:
        error = calibration.reprojection_error
    if error is not None and np.isfinite(error):
        return max(0.25, min(4.0, float(error) * 0.18))
    return max(0.5, min(4.0, (1.0 - float(calibration.confidence)) * 5.0))


def _make_hypothesis(
    target: TemporalCalibrationFrame,
    anchor: TemporalCalibrationFrame,
    anchor_calibration: PitchCalibration,
    motion_edges: dict[int, CameraMotionEstimate],
) -> CalibrationHypothesis | None:
    path = _motion_path(motion_edges, target.sample_index, anchor.sample_index)
    if path is None:
        return None
    temporal_distance = abs(float(target.scene_time) - float(anchor.scene_time))
    matrix = normalize_homography(anchor_calibration.image_to_pitch @ path.target_to_anchor)
    direction = "forward" if anchor.sample_index < target.sample_index else "backward"
    anchor_score = max(0.0, min(1.0, float(anchor_calibration.confidence)))
    inlier_ratio = anchor_calibration.inlier_ratio
    if inlier_ratio is not None:
        anchor_score *= 0.85 + 0.15 * max(0.0, min(1.0, float(inlier_ratio)))
    temporal_decay = exp(-temporal_distance / 7.0)
    score = anchor_score * (0.82 + 0.18 * path.confidence) * temporal_decay
    uncertainty = (
        _anchor_uncertainty(anchor_calibration)
        + temporal_distance * 0.38
        + path.residual_sum * 0.035
        + (1.0 - path.confidence) * 3.0
    )
    propagated = replace(
        anchor_calibration,
        image_to_pitch=matrix,
        confidence=max(0.0, min(0.99, score)),
        supported_lines=0,
        mean_line_score=0.0,
        matched_curves=0,
        method=f"temporal-{direction}",
        keypoint_count=0,
        inlier_count=0,
        reprojection_error=None,
        frame_index=target.source_frame_index,
        detected_keypoint_count=0,
        completed_keypoint_count=0,
        inlier_ratio=None,
        reprojection_p95=None,
        raw_line_count=0,
        ground_error_p50=None,
        ground_error_p95=None,
        raw_keypoints=(),
        confidence_kind="heuristic-temporal-hypothesis-score",
    )
    return CalibrationHypothesis(
        id=(
            f"temporal-{direction}-s{anchor.sample_index}"
            f"-to-s{target.sample_index}"
        ),
        target_sample_index=target.sample_index,
        anchor_sample_index=anchor.sample_index,
        anchor_source_frame_index=anchor.source_frame_index,
        anchor_scene_time=anchor.scene_time,
        direction=direction,
        calibration=propagated,
        score=max(0.0, min(0.99, score)),
        uncertainty_metres=max(0.25, uncertainty),
        motion_confidence=path.confidence,
        temporal_distance_seconds=temporal_distance,
        motion_edge_indices=path.edge_indices,
    )


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
    """Resolve direct and forward/backward temporal calibration hypotheses.

    Direct observations are never replaced.  Missing/rejected frames may be
    recovered only through a chain of reliable motion edges and within the
    configured temporal gap.  Competing candidates are retained in the result;
    close candidates form consensus, while similarly scored incompatible
    candidates remain explicitly ambiguous and publish no metric calibration.
    """

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
                uncertainty_metres=_anchor_uncertainty(direct),
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
                hypothesis := _make_hypothesis(
                    target,
                    anchor,
                    direct_calibrations[anchor.sample_index],
                    motion_edges,
                )
            )
            is not None
        ]
        candidates.sort(key=lambda item: (item.score, -item.uncertainty_metres), reverse=True)
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
                consensus_matrix = _bidirectional_consensus_homography(
                    target,
                    top,
                    consensus_peer,
                )
                if consensus_matrix is None:
                    rejection_reasons.append("temporal-bidirectional-consensus-failed")
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
                            min(top.uncertainty_metres, consensus_peer.uncertainty_metres)
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

    # A malformed caller should not silently lose frames.
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
