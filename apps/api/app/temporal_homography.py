from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from .camera_motion_contract import CameraMotionEstimate


@dataclass(frozen=True)
class MotionPath:
    target_to_anchor: np.ndarray
    confidence: float
    residual_sum: float
    edge_indices: tuple[int, ...]


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


def motion_path(
    motion_edges: dict[int, CameraMotionEstimate],
    target_sample_index: int,
    anchor_sample_index: int,
) -> MotionPath | None:
    if target_sample_index == anchor_sample_index:
        return MotionPath(np.eye(3, dtype=np.float64), 1.0, 0.0, ())
    lower = min(target_sample_index, anchor_sample_index)
    upper = max(target_sample_index, anchor_sample_index)
    edge_indices = tuple(range(lower + 1, upper + 1))
    edges = [motion_edges.get(index) for index in edge_indices]
    if any(edge is None or not edge.reliable for edge in edges):
        return None
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
    return MotionPath(
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
    path = motion_path(motion_edges, target_sample_index, anchor_sample_index)
    if path is None:
        return None
    return normalize_homography(anchor_image_to_pitch @ path.target_to_anchor)


def project_image_points(points: np.ndarray, image_to_pitch: np.ndarray) -> np.ndarray:
    source = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = source @ image_to_pitch.T
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-8
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def homography_disagreement_metres(
    left: np.ndarray, right: np.ndarray, width: int, height: int
) -> float | None:
    xs = np.linspace(width * 0.18, width * 0.82, 5)
    ys = np.linspace(height * 0.42, height * 0.92, 4)
    points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    left_pitch = project_image_points(points, left)
    right_pitch = project_image_points(points, right)
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
    normalized = project_image_points(points, transform)
    return (normalized, transform) if np.isfinite(normalized).all() else None


def fit_image_to_pitch_homography(
    image_points: np.ndarray, pitch_points: np.ndarray
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
    fitted = project_image_points(image_points, matrix)
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
