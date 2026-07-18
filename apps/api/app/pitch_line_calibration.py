from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, cos, degrees, hypot, pi, sin
from time import monotonic

import cv2
import numpy as np

from .pitch_calibration_contract import PitchCalibration
from .pitch_calibration_orientation import canonicalize_penalty_side
from .pitch_geometry import PITCH_CURVES, PITCH_LINES, RECTANGLES, project_points
from .pitch_image_evidence import pitch_line_mask


@dataclass(frozen=True)
class _ImageLine:
    rho: float
    theta: float
    support: int

    @property
    def orientation(self) -> float:
        return (degrees(self.theta) + 90.0) % 180.0


def _angle_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 180.0
    return min(difference, 180.0 - difference)


def _orientation_families(mask: np.ndarray) -> tuple[float, float] | None:
    segments = cv2.HoughLinesP(
        mask,
        1,
        pi / 180,
        threshold=26,
        minLineLength=max(32, int(mask.shape[1] * 0.04)),
        maxLineGap=15,
    )
    if segments is None:
        return None
    vectors = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        length = hypot(float(x2 - x1), float(y2 - y1))
        if length < 32:
            continue
        angle = atan2(float(y2 - y1), float(x2 - x1)) % pi
        repeat = max(1, min(10, int(length / 35)))
        vectors.extend([[cos(angle * 2), sin(angle * 2)]] * repeat)
    if len(vectors) < 8:
        return None
    cv2.setRNGSeed(11)
    _, labels, centers = cv2.kmeans(
        np.float32(vectors),
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001),
        8,
        cv2.KMEANS_PP_CENTERS,
    )
    weights = [int((labels.ravel() == index).sum()) for index in range(2)]
    if min(weights) < 2:
        return None
    orientations = [degrees(atan2(float(center[1]), float(center[0])) / 2) % 180 for center in centers]
    if not 12.0 <= _angle_distance(*orientations) <= 82.0:
        return None
    return orientations[0], orientations[1]


def _line_evidence(distance_map: np.ndarray, rho: float, theta: float) -> int:
    height, width = distance_map.shape
    direction_x, direction_y = -sin(theta), cos(theta)
    origin_x, origin_y = rho * cos(theta), rho * sin(theta)
    parameters = []
    for x in (0.0, float(width - 1)):
        if abs(direction_x) > 1e-6:
            parameters.append((x - origin_x) / direction_x)
    for y in (height * 0.18, float(height - 1)):
        if abs(direction_y) > 1e-6:
            parameters.append((y - origin_y) / direction_y)
    inside_parameters = []
    for parameter in parameters:
        x = origin_x + parameter * direction_x
        y = origin_y + parameter * direction_y
        if -1 <= x <= width and height * 0.18 - 1 <= y <= height:
            inside_parameters.append(parameter)
    if len(inside_parameters) < 2:
        return 0
    start, end = min(inside_parameters), max(inside_parameters)
    sample_count = max(2, min(1200, int(end - start) + 1))
    sample_parameters = np.linspace(start, end, sample_count)
    xs = np.clip(np.rint(origin_x + sample_parameters * direction_x).astype(int), 0, width - 1)
    ys = np.clip(np.rint(origin_y + sample_parameters * direction_y).astype(int), 0, height - 1)
    hits = distance_map[ys, xs] < 3.0
    if not hits.any():
        return 0
    locally_supported = np.convolve(hits.astype(np.uint8), np.ones(13, dtype=np.uint8), mode="same") >= 3
    gaps = np.flatnonzero(~locally_supported)
    boundaries = np.concatenate(([-1], gaps, [len(locally_supported)]))
    longest_run = int(np.diff(boundaries).max() - 1)
    return int(hits.sum() + longest_run * 2)


def _candidate_lines(
    mask: np.ndarray,
    orientation: float,
    other_orientation: float,
) -> list[_ImageLine]:
    edges = cv2.Canny(mask, 40, 130)
    raw = cv2.HoughLines(edges, 1, pi / 720, threshold=max(26, int(mask.shape[1] * 0.03)))
    if raw is None:
        return []
    distance_map = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    candidates = []
    coarse_keys: set[tuple[int, int]] = set()
    for rho, theta in raw.reshape(-1, 2)[:3000]:
        line = _ImageLine(float(rho), float(theta), 0)
        own_distance = _angle_distance(line.orientation, orientation)
        if own_distance > 24.0 or own_distance > _angle_distance(line.orientation, other_orientation):
            continue
        coarse_key = (round(line.rho / 4.0), round(line.orientation))
        if coarse_key in coarse_keys:
            continue
        coarse_keys.add(coarse_key)
        support = _line_evidence(distance_map, line.rho, line.theta)
        if support >= 80:
            candidates.append(_ImageLine(line.rho, line.theta, support))
    candidates.sort(key=lambda line: line.support, reverse=True)
    unique: list[_ImageLine] = []
    for candidate in candidates:
        if any(
            _angle_distance(candidate.orientation, existing.orientation) < 1.5
            and abs(candidate.rho - existing.rho) < 8
            for existing in unique
        ):
            continue
        unique.append(candidate)

    # Perspective convergence can rotate parallel pitch lines by several degrees.
    # Preserve both spatial and angular diversity instead of only taking the most
    # strongly supported (and often duplicated) Hough peaks.
    buckets: dict[int, list[_ImageLine]] = {}
    for candidate in unique:
        bucket = int(np.floor(candidate.rho / 44.0))
        buckets.setdefault(bucket, []).append(candidate)

    representatives: list[list[_ImageLine]] = []
    for selected in buckets.values():
        selected.sort(key=lambda line: line.support, reverse=True)
        choices = selected[:2]
        for target in (-16.0, -9.0, -5.0, 0.0, 5.0, 9.0, 16.0):
            angular = min(
                selected,
                key=lambda line: abs(
                    (((line.orientation - orientation + 90.0) % 180.0) - 90.0) - target
                ),
            )
            if angular not in choices:
                choices.append(angular)
        representatives.append(choices)

    diverse: list[_ImageLine] = []
    for rank in range(9):
        layer = [choices[rank] for choices in representatives if len(choices) > rank]
        layer.sort(key=lambda line: line.support, reverse=True)
        diverse.extend(layer)
    return diverse[:80]


def _candidate_pairs(
    lines: list[_ImageLine],
    limit: int,
) -> list[tuple[_ImageLine, _ImageLine]]:
    pairs: list[tuple[float, tuple[_ImageLine, _ImageLine]]] = []
    for left, right in combinations(lines, 2):
        angular_distance = _angle_distance(left.orientation, right.orientation)
        rho_distance = abs(left.rho - right.rho)
        if angular_distance > 18.0 or rho_distance < 16.0:
            continue
        evidence = np.log1p(left.support) + np.log1p(right.support)
        evidence += min(1.2, np.log1p(rho_distance) * 0.16)
        pairs.append((float(evidence), (left, right)))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [pair for _, pair in pairs[:limit]]


def _intersection(left: _ImageLine, right: _ImageLine) -> np.ndarray | None:
    matrix = np.array(
        [[cos(left.theta), sin(left.theta)], [cos(right.theta), sin(right.theta)]],
        dtype=np.float64,
    )
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 0.04:
        return None
    return np.linalg.solve(matrix, np.array([left.rho, right.rho], dtype=np.float64))


def _quad_points(first: tuple[_ImageLine, _ImageLine], second: tuple[_ImageLine, _ImageLine]) -> np.ndarray | None:
    points = []
    for first_line in first:
        for second_line in second:
            point = _intersection(first_line, second_line)
            if point is None:
                return None
            points.append(point)
    return np.float32(points)


def _valid_quad(points: np.ndarray, width: int, height: int) -> bool:
    if not np.isfinite(points).all():
        return False
    if (
        (points[:, 0] < -width * 0.18).any()
        or (points[:, 0] > width * 1.18).any()
        or (points[:, 1] < height * 0.12).any()
        or (points[:, 1] > height * 1.12).any()
    ):
        return False
    distances = [
        hypot(*(points[0] - points[1])),
        hypot(*(points[2] - points[3])),
        hypot(*(points[0] - points[2])),
        hypot(*(points[1] - points[3])),
    ]
    return min(distances) > 34 and max(distances) < hypot(width, height) * 1.4


def _score_homography(
    image_to_pitch: np.ndarray,
    distance_map: np.ndarray,
    expected_curve: str,
) -> tuple[float, int, float, int]:
    try:
        pitch_to_image = np.linalg.inv(image_to_pitch)
    except np.linalg.LinAlgError:
        return 0.0, 0, 0.0, 0
    marking_scores: list[tuple[str, float]] = []
    for _, start, end in PITCH_LINES:
        alpha = np.linspace(0.0, 1.0, 90)
        pitch_points = np.column_stack(
            [start[0] + (end[0] - start[0]) * alpha, start[1] + (end[1] - start[1]) * alpha]
        )
        marking_scores.extend(_score_projected_marking("line", pitch_points, pitch_to_image, distance_map))
    for name, pitch_points in PITCH_CURVES:
        if name == expected_curve:
            marking_scores.extend(
                _score_projected_marking("curve", pitch_points, pitch_to_image, distance_map)
            )
    if len(marking_scores) < 4:
        return 0.0, 0, 0.0, 0
    supported = sum(score >= 0.32 for _, score in marking_scores)
    matched_curves = sum(kind == "curve" and score >= 0.27 for kind, score in marking_scores)
    strongest = sorted((score for _, score in marking_scores), reverse=True)[: min(9, len(marking_scores))]
    mean_score = float(np.mean(strongest))
    curve_peak = max((score for kind, score in marking_scores if kind == "curve"), default=0.0)
    strongest_lines = sorted(
        (score for kind, score in marking_scores if kind == "line"), reverse=True
    )[:4]
    line_peak = float(np.mean(strongest_lines)) if strongest_lines else 0.0
    # Straight lines alone are ambiguous in broadcast footage (boards, mowing
    # stripes, goal frame). A circle or penalty arc is the discriminating cue.
    confidence = (curve_peak * 0.70 + line_peak * 0.30) * min(1.0, supported / 4.0)
    return confidence, supported, mean_score, matched_curves


def _plausible_camera(
    image_to_pitch: np.ndarray,
    image_points: np.ndarray,
    image_height: int,
) -> bool:
    try:
        pitch_to_image = np.linalg.inv(image_to_pitch)
    except np.linalg.LinAlgError:
        return False
    horizon = np.cross(pitch_to_image[:, 0], pitch_to_image[:, 1])
    if not np.isfinite(horizon).all() or abs(float(horizon[1])) < 1e-7:
        return False
    horizon_y = (-horizon[0] * image_points[:, 0] - horizon[2]) / horizon[1]
    # A broadcast camera is above the pitch plane: the ground-plane horizon must
    # stay above every corner used to calibrate the visible field rectangle.
    return bool(np.all(horizon_y < image_points[:, 1] - image_height * 0.025))


def _score_projected_marking(
    kind: str,
    pitch_points: np.ndarray,
    pitch_to_image: np.ndarray,
    distance_map: np.ndarray,
) -> list[tuple[str, float]]:
    height, width = distance_map.shape
    image_points = project_points(pitch_points, pitch_to_image)
    inside = (
        np.isfinite(image_points).all(axis=1)
        & (image_points[:, 0] >= 0)
        & (image_points[:, 0] < width)
        & (image_points[:, 1] >= height * 0.16)
        & (image_points[:, 1] < height)
    )
    minimum_samples = 14 if kind == "curve" else 12
    if int(inside.sum()) < minimum_samples:
        return []
    visible_points = image_points[inside]
    span = hypot(float(np.ptp(visible_points[:, 0])), float(np.ptp(visible_points[:, 1])))
    if span < (34.0 if kind == "curve" else 42.0):
        return []
    visible = visible_points.round().astype(int)
    visible[:, 0] = np.clip(visible[:, 0], 0, width - 1)
    visible[:, 1] = np.clip(visible[:, 1], 0, height - 1)
    distances = distance_map[visible[:, 1], visible[:, 0]]
    raw_score = float(np.exp(-distances / 4.2).mean())
    coverage = min(1.0, float(inside.sum()) / max(minimum_samples * 2.0, len(pitch_points) * 0.24))
    span_factor = min(1.0, span / (68.0 if kind == "curve" else 90.0))
    return [(kind, raw_score * (0.55 + 0.45 * min(coverage, span_factor)))]


def calibrate_pitch(
    image: np.ndarray,
    ground_points: np.ndarray | None = None,
    *,
    max_quad_candidates: int = 3000,
    deadline: float | None = None,
    diagnostics: dict | None = None,
) -> PitchCalibration | None:
    quad_candidate_limit = max(1, int(max_quad_candidates))
    candidate_pool_limit = quad_candidate_limit * 12
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "budgetExhausted": False,
                "deadlineExceeded": False,
                "candidateLimitReached": False,
                "candidatePoolLimitReached": False,
                "candidateEvaluationLimitReached": False,
                "quadCandidateLimit": quad_candidate_limit,
                "candidatePoolLimit": candidate_pool_limit,
                "quadCandidatesGenerated": 0,
                "quadCandidatesEvaluated": 0,
            }
        )

    deadline_exceeded = False
    candidate_pool_limit_reached = False

    def past_deadline() -> bool:
        nonlocal deadline_exceeded
        if deadline is not None and monotonic() >= deadline:
            deadline_exceeded = True
            return True
        return False

    mask = pitch_line_mask(image)
    families = _orientation_families(mask)
    if families is None:
        return None
    first_lines = _candidate_lines(mask, families[0], families[1])
    second_lines = _candidate_lines(mask, families[1], families[0])
    if len(first_lines) < 2 or len(second_lines) < 2:
        return None
    first_span = float(np.ptp([line.rho for line in first_lines]))
    second_span = float(np.ptp([line.rho for line in second_lines]))
    first_pairs = _candidate_pairs(first_lines, 1800 if first_span > 500 else 260)
    second_pairs = _candidate_pairs(second_lines, 1800 if second_span > 500 else 260)
    if not first_pairs or not second_pairs:
        return None

    inverted = cv2.bitwise_not(mask)
    distance_map = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
    height, width = mask.shape
    best: PitchCalibration | None = None
    quad_candidates: list[tuple[float, np.ndarray]] = []
    for first_pair in first_pairs:
        if past_deadline():
            break
        if len(quad_candidates) >= candidate_pool_limit:
            candidate_pool_limit_reached = True
            break
        for second_pair in second_pairs:
            if past_deadline():
                break
            if len(quad_candidates) >= candidate_pool_limit:
                candidate_pool_limit_reached = True
                break
            image_points = _quad_points(first_pair, second_pair)
            if image_points is None or not _valid_quad(image_points, width, height):
                continue
            area = float(cv2.contourArea(cv2.convexHull(image_points)))
            vertical_span = float(np.ptp(image_points[:, 1]))
            if area < width * height * 0.035 or vertical_span < height * 0.15:
                continue
            evidence = sum(np.log1p(line.support) for line in (*first_pair, *second_pair))
            quad_candidates.append((float(evidence + np.log1p(area) * 0.18), image_points))
    quad_candidates.sort(key=lambda item: item[0], reverse=True)
    evaluated = 0
    for _, image_points in quad_candidates[:quad_candidate_limit]:
        if past_deadline():
            break
        evaluated += 1
        for rectangle_name, x_values, z_values, expected_curve in RECTANGLES:
            if past_deadline():
                break
            for swap_x in (False, True):
                if past_deadline():
                    break
                for swap_z in (False, True):
                    if past_deadline():
                        break
                    xs = x_values[::-1] if swap_x else x_values
                    zs = z_values[::-1] if swap_z else z_values
                    base_pitch_points = np.float32(
                        [[xs[0], zs[0]], [xs[0], zs[1]], [xs[1], zs[0]], [xs[1], zs[1]]]
                    )
                    for transpose_axes in (False, True):
                        if past_deadline():
                            break
                        pitch_points = (
                            base_pitch_points[[0, 2, 1, 3]] if transpose_axes else base_pitch_points
                        )
                        homography = cv2.getPerspectiveTransform(image_points, pitch_points)
                        if not _plausible_camera(homography, image_points, height):
                            continue
                        player_fit = 1.0
                        if ground_points is not None and len(ground_points) >= 4:
                            projected_players = project_points(
                                np.asarray(ground_points, dtype=np.float64), homography
                            )
                            inside = (
                                np.isfinite(projected_players).all(axis=1)
                                & (np.abs(projected_players[:, 0]) <= 53.5)
                                & (np.abs(projected_players[:, 1]) <= 35.0)
                            )
                            player_fit = float(inside.mean())
                            if player_fit < 0.68:
                                continue
                            fitted_players = projected_players[inside]
                            if len(fitted_players) >= 8:
                                spread = np.ptp(fitted_players, axis=0)
                                if float(spread[0]) < 18.0 or float(spread[1]) < 16.0:
                                    continue
                        confidence, supported, mean_score, matched_curves = _score_homography(
                            homography, distance_map, expected_curve
                        )
                        confidence *= 0.86 + player_fit * 0.14
                        if supported < 4 or matched_curves < 1 or confidence < 0.50:
                            continue
                        candidate = PitchCalibration(
                            image_to_pitch=homography,
                            confidence=confidence,
                            supported_lines=supported,
                            mean_line_score=mean_score,
                            rectangle=rectangle_name,
                            matched_curves=matched_curves,
                        )
                        if best is None or candidate.confidence > best.confidence:
                            best = candidate
    candidate_evaluation_limit_reached = (
        not deadline_exceeded
        and evaluated >= quad_candidate_limit
        and len(quad_candidates) > evaluated
    )
    candidate_limit_reached = (
        candidate_pool_limit_reached or candidate_evaluation_limit_reached
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "budgetExhausted": deadline_exceeded or candidate_limit_reached,
                "deadlineExceeded": deadline_exceeded,
                "candidateLimitReached": candidate_limit_reached,
                "candidatePoolLimitReached": candidate_pool_limit_reached,
                "candidateEvaluationLimitReached": candidate_evaluation_limit_reached,
                "quadCandidatesGenerated": len(quad_candidates),
                "quadCandidatesEvaluated": evaluated,
            }
        )
    return canonicalize_penalty_side(best, width) if best is not None else None

