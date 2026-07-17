from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, cos, degrees, exp, hypot, pi, sin
from time import monotonic

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchCalibration:
    image_to_pitch: np.ndarray
    confidence: float
    supported_lines: int
    mean_line_score: float
    rectangle: str
    matched_curves: int = 0
    method: str = "pitch-lines-ransac"
    keypoint_count: int = 0
    inlier_count: int = 0
    reprojection_error: float | None = None
    frame_index: int | None = None
    detected_keypoint_count: int = 0
    completed_keypoint_count: int = 0
    inlier_ratio: float | None = None
    reprojection_p95: float | None = None
    raw_line_count: int = 0
    ground_error_p50: float | None = None
    ground_error_p95: float | None = None
    raw_keypoints: tuple[dict, ...] = ()
    raw_lines: tuple[dict, ...] = ()
    confidence_kind: str = "heuristic-quality-score"
    backend_diagnostics: dict | None = None

    def as_dict(self) -> dict:
        return {
            "status": "ready",
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "supportedLines": self.supported_lines,
            "matchedCurves": self.matched_curves,
            "meanLineScore": round(self.mean_line_score, 3),
            "rectangle": self.rectangle,
            "pitchSide": pitch_side(self.rectangle),
            "keypointCount": self.keypoint_count,
            "inlierCount": self.inlier_count,
            "reprojectionError": (
                round(self.reprojection_error, 3)
                if self.reprojection_error is not None
                else None
            ),
            "frameIndex": self.frame_index,
            "detectedKeypointCount": self.detected_keypoint_count,
            "completedKeypointCount": self.completed_keypoint_count,
            "inlierRatio": (
                round(self.inlier_ratio, 5) if self.inlier_ratio is not None else None
            ),
            "reprojectionP95": (
                round(self.reprojection_p95, 3) if self.reprojection_p95 is not None else None
            ),
            "rawLineCount": self.raw_line_count,
            "groundErrorP50Metres": (
                round(self.ground_error_p50, 4) if self.ground_error_p50 is not None else None
            ),
            "groundErrorP95Metres": (
                round(self.ground_error_p95, 4) if self.ground_error_p95 is not None else None
            ),
            "rawKeypoints": [dict(item) for item in self.raw_keypoints],
            "rawLines": [dict(item) for item in self.raw_lines],
            "confidenceKind": self.confidence_kind,
            "backendDiagnostics": self.backend_diagnostics,
            "imageToPitch": [[round(float(value), 8) for value in row] for row in self.image_to_pitch],
        }


@dataclass(frozen=True)
class CalibrationAlignmentMetrics:
    """Bidirectional image-space agreement between field evidence and a camera fit.

    Precision asks whether projected model markings are supported by observed
    white field pixels. Recall asks whether the observed field pixels are
    explained by the projected model. The old median-only score answered only
    the first question and could therefore reward a tiny, incomplete overlay.
    """

    precision: float
    recall: float
    f1: float
    residual_p50: float
    residual_p95: float
    model_sample_count: int
    observed_sample_count: int
    tolerance_pixels: float

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "f1": round(self.f1, 5),
            "residualP50": round(self.residual_p50, 3),
            "residualP95": round(self.residual_p95, 3),
            "modelSampleCount": self.model_sample_count,
            "observedSampleCount": self.observed_sample_count,
            "tolerancePixels": round(self.tolerance_pixels, 2),
        }


def pitch_side(rectangle: str | None) -> str | None:
    if rectangle and rectangle.endswith("-left"):
        return "left"
    if rectangle and rectangle.endswith("-right"):
        return "right"
    return None


def opposite_pitch_preset(preset: str) -> str:
    if preset.endswith("-left"):
        return f"{preset[:-5]}-right"
    if preset.endswith("-right"):
        return f"{preset[:-6]}-left"
    return preset


def flip_pitch_calibration(calibration: PitchCalibration) -> PitchCalibration:
    pitch_flip = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PitchCalibration(
        image_to_pitch=pitch_flip @ calibration.image_to_pitch,
        confidence=calibration.confidence,
        supported_lines=calibration.supported_lines,
        mean_line_score=calibration.mean_line_score,
        rectangle=opposite_pitch_preset(calibration.rectangle),
        matched_curves=calibration.matched_curves,
        method=calibration.method,
        keypoint_count=calibration.keypoint_count,
        inlier_count=calibration.inlier_count,
        reprojection_error=calibration.reprojection_error,
        frame_index=calibration.frame_index,
        detected_keypoint_count=calibration.detected_keypoint_count,
        completed_keypoint_count=calibration.completed_keypoint_count,
        inlier_ratio=calibration.inlier_ratio,
        reprojection_p95=calibration.reprojection_p95,
        raw_line_count=calibration.raw_line_count,
        ground_error_p50=calibration.ground_error_p50,
        ground_error_p95=calibration.ground_error_p95,
        raw_keypoints=calibration.raw_keypoints,
        raw_lines=calibration.raw_lines,
        confidence_kind=calibration.confidence_kind,
        backend_diagnostics=calibration.backend_diagnostics,
    )


def canonicalize_penalty_side(
    calibration: PitchCalibration,
    image_width: int,
) -> PitchCalibration:
    """Keep a screen-left goal on the left pitch half and screen-right on the right.

    Penalty and goal-area markings are mirror-symmetric, so line scoring alone
    cannot distinguish pitch halves. Use the fitted landmark's image location to
    resolve the sign of pitch X. Attack direction remains an explicit scene setting.
    """
    side = pitch_side(calibration.rectangle)
    if side is None or not calibration.rectangle.startswith(("penalty-area-", "goal-area-")):
        return calibration
    center_magnitude = 49.75 if calibration.rectangle.startswith("goal-area-") else 44.25
    pitch_center_x = -center_magnitude if side == "left" else center_magnitude
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return calibration
    image_center = _project(np.array([[pitch_center_x, 0.0]]), pitch_to_image)[0]
    if not np.isfinite(image_center).all():
        return calibration
    offset = float(image_center[0]) - image_width / 2
    if abs(offset) < image_width * 0.04:
        return calibration
    expected_right = offset > 0
    current_right = side == "right"
    if expected_right == current_right:
        return calibration
    return flip_pitch_calibration(calibration)


@dataclass(frozen=True)
class _ImageLine:
    rho: float
    theta: float
    support: int

    @property
    def orientation(self) -> float:
        return (degrees(self.theta) + 90.0) % 180.0


PITCH_LINES = [
    ("touch-top", (-52.5, -34.0), (52.5, -34.0)),
    ("touch-bottom", (-52.5, 34.0), (52.5, 34.0)),
    ("goal-left", (-52.5, -34.0), (-52.5, 34.0)),
    ("goal-right", (52.5, -34.0), (52.5, 34.0)),
    ("halfway", (0.0, -34.0), (0.0, 34.0)),
    ("penalty-left-main", (-36.0, -20.16), (-36.0, 20.16)),
    ("penalty-left-top", (-52.5, -20.16), (-36.0, -20.16)),
    ("penalty-left-bottom", (-52.5, 20.16), (-36.0, 20.16)),
    ("penalty-right-main", (36.0, -20.16), (36.0, 20.16)),
    ("penalty-right-top", (36.0, -20.16), (52.5, -20.16)),
    ("penalty-right-bottom", (36.0, 20.16), (52.5, 20.16)),
    ("goal-area-left-main", (-47.0, -9.16), (-47.0, 9.16)),
    ("goal-area-left-top", (-52.5, -9.16), (-47.0, -9.16)),
    ("goal-area-left-bottom", (-52.5, 9.16), (-47.0, 9.16)),
    ("goal-area-right-main", (47.0, -9.16), (47.0, 9.16)),
    ("goal-area-right-top", (47.0, -9.16), (52.5, -9.16)),
    ("goal-area-right-bottom", (47.0, 9.16), (52.5, 9.16)),
]

# Official PnLCalib/SoccerNet labels mapped to the same semantic markings used
# by our projected pitch model.  This enables per-class diagnostics instead of
# relying only on a noisy global white-pixel mask.
PNLCALIB_LINE_TO_PITCH_LINE = {
    "Big rect. left bottom": "penalty-left-bottom",
    "Big rect. left main": "penalty-left-main",
    "Big rect. left top": "penalty-left-top",
    "Big rect. right bottom": "penalty-right-bottom",
    "Big rect. right main": "penalty-right-main",
    "Big rect. right top": "penalty-right-top",
    "Middle line": "halfway",
    "Side line bottom": "touch-bottom",
    "Side line left": "goal-left",
    "Side line right": "goal-right",
    "Side line top": "touch-top",
    "Small rect. left bottom": "goal-area-left-bottom",
    "Small rect. left main": "goal-area-left-main",
    "Small rect. left top": "goal-area-left-top",
    "Small rect. right bottom": "goal-area-right-bottom",
    "Small rect. right main": "goal-area-right-main",
    "Small rect. right top": "goal-area-right-top",
}


def _curve_points(center_x: float, center_z: float, radius: float, side: str | None = None) -> np.ndarray:
    angles = np.linspace(0.0, pi * 2.0, 180)
    points = np.column_stack(
        [center_x + np.cos(angles) * radius, center_z + np.sin(angles) * radius]
    )
    if side == "left":
        points = points[points[:, 0] >= -36.0]
    elif side == "right":
        points = points[points[:, 0] <= 36.0]
    return points


PITCH_CURVES = [
    ("center-circle", _curve_points(0.0, 0.0, 9.15)),
    ("penalty-arc-left", _curve_points(-41.5, 0.0, 9.15, "left")),
    ("penalty-arc-right", _curve_points(41.5, 0.0, 9.15, "right")),
]


RECTANGLES = [
    ("penalty-area-right", (36.0, 52.5), (-20.16, 20.16), "penalty-arc-right"),
    ("penalty-area-left", (-52.5, -36.0), (-20.16, 20.16), "penalty-arc-left"),
]


ANCHOR_PRESETS: dict[str, list[tuple[str, str, tuple[float, float]]]] = {
    "penalty-area-right": [
        ("front-far", "Penalty front · far", (36.0, -20.16)),
        ("front-near", "Penalty front · near", (36.0, 20.16)),
        ("goal-far", "Goal line · far", (52.5, -20.16)),
        ("goal-near", "Goal line · near", (52.5, 20.16)),
    ],
    "goal-area-right": [
        ("front-far", "Goal area front · far", (47.0, -9.16)),
        ("front-near", "Goal area front · near", (47.0, 9.16)),
        ("goal-far", "Goal line · far", (52.5, -9.16)),
        ("goal-near", "Goal line · near", (52.5, 9.16)),
    ],
    "penalty-area-left": [
        ("goal-far", "Goal line · far", (-52.5, -20.16)),
        ("goal-near", "Goal line · near", (-52.5, 20.16)),
        ("front-far", "Penalty front · far", (-36.0, -20.16)),
        ("front-near", "Penalty front · near", (-36.0, 20.16)),
    ],
    "goal-area-left": [
        ("goal-far", "Goal line · far", (-52.5, -9.16)),
        ("goal-near", "Goal line · near", (-52.5, 9.16)),
        ("front-far", "Goal area front · far", (-47.0, -9.16)),
        ("front-near", "Goal area front · near", (-47.0, 9.16)),
    ],
    "center-circle": [
        ("circle-left", "Circle · left", (-9.15, 0.0)),
        ("circle-top", "Circle · far", (0.0, -9.15)),
        ("circle-right", "Circle · right", (9.15, 0.0)),
        ("circle-bottom", "Circle · near", (0.0, 9.15)),
    ],
}


def calibration_from_anchors(
    anchors: list[dict],
    preset: str,
    confidence: float = 0.9,
) -> PitchCalibration:
    if preset not in ANCHOR_PRESETS:
        raise ValueError("Unsupported pitch anchor preset")
    if len(anchors) < 4:
        raise ValueError("At least four pitch anchors are required")
    image_points = np.float64(
        [[float(anchor["image"]["x"]), float(anchor["image"]["y"])] for anchor in anchors]
    )
    pitch_points = np.float64(
        [[float(anchor["pitch"]["x"]), float(anchor["pitch"]["z"])] for anchor in anchors]
    )
    if cv2.contourArea(cv2.convexHull(np.float32(image_points))) < 80.0:
        raise ValueError("Pitch anchors are too close together")
    if cv2.contourArea(cv2.convexHull(np.float32(pitch_points))) < 8.0:
        raise ValueError("Pitch anchors do not cover a stable area of the pitch")
    method = cv2.RANSAC if len(anchors) > 4 else 0
    homography, mask = cv2.findHomography(image_points, pitch_points, method, 2.5)
    if homography is None or mask is None or not np.isfinite(homography).all():
        raise ValueError("Pitch anchors do not define a valid plane")
    determinant = float(np.linalg.det(homography))
    if abs(determinant) < 1e-10:
        raise ValueError("Pitch anchors produce a singular projection")
    homography /= homography[2, 2]
    try:
        pitch_to_image = np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Pitch anchors produce a singular inverse projection") from exc
    image_reprojected = _project(pitch_points, pitch_to_image)
    image_residuals = np.linalg.norm(image_reprojected - image_points, axis=1)
    if not np.isfinite(image_residuals).all():
        raise ValueError("Pitch anchors produce non-finite reprojection residuals")
    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 4:
        raise ValueError("Pitch anchors do not contain four geometric inliers")
    rectangle = preset if preset in {"penalty-area-left", "penalty-area-right"} else preset
    return PitchCalibration(
        image_to_pitch=homography,
        confidence=max(0.0, min(1.0, confidence)),
        supported_lines=len(anchors),
        mean_line_score=0.0,
        rectangle=rectangle,
        matched_curves=1 if preset == "center-circle" else 0,
        keypoint_count=len(anchors),
        inlier_count=inlier_count,
        reprojection_error=float(np.median(image_residuals[inlier_mask])),
    )


def projected_pitch_markings(
    calibration: PitchCalibration,
    width: int,
    height: int,
) -> list[dict]:
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return []
    sources: list[tuple[str, str, np.ndarray]] = []
    for name, start, end in PITCH_LINES:
        alpha = np.linspace(0.0, 1.0, 90)
        points = np.column_stack(
            [start[0] + (end[0] - start[0]) * alpha, start[1] + (end[1] - start[1]) * alpha]
        )
        sources.append((name, "line", points))
    sources.extend((name, "curve", points) for name, points in PITCH_CURVES)

    markings = []
    for name, kind, pitch_points in sources:
        image_points = _project(pitch_points, pitch_to_image)
        valid = (
            np.isfinite(image_points).all(axis=1)
            & (image_points[:, 0] > -width * 0.2)
            & (image_points[:, 0] < width * 1.2)
            & (image_points[:, 1] > -height * 0.2)
            & (image_points[:, 1] < height * 1.2)
        )
        visible = image_points[valid]
        if len(visible) < (8 if kind == "curve" else 2):
            continue
        markings.append(
            {
                "id": name,
                "kind": kind,
                "points": [
                    {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}
                    for point in visible
                ],
            }
        )
    return markings


def calibration_alignment_metrics(
    image: np.ndarray,
    calibration: PitchCalibration,
    tolerance_pixels: float = 3.0,
) -> CalibrationAlignmentMetrics | None:
    observed_mask = pitch_line_mask(image)
    height, width = observed_mask.shape
    model_mask = np.zeros_like(observed_mask)
    for marking in projected_pitch_markings(calibration, width, height):
        points = np.float32([[point["x"], point["y"]] for point in marking["points"]])
        inside = (
            np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= height * 0.16)
            & (points[:, 1] < height)
        )
        visible = points[inside].round().astype(np.int32)
        if len(visible) < (8 if marking["kind"] == "curve" else 2):
            continue
        cv2.polylines(model_mask, [visible], False, 255, 1, cv2.LINE_AA)

    # OpenCV distanceTransform measures distance to zero pixels. Normalize
    # anti-aliased raster values first; otherwise only fully opaque (`255`)
    # model pixels become zero after inversion and valid edge pixels disappear.
    model_mask = np.where(model_mask > 0, 255, 0).astype(np.uint8)
    observed_mask = np.where(observed_mask > 0, 255, 0).astype(np.uint8)
    model_pixels = model_mask > 0
    observed_pixels = observed_mask > 0
    model_count = int(model_pixels.sum())
    observed_count = int(observed_pixels.sum())
    if model_count < 20 or observed_count < 20:
        return None

    distance_to_observed = cv2.distanceTransform(
        cv2.bitwise_not(observed_mask), cv2.DIST_L2, 3
    )
    distance_to_model = cv2.distanceTransform(cv2.bitwise_not(model_mask), cv2.DIST_L2, 3)
    model_residuals = np.clip(distance_to_observed[model_pixels], 0.0, 80.0)
    observed_residuals = np.clip(distance_to_model[observed_pixels], 0.0, 80.0)
    precision = float(np.mean(model_residuals <= tolerance_pixels))
    recall = float(np.mean(observed_residuals <= tolerance_pixels))
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    return CalibrationAlignmentMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        residual_p50=float(np.median(model_residuals)),
        residual_p95=float(np.percentile(model_residuals, 95)),
        model_sample_count=model_count,
        observed_sample_count=observed_count,
        tolerance_pixels=float(tolerance_pixels),
    )


def calibration_alignment_error(image: np.ndarray, calibration: PitchCalibration) -> float | None:
    """Compatibility wrapper for callers that still display one residual."""
    metrics = calibration_alignment_metrics(image, calibration)
    return round(metrics.residual_p50, 2) if metrics is not None else None


def calibration_horizon(
    calibration: PitchCalibration,
    image_width: int,
) -> dict | None:
    """Return the image-space ground-plane horizon for calibration QA."""
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return None
    line = np.cross(pitch_to_image[:, 0], pitch_to_image[:, 1])
    if not np.isfinite(line).all() or abs(float(line[1])) < 1e-8:
        return None
    x1 = 0.0
    x2 = float(max(1, image_width - 1))
    y1 = float((-line[0] * x1 - line[2]) / line[1])
    y2 = float((-line[0] * x2 - line[2]) / line[1])
    if not np.isfinite([y1, y2]).all():
        return None
    return {
        "start": {"x": x1, "y": round(y1, 3)},
        "end": {"x": x2, "y": round(y2, 3)},
    }


def pitch_line_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 100, 255]))
    green = cv2.inRange(hsv, np.array([25, 25, 20]), np.array([105, 255, 255]))
    # Bridge painted field lines before connected-component selection. Without
    # this, a halfway/touch line can split the grass into several components and
    # the old "largest component" rule keeps only one side of the actual pitch.
    connected_green = cv2.morphologyEx(
        green,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(connected_green, 8)
    pitch_green = np.zeros_like(green)
    if component_count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pitch_green[labels == largest] = 255
    near_pitch = cv2.dilate(pitch_green, np.ones((9, 9), np.uint8))
    mask = cv2.bitwise_and(white, near_pitch)
    mask[: int(image.shape[0] * 0.18)] = 0
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


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


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    source = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = source @ homography.T
    valid = np.abs(projected[:, 2]) > 1e-8
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    output[valid] = projected[valid, :2] / projected[valid, 2:3]
    return output


def semantic_line_evidence(calibration: PitchCalibration) -> list[dict]:
    """Add an image residual to every observed PnL semantic line.

    Detector endpoints delimit only the visible part of a marking.  They are
    therefore compared with the infinite projected line, not with fixed model
    endpoints.  Goal posts and crossbars remain visible 3D evidence and are
    intentionally not scored by the grass-plane homography.
    """

    if not calibration.raw_lines:
        return []
    pitch_lines = {
        name: (np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64))
        for name, start, end in PITCH_LINES
    }
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return [dict(line) for line in calibration.raw_lines]

    result: list[dict] = []
    for raw_line in calibration.raw_lines:
        evidence = dict(raw_line)
        evidence["residualP50"] = None
        evidence["residualP95"] = None
        if evidence.get("groundPlane") is False:
            evidence["residualStatus"] = "not-scored-3d"
            result.append(evidence)
            continue
        pitch_line_name = PNLCALIB_LINE_TO_PITCH_LINE.get(str(evidence.get("name") or ""))
        pitch_segment = pitch_lines.get(pitch_line_name or "")
        start = evidence.get("start")
        end = evidence.get("end")
        if pitch_segment is None or not isinstance(start, dict) or not isinstance(end, dict):
            evidence["residualStatus"] = "not-scored"
            result.append(evidence)
            continue
        model_image = _project(np.vstack(pitch_segment), pitch_to_image)
        if not np.isfinite(model_image).all():
            evidence["residualStatus"] = "not-scored"
            result.append(evidence)
            continue
        direction = model_image[1] - model_image[0]
        denominator = hypot(float(direction[0]), float(direction[1]))
        if denominator < 1e-7:
            evidence["residualStatus"] = "not-scored"
            result.append(evidence)
            continue
        observed = np.asarray(
            [
                [float(start["x"]), float(start["y"])],
                [float(end["x"]), float(end["y"])],
            ],
            dtype=np.float64,
        )
        relative = observed - model_image[0]
        residuals = np.abs(
            direction[0] * relative[:, 1] - direction[1] * relative[:, 0]
        ) / denominator
        evidence["residualP50"] = round(float(np.median(residuals)), 3)
        evidence["residualP95"] = round(float(np.percentile(residuals, 95)), 3)
        evidence["residualStatus"] = "scored"
        result.append(evidence)
    return result


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
    image_points = _project(pitch_points, pitch_to_image)
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
                            projected_players = _project(
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


def calibration_overlay(image: np.ndarray, calibration: PitchCalibration) -> np.ndarray:
    overlay = image.copy()
    pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    palette = (80, 240, 255)
    for _, start, end in PITCH_LINES:
        alpha = np.linspace(0.0, 1.0, 80)
        pitch_points = np.column_stack(
            [start[0] + (end[0] - start[0]) * alpha, start[1] + (end[1] - start[1]) * alpha]
        )
        image_points = _project(pitch_points, pitch_to_image)
        valid = np.isfinite(image_points).all(axis=1)
        points = image_points[valid].round().astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, palette, 2, cv2.LINE_AA)
    return cv2.addWeighted(image, 0.72, overlay, 0.58, 0)
