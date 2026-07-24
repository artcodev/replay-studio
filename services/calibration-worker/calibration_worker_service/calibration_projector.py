from __future__ import annotations

from math import exp
from typing import Any

import numpy as np

from .calibration_contract import (
    DecodedFrame,
    FrameCalibration,
    RawKeypointObservation,
    RawLineObservation,
)
from .pnlcalib_constants import (
    GOAL_FRAME_LINE_IDS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SEMANTIC_LINE_NAMES,
)
from .pnlcalib_runtime import PnLCalibRuntime


class CalibrationProjector:
    """Turn PnLCalib semantic evidence into a quality-gated homography."""

    def __init__(self, runtime: PnLCalibRuntime) -> None:
        self._complete_keypoints = runtime.complete_keypoints
        self._frame_calibrator_factory = runtime.frame_calibrator_factory

    @staticmethod
    def _pitch_side(keypoints: dict[int, dict]) -> str | None:
        visible = [
            (float(item.get("xw", item.get("x"))), float(item.get("p", 1.0)))
            for item in keypoints.values()
            if ("xw" in item or "x" in item) and float(item.get("p", 1.0)) > 0.0
        ]
        if not visible:
            return None
        weighted_x = sum(x * weight for x, weight in visible) / sum(
            weight for _, weight in visible
        )
        if abs(weighted_x) < 4.0:
            return None
        return "left" if weighted_x < 0.0 else "right"

    @staticmethod
    def _original_homography(
        homography: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        original_to_resized = np.array(
            [
                [INPUT_WIDTH / width, 0.0, 0.0],
                [0.0, INPUT_HEIGHT / height, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = np.asarray(homography, dtype=np.float64) @ original_to_resized
        result /= result[2, 2]
        return result

    @staticmethod
    def _raw_line_evidence(
        frame: DecodedFrame,
        detected_lines: dict[int, dict],
    ) -> tuple[RawLineObservation, ...]:
        """Snapshot source evidence before PnLCalib mutates line dictionaries."""

        scale_x = frame.width / INPUT_WIDTH
        scale_y = frame.height / INPUT_HEIGHT
        evidence: list[RawLineObservation] = []
        for line_id, item in sorted(detected_lines.items()):
            if not 1 <= int(line_id) <= len(SEMANTIC_LINE_NAMES):
                continue
            evidence.append(
                RawLineObservation(
                    line_id=int(line_id),
                    name=SEMANTIC_LINE_NAMES[int(line_id) - 1],
                    start_x=round(float(item["x_1"]) * scale_x, 3),
                    start_y=round(float(item["y_1"]) * scale_y, 3),
                    end_x=round(float(item["x_2"]) * scale_x, 3),
                    end_y=round(float(item["y_2"]) * scale_y, 3),
                    confidence=round(
                        min(
                            float(item.get("p_1", 0.0)),
                            float(item.get("p_2", 0.0)),
                        ),
                        5,
                    ),
                    ground_plane=int(line_id) not in GOAL_FRAME_LINE_IDS,
                )
            )
        return tuple(evidence)

    def project(
        self,
        frame: DecodedFrame,
        keypoints: dict[int, dict],
        detected_lines: dict[int, dict],
    ) -> FrameCalibration | None:
        raw_lines = self._raw_line_evidence(frame, detected_lines)
        ground_lines = {
            key: value
            for key, value in detected_lines.items()
            if key not in GOAL_FRAME_LINE_IDS
        }
        completed_keypoints, normalized_lines = self._complete_keypoints(
            keypoints,
            ground_lines,
            w=INPUT_WIDTH,
            h=INPUT_HEIGHT,
            normalize=True,
        )
        camera: Any = self._frame_calibrator_factory(
            INPUT_WIDTH,
            INPUT_HEIGHT,
            denormalize=True,
        )
        camera.update(completed_keypoints, normalized_lines)
        result = camera.heuristic_voting_ground(refine_lines=True)
        if result is None or result.get("homography") is None:
            return None

        matrix = self._original_homography(
            result["homography"],
            frame.width,
            frame.height,
        )
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
            return None
        raw_error = result.get("rep_err")
        error = float(raw_error) if raw_error is not None else 999.0
        if not np.isfinite(error):
            return None

        ground = camera.subsets.get("ground_plane") or {}
        raw_ground = {
            key: ground[key]
            for key in keypoints
            if key in ground
            and all(name in ground[key] for name in ("xi", "yi", "xw", "yw"))
        }
        detected_count = len(raw_ground)
        if detected_count < 6 or error > 18.0:
            return None

        image_points = np.array(
            [
                [float(item["xi"]), float(item["yi"]), 1.0]
                for item in raw_ground.values()
            ],
            dtype=np.float64,
        )
        world_points = np.array(
            [[float(item["xw"]), float(item["yw"])] for item in raw_ground.values()],
            dtype=np.float64,
        )
        projected = image_points @ np.asarray(result["homography"], dtype=np.float64).T
        valid = np.abs(projected[:, 2]) > 1e-8
        world_error = np.full(len(projected), np.inf, dtype=np.float64)
        world_error[valid] = np.linalg.norm(
            projected[valid, :2] / projected[valid, 2:3] - world_points[valid],
            axis=1,
        )
        try:
            pitch_to_image = np.linalg.inv(
                np.asarray(result["homography"], dtype=np.float64)
            )
            world_homogeneous = np.column_stack(
                [world_points, np.ones(len(world_points), dtype=np.float64)]
            )
            reprojected_image = world_homogeneous @ pitch_to_image.T
            image_valid = np.abs(reprojected_image[:, 2]) > 1e-8
            image_error = np.full(
                len(reprojected_image), np.inf, dtype=np.float64
            )
            image_error[image_valid] = np.linalg.norm(
                reprojected_image[image_valid, :2]
                / reprojected_image[image_valid, 2:3]
                - image_points[image_valid, :2],
                axis=1,
            )
            finite_image_error = image_error[np.isfinite(image_error)]
        except np.linalg.LinAlgError:
            finite_image_error = np.asarray([], dtype=np.float64)
        if not len(finite_image_error):
            return None
        inlier_count = int((world_error <= 1.5).sum())
        if inlier_count < 6 or inlier_count / detected_count < 0.65:
            return None

        inlier_ratio = inlier_count / detected_count
        finite_world_error = world_error[np.isfinite(world_error)]
        confidence = float(
            np.clip(
                0.25 * min(1.0, detected_count / 10.0)
                + 0.25 * inlier_ratio
                + 0.35 * exp(-error / 8.0)
                + 0.15 * min(1.0, len(ground_lines) / 8.0),
                0.0,
                0.99,
            )
        )
        scale_x = frame.width / INPUT_WIDTH
        scale_y = frame.height / INPUT_HEIGHT
        raw_keypoints = tuple(
            RawKeypointObservation(
                keypoint_id=int(key),
                image_x=round(float(item["xi"]) * scale_x, 3),
                image_y=round(float(item["yi"]) * scale_y, 3),
                pitch_x=round(float(item["xw"]), 4),
                pitch_z=round(float(item["yw"]), 4),
                confidence=round(float(keypoints[key].get("p", 1.0)), 5),
                inlier=bool(np.isfinite(residual) and residual <= 1.5),
                ground_residual_metres=(
                    round(float(residual), 5) if np.isfinite(residual) else None
                ),
            )
            for (key, item), residual in zip(raw_ground.items(), world_error)
        )
        return FrameCalibration(
            frame_index=frame.frame_index,
            confidence=round(confidence, 5),
            detected_keypoint_count=detected_count,
            completed_keypoint_count=len(ground),
            inlier_count=inlier_count,
            inlier_ratio=round(inlier_ratio, 5),
            line_count=len(ground_lines),
            detected_line_count=len(raw_lines),
            raw_lines=raw_lines,
            matched_curves=sum(key in keypoints for key in range(31, 58)),
            completed_curve_count=sum(
                key in completed_keypoints for key in range(31, 58)
            ),
            reprojection_error=round(error, 5),
            reprojection_p95=round(
                float(np.percentile(finite_image_error, 95)), 5
            ),
            ground_error_p50_metres=round(float(np.median(finite_world_error)), 5),
            ground_error_p95_metres=round(
                float(np.percentile(finite_world_error, 95)),
                5,
            ),
            pitch_side=self._pitch_side(raw_ground),
            raw_keypoints=raw_keypoints,
            image_to_pitch=tuple(
                tuple(round(float(value), 10) for value in row) for row in matrix
            ),
        )
