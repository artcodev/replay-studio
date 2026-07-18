from __future__ import annotations

"""Pure diagnostic payloads and ranking for pitch-calibration observations."""

from copy import deepcopy
from math import hypot

import numpy as np

from .pitch_calibration_contract import PitchCalibration, pitch_side
from .pitch_calibration_quality import semantic_line_evidence
from .pitch_geometry import calibration_horizon, projected_pitch_markings
from .reconstruction_frame_calibration_quality import direct_calibration_qa
from .reconstruction_person_detection_contract import Detection


def matrix_payload(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


def keypoint_evidence(calibration: PitchCalibration) -> list[dict]:
    try:
        pitch_to_image = np.linalg.inv(calibration.image_to_pitch)
    except np.linalg.LinAlgError:
        return [dict(item) for item in calibration.raw_keypoints]
    result: list[dict] = []
    for raw in calibration.raw_keypoints:
        item = deepcopy(raw)
        image_point = item.get("image") or {}
        pitch_point = item.get("pitch") or {}
        try:
            projected = pitch_to_image @ np.array(
                [float(pitch_point["x"]), float(pitch_point["z"]), 1.0],
                dtype=np.float64,
            )
            if abs(float(projected[2])) < 1e-8:
                raise ValueError
            projected_x = float(projected[0] / projected[2])
            projected_y = float(projected[1] / projected[2])
            observed_x = float(image_point["x"])
            observed_y = float(image_point["y"])
            dx = projected_x - observed_x
            dy = projected_y - observed_y
            item["projectedImage"] = {
                "x": round(projected_x, 3),
                "y": round(projected_y, 3),
            }
            item["residualVector"] = {
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "magnitude": round(hypot(dx, dy), 3),
            }
        except (KeyError, TypeError, ValueError):
            item["projectedImage"] = None
            item["residualVector"] = None
        result.append(item)
    return result


def frame_calibration_evidence(
    scene: dict,
    sample_index: int,
    scene_time: float,
    image: np.ndarray,
    calibration: PitchCalibration | None,
    *,
    projection_source: str,
    people: list[Detection] | None = None,
    pitch: dict | None = None,
    source_frame_index: int,
    manual: bool = False,
) -> dict:
    source_frame_index = int(source_frame_index)
    height, width = image.shape[:2]
    source_start = float(
        scene.get("payload", {}).get("videoAsset", {}).get("sourceStart") or 0.0
    )
    if calibration is None:
        return {
            "sourceFrameIndex": source_frame_index,
            "sampleIndex": sample_index,
            "sceneTime": round(float(scene_time), 3),
            "sourceTime": round(source_start + float(scene_time), 3),
            "frameWidth": width,
            "frameHeight": height,
            "status": "missing",
            "solutionStatus": "unresolved",
            "source": "none",
            "projectionSource": "none",
            "backend": None,
            "confidence": None,
            "confidenceKind": None,
            "imageToPitch": None,
            "keypointCount": 0,
            "detectedKeypointCount": 0,
            "completedKeypointCount": 0,
            "inlierCount": 0,
            "inlierRatio": None,
            "rawLineCount": 0,
            "rawKeypoints": [],
            "keypoints": [],
            "reprojectionError": None,
            "reprojectionP95": None,
            "groundErrorP50Metres": None,
            "groundErrorP95Metres": None,
            "visiblePitchSide": None,
            "rectangle": None,
            "alignmentMetrics": None,
            "horizon": None,
            "markings": [],
            "rejectionReasons": ["no-automatic-calibration-candidate"],
            "personSupport": None,
            "qualityGates": [],
        }

    qa = direct_calibration_qa(
        image,
        calibration,
        people=people,
        pitch=pitch,
        manual=manual,
    )
    payload = calibration.as_dict()
    alignment = qa["alignmentMetrics"]
    accepted = not qa["rejectionReasons"]
    frame_reprojection_error = (
        alignment.get("residualP50")
        if alignment is not None
        else payload.get("reprojectionError")
    )
    frame_reprojection_p95 = (
        alignment.get("residualP95")
        if alignment is not None
        else payload.get("reprojectionP95") or payload.get("reprojectionError")
    )
    return {
        "sourceFrameIndex": source_frame_index,
        "sampleIndex": sample_index,
        "sceneTime": round(float(scene_time), 3),
        "sourceTime": round(source_start + float(scene_time), 3),
        "frameWidth": width,
        "frameHeight": height,
        "status": "accepted" if accepted else "rejected",
        "solutionStatus": "direct-accepted" if accepted else "direct-rejected",
        "source": calibration.method,
        "projectionSource": projection_source,
        "backend": calibration.method,
        "confidence": round(float(calibration.confidence), 3),
        "confidenceKind": payload.get("confidenceKind"),
        "backendDiagnostics": deepcopy(payload.get("backendDiagnostics")),
        "imageToPitch": matrix_payload(calibration.image_to_pitch),
        "keypointCount": payload.get("keypointCount", 0),
        "detectedKeypointCount": payload.get("detectedKeypointCount", 0),
        "completedKeypointCount": payload.get("completedKeypointCount", 0),
        "inlierCount": payload.get("inlierCount", 0),
        "inlierRatio": payload.get("inlierRatio"),
        "rawLineCount": payload.get("rawLineCount", 0),
        "rawKeypoints": payload.get("rawKeypoints", []),
        "rawLines": semantic_line_evidence(calibration),
        "keypoints": keypoint_evidence(calibration),
        "reprojectionError": frame_reprojection_error,
        "reprojectionP95": frame_reprojection_p95,
        "groundErrorP50Metres": payload.get("groundErrorP50Metres"),
        "groundErrorP95Metres": payload.get("groundErrorP95Metres"),
        "visiblePitchSide": pitch_side(calibration.rectangle),
        "rectangle": calibration.rectangle,
        "alignmentMetrics": alignment,
        "horizon": calibration_horizon(calibration, width),
        "markings": projected_pitch_markings(calibration, width, height),
        "rejectionReasons": qa["rejectionReasons"],
        "personSupport": qa["personSupport"],
        "qualityGates": qa["qualityGates"],
    }


def calibration_attempt_payload(evidence: dict) -> dict:
    return {
        "backend": evidence.get("backend"),
        "status": evidence.get("status"),
        "confidence": evidence.get("confidence"),
        "reprojectionError": evidence.get("reprojectionError"),
        "reprojectionP95": evidence.get("reprojectionP95"),
        "visiblePitchSide": evidence.get("visiblePitchSide"),
        "rejectionReasons": list(evidence.get("rejectionReasons") or []),
        "backendDiagnostics": deepcopy(evidence.get("backendDiagnostics")),
    }


def calibration_backend_rank(evidence: dict) -> float:
    backend = str(evidence.get("backend") or evidence.get("source") or "")
    if backend.startswith("pnlcalib"):
        return 3.0
    if "keypoint" in backend:
        return 2.0
    if backend == "pitch-lines-ransac":
        return 0.0
    return 1.0


def calibration_evidence_rank(evidence: dict) -> tuple[float, float, float, float, float]:
    alignment = evidence.get("alignmentMetrics") or {}
    residual = evidence.get("reprojectionP95")
    return (
        1.0 if evidence.get("status") == "accepted" else 0.0,
        calibration_backend_rank(evidence),
        float(alignment.get("f1") or 0.0),
        float(evidence.get("confidence") or 0.0),
        -float(residual) if residual is not None else -1e9,
    )
