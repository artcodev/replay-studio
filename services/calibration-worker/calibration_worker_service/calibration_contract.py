from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol


BACKEND_NAME = "pnlcalib-points-lines"


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    frame_index: int
    width: int
    height: int
    tensor: Any
    content_sha256: str


@dataclass(frozen=True, slots=True)
class RawLineObservation:
    line_id: int
    name: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    confidence: float
    ground_plane: bool

    def to_wire(self) -> dict:
        return {
            "id": self.line_id,
            "name": self.name,
            "start": {"x": self.start_x, "y": self.start_y},
            "end": {"x": self.end_x, "y": self.end_y},
            "confidence": self.confidence,
            "groundPlane": self.ground_plane,
        }


@dataclass(frozen=True, slots=True)
class RawKeypointObservation:
    keypoint_id: int
    image_x: float
    image_y: float
    pitch_x: float
    pitch_z: float
    confidence: float
    inlier: bool
    ground_residual_metres: float | None

    def to_wire(self) -> dict:
        return {
            "id": self.keypoint_id,
            "image": {"x": self.image_x, "y": self.image_y},
            "pitch": {"x": self.pitch_x, "z": self.pitch_z},
            "confidence": self.confidence,
            "inlier": self.inlier,
            "groundResidualMetres": self.ground_residual_metres,
        }


@dataclass(frozen=True, slots=True)
class FrameCalibration:
    frame_index: int
    confidence: float
    detected_keypoint_count: int
    completed_keypoint_count: int
    inlier_count: int
    inlier_ratio: float
    line_count: int
    detected_line_count: int
    raw_lines: tuple[RawLineObservation, ...]
    matched_curves: int
    completed_curve_count: int
    reprojection_error: float
    ground_error_p50_metres: float
    ground_error_p95_metres: float
    pitch_side: str | None
    raw_keypoints: tuple[RawKeypointObservation, ...]
    image_to_pitch: tuple[tuple[float, float, float], ...]

    def for_frame(self, frame_index: int) -> FrameCalibration:
        return replace(self, frame_index=frame_index)

    def to_wire(self) -> dict:
        return {
            "frameIndex": self.frame_index,
            "method": BACKEND_NAME,
            "confidence": self.confidence,
            "confidenceKind": "heuristic-quality-score",
            "keypointCount": self.detected_keypoint_count,
            "detectedKeypointCount": self.detected_keypoint_count,
            "completedKeypointCount": self.completed_keypoint_count,
            "inlierCount": self.inlier_count,
            "inlierRatio": self.inlier_ratio,
            "lineCount": self.line_count,
            "detectedLineCount": self.detected_line_count,
            "rawLines": [line.to_wire() for line in self.raw_lines],
            "matchedCurves": self.matched_curves,
            "completedCurveCount": self.completed_curve_count,
            "reprojectionError": self.reprojection_error,
            "groundErrorP50Metres": self.ground_error_p50_metres,
            "groundErrorP95Metres": self.ground_error_p95_metres,
            "pitchSide": self.pitch_side,
            "rawKeypoints": [point.to_wire() for point in self.raw_keypoints],
            "imageToPitch": [list(row) for row in self.image_to_pitch],
        }


@dataclass(slots=True)
class InferenceTimings:
    tensor_assembly_seconds: float = 0.0
    keypoint_inference_seconds: float = 0.0
    line_inference_seconds: float = 0.0
    heatmap_decode_seconds: float = 0.0
    geometry_seconds: float = 0.0

    def snapshot(self) -> InferenceTimingReport:
        return InferenceTimingReport(
            tensor_assembly_seconds=self.tensor_assembly_seconds,
            keypoint_inference_seconds=self.keypoint_inference_seconds,
            line_inference_seconds=self.line_inference_seconds,
            heatmap_decode_seconds=self.heatmap_decode_seconds,
            geometry_seconds=self.geometry_seconds,
        )


@dataclass(frozen=True, slots=True)
class InferenceTimingReport:
    tensor_assembly_seconds: float
    keypoint_inference_seconds: float
    line_inference_seconds: float
    heatmap_decode_seconds: float
    geometry_seconds: float


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    model_version: str
    requested_frame_count: int
    unique_frame_count: int
    cache_hit_count: int
    cache_miss_count: int
    deduplicated_frame_count: int
    inference_batch_count: int
    cache_entry_count: int
    lock_wait_seconds: float
    inference_timings: InferenceTimingReport
    engine_seconds: float

    def to_wire(self) -> dict:
        timings = self.inference_timings
        return {
            "modelVersion": self.model_version,
            "requestedFrameCount": self.requested_frame_count,
            "uniqueFrameCount": self.unique_frame_count,
            "cacheHitCount": self.cache_hit_count,
            "cacheMissCount": self.cache_miss_count,
            "deduplicatedFrameCount": self.deduplicated_frame_count,
            "inferenceBatchCount": self.inference_batch_count,
            "cacheEntryCount": self.cache_entry_count,
            "lockWaitSeconds": round(self.lock_wait_seconds, 6),
            "tensorAssemblySeconds": round(timings.tensor_assembly_seconds, 6),
            "keypointInferenceSeconds": round(timings.keypoint_inference_seconds, 6),
            "lineInferenceSeconds": round(timings.line_inference_seconds, 6),
            "heatmapDecodeSeconds": round(timings.heatmap_decode_seconds, 6),
            "geometrySeconds": round(timings.geometry_seconds, 6),
            "modelInferenceSeconds": round(
                timings.keypoint_inference_seconds + timings.line_inference_seconds,
                6,
            ),
            "engineSeconds": round(self.engine_seconds, 6),
        }


@dataclass(frozen=True, slots=True)
class CalibrationBatchResult:
    frames: tuple[FrameCalibration, ...]
    diagnostics: CalibrationDiagnostics


@dataclass(frozen=True, slots=True)
class CalibrationReadiness:
    device: str
    batch_size: int
    model_version: str
    model_load_seconds: float
    cache_max_entries: int
    cache_ttl_seconds: float
    cache_entry_count: int

    def to_wire(self) -> dict:
        return {
            "backend": BACKEND_NAME,
            "device": self.device,
            "batchSize": self.batch_size,
            "modelVersion": self.model_version,
            "modelLoadSeconds": round(self.model_load_seconds, 3),
            "cacheMaxEntries": self.cache_max_entries,
            "cacheTtlSeconds": self.cache_ttl_seconds,
            "cacheEntryCount": self.cache_entry_count,
        }


class CalibrationEngine(Protocol):
    def calibrate(self, frames: list[DecodedFrame]) -> CalibrationBatchResult: ...

    def readiness(self) -> CalibrationReadiness: ...


class CalibrationEngineProvider(Protocol):
    def get_engine(self) -> CalibrationEngine: ...


class CalibrationBatchInference(Protocol):
    def infer(
        self,
        frames: list[DecodedFrame],
        timings: InferenceTimings,
    ) -> list[FrameCalibration | None]: ...
