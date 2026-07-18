from __future__ import annotations

from io import BytesIO

from PIL import Image

from calibration_worker_service.calibration_contract import (
    CalibrationBatchResult,
    CalibrationDiagnostics,
    CalibrationReadiness,
    FrameCalibration,
    InferenceTimings,
)
from calibration_worker_service.calibration_service import (
    CalibrationRequestError,
    CalibrationService,
)


def _calibration(frame_index: int) -> FrameCalibration:
    return FrameCalibration(
        frame_index=frame_index,
        confidence=0.9,
        detected_keypoint_count=6,
        completed_keypoint_count=8,
        inlier_count=6,
        inlier_ratio=1.0,
        line_count=3,
        detected_line_count=4,
        raw_lines=(),
        matched_curves=0,
        completed_curve_count=0,
        reprojection_error=1.0,
        ground_error_p50_metres=0.1,
        ground_error_p95_metres=0.2,
        pitch_side="right",
        raw_keypoints=(),
        image_to_pitch=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )


class FakeEngine:
    def __init__(self) -> None:
        self.received = []

    def readiness(self) -> CalibrationReadiness:
        return CalibrationReadiness("cpu", 2, "fake-v1", 0.1, 8, 60.0, 0)

    def calibrate(self, frames) -> CalibrationBatchResult:
        self.received = frames
        diagnostics = CalibrationDiagnostics(
            model_version="fake-v1",
            requested_frame_count=len(frames),
            unique_frame_count=len(frames),
            cache_hit_count=0,
            cache_miss_count=len(frames),
            deduplicated_frame_count=0,
            inference_batch_count=1,
            cache_entry_count=len(frames),
            lock_wait_seconds=0.0,
            inference_timings=InferenceTimings().snapshot(),
            engine_seconds=0.01,
        )
        return CalibrationBatchResult(
            frames=tuple(_calibration(frame.frame_index) for frame in frames),
            diagnostics=diagnostics,
        )


class FakeRuntime:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine

    def get_engine(self) -> FakeEngine:
        return self.engine


def _jpeg(width: int = 16, height: int = 9) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color=(30, 120, 30)).save(output, "JPEG")
    return output.getvalue()


def test_service_decodes_frames_and_serializes_engine_dtos() -> None:
    engine = FakeEngine()
    service = CalibrationService(FakeRuntime(engine))

    result = service.calibrate("[7]", [_jpeg()])

    assert result["backend"] == "pnlcalib-points-lines"
    assert result["requestedFrameCount"] == 1
    assert result["calibratedFrameCount"] == 1
    assert result["frames"][0]["frameIndex"] == 7
    assert result["frames"][0]["pitchSide"] == "right"
    assert result["diagnostics"]["modelVersion"] == "fake-v1"
    assert engine.received[0].width == 16
    assert engine.received[0].height == 9


def test_service_rejects_mismatched_frame_indices_before_loading_engine() -> None:
    engine = FakeEngine()
    service = CalibrationService(FakeRuntime(engine))

    try:
        service.calibrate("[]", [_jpeg()])
    except CalibrationRequestError as exc:
        assert str(exc) == "frame_indices must match the uploaded frames"
    else:
        raise AssertionError("mismatched frame metadata must fail")

    assert engine.received == []
