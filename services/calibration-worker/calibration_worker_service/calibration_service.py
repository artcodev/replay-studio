from __future__ import annotations

import json
from time import perf_counter

from .calibration_contract import BACKEND_NAME, CalibrationEngineProvider
from .frame_decoder import decode_frame


class CalibrationRequestError(ValueError):
    """The multipart request cannot be mapped to a frame batch."""


class CalibrationInferenceError(RuntimeError):
    """The model runtime failed after request validation completed."""


class CalibrationService:
    def __init__(self, runtime: CalibrationEngineProvider) -> None:
        self._runtime = runtime

    def readiness(self) -> dict:
        return self._runtime.get_engine().readiness().to_wire()

    def calibrate(self, frame_indices: str, payloads: list[bytes]) -> dict:
        request_started = perf_counter()
        try:
            indices = json.loads(frame_indices)
            if not isinstance(indices, list) or len(indices) != len(payloads):
                raise ValueError
            decode_started = perf_counter()
            decoded = [
                decode_frame(int(index), payload)
                for index, payload in zip(indices, payloads)
            ]
            decode_seconds = perf_counter() - decode_started
        except (TypeError, ValueError) as exc:
            raise CalibrationRequestError(
                "frame_indices must match the uploaded frames"
            ) from exc

        try:
            acquire_started = perf_counter()
            engine = self._runtime.get_engine()
            engine_acquire_seconds = perf_counter() - acquire_started
            result = engine.calibrate(decoded)
        except Exception as exc:
            raise CalibrationInferenceError(
                f"PnLCalib inference failed: {exc}"
            ) from exc

        diagnostics = {
            **result.diagnostics.to_wire(),
            "decodeSeconds": round(decode_seconds, 6),
            "engineAcquireSeconds": round(engine_acquire_seconds, 6),
            "totalSeconds": round(perf_counter() - request_started, 6),
        }
        return {
            "backend": BACKEND_NAME,
            "requestedFrameCount": len(decoded),
            "calibratedFrameCount": len(result.frames),
            "diagnostics": diagnostics,
            "frames": [frame.to_wire() for frame in result.frames],
        }
