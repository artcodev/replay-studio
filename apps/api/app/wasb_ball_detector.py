from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from .ball_detection_contract import (
    BallDetectionBatch,
    BallDetectionError,
    BallDetector,
    BallDetectorConfigurationError,
    BallDetectorUnavailable,
    FailurePolicy,
    FrameInput,
)
from .wasb_ball_protocol import (
    WasbMultipartRequest,
    build_wasb_multipart_request,
    parse_wasb_target_response,
)
from .wasb_ball_transport import wasb_http_transport


WasbServiceTransport = Callable[
    [str, WasbMultipartRequest, float],
    Mapping[str, Any],
]


class WasbServiceBallDetector:
    """Canonical multipart client for the isolated WASB-SBDT worker."""

    backend_name = "wasb-service"

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        max_candidates: int = 12,
        nms_iou: float = 0.1,
        failure_policy: FailurePolicy = "raise",
        fallback: BallDetector | None = None,
        transport: WasbServiceTransport | None = None,
    ) -> None:
        parsed_url = urlsplit(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise BallDetectorConfigurationError(
                "WASB service URL must use http or https"
            )
        if (
            parsed_url.path.rstrip("/") != "/v1/detections"
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise BallDetectorConfigurationError(
                "WASB service URL must target /v1/detections"
            )
        if timeout <= 0:
            raise BallDetectorConfigurationError("WASB timeout must be positive")
        if max_candidates <= 0:
            raise BallDetectorConfigurationError(
                "max_candidates must be positive"
            )
        if not 0.0 <= nms_iou <= 1.0:
            raise BallDetectorConfigurationError(
                "nms_iou must be between 0 and 1"
            )
        if failure_policy not in ("raise", "fallback"):
            raise BallDetectorConfigurationError(
                "failure_policy must be raise or fallback"
            )
        if failure_policy == "fallback" and fallback is None:
            raise BallDetectorConfigurationError(
                "failure_policy=fallback requires an explicit fallback detector"
            )
        self._url = url
        self._timeout = timeout
        self._max_candidates = max_candidates
        self._nms_iou = nms_iou
        self._failure_policy = failure_policy
        self._fallback = fallback
        self._transport = transport or wasb_http_transport

    def detect(
        self,
        frame: FrameInput,
        *,
        frame_index: int | None = None,
        timestamp: float | None = None,
        context_frames: Sequence[FrameInput] = (),
    ) -> BallDetectionBatch:
        try:
            request = build_wasb_multipart_request(
                frame,
                context_frames,
                frame_index=frame_index,
                timestamp=timestamp,
                max_candidates=self._max_candidates,
            )
            response = self._transport(self._url, request, self._timeout)
            target = parse_wasb_target_response(
                response,
                request=request,
                backend_name=self.backend_name,
                max_candidates=self._max_candidates,
                nms_iou=self._nms_iou,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            return BallDetectionBatch(
                candidates=target.candidates,
                image_size=target.image_size,
                backend=self.backend_name,
                metadata={
                    "temporalContextFrames": len(context_frames),
                    "temporalContextMode": request.context_mode,
                    "worker": dict(target.worker_metadata),
                },
            )
        except Exception as exc:
            if self._failure_policy == "fallback" and self._fallback is not None:
                fallback_result = self._fallback.detect(
                    frame,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    context_frames=context_frames,
                )
                return BallDetectionBatch(
                    candidates=fallback_result.candidates,
                    image_size=fallback_result.image_size,
                    backend=fallback_result.backend,
                    metadata={
                        **dict(fallback_result.metadata),
                        "requestedBackend": self.backend_name,
                        "fallbackReason": f"{type(exc).__name__}: {exc}",
                    },
                )
            if isinstance(exc, BallDetectionError):
                raise
            raise BallDetectorUnavailable(
                f"{self.backend_name} failed: {exc}"
            ) from exc
