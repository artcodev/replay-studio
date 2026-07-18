from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .provider_contract import (
    BallCandidate,
    BallDetectionProvider,
    BallProviderInfo,
    ProviderUnavailable,
)
from .request_contract import (
    CONTRACT_VERSION,
    BallRequestError,
    decode_image,
    parse_manifest,
)
from .settings import BallWorkerSettings


class BallInferenceError(RuntimeError):
    """The temporal model failed or violated the worker contract."""


def _candidate_json(
    candidate: BallCandidate,
    *,
    provider: BallDetectionProvider,
    model_version: str,
    source_frame_index: int,
    radius: float,
) -> dict[str, Any]:
    confidence = float(np.clip(candidate.confidence, 0.0, 1.0))
    return {
        "position": [round(candidate.x, 4), round(candidate.y, 4)],
        "x": round(candidate.x, 4),
        "y": round(candidate.y, 4),
        "radius": radius,
        "confidence": round(confidence, 6),
        "backend": provider.backend,
        "modelVersion": model_version,
        "observed": True,
        "sourceFrameIndex": source_frame_index,
        "heatmapPeak": round(candidate.heatmap_peak, 6),
        "componentScore": round(candidate.component_score, 6),
        "componentArea": candidate.component_area,
        **candidate.metadata,
    }


def _provider_identity(
    provider: BallDetectionProvider,
) -> BallProviderInfo:
    info = provider.info()
    if not isinstance(info, BallProviderInfo):
        raise ProviderUnavailable("Ball provider returned an invalid status contract")
    if not info.model_version:
        raise ProviderUnavailable("Ball provider did not report modelVersion")
    return info


def _detect_sequence(
    provider: BallDetectionProvider,
    images: Sequence[np.ndarray],
    manifest_frames: Sequence[dict[str, Any]],
    *,
    max_candidates: int,
    radius: float,
) -> tuple[list[dict[str, Any]], int]:
    if provider.frames_in != provider.frames_out:
        raise ProviderUnavailable("This worker requires equal WASB framesIn and framesOut")
    window_size = provider.frames_in
    model_version = _provider_identity(provider).model_version
    responses: list[dict[str, Any]] = []
    window_count = 0
    for start in range(0, len(images), window_size):
        source_indices = [
            min(start + offset, len(images) - 1) for offset in range(window_size)
        ]
        window = [images[index] for index in source_indices]
        source_file_indices = [
            manifest_frames[index]["fileIndex"] for index in source_indices
        ]
        temporal_padding = (
            len(set(source_indices)) != len(source_indices)
            or len(set(source_file_indices)) != len(source_file_indices)
        )
        detections = provider.detect_window(window, max_candidates=max_candidates)
        if len(detections) != provider.frames_out:
            raise ProviderUnavailable(
                "Ball provider returned an incomplete temporal window"
            )
        window_count += 1
        for offset in range(min(window_size, len(images) - start)):
            sequence_index = start + offset
            frame = manifest_frames[sequence_index]
            image = images[sequence_index]
            frame_index = int(frame["frameIndex"])
            responses.append(
                {
                    **frame,
                    "imageSize": [int(image.shape[1]), int(image.shape[0])],
                    "temporalPadding": temporal_padding,
                    "candidates": [
                        _candidate_json(
                            candidate,
                            provider=provider,
                            model_version=model_version,
                            source_frame_index=frame_index,
                            radius=radius,
                        )
                        for candidate in detections[offset]
                    ],
                }
            )
    return responses, window_count


class BallDetectionService:
    """Decode temporal contracts and coordinate WASB provider inference."""

    def __init__(
        self,
        provider: BallDetectionProvider,
        settings: BallWorkerSettings,
    ) -> None:
        self.provider = provider
        self.settings = settings

    def _inference(self, call):
        try:
            return call()
        except ProviderUnavailable as exc:
            raise BallInferenceError(str(exc)) from exc
        except Exception as exc:
            raise BallInferenceError(
                f"WASB inference failed: {type(exc).__name__}: {exc}"
            ) from exc

    def batch_detections(
        self,
        frame_bytes: list[bytes],
        manifest: str,
    ) -> dict[str, Any]:
        manifest_frames, max_candidates, target_index = parse_manifest(
            manifest,
            len(frame_bytes),
            self.settings.max_candidates_default,
        )
        if len(manifest_frames) > self.settings.max_batch_frames:
            raise BallRequestError(
                413, f"At most {self.settings.max_batch_frames} frames are allowed"
            )
        decoded_by_file_index = [
            decode_image(
                data,
                label=f"frames[{file_index}]",
                max_bytes=self.settings.max_frame_bytes,
                max_pixels=self.settings.max_frame_pixels,
            )
            for file_index, data in enumerate(frame_bytes)
        ]
        images = [
            decoded_by_file_index[item["fileIndex"]] for item in manifest_frames
        ]
        result_frames, window_count = self._inference(
            lambda: _detect_sequence(
                self.provider,
                images,
                manifest_frames,
                max_candidates=max_candidates,
                radius=self.settings.candidate_radius,
            )
        )
        info = self._inference(lambda: _provider_identity(self.provider))
        return {
            "contractVersion": CONTRACT_VERSION,
            "backend": self.provider.backend,
            "modelVersion": info.model_version,
            "frames": result_frames,
            "metadata": {
                **info.to_wire(),
                "windowCount": window_count,
                "requestedFrameCount": len(manifest_frames),
                "targetIndex": target_index,
            },
        }
