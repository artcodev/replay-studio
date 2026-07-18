from __future__ import annotations

from threading import Lock
from typing import Sequence

import numpy as np

from .provider_contract import (
    BallCandidate,
    BallProviderInfo,
    ProviderUnavailable,
)
from .wasb_configuration import (
    BACKEND_NAME,
    FRAMES_IN,
    FRAMES_OUT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    WasbConfiguration,
)
from .wasb_geometry import affine_transforms, component_candidates
from .wasb_model_loading import LoadedWasbModel, load_wasb_model


class WasbSoccerProvider:
    """Thread-safe inference runtime for the pinned WASB soccer model."""

    backend = BACKEND_NAME
    frames_in = FRAMES_IN
    frames_out = FRAMES_OUT

    def __init__(self, configuration: WasbConfiguration | None = None) -> None:
        self.configuration = configuration or WasbConfiguration.from_environment()
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._runtime: LoadedWasbModel | None = None

    @property
    def loaded(self) -> bool:
        return self._runtime is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            self._runtime = load_wasb_model(self.configuration)

    @property
    def model_version(self) -> str:
        digest = (
            self._runtime.checkpoint_sha256
            if self._runtime is not None
            else self.configuration.expected_sha256
        )
        return f"wasb-soccer@sha256:{digest[:12]}"

    def info(self) -> BallProviderInfo:
        return BallProviderInfo(
            backend=self.backend,
            model_version=self.model_version,
            checkpoint_sha256=(
                self._runtime.checkpoint_sha256 if self._runtime is not None else None
            ),
            device=self.configuration.device_name,
            frames_in=self.frames_in,
            frames_out=self.frames_out,
            input_size=(INPUT_WIDTH, INPUT_HEIGHT),
            score_threshold=self.configuration.score_threshold,
            model_load_seconds=(
                self._runtime.load_seconds if self._runtime is not None else None
            ),
        )

    def detect_window(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        max_candidates: int,
    ) -> list[list[BallCandidate]]:
        if not self.loaded:
            self.load()
        if len(frames_rgb) != self.frames_in:
            raise ProviderUnavailable(
                f"WASB requires exactly {self.frames_in} temporal frames, "
                f"received {len(frames_rgb)}"
            )
        if max_candidates <= 0:
            raise ProviderUnavailable("max_candidates must be positive")
        first = np.asarray(frames_rgb[0])
        if first.ndim != 3 or first.shape[2] != 3:
            raise ProviderUnavailable(
                "WASB frames must be RGB arrays with three channels"
            )
        image_height, image_width = first.shape[:2]
        if image_width <= 0 or image_height <= 0:
            raise ProviderUnavailable("WASB received an empty frame")
        for frame in frames_rgb:
            if np.asarray(frame).shape != first.shape:
                raise ProviderUnavailable(
                    "All frames in a WASB temporal window must have one size"
                )

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - image declares OpenCV
            raise ProviderUnavailable(f"OpenCV runtime is unavailable: {exc}") from exc

        forward_affine, inverse_affine = affine_transforms(image_width, image_height)
        means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        deviations = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        tensors: list[np.ndarray] = []
        for frame in frames_rgb:
            warped = cv2.warpAffine(
                np.asarray(frame, dtype=np.uint8),
                forward_affine,
                (INPUT_WIDTH, INPUT_HEIGHT),
                flags=cv2.INTER_LINEAR,
            )
            channels_first = np.transpose(
                warped.astype(np.float32) / 255.0,
                (2, 0, 1),
            )
            tensors.append((channels_first - means) / deviations)
        input_array = np.concatenate(tensors, axis=0)[None, ...]

        runtime = self._runtime
        if runtime is None:  # Defensive guard for type narrowing and failed loads.
            raise ProviderUnavailable("WASB runtime did not become ready")
        with self._inference_lock, runtime.torch.inference_mode():
            tensor = runtime.torch.from_numpy(input_array).to(runtime.device)
            outputs = runtime.model(tensor)
            logits = outputs[0] if isinstance(outputs, dict) else None
            if logits is None or logits.ndim != 4 or logits.shape[1] != self.frames_out:
                raise ProviderUnavailable(
                    "WASB model returned an unexpected heatmap shape"
                )
            heatmaps = runtime.torch.sigmoid(logits)[0].detach().cpu().numpy()

        return [
            component_candidates(
                heatmaps[frame_offset],
                inverse_affine,
                (image_width, image_height),
                score_threshold=self.configuration.score_threshold,
                max_candidates=max_candidates,
            )
            for frame_offset in range(self.frames_out)
        ]
