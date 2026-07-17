from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from types import ModuleType
from typing import Any, Protocol, Sequence

import numpy as np


BACKEND_NAME = "wasb-sbdt-soccer"
FRAMES_IN = 3
FRAMES_OUT = 3
INPUT_WIDTH = 512
INPUT_HEIGHT = 288
CHECKPOINT_SHA256 = "d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c"


class ProviderUnavailable(RuntimeError):
    """The configured WASB runtime or verified soccer checkpoint is unavailable."""


@dataclass(frozen=True, slots=True)
class BallCandidate:
    x: float
    y: float
    confidence: float
    heatmap_peak: float
    component_score: float
    component_area: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BallDetectionProvider(Protocol):
    backend: str
    frames_in: int
    frames_out: int

    @property
    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def info(self) -> dict[str, Any]: ...

    def detect_window(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        max_candidates: int,
    ) -> list[list[BallCandidate]]: ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ConfigNode(dict):
    """Minimal Hydra-compatible mapping used by the pinned HRNet module."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _config_node(value: Any) -> Any:
    if isinstance(value, dict):
        return _ConfigNode({key: _config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config_node(item) for item in value]
    return value


def _wasb_model_config() -> _ConfigNode:
    """Concrete copy of the pinned upstream ``configs/model/wasb.yaml``."""

    return _config_node({
        "name": "hrnet",
        "frames_in": FRAMES_IN,
        "frames_out": FRAMES_OUT,
        "inp_height": INPUT_HEIGHT,
        "inp_width": INPUT_WIDTH,
        "out_height": INPUT_HEIGHT,
        "out_width": INPUT_WIDTH,
        "rgb_diff": False,
        "out_scales": [0],
        "MODEL": {
            "EXTRA": {
                "FINAL_CONV_KERNEL": 1,
                "PRETRAINED_LAYERS": ["*"],
                "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
                "STAGE1": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 1,
                    "BLOCK": "BOTTLENECK",
                    "NUM_BLOCKS": [1],
                    "NUM_CHANNELS": [32],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE2": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 2,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [2, 2],
                    "NUM_CHANNELS": [16, 32],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE3": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 3,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64],
                    "FUSE_METHOD": "SUM",
                },
                "STAGE4": {
                    "NUM_MODULES": 1,
                    "NUM_BRANCHES": 4,
                    "BLOCK": "BASIC",
                    "NUM_BLOCKS": [2, 2, 2, 2],
                    "NUM_CHANNELS": [16, 32, 64, 128],
                    "FUSE_METHOD": "SUM",
                },
                "DECONV": {
                    "NUM_DECONVS": 0,
                    "KERNEL_SIZE": [],
                    "NUM_BASIC_BLOCKS": 2,
                },
            },
            "INIT_WEIGHTS": True,
        },
    })


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.asarray([-direction[1], direction[0]], dtype=np.float32)


def _affine_transforms(
    image_width: int,
    image_height: int,
    output_width: int = INPUT_WIDTH,
    output_height: int = INPUT_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Match upstream ``dataloaders.dataset_loader.get_transform`` exactly."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - image declares OpenCV
        raise ProviderUnavailable(f"OpenCV runtime is unavailable: {exc}") from exc

    center = np.asarray([image_width / 2.0, image_height / 2.0], dtype=np.float32)
    scale = float(max(image_height, image_width))
    source_direction = np.asarray([0.0, -scale * 0.5], dtype=np.float32)
    target_center = np.asarray([output_width * 0.5, output_height * 0.5], dtype=np.float32)
    target_direction = np.asarray([0.0, -output_width * 0.5], dtype=np.float32)

    source = np.zeros((3, 2), dtype=np.float32)
    target = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + source_direction
    source[2] = _third_point(source[0], source[1])
    target[0] = target_center
    target[1] = target_center + target_direction
    target[2] = _third_point(target[0], target[1])
    return (
        cv2.getAffineTransform(source, target),
        cv2.getAffineTransform(target, source),
    )


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("replay_studio_wasb_hrnet", path)
    if spec is None or spec.loader is None:
        raise ProviderUnavailable(f"Could not import pinned WASB HRNet source: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ProviderUnavailable(f"Pinned WASB HRNet source failed to import: {exc}") from exc
    return module


def _component_candidates(
    heatmap: np.ndarray,
    inverse_affine: np.ndarray,
    image_size: tuple[int, int],
    *,
    score_threshold: float,
    max_candidates: int,
) -> list[BallCandidate]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - image declares OpenCV
        raise ProviderUnavailable(f"OpenCV runtime is unavailable: {exc}") from exc

    mask = (heatmap > score_threshold).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(mask)
    image_width, image_height = image_size
    candidates: list[BallCandidate] = []
    for component_index in range(1, component_count):
        ys, xs = np.where(labels == component_index)
        if xs.size == 0:
            continue
        weights = heatmap[ys, xs].astype(np.float64)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            continue
        heatmap_x = float(np.sum(xs * weights) / weight_sum)
        heatmap_y = float(np.sum(ys * weights) / weight_sum)
        source = inverse_affine @ np.asarray([heatmap_x, heatmap_y, 1.0], dtype=np.float64)
        source_x = float(source[0])
        source_y = float(source[1])
        if not np.isfinite((source_x, source_y)).all():
            continue
        # Dark padding can still form a weak component just outside non-16:9
        # images. Do not turn it into a seemingly valid edge detection.
        if source_x < 0 or source_y < 0 or source_x >= image_width or source_y >= image_height:
            continue
        peak = float(weights.max())
        candidates.append(
            BallCandidate(
                x=source_x,
                y=source_y,
                confidence=peak,
                heatmap_peak=peak,
                component_score=weight_sum,
                component_area=int(xs.size),
                metadata={"heatmapComponent": component_index},
            )
        )
    return sorted(candidates, key=lambda item: item.confidence, reverse=True)[:max_candidates]


class WasbSoccerProvider:
    """Pinned WASB HRNet soccer checkpoint with lazy, explicit readiness.

    The upstream detector hard-asserts CUDA. This adapter uses the same model,
    checkpoint, affine transform, ImageNet normalization, sigmoid heatmaps and
    connected-component postprocessing, but allows CPU inference for local
    verification. CPU is correct but intentionally documented as slow.
    """

    backend = BACKEND_NAME
    frames_in = FRAMES_IN
    frames_out = FRAMES_OUT

    def __init__(self) -> None:
        root = _repo_root()
        self.weights_path = Path(
            os.environ.get(
                "WASB_WEIGHTS",
                str(root / "models" / "wasb-soccer-best.pth.tar"),
            )
        )
        self.source_path = Path(
            os.environ.get(
                "WASB_HRNET_SOURCE",
                str(root / ".references" / "WASB-SBDT" / "src" / "models" / "hrnet.py"),
            )
        )
        self.expected_sha256 = os.environ.get("WASB_WEIGHTS_SHA256", CHECKPOINT_SHA256)
        self.device_name = os.environ.get("WASB_DEVICE", "cpu")
        self.score_threshold = float(os.environ.get("WASB_SCORE_THRESHOLD", "0.5"))
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ProviderUnavailable("WASB_SCORE_THRESHOLD must be between 0 and 1")
        self._loaded = False
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self._torch = None
        self._model = None
        self._device = None
        self._checkpoint_sha256: str | None = None
        self._model_load_seconds: float | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _verify_assets(self) -> str:
        if not self.weights_path.is_file() or self.weights_path.stat().st_size == 0:
            raise ProviderUnavailable(f"WASB soccer checkpoint is missing: {self.weights_path}")
        actual_sha256 = _file_sha256(self.weights_path)
        if self.expected_sha256 and actual_sha256.lower() != self.expected_sha256.lower():
            raise ProviderUnavailable(
                "WASB soccer checkpoint checksum mismatch: "
                f"expected {self.expected_sha256}, received {actual_sha256}"
            )
        if not self.source_path.is_file() or self.source_path.stat().st_size == 0:
            raise ProviderUnavailable(f"Pinned WASB HRNet source is missing: {self.source_path}")
        return actual_sha256

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            started = perf_counter()
            checkpoint_sha256 = self._verify_assets()
            try:
                import torch
            except ImportError as exc:
                raise ProviderUnavailable(f"PyTorch runtime is unavailable: {exc}") from exc

            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise ProviderUnavailable(
                    f"WASB_DEVICE={self.device_name} was requested but CUDA is unavailable"
                )
            if self.device_name.startswith("mps") and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                raise ProviderUnavailable(
                    f"WASB_DEVICE={self.device_name} was requested but MPS is unavailable"
                )
            try:
                device = torch.device(self.device_name)
            except Exception as exc:
                raise ProviderUnavailable(f"Invalid WASB_DEVICE={self.device_name}: {exc}") from exc

            module = _load_module(self.source_path)
            model_class = getattr(module, "HRNet", None)
            if model_class is None:
                raise ProviderUnavailable("Pinned WASB source does not expose HRNet")
            try:
                model = model_class(_wasb_model_config())
                try:
                    checkpoint = torch.load(
                        self.weights_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:  # PyTorch 1.11 has no weights_only argument.
                    checkpoint = torch.load(self.weights_path, map_location="cpu")
                state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
                if not isinstance(state, dict):
                    raise ProviderUnavailable(
                        "WASB checkpoint has no model_state_dict; refusing an unverified format"
                    )
                model.load_state_dict(state, strict=True)
                model.to(device)
                model.eval()
            except ProviderUnavailable:
                raise
            except Exception as exc:
                raise ProviderUnavailable(f"WASB soccer model failed to load: {exc}") from exc

            self._torch = torch
            self._model = model
            self._device = device
            self._checkpoint_sha256 = checkpoint_sha256
            self._model_load_seconds = perf_counter() - started
            self._loaded = True

    @property
    def model_version(self) -> str:
        digest = self._checkpoint_sha256 or self.expected_sha256
        return f"wasb-soccer@sha256:{digest[:12]}"

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "modelVersion": self.model_version,
            "checkpointSha256": self._checkpoint_sha256,
            "device": self.device_name,
            "framesIn": self.frames_in,
            "framesOut": self.frames_out,
            "inputSize": [INPUT_WIDTH, INPUT_HEIGHT],
            "scoreThreshold": self.score_threshold,
            "modelLoadSeconds": (
                round(self._model_load_seconds, 4)
                if self._model_load_seconds is not None
                else None
            ),
        }

    def detect_window(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        max_candidates: int,
    ) -> list[list[BallCandidate]]:
        if not self._loaded:
            self.load()
        if len(frames_rgb) != self.frames_in:
            raise ProviderUnavailable(
                f"WASB requires exactly {self.frames_in} temporal frames, received {len(frames_rgb)}"
            )
        if max_candidates <= 0:
            raise ProviderUnavailable("max_candidates must be positive")
        first = np.asarray(frames_rgb[0])
        if first.ndim != 3 or first.shape[2] != 3:
            raise ProviderUnavailable("WASB frames must be RGB arrays with three channels")
        image_height, image_width = first.shape[:2]
        if image_width <= 0 or image_height <= 0:
            raise ProviderUnavailable("WASB received an empty frame")
        for frame in frames_rgb:
            array = np.asarray(frame)
            if array.shape != first.shape:
                raise ProviderUnavailable("All frames in a WASB temporal window must have one size")

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - image declares OpenCV
            raise ProviderUnavailable(f"OpenCV runtime is unavailable: {exc}") from exc

        forward_affine, inverse_affine = _affine_transforms(image_width, image_height)
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
            channels_first = np.transpose(warped.astype(np.float32) / 255.0, (2, 0, 1))
            tensors.append((channels_first - means) / deviations)
        input_array = np.concatenate(tensors, axis=0)[None, ...]

        assert self._torch is not None and self._model is not None and self._device is not None
        with self._inference_lock, self._torch.inference_mode():
            tensor = self._torch.from_numpy(input_array).to(self._device)
            outputs = self._model(tensor)
            logits = outputs[0] if isinstance(outputs, dict) else None
            if logits is None or logits.ndim != 4 or logits.shape[1] != self.frames_out:
                raise ProviderUnavailable("WASB model returned an unexpected heatmap shape")
            heatmaps = self._torch.sigmoid(logits)[0].detach().cpu().numpy()

        return [
            _component_candidates(
                heatmaps[frame_offset],
                inverse_affine,
                (image_width, image_height),
                score_threshold=self.score_threshold,
                max_candidates=max_candidates,
            )
            for frame_offset in range(self.frames_out)
        ]
