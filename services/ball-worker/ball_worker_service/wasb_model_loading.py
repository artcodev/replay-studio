from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any

from .provider_contract import ProviderUnavailable
from .wasb_configuration import (
    FRAMES_IN,
    FRAMES_OUT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    WasbConfiguration,
)


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

    return _config_node(
        {
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
        }
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_assets(configuration: WasbConfiguration) -> str:
    if (
        not configuration.weights_path.is_file()
        or configuration.weights_path.stat().st_size == 0
    ):
        raise ProviderUnavailable(
            f"WASB soccer checkpoint is missing: {configuration.weights_path}"
        )
    actual_sha256 = _file_sha256(configuration.weights_path)
    if actual_sha256 != configuration.expected_sha256:
        raise ProviderUnavailable(
            "WASB soccer checkpoint checksum mismatch: "
            f"expected {configuration.expected_sha256}, received {actual_sha256}"
        )
    if (
        not configuration.source_path.is_file()
        or configuration.source_path.stat().st_size == 0
    ):
        raise ProviderUnavailable(
            f"Pinned WASB HRNet source is missing: {configuration.source_path}"
        )
    return actual_sha256


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("replay_studio_wasb_hrnet", path)
    if spec is None or spec.loader is None:
        raise ProviderUnavailable(f"Could not import pinned WASB HRNet source: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ProviderUnavailable(
            f"Pinned WASB HRNet source failed to import: {exc}"
        ) from exc
    return module


@dataclass(frozen=True, slots=True)
class LoadedWasbModel:
    torch: Any
    model: Any
    device: Any
    checkpoint_sha256: str
    load_seconds: float


def load_wasb_model(configuration: WasbConfiguration) -> LoadedWasbModel:
    """Verify pinned assets and build an inference-only HRNet runtime."""

    started = perf_counter()
    checkpoint_sha256 = _verify_assets(configuration)
    try:
        import torch
    except ImportError as exc:
        raise ProviderUnavailable(f"PyTorch runtime is unavailable: {exc}") from exc

    if (
        configuration.device_name.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise ProviderUnavailable(
            f"WASB_DEVICE={configuration.device_name} was requested but CUDA is unavailable"
        )
    if configuration.device_name.startswith("mps") and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ProviderUnavailable(
            f"WASB_DEVICE={configuration.device_name} was requested "
            "but MPS is unavailable"
        )
    try:
        device = torch.device(configuration.device_name)
    except Exception as exc:
        raise ProviderUnavailable(
            f"Invalid WASB_DEVICE={configuration.device_name}: {exc}"
        ) from exc

    module = _load_module(configuration.source_path)
    model_class = getattr(module, "HRNet", None)
    if model_class is None:
        raise ProviderUnavailable("Pinned WASB source does not expose HRNet")
    try:
        model = model_class(_wasb_model_config())
        try:
            checkpoint = torch.load(
                configuration.weights_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # PyTorch 1.11 has no weights_only argument.
            checkpoint = torch.load(configuration.weights_path, map_location="cpu")
        state = (
            checkpoint.get("model_state_dict")
            if isinstance(checkpoint, dict)
            else None
        )
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

    return LoadedWasbModel(
        torch=torch,
        model=model,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        load_seconds=perf_counter() - started,
    )
