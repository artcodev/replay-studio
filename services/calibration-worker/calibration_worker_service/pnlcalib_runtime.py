from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from .pnlcalib_constants import CACHE_SCHEMA_VERSION, INPUT_HEIGHT, INPUT_WIDTH


class AttrDict(dict):
    """Small yacs-compatible subset required by the pinned HRNet code."""

    __getattr__ = dict.__getitem__


def _attr_dict(value: object) -> object:
    if isinstance(value, dict):
        return AttrDict({key: _attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attr_dict(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PnLCalibRuntime:
    keypoint_model_factory: Callable[..., torch.nn.Module]
    line_model_factory: Callable[..., torch.nn.Module]
    frame_calibrator_factory: Callable[..., Any]
    complete_keypoints: Callable[..., tuple[dict, dict]]
    coords_to_dict: Callable[..., list[dict]]
    decode_keypoints: Callable[..., object]
    decode_lines: Callable[..., object]


@dataclass(frozen=True, slots=True)
class LoadedPnLCalibModels:
    runtime: PnLCalibRuntime
    device: torch.device
    keypoint_model: torch.nn.Module
    line_model: torch.nn.Module
    model_version: str


def plugin_root() -> Path:
    soccernet_root = Path(os.environ.get("SOCCERNET_ROOT", "/opt/sn-gamestate"))
    return soccernet_root / "plugins" / "calibration"


def load_pnlcalib_runtime(root: Path | None = None) -> PnLCalibRuntime:
    """Load only the pinned SoccerNet plugin selected by the worker image."""

    resolved_root = root or plugin_root()
    if str(resolved_root) not in sys.path:
        sys.path.insert(0, str(resolved_root))
    try:
        from pnlcalib.model.cls_hrnet import get_cls_net
        from pnlcalib.model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from pnlcalib.utils.utils_calib import FramebyFrameCalib
        from pnlcalib.utils.utils_heatmap import (
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"PnLCalib runtime is unavailable under {resolved_root}"
        ) from exc
    return PnLCalibRuntime(
        keypoint_model_factory=get_cls_net,
        line_model_factory=get_cls_net_l,
        frame_calibrator_factory=FramebyFrameCalib,
        complete_keypoints=complete_keypoints,
        coords_to_dict=coords_to_dict,
        decode_keypoints=get_keypoints_from_heatmap_batch_maxpool,
        decode_lines=get_keypoints_from_heatmap_batch_maxpool_l,
    )


def model_version(*weights_paths: Path) -> str:
    """Build a process-local model identity without rereading large weights."""

    identity = [CACHE_SCHEMA_VERSION, str(INPUT_WIDTH), str(INPUT_HEIGHT)]
    for path in weights_paths:
        stat = path.stat()
        identity.extend([str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns)])
    return sha256("\n".join(identity).encode("utf-8")).hexdigest()[:16]


def _load_model(
    config_path: Path,
    weights_path: Path,
    factory: Callable[..., torch.nn.Module],
    device: torch.device,
) -> torch.nn.Module:
    if not weights_path.is_file():
        raise RuntimeError(f"PnLCalib weights are missing: {weights_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = _attr_dict(yaml.safe_load(handle))
    model = factory(config)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_pnlcalib_models() -> LoadedPnLCalibModels:
    root = plugin_root()
    runtime = load_pnlcalib_runtime(root)
    device = torch.device(os.environ.get("PNLCALIB_DEVICE", "cpu"))
    keypoint_weights = Path(
        os.environ.get("PNLCALIB_KEYPOINT_WEIGHTS", "/models/pnl_SV_kp")
    )
    line_weights = Path(
        os.environ.get("PNLCALIB_LINE_WEIGHTS", "/models/pnl_SV_lines")
    )
    model_root = root / "pnlcalib"
    version = model_version(keypoint_weights, line_weights)
    return LoadedPnLCalibModels(
        runtime=runtime,
        device=device,
        keypoint_model=_load_model(
            model_root / "config" / "hrnetv2_w48.yaml",
            keypoint_weights,
            runtime.keypoint_model_factory,
            device,
        ),
        line_model=_load_model(
            model_root / "config" / "hrnetv2_w48_l.yaml",
            line_weights,
            runtime.line_model_factory,
            device,
        ),
        model_version=version,
    )
