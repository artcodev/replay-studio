from __future__ import annotations

"""Long-lived Ultralytics engine with explicit accelerator validation."""

from hashlib import sha256
from math import isfinite
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np


PERSON_NAMES = frozenset(
    {"person", "player", "goalkeeper", "referee", "staff"}
)
BALL_NAMES = frozenset({"ball", "sports ball"})


class DetectionEngineUnavailable(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_torch_device(torch_module, requested: str):
    normalized = requested.strip().lower()
    if normalized.startswith("cuda") and not torch_module.cuda.is_available():
        raise DetectionEngineUnavailable(
            f"PERSON_DETECTION_DEVICE={requested} requested but CUDA is unavailable"
        )
    if normalized == "mps":
        backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        if (
            backend is None
            or not backend.is_built()
            or not backend.is_available()
        ):
            raise DetectionEngineUnavailable(
                "PERSON_DETECTION_DEVICE=mps requested but Metal/MPS is unavailable"
            )
    if normalized not in {"cpu", "mps"} and not normalized.startswith("cuda"):
        raise DetectionEngineUnavailable(
            f"Unsupported PERSON_DETECTION_DEVICE={requested}"
        )
    return torch_module.device(normalized)


class UltralyticsDetectionEngine:
    backend = "ultralytics-yolo"

    def __init__(self) -> None:
        self.weights_path = Path(
            os.environ.get("PERSON_DETECTION_WEIGHTS", "yolo26m.pt")
        ).expanduser()
        self.device_name = os.environ.get("PERSON_DETECTION_DEVICE", "cpu")
        self._model = None
        self._torch = None
        self._provider_version: str | None = None
        self._checkpoint_sha256: str | None = None
        self._model_load_seconds: float | None = None
        self._model_warmup_seconds: float | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if not self.weights_path.is_file() or self.weights_path.stat().st_size <= 0:
                raise DetectionEngineUnavailable(
                    f"Person detector checkpoint is missing: {self.weights_path}"
                )
            started = perf_counter()
            try:
                import torch
                import ultralytics
                from ultralytics import YOLO
            except Exception as exc:
                raise DetectionEngineUnavailable(
                    f"Ultralytics runtime is unavailable: {exc}"
                ) from exc
            resolve_torch_device(torch, self.device_name)
            try:
                model = YOLO(str(self.weights_path))
                warmup_started = perf_counter()
                model.predict(
                    np.zeros((1080, 1920, 3), dtype=np.uint8),
                    imgsz=1280,
                    conf=0.5,
                    classes=[0],
                    iou=0.7,
                    max_det=1,
                    device=self.device_name,
                    verbose=False,
                )
                if self.device_name == "mps":
                    torch.mps.synchronize()
            except Exception as exc:
                raise DetectionEngineUnavailable(
                    f"Person detector failed to load: {exc}"
                ) from exc
            self._model_warmup_seconds = perf_counter() - warmup_started
            self._torch = torch
            self._provider_version = str(ultralytics.__version__)
            self._checkpoint_sha256 = _file_sha256(self.weights_path)
            self._model = model
            self._model_load_seconds = perf_counter() - started

    def info(self) -> dict:
        checkpoint_sha = self._checkpoint_sha256
        return {
            "schemaVersion": 1,
            "backend": self.backend,
            "providerVersion": self._provider_version,
            "modelVersion": checkpoint_sha[:16] if checkpoint_sha else None,
            "checkpoint": {
                "name": self.weights_path.name,
                "size": (
                    int(self.weights_path.stat().st_size)
                    if self.weights_path.is_file()
                    else 0
                ),
                "sha256": checkpoint_sha,
            },
            "device": self.device_name,
            "batchSize": 1,
            "torchVersion": (
                str(self._torch.__version__) if self._torch is not None else None
            ),
            "mpsFallbackEnabled": (
                os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
                not in {"0", "false", "False", ""}
            ),
            "modelLoadSeconds": self._model_load_seconds,
            "modelWarmupSeconds": self._model_warmup_seconds,
        }

    def predict(self, image, policy: dict):
        self.load()
        assert self._model is not None
        assert self._torch is not None
        names = {
            int(index): str(name)
            for index, name in (getattr(self._model, "names", None) or {}).items()
        }
        class_ids = [
            index
            for index, name in names.items()
            if name.strip().lower() in PERSON_NAMES | BALL_NAMES
        ]
        if not class_ids:
            class_ids = [0, 32]
        with self._inference_lock:
            if self.device_name == "mps":
                self._torch.mps.synchronize()
            started = perf_counter()
            try:
                result = self._model.predict(
                    image,
                    imgsz=policy["imageSize"],
                    conf=policy["confidence"],
                    classes=sorted(class_ids),
                    iou=policy["nmsIou"],
                    max_det=policy["maxDetections"],
                    device=self.device_name,
                    verbose=False,
                )[0]
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy()
                confidences = result.boxes.conf.detach().cpu().numpy()
                if self.device_name == "mps":
                    self._torch.mps.synchronize()
            except Exception as exc:
                raise DetectionEngineUnavailable(
                    f"Person detection inference failed: {exc}"
                ) from exc
            inference_seconds = perf_counter() - started
        height, width = image.shape[:2]
        normalized_boxes: list[dict] = []
        degenerate_box_count = 0
        for box, class_id, confidence in zip(boxes, classes, confidences):
            values = [float(value) for value in box]
            score = float(confidence)
            if not all(isfinite(value) for value in (*values, score)):
                degenerate_box_count += 1
                continue
            x1 = max(0.0, min(float(width), values[0]))
            y1 = max(0.0, min(float(height), values[1]))
            x2 = max(0.0, min(float(width), values[2]))
            y2 = max(0.0, min(float(height), values[3]))
            if x2 <= x1 or y2 <= y1:
                degenerate_box_count += 1
                continue
            normalized_boxes.append(
                {
                    "classId": int(class_id),
                    "confidence": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        return (
            names,
            normalized_boxes,
            inference_seconds,
            degenerate_box_count,
        )
