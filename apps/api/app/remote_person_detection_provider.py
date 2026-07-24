from __future__ import annotations

"""Strict multipart client for the native person-detection worker."""

import json
from math import isfinite
from pathlib import Path
from typing import Mapping

import cv2
import httpx

from .person_detection_policy import (
    DETECTOR_MAX_DETECTIONS,
    DETECTOR_PROVIDER_NMS_IOU,
    GENERIC_ULTRALYTICS_CONFIDENCE,
    GENERIC_ULTRALYTICS_IMAGE_SIZE,
)
from .person_detection_provider_contract import (
    PersonDetectionProvider,
    RawDetectionBox,
    RawFramePrediction,
)
from .reconstruction_errors import ReconstructionError


READINESS_FIELDS = frozenset(
    {
        "schemaVersion",
        "status",
        "backend",
        "providerVersion",
        "modelVersion",
        "checkpoint",
        "device",
        "batchSize",
        "torchVersion",
        "mpsFallbackEnabled",
        "modelLoadSeconds",
        "modelWarmupSeconds",
    }
)
ANALYSIS_FIELDS = READINESS_FIELDS - {"status", "modelLoadSeconds"} | {
    "image",
    "names",
    "boxes",
    "diagnostics",
}
BOX_FIELDS = frozenset({"classId", "confidence", "x1", "y1", "x2", "y2"})
DIAGNOSTIC_FIELDS = frozenset(
    {
        "decodeSeconds",
        "inferenceSeconds",
        "requestSeconds",
        "boxCount",
        "degenerateBoxCount",
    }
)


def _reject_unknown(value: Mapping, fields: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ReconstructionError(
            f"Person detection worker returned unknown {label} fields: "
            + ", ".join(unknown)
        )


def _validate_runtime_payload(
    payload: object,
    *,
    fields: frozenset[str],
) -> dict:
    if not isinstance(payload, dict):
        raise ReconstructionError("Person detection worker returned malformed JSON")
    _reject_unknown(payload, fields, "response")
    if (
        payload.get("schemaVersion") != 1
        or payload.get("backend") != "ultralytics-yolo"
        or not isinstance(payload.get("providerVersion"), str)
        or not isinstance(payload.get("modelVersion"), str)
        or not str(payload.get("modelVersion")).strip()
        or not isinstance(payload.get("device"), str)
        or not str(payload.get("device")).strip()
        or isinstance(payload.get("batchSize"), bool)
        or not isinstance(payload.get("batchSize"), int)
        or int(payload.get("batchSize")) < 1
        or not isinstance(payload.get("mpsFallbackEnabled"), bool)
    ):
        raise ReconstructionError(
            "Person detection worker returned an unsupported runtime contract"
        )
    checkpoint = payload.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("name"), str)
        or not isinstance(checkpoint.get("sha256"), str)
        or len(str(checkpoint.get("sha256"))) != 64
        or isinstance(checkpoint.get("size"), bool)
        or not isinstance(checkpoint.get("size"), int)
        or int(checkpoint.get("size")) <= 0
    ):
        raise ReconstructionError(
            "Person detection worker returned invalid checkpoint provenance"
        )
    return payload


def validate_person_detection_readiness(payload: object) -> dict:
    value = _validate_runtime_payload(payload, fields=READINESS_FIELDS)
    if value.get("status") != "ready":
        raise ReconstructionError("Person detection worker is not ready")
    return value


def validate_person_detection_payload(payload: object) -> dict:
    value = _validate_runtime_payload(payload, fields=ANALYSIS_FIELDS)
    image = value.get("image")
    if (
        not isinstance(image, dict)
        or set(image) != {"width", "height"}
        or any(
            isinstance(image.get(field), bool)
            or not isinstance(image.get(field), int)
            or int(image.get(field)) <= 0
            for field in ("width", "height")
        )
    ):
        raise ReconstructionError(
            "Person detection worker returned invalid image dimensions"
        )
    names = value.get("names")
    if not isinstance(names, dict) or any(
        not str(index).isdigit() or not isinstance(name, str)
        for index, name in names.items()
    ):
        raise ReconstructionError(
            "Person detection worker returned invalid class names"
        )
    boxes = value.get("boxes")
    if not isinstance(boxes, list):
        raise ReconstructionError("Person detection worker returned no boxes array")
    for box in boxes:
        if not isinstance(box, dict):
            raise ReconstructionError("Person detection worker returned invalid box")
        _reject_unknown(box, BOX_FIELDS, "box")
        if (
            isinstance(box.get("classId"), bool)
            or not isinstance(box.get("classId"), int)
            or any(
                isinstance(box.get(field), bool)
                or not isinstance(box.get(field), (int, float))
                or not isfinite(float(box.get(field)))
                for field in ("confidence", "x1", "y1", "x2", "y2")
            )
            or not 0.0 <= float(box["confidence"]) <= 1.0
            or float(box["x2"]) <= float(box["x1"])
            or float(box["y2"]) <= float(box["y1"])
        ):
            raise ReconstructionError("Person detection worker returned invalid box")
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ReconstructionError(
            "Person detection worker returned malformed diagnostics"
        )
    _reject_unknown(diagnostics, DIAGNOSTIC_FIELDS, "diagnostic")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not isfinite(float(item))
        or float(item) < 0
        for item in diagnostics.values()
    ):
        raise ReconstructionError(
            "Person detection worker returned invalid diagnostics"
        )
    return value


def require_person_detection_checkpoint(
    payload: Mapping,
    requested_model: str,
) -> None:
    """Fail closed when a fixed remote worker loaded a different checkpoint."""

    requested_name = Path(requested_model).name
    checkpoint = payload.get("checkpoint")
    actual_name = (
        str(checkpoint.get("name"))
        if isinstance(checkpoint, Mapping)
        and isinstance(checkpoint.get("name"), str)
        else ""
    )
    if actual_name == requested_name:
        return
    raise ReconstructionError(
        "Person detection worker model mismatch: reconstruction requested "
        f"{requested_name}, but the worker loaded {actual_name or 'an unknown checkpoint'}. "
        "Restart the native person-detection worker with the requested checkpoint; "
        "the selected model was not used."
    )


class RemotePersonDetectionProvider(PersonDetectionProvider):
    def __init__(
        self,
        worker_url: str,
        *,
        timeout: float,
        expected_checkpoint: str,
    ) -> None:
        self._worker_url = worker_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        try:
            response = self._client.get(f"{self._worker_url}/health/ready")
            response.raise_for_status()
            self._info = validate_person_detection_readiness(response.json())
            require_person_detection_checkpoint(
                self._info,
                expected_checkpoint,
            )
        except (httpx.HTTPError, ValueError, ReconstructionError) as exc:
            self._client.close()
            raise ReconstructionError(
                f"Person detection worker is unavailable: {exc}"
            ) from exc

    def info(self) -> dict:
        return dict(self._info)

    def predict(self, path: Path) -> RawFramePrediction:
        manifest = json.dumps(
            {
                "contractVersion": 1,
                "imageSize": int(GENERIC_ULTRALYTICS_IMAGE_SIZE),
                "confidence": float(GENERIC_ULTRALYTICS_CONFIDENCE),
                "nmsIou": float(DETECTOR_PROVIDER_NMS_IOU),
                "maxDetections": int(DETECTOR_MAX_DETECTIONS),
            },
            separators=(",", ":"),
        )
        try:
            with path.open("rb") as handle:
                response = self._client.post(
                    f"{self._worker_url}/v1/detections",
                    files={"frame": (path.name, handle, "image/jpeg")},
                    data={"manifest": manifest},
                )
            response.raise_for_status()
            payload = validate_person_detection_payload(response.json())
        except (OSError, httpx.HTTPError, ValueError, ReconstructionError) as exc:
            raise ReconstructionError(
                f"Person detection worker failed for {path.name}: {exc}"
            ) from exc
        image = cv2.imread(str(path))
        if image is None:
            raise ReconstructionError(f"Could not decode sampled frame {path}")
        if (
            image.shape[1] != int(payload["image"]["width"])
            or image.shape[0] != int(payload["image"]["height"])
        ):
            raise ReconstructionError(
                "Person detection worker decoded different frame dimensions"
            )
        return RawFramePrediction(
            image_bgr=image,
            names={int(index): str(name) for index, name in payload["names"].items()},
            boxes=tuple(
                RawDetectionBox(
                    class_id=int(box["classId"]),
                    confidence=float(box["confidence"]),
                    x1=float(box["x1"]),
                    y1=float(box["y1"]),
                    x2=float(box["x2"]),
                    y2=float(box["y2"]),
                )
                for box in payload["boxes"]
            ),
            diagnostics=dict(payload["diagnostics"]),
        )


def person_detection_worker_readiness(
    worker_url: str | None,
    *,
    timeout: float = 2.0,
) -> dict:
    if not worker_url:
        return {"configured": False, "status": "in-process", "backend": "ultralytics-yolo"}
    try:
        response = httpx.get(
            f"{worker_url.rstrip('/')}/health/ready",
            timeout=timeout,
        )
        response.raise_for_status()
        value = validate_person_detection_readiness(response.json())
    except (httpx.HTTPError, ValueError, ReconstructionError) as exc:
        return {
            "configured": True,
            "status": "unavailable",
            "backend": "ultralytics-yolo",
            "detail": str(exc),
        }
    return {
        "configured": True,
        "status": "ready",
        "backend": value["backend"],
        "device": value["device"],
        "batchSize": value["batchSize"],
        "modelVersion": value["modelVersion"],
        "providerVersion": value["providerVersion"],
        "torchVersion": value.get("torchVersion"),
        "mpsFallbackEnabled": value["mpsFallbackEnabled"],
        "modelLoadSeconds": value.get("modelLoadSeconds"),
        "modelWarmupSeconds": value.get("modelWarmupSeconds"),
        "checkpoint": dict(value["checkpoint"]),
    }


__all__ = (
    "RemotePersonDetectionProvider",
    "person_detection_worker_readiness",
    "require_person_detection_checkpoint",
    "validate_person_detection_payload",
    "validate_person_detection_readiness",
)
