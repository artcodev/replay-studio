from __future__ import annotations

"""In-process Ultralytics implementation of the person detector contract."""

from pathlib import Path
from time import perf_counter

from .person_detection_provider_contract import (
    PersonDetectionProvider,
    RawFramePrediction,
)
from .person_detector_provenance import local_person_detection_provider_info
from .ultralytics_person_inference import (
    prediction_from_ultralytics_result,
    predict_frame,
)


class LocalUltralyticsPersonDetectionProvider(PersonDetectionProvider):
    def __init__(self, model_name: str, model: object) -> None:
        self._model = model
        self._info = local_person_detection_provider_info(model_name, model)

    def info(self) -> dict:
        return dict(self._info)

    def predict(self, path: Path) -> RawFramePrediction:
        started = perf_counter()
        result = predict_frame(self._model, path)
        prediction = prediction_from_ultralytics_result(result)
        prediction.diagnostics = {
            "requestSeconds": round(perf_counter() - started, 6),
            "inferenceSeconds": round(perf_counter() - started, 6),
            "boxCount": len(prediction.boxes),
        }
        return prediction


__all__ = ("LocalUltralyticsPersonDetectionProvider",)
