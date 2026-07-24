"""Ultralytics-specific local inference and result adaptation."""

from __future__ import annotations

from pathlib import Path

from .config import get_settings
from .person_detection_policy import (
    DETECTOR_MAX_DETECTIONS,
    DETECTOR_PROVIDER_NMS_IOU,
    GENERIC_ULTRALYTICS_CONFIDENCE,
    GENERIC_ULTRALYTICS_IMAGE_SIZE,
)
from .person_detection_candidate_selection import (
    detection_class_ids,
    parse_person_prediction,
)
from .person_detection_provider_contract import (
    RawDetectionBox,
    RawFramePrediction,
)
from .reconstruction_person_detection_contract import Detection


def predict_frame(model, path: Path | str):
    person_ids, ball_ids = detection_class_ids(getattr(model, "names", None))
    return model.predict(
        str(path),
        imgsz=GENERIC_ULTRALYTICS_IMAGE_SIZE,
        conf=GENERIC_ULTRALYTICS_CONFIDENCE,
        classes=sorted(person_ids | ball_ids),
        iou=DETECTOR_PROVIDER_NMS_IOU,
        max_det=DETECTOR_MAX_DETECTIONS,
        device=get_settings().reconstruction_device,
        verbose=False,
    )[0]


def prediction_from_ultralytics_result(result) -> RawFramePrediction:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    return RawFramePrediction(
        image_bgr=result.orig_img,
        names={
            int(index): str(name)
            for index, name in (getattr(result, "names", None) or {}).items()
        },
        boxes=tuple(
            RawDetectionBox(
                class_id=int(class_id),
                confidence=float(confidence),
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
            )
            for box, class_id, confidence in zip(boxes, classes, confidences)
        ),
    )


def parse_person_detections(
    result,
    *,
    debug_log: list[dict] | None = None,
) -> tuple[list[Detection], list[dict]]:
    """Compatibility-free Ultralytics adapter for non-provider consumers."""

    return parse_person_prediction(
        prediction_from_ultralytics_result(result),
        debug_log=debug_log,
    )
