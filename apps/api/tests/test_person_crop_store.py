from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import app.reconstruction_sampled_frame_detection as sampled_detection
from app.person_crop_store import (
    PersonCropPolicy,
    extract_and_store_person_crops,
    extract_person_crop,
    lookup_person_crop_envelope,
)
from app.person_detection_cache import frame_content_sha256
from app.reconstruction_person_detection_contract import Detection


def _image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


def _detection(observation_id: str, x: float = 60.0) -> Detection:
    detection = Detection(
        x=x,
        y=100.0,
        width=30.0,
        height=60.0,
        confidence=0.9,
        feature=np.zeros(12, dtype=np.float32),
    )
    detection.image_x = x
    detection.image_y = 100.0
    detection.observation_id = observation_id
    return detection


def test_extract_store_and_lookup_round_trip(tmp_path):
    frame_sha = "a" * 64
    diagnostics: dict = {}
    records = extract_and_store_person_crops(
        tmp_path,
        image=_image(),
        frame_sha256=frame_sha,
        detections=[_detection("obs-1"), _detection("obs-2", x=120.0)],
        policy=PersonCropPolicy(),
        diagnostics=diagnostics,
    )
    assert diagnostics == {"stores": 1}
    assert set(records) == {"obs-1", "obs-2"}
    assert records["obs-1"].crop_jpeg
    assert len(records["obs-1"].crop_sha256) == 64

    stored = lookup_person_crop_envelope(
        tmp_path, frame_sha256=frame_sha, policy=PersonCropPolicy()
    )
    assert stored is not None
    assert stored["obs-1"].crop_jpeg == records["obs-1"].crop_jpeg
    # The published bytes decode back into the padded crop.
    decoded = cv2.imdecode(
        np.frombuffer(stored["obs-1"].crop_jpeg, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    x1, y1, x2, y2 = stored["obs-1"].padded_rect
    assert decoded.shape[:2] == (y2 - y1, x2 - x1)

    # A tampered envelope is an ordinary miss, not an error.
    envelope_path = next(tmp_path.rglob("*.json"))
    envelope_path.write_text(envelope_path.read_text()[:-30])
    assert (
        lookup_person_crop_envelope(
            tmp_path, frame_sha256=frame_sha, policy=PersonCropPolicy()
        )
        is None
    )


def test_second_pass_reuses_the_envelope_until_observations_change(tmp_path):
    frame_sha = "b" * 64
    image = _image()
    diagnostics: dict = {}
    extract_and_store_person_crops(
        tmp_path,
        image=image,
        frame_sha256=frame_sha,
        detections=[_detection("obs-1")],
        policy=PersonCropPolicy(),
        diagnostics=diagnostics,
    )
    reused = extract_and_store_person_crops(
        tmp_path,
        image=image,
        frame_sha256=frame_sha,
        detections=[_detection("obs-1")],
        policy=PersonCropPolicy(),
        diagnostics=diagnostics,
    )
    assert diagnostics == {"stores": 1, "hits": 1}
    assert set(reused) == {"obs-1"}

    # A new manual annotation adds an observation: the envelope no longer
    # covers the frame and is rebuilt from pixels.
    extract_and_store_person_crops(
        tmp_path,
        image=image,
        frame_sha256=frame_sha,
        detections=[_detection("obs-1"), _detection("obs-annotation", x=30.0)],
        policy=PersonCropPolicy(),
        diagnostics=diagnostics,
    )
    assert diagnostics == {"stores": 2, "hits": 1}
    stored = lookup_person_crop_envelope(
        tmp_path, frame_sha256=frame_sha, policy=PersonCropPolicy()
    )
    assert set(stored) == {"obs-1", "obs-annotation"}


def test_crop_geometry_clamps_pads_and_reports_qa(tmp_path):
    image = _image()
    # A bbox flush with the top-left corner: padding is clipped.
    crop, rect, quality, reasons = extract_person_crop(
        image,
        {"x": 0.0, "y": 0.0, "width": 30.0, "height": 60.0},
        PersonCropPolicy(minimum_sharpness=0.0),
    )
    assert rect[0] == 0 and rect[1] == 0
    assert quality["borderClipped"] is True
    assert crop.shape[0] == rect[3] - rect[1]
    assert reasons == []

    # A tiny detection fails the minimum-size gate but is still cut.
    _crop, _rect, quality, reasons = extract_person_crop(
        image,
        {"x": 50.0, "y": 50.0, "width": 4.0, "height": 6.0},
        PersonCropPolicy(minimum_sharpness=0.0),
    )
    assert "crop-too-small" in reasons
    assert quality["sourceBoxWidth"] == 4.0

    # A bbox fully outside the frame has no pixels.
    _crop, _rect, _quality, reasons = extract_person_crop(
        image,
        {"x": 500.0, "y": 500.0, "width": 30.0, "height": 60.0},
        PersonCropPolicy(),
    )
    assert "crop-outside-frame" in reasons


def test_detection_pass_attaches_crop_identity(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame_00007.jpg"
    assert cv2.imwrite(str(frame_path), _image())
    image = cv2.imread(str(frame_path))
    detection = _detection("")
    detection.observation_id = None

    monkeypatch.setattr(
        sampled_detection.person_base_detection_cache,
        "cached_base_frame_detections",
        lambda *_args: (image, [detection], []),
    )
    monkeypatch.setattr(sampled_detection, "frame_annotations", lambda *_: [])
    monkeypatch.setattr(
        sampled_detection,
        "apply_person_annotations",
        lambda _image, people, _annotations: people,
    )

    class Accumulator:
        def __init__(self, *_args) -> None:
            pass

        def add_frame(self, **_values) -> None:
            pass

        def result(self):
            return SimpleNamespace()

    monkeypatch.setattr(
        sampled_detection, "SampledCalibrationAccumulator", Accumulator
    )
    store_directory = tmp_path / "person-crops"
    monkeypatch.setattr(
        sampled_detection,
        "person_crop_store_runtime",
        lambda: (store_directory, PersonCropPolicy(minimum_sharpness=0.0)),
    )

    runtime = SimpleNamespace(
        model=object(),
        person_cache_directory=tmp_path,
        person_detector_input={"fingerprint": "detector-v1"},
        person_cache_diagnostics={"errors": [], "hits": 1},
    )
    sampled_detection.analyze_sampled_frames(
        {"payload": {"videoAsset": {}}},
        [(frame_path, 0.0)],
        runtime,
        SimpleNamespace(),
        SimpleNamespace(update=lambda *args, **kwargs: None),
    )

    assert detection.observation_id
    assert detection.crop_frame_sha256 == frame_content_sha256(frame_path)
    assert detection.crop_sha256
    assert detection.crop_rejection_reasons == ()
    stored = lookup_person_crop_envelope(
        store_directory,
        frame_sha256=detection.crop_frame_sha256,
        policy=PersonCropPolicy(minimum_sharpness=0.0),
    )
    assert stored is not None
    assert stored[str(detection.observation_id)].crop_sha256 == (
        detection.crop_sha256
    )
    assert runtime.person_cache_diagnostics["personCropStore"] == {
        "hits": 0,
        "stores": 1,
        "storeErrors": 0,
    }
