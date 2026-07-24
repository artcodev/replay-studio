from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_errors import ReconstructionError
from app.reconstruction_inputs import require_model_weights_available
from app.ultralytics_person_inference import (
    detection_class_ids,
    parse_person_detections,
)


def test_class_ids_resolve_by_name_with_coco_fallback():
    coco = {0: "person", 32: "sports ball", 5: "bus"}
    assert detection_class_ids(coco) == (frozenset({0}), frozenset({32}))

    football = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    assert detection_class_ids(football) == (
        frozenset({0, 1, 2}),
        frozenset({3}),
    )

    # A checkpoint without a usable name map keeps the historical COCO ids.
    assert detection_class_ids(None) == (frozenset({0}), frozenset({32}))
    assert detection_class_ids({7: "cone"}) == (frozenset({0}), frozenset({32}))


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Result:
    def __init__(self, image, names, boxes, classes, confidences):
        self.orig_img = image
        self.names = names

        class _Boxes:
            xyxy = _Tensor(boxes)
            cls = _Tensor(classes)
            conf = _Tensor(confidences)

        self.boxes = _Boxes()


def _pitch_image() -> np.ndarray:
    # A green pitch keeps is_pitch_person and the ball grass gate satisfied.
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    image[:, :] = (60, 160, 60)
    return image


def test_football_class_names_keep_officials_and_the_ball():
    result = _Result(
        _pitch_image(),
        names={0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"},
        boxes=[
            [100.0, 100.0, 130.0, 190.0],
            [200.0, 90.0, 228.0, 180.0],
            [300.0, 200.0, 308.0, 208.0],
        ],
        classes=[1, 2, 3],
        confidences=[0.9, 0.85, 0.7],
    )

    people, balls = parse_person_detections(result)

    assert len(people) == 2
    assert {round(person.confidence, 2) for person in people} == {0.9, 0.85}
    assert len(balls) == 1
    assert balls[0]["confidence"] == pytest.approx(0.7)


def test_debug_log_explains_every_raw_box():
    # One kept person, one duplicate person suppressed by NMS, one spectator
    # off the grass, and one ball: every raw box must carry a verdict.
    image = _pitch_image()
    image[0:40, :] = (40, 40, 200)  # a red stand strip at the top
    result = _Result(
        image,
        names={0: "player", 3: "ball"},
        boxes=[
            [100.0, 100.0, 130.0, 190.0],
            [101.0, 101.0, 131.0, 191.0],
            [200.0, 2.0, 220.0, 38.0],
            [300.0, 200.0, 308.0, 208.0],
        ],
        classes=[0, 0, 0, 3],
        confidences=[0.9, 0.5, 0.9, 0.7],
    )

    log: list[dict] = []
    people, balls = parse_person_detections(result, debug_log=log)

    assert len(people) == 1 and len(balls) == 1
    verdicts = [record["verdict"] for record in log]
    assert verdicts == [
        "accepted-person",
        "rejected-person-nms",
        "rejected-foot-above-horizon",
        "accepted-ball",
    ]
    assert log[0]["className"] == "player"
    assert log[0]["gate"]["pitchRatio"] > 0.9
    assert log[2]["gate"]["verdict"] == "rejected-foot-above-horizon"
    assert log[3]["ballGrassRatio"] > 0.9

    # The debug channel must not change the parsed result itself.
    silent_people, silent_balls = parse_person_detections(result)
    assert len(silent_people) == len(people) and len(silent_balls) == len(balls)


def test_frame_analysis_dump_is_persisted_atomically(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.reconstruction_frame_analysis import persist_frame_analysis

    monkeypatch.setattr(
        "app.reconstruction_frame_analysis.get_settings",
        lambda: SimpleNamespace(analysis_run_log_directory=str(tmp_path)),
    )
    path = persist_frame_analysis({"debug": {"ok": True}}, "scene-1", 42)
    assert path is not None
    import json

    stored = json.loads((tmp_path / "frame-analysis" / "scene-1-frame-00042.json").read_text())
    assert stored == {"debug": {"ok": True}}

    # IO failures must degrade to None, never break the analysis response.
    monkeypatch.setattr(
        "app.reconstruction_frame_analysis.get_settings",
        lambda: SimpleNamespace(
            analysis_run_log_directory=str(tmp_path / "frame-analysis" / "scene-1-frame-00042.json")
        ),
    )
    assert persist_frame_analysis({}, "scene-1", 1) is None


def test_owner_supplied_weights_fail_early_with_an_installation_hint(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "app.reconstruction_inputs.get_settings",
        lambda: SimpleNamespace(football_detector_weights=None),
    )
    with pytest.raises(ReconstructionError, match="football.pt.*not installed"):
        require_model_weights_available("football.pt")

    # The baked-image layout (./models) is searched alongside the repo root.
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "football.pt").write_bytes(b"weights")
    require_model_weights_available("football.pt")

    # An explicit FOOTBALL_DETECTOR_WEIGHTS path wins over the layouts.
    elsewhere = tmp_path / "elsewhere.pt"
    elsewhere.write_bytes(b"weights-2")
    monkeypatch.setattr(
        "app.reconstruction_inputs.get_settings",
        lambda: SimpleNamespace(football_detector_weights=str(elsewhere)),
    )
    from app.reconstruction_inputs import resolve_custom_model_checkpoint

    require_model_weights_available("football.pt")
    assert resolve_custom_model_checkpoint("football.pt") == elsewhere.resolve()

    # Stock checkpoints keep the auto-download path untouched.
    require_model_weights_available("yolo26m.pt")
