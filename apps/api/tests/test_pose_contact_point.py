from __future__ import annotations

import numpy as np

from app.person_crop_store import (
    PersonCropPolicy,
    extract_and_store_person_crops,
)
from app.pose_contact_point import (
    ContactPointPolicy,
    contact_point_from_pose,
    resolve_pose_contact_points,
)
from app.reconstruction_metric_projection import attach_metric_positions
from app.reconstruction_person_detection_contract import Detection


def _pose(feet: dict[int, tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((26, 2), dtype=np.float64)
    scores = np.zeros(26, dtype=np.float64)
    for index, (x, y, score) in feet.items():
        keypoints[index] = (x, y)
        scores[index] = score
    return keypoints, scores


def test_contact_point_math_prefers_lowest_confident_toe_or_heel():
    policy = ContactPointPolicy(minimum_keypoint_score=0.3)
    # Left foot: heel (24) is lower than big toe (20); right foot only the
    # ankle (16) is confident.
    keypoints, scores = _pose(
        {
            20: (10.0, 90.0, 0.9),
            24: (12.0, 95.0, 0.8),
            16: (30.0, 92.0, 0.7),
        }
    )
    contact = contact_point_from_pose(keypoints, scores, policy=policy)
    assert contact is not None
    x, y, score = contact
    assert (x, y) == ((12.0 + 30.0) / 2, (95.0 + 92.0) / 2)
    assert score == 0.7

    # No confident foot evidence at all -> no contact point.
    keypoints, scores = _pose({20: (10.0, 90.0, 0.1)})
    assert contact_point_from_pose(keypoints, scores, policy=policy) is None


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


def _seed_store(tmp_path, monkeypatch, detections):
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (140, 200, 3), dtype=np.uint8)
    store = tmp_path / "person-crops"
    frame_sha = "e" * 64
    records = extract_and_store_person_crops(
        store,
        image=image,
        frame_sha256=frame_sha,
        detections=detections,
        policy=PersonCropPolicy(minimum_sharpness=0.0),
    )
    for detection in detections:
        record = records[str(detection.observation_id)]
        detection.crop_frame_sha256 = frame_sha
        detection.crop_sha256 = record.crop_sha256
        detection.crop_quality = dict(record.quality)
    monkeypatch.setattr(
        "app.person_crop_store.person_crop_store_runtime",
        lambda: (store, PersonCropPolicy(minimum_sharpness=0.0)),
    )
    return records


def test_resolver_stamps_frame_space_contact_and_reports_coverage(
    tmp_path, monkeypatch
):
    detection = _detection("obs-1")
    records = _seed_store(tmp_path, monkeypatch, [detection])
    record = records["obs-1"]
    crop_height = record.padded_rect[3] - record.padded_rect[1]
    crop_width = record.padded_rect[2] - record.padded_rect[0]
    feet_crop_x = crop_width / 2 + 3.0
    feet_crop_y = crop_height - 2.0

    def backend(_crop):
        return _pose(
            {
                20: (feet_crop_x - 2.0, feet_crop_y, 0.9),
                21: (feet_crop_x + 2.0, feet_crop_y, 0.9),
            }
        )

    diagnostics = resolve_pose_contact_points(
        [([detection], 0.0)],
        policy=ContactPointPolicy(minimum_crop_height=8),
        backend_factory=lambda: backend,
    )

    assert diagnostics["status"] == "ready"
    assert diagnostics["poseFeetCount"] == 1
    assert diagnostics["poseFeetRatio"] == 1.0
    assert detection.contact_source == "pose-feet"
    assert detection.contact_image_x == record.padded_rect[0] + feet_crop_x
    assert detection.contact_image_y == record.padded_rect[1] + feet_crop_y


def test_resolver_degrades_explicitly_per_observation(tmp_path, monkeypatch):
    eligible = _detection("obs-ok", x=60.0)
    tiny = _detection("obs-tiny", x=140.0)
    _seed_store(tmp_path, monkeypatch, [eligible, tiny])
    tiny.crop_quality = {**(tiny.crop_quality or {}), "cropHeight": 4}

    def broken_feet_backend(_crop):
        # Feet far outside the bbox deviation gate -> implausible pose.
        return _pose({20: (500.0, 500.0, 0.9), 21: (504.0, 500.0, 0.9)})

    diagnostics = resolve_pose_contact_points(
        [([eligible, tiny], 0.0)],
        policy=ContactPointPolicy(minimum_crop_height=8),
        backend_factory=lambda: broken_feet_backend,
    )
    assert diagnostics["implausibleFeetCount"] == 1
    assert diagnostics["cropTooSmallCount"] == 1
    assert diagnostics["poseFeetCount"] == 0
    assert eligible.contact_source is None

    # A missing pose runtime is a single explicit status, never a crash.
    unavailable = resolve_pose_contact_points(
        [([eligible], 0.0)],
        policy=ContactPointPolicy(),
        backend_factory=lambda: None,
    )
    assert unavailable["status"] == "pose-runtime-unavailable"


def test_metric_projection_prefers_the_pose_contact_point():
    class _Calibration:
        image_to_pitch = np.eye(3, dtype=np.float64)

    pitch = {"length": 1000, "width": 1000}
    with_pose = _detection("obs-a")
    with_pose.contact_image_x = 40.0
    with_pose.contact_image_y = 90.0
    with_pose.contact_source = "pose-feet"
    without_pose = _detection("obs-b")

    attach_metric_positions([with_pose, without_pose], [], _Calibration(), pitch)

    assert (with_pose.pitch_x, with_pose.pitch_z) == (40.0, 90.0)
    assert (without_pose.pitch_x, without_pose.pitch_z) == (60.0, 100.0)
