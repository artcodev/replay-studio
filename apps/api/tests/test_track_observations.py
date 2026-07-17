from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.reconstruction import (
    Detection,
    ReconstructionError,
    TrackState,
    _capture_detection_observations,
    _merge_scene_track_documents,
    _scene_tracks,
    analyze_scene_frame,
)


FRAME_INDEX = 101
FRAME_PATH = Path(f"/tmp/frame_{FRAME_INDEX:05d}.jpg")


def _automatic_detection(x: float, y: float) -> Detection:
    return Detection(
        x=x,
        y=y,
        width=20.0,
        height=40.0,
        confidence=0.8,
        feature=np.zeros(12, dtype=np.float32),
    )


def _track(
    track_id: str,
    x: float,
    z: float,
    *,
    observations: list[dict] | None = None,
    observed: bool = True,
) -> dict:
    return {
        "id": track_id,
        "label": track_id,
        "teamId": "home",
        "color": "#ffffff",
        "keyframes": [
            {
                "t": 0.0,
                "x": x,
                "z": z,
                "confidence": 0.9 if observed else 0.18,
                "observed": observed,
                "presenceState": "observed" if observed else "inferred-gap",
            }
        ],
        **({"observations": observations} if observations is not None else {}),
    }


def _observation(x: float, y: float, pitch_x: float, pitch_z: float, **values) -> dict:
    return {
        "frameIndex": FRAME_INDEX,
        "sceneTime": 0.0,
        "bbox": {"x": x, "y": y, "width": 20.0, "height": 40.0},
        "pitch": {"x": pitch_x, "z": pitch_z},
        "confidence": 0.8,
        "annotationId": None,
        **values,
    }


def _scene(tracks: list[dict], *, observation_schema: bool) -> dict:
    reconstruction = {
        "status": "ready",
        "model": "test-model",
        "pitchCalibration": {"status": "fallback"},
    }
    if observation_schema:
        reconstruction["trackObservationSchemaVersion"] = 2
    return {
        "id": "scene-observations",
        "duration": 1.0,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "tracks": tracks,
            "ball": {"keyframes": []},
            "videoAsset": {
                "sourceStart": 0.0,
                "reconstruction": reconstruction,
            },
        },
    }


def _patch_analysis(monkeypatch, detections: list[Detection]) -> None:
    result = SimpleNamespace(orig_img=np.zeros((540, 960, 3), dtype=np.uint8))
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(FRAME_PATH, 0.0)])
    monkeypatch.setattr("app.reconstruction._load_model", lambda _: object())
    monkeypatch.setattr("app.reconstruction._predict_frame", lambda *_: result)
    monkeypatch.setattr(
        "app.reconstruction._person_detections",
        lambda _: (detections, []),
    )


def test_raw_bbox_and_source_frame_survive_camera_stabilization(monkeypatch):
    detection = Detection(
        x=110.0,
        y=200.0,
        width=20.0,
        height=40.0,
        confidence=0.8,
        feature=np.zeros(12, dtype=np.float32),
        annotation_id="manual-player",
        annotation_kind="home-player",
        pitch_x=4.0,
        pitch_z=2.0,
    )
    _capture_detection_observations([detection], FRAME_INDEX)
    detection.x = 410.0
    detection.y = 350.0
    track = TrackState(id=1)
    track.append(detection, frame_index=0, time=0.0)

    point = track.points[0]
    assert point["frameIndex"] == FRAME_INDEX
    assert point["bbox"] == {"x": 100.0, "y": 160.0, "width": 20.0, "height": 40.0}
    assert (point["px"], point["py"]) == (410.0, 350.0)

    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(FRAME_PATH, 0.0)])
    rendered = _scene_tracks(
        [track],
        {1: "home"},
        {"home": "#ffffff"},
        (960, 540),
        {
            "duration": 1.0,
            "payload": {
                "pitch": {"length": 105, "width": 68},
                "videoAsset": {"sourceStart": 0.0},
            },
        },
        coordinate_mode="metric",
    )

    assert rendered[0]["observations"] == [
        {
            "frameIndex": FRAME_INDEX,
            "sceneTime": 0.0,
            "bbox": {"x": 100.0, "y": 160.0, "width": 20.0, "height": 40.0},
            "pitch": {"x": 4.0, "z": 2.0},
            "confidence": 0.8,
            "annotationId": "manual-player",
            "metricStatus": "accepted",
            "metricReason": None,
            "projectionSource": "direct",
        }
    ]


def test_automatic_observation_ids_do_not_depend_on_detector_result_order() -> None:
    first = _automatic_detection(120.0, 210.0)
    second = _automatic_detection(420.0, 310.0)
    _capture_detection_observations([first, second], FRAME_INDEX)
    ids_by_x = {item.x: item.observation_id for item in (first, second)}

    rebuilt_first = _automatic_detection(120.0, 210.0)
    rebuilt_second = _automatic_detection(420.0, 310.0)
    _capture_detection_observations(
        [rebuilt_second, rebuilt_first],
        FRAME_INDEX,
    )

    assert rebuilt_first.observation_id == ids_by_x[120.0]
    assert rebuilt_second.observation_id == ids_by_x[420.0]
    assert rebuilt_first.observation_id != rebuilt_second.observation_id


def test_indistinguishable_duplicate_observations_fail_closed() -> None:
    first = _automatic_detection(120.0, 210.0)
    duplicate = _automatic_detection(120.0, 210.0)

    with pytest.raises(ReconstructionError, match="indistinguishable duplicate"):
        _capture_detection_observations([first, duplicate], FRAME_INDEX)


def test_rejected_metric_fragment_keeps_authoritative_video_identity(monkeypatch):
    track = TrackState(id=1)
    samples = [
        (101, 0.0, 110.0, -40.0),
        (102, 0.1, 210.0, 40.0),
        (103, 0.2, 310.0, 40.5),
        (104, 0.3, 410.0, 41.0),
    ]
    for index, (frame_index, time, image_x, pitch_x) in enumerate(samples):
        detection = Detection(
            x=image_x,
            y=200.0,
            width=20.0,
            height=40.0,
            confidence=0.8,
            feature=np.zeros(12, dtype=np.float32),
            annotation_id="manual-player" if index == 0 else None,
            annotation_kind="home-player" if index == 0 else None,
            pitch_x=pitch_x,
            pitch_z=0.0,
            projection_source="direct",
            calibration_frame_index=frame_index,
            position_uncertainty_metres=0.8,
        )
        _capture_detection_observations([detection], frame_index)
        track.append(detection, frame_index=index, time=time)

    monkeypatch.setattr(
        "app.reconstruction._frame_paths",
        lambda _: [(Path(f"/tmp/frame_{index:05d}.jpg"), time) for index, time, *_ in samples],
    )
    rendered = _scene_tracks(
        [track],
        {1: "home"},
        {"home": "#ffffff"},
        (960, 540),
        {
            "duration": 1.0,
            "payload": {
                "pitch": {"length": 105, "width": 68},
                "videoAsset": {"sourceStart": 0.0},
            },
        },
        coordinate_mode="metric",
    )

    published = rendered[0]
    assert len(published["observations"]) == 4
    rejected = published["observations"][0]
    assert rejected["frameIndex"] == FRAME_INDEX
    assert rejected["bbox"]["x"] == 100.0
    assert rejected["metricStatus"] == "rejected"
    assert rejected["metricReason"] == "trajectory-fragment-rejected"
    assert rejected["rawPitch"] == {"x": -40.0, "z": 0.0}
    assert "pitch" not in rejected
    assert published["trajectoryQa"]["publishedIdentityObservationCount"] == 4
    assert published["trajectoryQa"]["metricRejectedObservationCount"] == 1

    inferred_competitor = _track(
        "inferred-competitor",
        -40.0,
        0.0,
        observations=[],
        observed=False,
    )
    scene = _scene([published, inferred_competitor], observation_schema=True)
    _patch_analysis(
        monkeypatch,
        [Detection(110, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32))],
    )

    analysis = analyze_scene_frame(scene, 0.0)
    person = next(item for item in analysis["people"] if item["bbox"]["x"] == 100.0)
    assert person["matchedTrackId"] == published["id"]
    assert person["metricStatus"] == "rejected"
    assert person["metricReason"] == "trajectory-fragment-rejected"
    assert person["positionSource"] == "track-inferred"
    assert person["rawPitch"] == {"x": -40.0, "z": 0.0}


def test_unprojected_image_observation_is_published_without_fake_pitch(monkeypatch):
    track = TrackState(id=1)
    projected = Detection(
        110,
        200,
        20,
        40,
        0.8,
        np.zeros(12, dtype=np.float32),
        annotation_id="manual-player",
        annotation_kind="home-player",
        pitch_x=1.0,
        pitch_z=2.0,
    )
    missing_metric = Detection(
        210,
        200,
        20,
        40,
        0.8,
        np.zeros(12, dtype=np.float32),
    )
    for sample_index, detection in enumerate((projected, missing_metric)):
        _capture_detection_observations([detection], FRAME_INDEX + sample_index)
        track.append(detection, frame_index=sample_index, time=sample_index * 0.1)
    monkeypatch.setattr("app.reconstruction._frame_paths", lambda _: [(FRAME_PATH, 0.0)])

    published = _scene_tracks(
        [track],
        {1: "home"},
        {"home": "#ffffff"},
        (960, 540),
        {
            "duration": 1.0,
            "payload": {
                "pitch": {"length": 105, "width": 68},
                "videoAsset": {"sourceStart": 0.0},
            },
        },
        coordinate_mode="metric",
    )[0]

    unprojected = published["observations"][1]
    assert unprojected["metricStatus"] == "unprojected"
    assert unprojected["metricReason"] == "metric-projection-unavailable"
    assert "pitch" not in unprojected
    assert "rawPitch" not in unprojected


def test_persisted_bbox_identity_cannot_be_swapped_by_wrong_pitch_positions(monkeypatch):
    tracks = [
        _track("track-left-video", 35.0, 0.0, observations=[_observation(100, 160, 35, 0)]),
        _track("track-right-video", -39.0, 0.0, observations=[_observation(800, 160, -39, 0)]),
    ]
    detections = [
        Detection(110, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32)),
        Detection(810, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32)),
    ]
    _patch_analysis(monkeypatch, detections)

    result = analyze_scene_frame(_scene(tracks, observation_schema=True), 0.0)
    by_x = {person["bbox"]["x"]: person for person in result["people"]}

    assert by_x[100]["matchedTrackId"] == "track-left-video"
    assert by_x[800]["matchedTrackId"] == "track-right-video"
    assert {person["matchSource"] for person in result["people"]} == {
        "persisted-observation"
    }


def test_stored_observation_is_returned_when_fresh_detector_misses(monkeypatch):
    tracks = [
        _track("visible", 1.0, 1.0, observations=[_observation(100, 160, 1, 1)]),
        _track("detector-missed", 3.0, 2.0, observations=[_observation(300, 180, 3, 2)]),
    ]
    _patch_analysis(
        monkeypatch,
        [Detection(110, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32))],
    )

    result = analyze_scene_frame(_scene(tracks, observation_schema=True), 0.0)

    assert {person["matchedTrackId"] for person in result["people"]} == {
        "visible",
        "detector-missed",
    }
    missed = next(person for person in result["people"] if person["matchedTrackId"] == "detector-missed")
    assert missed["bbox"]["x"] == 300
    assert missed["matchSource"] == "persisted-observation"


def test_inferred_track_without_observation_never_claims_video_bbox(monkeypatch):
    track = _track("inferred-only", -39.0, -8.8, observations=[], observed=False)
    _patch_analysis(
        monkeypatch,
        [Detection(110, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32))],
    )

    result = analyze_scene_frame(_scene([track], observation_schema=True), 0.0)

    assert len(result["people"]) == 1
    assert result["people"][0]["matchedTrackId"] is None
    assert result["people"][0]["observationId"] is None


def test_legacy_scene_without_exact_frame_calibration_remains_unmatched(monkeypatch):
    track = _track("legacy-nearby", -39.0, -8.8, observed=True)
    _patch_analysis(
        monkeypatch,
        [Detection(110, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32))],
    )

    result = analyze_scene_frame(_scene([track], observation_schema=False), 0.0)

    assert result["people"][0]["matchedTrackId"] is None
    assert result["identityLinking"]["mode"] == "rebuild-required"
    assert any("rebuild tracks" in warning.lower() for warning in result["warnings"])


def test_legacy_exact_observed_frame_match_is_conservative(monkeypatch):
    scene = _scene([_track("legacy-exact", 0.0, 0.0, observed=True)], observation_schema=False)
    scene["payload"]["videoAsset"]["reconstruction"]["calibration"] = {
        "frameEvidence": [
            {
                "sourceFrameIndex": FRAME_INDEX,
                "status": "accepted",
                "solutionStatus": "direct-accepted",
                "confidence": 0.9,
                "imageToPitch": [
                    [0.1, 0.0, -10.0],
                    [0.0, 0.1, -20.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        ]
    }
    _patch_analysis(
        monkeypatch,
        [Detection(100, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32))],
    )

    result = analyze_scene_frame(scene, 0.0)

    assert result["people"][0]["matchedTrackId"] == "legacy-exact"
    assert result["people"][0]["matchSource"] == "legacy-observed-frame"


def test_legacy_exact_frame_ambiguity_fails_closed(monkeypatch):
    scene = _scene(
        [
            _track("legacy-a", 0.0, 0.0, observed=True),
            _track("legacy-b", 0.5, 0.0, observed=True),
        ],
        observation_schema=False,
    )
    scene["payload"]["videoAsset"]["reconstruction"]["calibration"] = {
        "frameEvidence": [
            {
                "sourceFrameIndex": FRAME_INDEX,
                "status": "accepted",
                "solutionStatus": "direct-accepted",
                "confidence": 0.9,
                "imageToPitch": [
                    [0.1, 0.0, -10.0],
                    [0.0, 0.1, -20.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        ]
    }
    _patch_analysis(
        monkeypatch,
        [Detection(100, 200, 20, 40, 0.8, np.zeros(12, dtype=np.float32))],
    )

    result = analyze_scene_frame(scene, 0.0)

    assert result["people"][0]["matchedTrackId"] is None


def test_scene_identity_merge_unions_observations_by_frame_and_prefers_manual():
    target_observation = _observation(
        100,
        160,
        0,
        0,
        confidence=0.95,
        metricStatus="accepted",
        metricReason=None,
    )
    manual_rejected = _observation(
        120,
        160,
        0.2,
        0,
        confidence=0.4,
        annotationId="manual-source",
        metricStatus="rejected",
        metricReason="trajectory-fragment-rejected",
        rawPitch={"x": 0.2, "z": 0.0},
    )
    manual_rejected.pop("pitch")
    target = {
        "id": "target",
        "annotationIds": [],
        "keyframes": [{"t": 0.0, "x": 0.0, "z": 0.0, "confidence": 0.9, "observed": True}],
        "observations": [target_observation],
    }
    source = {
        "id": "source",
        "annotationIds": ["manual-source"],
        "keyframes": [{"t": 0.2, "x": 1.0, "z": 0.0, "confidence": 0.8, "observed": True}],
        "observations": [
            manual_rejected,
            {
                **_observation(300, 160, 1, 0),
                "frameIndex": 102,
                "sceneTime": 0.2,
            },
        ],
    }

    merged = _merge_scene_track_documents(
        target,
        source,
        {"id": "manual-source"},
        {"duration": 1.0, "payload": {"pitch": {"length": 105, "width": 68}}},
    )

    assert [item["frameIndex"] for item in merged["observations"]] == [101, 102]
    assert merged["observations"][0]["annotationId"] == "manual-source"
    assert merged["observations"][0]["bbox"]["x"] == 120
    assert merged["observations"][0]["metricStatus"] == "rejected"
    assert merged["observations"][0]["rawPitch"] == {"x": 0.2, "z": 0.0}
