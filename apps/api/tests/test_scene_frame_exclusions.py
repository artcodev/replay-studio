from __future__ import annotations

from copy import deepcopy

import pytest

from app import reconstruction_inputs
from app.reconstruction_calibration_fingerprint import (
    calibration_input_fingerprint,
)
from app.reconstruction_errors import ReconstructionError
from app.scene_analysis_frame_read import exact_scene_analysis_frame
from app.scene_document import reconstruction_input_fingerprint
import app.scene_frame_exclusion_command as frame_exclusion_command
from app.scene_frame_exclusion_command import set_scene_frame_excluded


def _scene() -> dict:
    return {
        "id": "scene-1",
        "duration": 0.3,
        "payload": {
            "pitch": {"length": 105, "width": 68},
            "videoAsset": {
                "id": "asset-1",
                "fps": 10.0,
                "analysisFps": 10.0,
                "sourceStart": 0.0,
                "sourceEnd": 0.3,
                "generationKey": "generation-1",
                "analysisFrameInput": {
                    "schemaVersion": 1,
                    "source": "uploaded-video",
                    "coordinateSpace": "source-video-pixels",
                    "width": 1920,
                    "height": 1080,
                    "resize": "none",
                },
                "reconstruction": {
                    "status": "ready",
                    "calibrationInputFingerprint": "old-calibration",
                    "calibrationFallbackConsent": {"policy": "explicit-image-fallback"},
                    "pendingCalibrationEditSession": {
                        "schemaVersion": 1,
                        "edits": [],
                    },
                    "pitchCalibrationOverrides": [
                        {
                            "sourceFrameIndex": 2,
                            "sceneTime": 0.1,
                            "imageToPitch": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        },
                        {
                            "sourceFrameIndex": 3,
                            "sceneTime": 0.2,
                            "imageToPitch": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        },
                    ],
                },
            },
        },
    }


@pytest.fixture
def frame_scene_runtime(monkeypatch, tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(1, 5):
        (frames / f"frame_{index:05d}.jpg").write_bytes(f"jpeg-{index}".encode())
    monkeypatch.setattr(
        reconstruction_inputs,
        "video_generation_directory",
        lambda _asset_id, _generation_key: tmp_path,
    )
    monkeypatch.setattr(
        frame_exclusion_command.scenes,
        "put",
        lambda scene: scene,
    )
    return tmp_path


def test_exclusion_is_one_reversible_input_for_calibration_and_reconstruction(
    frame_scene_runtime,
):
    scene = _scene()
    reconstruction_before = reconstruction_input_fingerprint(scene)
    calibration_before = calibration_input_fingerprint(scene)

    set_scene_frame_excluded(scene, 2, excluded=True)

    video = scene["payload"]["videoAsset"]
    reconstruction = video["reconstruction"]
    assert video["frameExclusions"][0]["sourceFrameIndex"] == 2
    assert [path.name for path, _ in reconstruction_inputs.frame_paths(scene)] == [
        "frame_00001.jpg",
        "frame_00003.jpg",
        "frame_00004.jpg",
    ]
    assert [
        item["sourceFrameIndex"]
        for item in reconstruction["pitchCalibrationOverrides"]
    ] == [3]
    assert "pendingCalibrationEditSession" not in reconstruction
    assert "calibrationInputFingerprint" not in reconstruction
    assert "calibrationFallbackConsent" not in reconstruction
    assert reconstruction_input_fingerprint(scene) != reconstruction_before
    assert calibration_input_fingerprint(scene) != calibration_before

    set_scene_frame_excluded(scene, 2, excluded=False)

    assert "frameExclusions" not in video
    assert [path.name for path, _ in reconstruction_inputs.frame_paths(scene)] == [
        "frame_00001.jpg",
        "frame_00002.jpg",
        "frame_00003.jpg",
        "frame_00004.jpg",
    ]


def test_exact_frame_read_is_generation_pinned_and_keeps_excluded_frames(
    frame_scene_runtime,
):
    scene = _scene()
    set_scene_frame_excluded(scene, 2, excluded=True)

    exact = exact_scene_analysis_frame(scene, "generation-1", 2)

    assert exact.name == "frame_00002.jpg"
    with pytest.raises(ReconstructionError, match="older video generation"):
        exact_scene_analysis_frame(scene, "generation-old", 2)


def test_exclusion_refuses_to_remove_every_frame(frame_scene_runtime):
    scene = _scene()
    for index in (1, 2, 3):
        set_scene_frame_excluded(scene, index, excluded=True)

    snapshot = deepcopy(scene)
    with pytest.raises(ReconstructionError, match="retain at least one"):
        set_scene_frame_excluded(scene, 4, excluded=True)
    assert scene == snapshot
