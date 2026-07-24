from __future__ import annotations

import pytest

import app.segment_layout_command as segment_layout_command
from app.reconstruction_errors import ReconstructionError
from app.segment_layout_command import set_scene_segment_layout


@pytest.fixture(autouse=True)
def captured_writes(monkeypatch):
    written: list[dict] = []
    monkeypatch.setattr(
        segment_layout_command.scenes,
        "put",
        lambda scene: (written.append(scene), scene)[1],
    )
    return written


def _scene(status: str | None = None) -> dict:
    return {
        "id": "video-1",
        "payload": {
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {"status": status} if status else {},
                "segments": [
                    {"id": "shot-01", "layout": {"group": 1, "variant": "A"}},
                    {"id": "shot-02", "layout": {"group": 1, "variant": "B"}},
                ],
                "segmentLayout": {"status": "proposed", "groups": []},
            }
        },
    }


def _entry(segment_id: str, group: int, role: str = "original") -> dict:
    return {
        "id": segment_id,
        "group": group,
        "variant": "A",
        "label": f"{group}-A",
        "role": role,
        "confidence": 1.0,
    }


def test_layout_is_applied_onto_the_loaded_scene_regardless_of_client_revision():
    # The command never inspects a client-supplied revision: it edits the
    # scene the route just loaded, so a stale editor still saves.
    scene = _scene()
    saved = set_scene_segment_layout(
        scene,
        [_entry("shot-02", 2, role="replay")],
        "confirmed",
    )

    video = saved["payload"]["videoAsset"]
    assert video["segments"][1]["layout"] == {
        "group": 2,
        "variant": "A",
        "label": "2-A",
        "role": "replay",
        "confidence": 1.0,
    }
    # Untouched segments keep their stored layout.
    assert video["segments"][0]["layout"] == {"group": 1, "variant": "A"}
    assert video["segmentLayout"]["status"] == "confirmed"


def test_unknown_segments_fail_closed_instead_of_silently_dropping_edits():
    with pytest.raises(ReconstructionError, match="Unknown timeline segments: shot-99"):
        set_scene_segment_layout(_scene(), [_entry("shot-99", 2)], "edited")


def test_running_reconstruction_blocks_the_layout_write():
    with pytest.raises(ReconstructionError, match="Wait for reconstruction"):
        set_scene_segment_layout(
            _scene(status="processing"), [_entry("shot-01", 1)], "edited"
        )


def test_scene_without_segments_is_rejected():
    scene = _scene()
    scene["payload"]["videoAsset"]["segments"] = []
    with pytest.raises(ReconstructionError, match="no timeline segments"):
        set_scene_segment_layout(scene, [_entry("shot-01", 1)], "edited")
