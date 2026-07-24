from __future__ import annotations

import pytest

import app.scene_metadata_command as scene_metadata_command
from app.reconstruction_errors import ReconstructionError
from app.scene_metadata_command import (
    set_scene_event_bindings,
    set_scene_title,
    set_track_metadata,
)


@pytest.fixture(autouse=True)
def stored_writes(monkeypatch):
    written: list[dict] = []
    monkeypatch.setattr(
        scene_metadata_command.scenes,
        "put",
        lambda scene: (written.append(scene), scene)[1],
    )
    return written


def _scene(status: str | None = None) -> dict:
    return {
        "id": "video-1",
        "title": "Old title",
        "duration": 10.0,
        "payload": {
            "eventBindings": [],
            "tracks": [
                {"id": "auto-home-02", "label": "Home track 02", "number": 2},
            ],
            "videoAsset": {
                "id": "asset-1",
                "reconstruction": {"status": status} if status else {},
            },
        },
    }


def test_each_command_edits_only_its_own_domain(stored_writes):
    scene = _scene()

    set_scene_title(scene, "  Shot 02 review  ")
    set_scene_event_bindings(
        scene,
        [
            {
                "sceneTime": 2.5,
                "externalEventId": "event-9",
                "label": "Goal",
                "type": "goal",
            }
        ],
    )
    set_track_metadata(scene, "auto-home-02", label="Pedri", number=9)

    assert scene["title"] == "Shot 02 review"
    assert scene["payload"]["eventBindings"] == [
        {
            "sceneTime": 2.5,
            "externalEventId": "event-9",
            "label": "Goal",
            "type": "goal",
        }
    ]
    track = scene["payload"]["tracks"][0]
    assert track["label"] == "Pedri" and track["number"] == 9
    assert len(stored_writes) == 3


def test_commands_never_require_a_client_revision():
    # The command signature carries no revision at all: the caller edits the
    # scene the route just loaded, so a stale editor cannot be refused.
    scene = _scene()
    scene["revision"] = 3
    set_scene_title(scene, "Renamed")
    assert scene["revision"] == 3


def test_running_reconstruction_blocks_every_command():
    for command in (
        lambda scene: set_scene_title(scene, "x"),
        lambda scene: set_scene_event_bindings(scene, []),
        lambda scene: set_track_metadata(scene, "auto-home-02", number=1),
    ):
        with pytest.raises(ReconstructionError, match="Wait for reconstruction"):
            command(_scene(status="processing"))


def test_invalid_edits_fail_closed():
    with pytest.raises(ReconstructionError, match="title cannot be empty"):
        set_scene_title(_scene(), "   ")
    with pytest.raises(ReconstructionError, match="Unknown track"):
        set_track_metadata(_scene(), "auto-away-99", number=4)
    with pytest.raises(ReconstructionError, match="outside the scene"):
        set_scene_event_bindings(
            _scene(),
            [
                {
                    "sceneTime": -1.0,
                    "externalEventId": "e",
                    "label": "l",
                    "type": "t",
                }
            ],
        )
