from pathlib import Path

from app.reconstruction_frame_annotation_contract import FrameAnnotationTarget
from app.reconstruction_identity_annotation_undo_planning import (
    plan_frame_person_annotation_delete,
)
from app.reconstruction_identity_annotation_upsert_planning import (
    plan_frame_person_annotation_upsert,
)
import app.reconstruction_identity_annotation_delete_command as delete_command
import app.reconstruction_identity_annotation_upsert_command as upsert_command


def test_pure_annotation_planners_upsert_and_delete_without_io() -> None:
    scene = {
        "id": "scene-plan",
        "duration": 2.0,
        "payload": {
            "videoAsset": {
                "sourceStart": 10.0,
                "reconstruction": {"status": "ready", "frameAnnotations": []},
            }
        },
    }
    annotation = plan_frame_person_annotation_upsert(
        scene,
        {
            "scene_time": 0.0,
            "bbox": {"x": 20, "y": 30, "width": 18, "height": 42},
            "kind": "home-player",
            "action": "confirm",
            "scope": "observation",
        },
        target=FrameAnnotationTarget(
            path=Path("frame_00001.jpg"),
            scene_time=0.0,
            frame_index=1,
            x=20.0,
            y=30.0,
            width=18.0,
            height=42.0,
        ),
        annotation_id="annotation-plan",
        updated_at="2026-07-18T00:00:00+00:00",
    )

    assert annotation["id"] == "annotation-plan"
    assert plan_frame_person_annotation_delete(scene, annotation["id"]) == annotation
    assert scene["payload"]["videoAsset"]["reconstruction"]["frameAnnotations"] == []


def test_annotation_commands_always_commit_their_draft(monkeypatch) -> None:
    scene = {"id": "scene-command"}
    committed: list[dict] = []
    monkeypatch.setattr(
        upsert_command,
        "draft_frame_person_annotation_upsert",
        lambda received, _values: {"id": "created", "scene": received["id"]},
    )
    monkeypatch.setattr(
        delete_command,
        "draft_frame_person_annotation_delete",
        lambda received, annotation_id: {"id": annotation_id, "scene": received["id"]},
    )
    monkeypatch.setattr(
        upsert_command,
        "commit_identity_annotation_scene",
        lambda received: committed.append(received),
    )
    monkeypatch.setattr(
        delete_command,
        "commit_identity_annotation_scene",
        lambda received: committed.append(received),
    )

    assert upsert_command.upsert_frame_person_annotation(scene, {})["id"] == "created"
    assert delete_command.delete_frame_person_annotation(scene, "deleted")["id"] == "deleted"
    assert committed == [scene, scene]
