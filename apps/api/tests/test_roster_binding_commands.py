from __future__ import annotations

import pytest

import app.reconstruction_identity_roster_commands as commands
from app.reconstruction_errors import ReconstructionError


def test_set_roster_binding_command_always_commits_planned_scene(monkeypatch) -> None:
    scene = {"id": "scene-1"}
    annotation = {"id": "binding-1"}
    calls: list[tuple[str, object]] = []

    def draft(value, canonical_person_id, external_player_id, *, match_snapshot=None):
        calls.append(
            (
                "draft",
                (value, canonical_person_id, external_player_id, match_snapshot),
            )
        )
        return annotation

    monkeypatch.setattr(commands, "draft_canonical_roster_binding", draft)
    monkeypatch.setattr(
        commands,
        "commit_identity_annotation_scene",
        lambda value: calls.append(("commit", value)),
    )

    result = commands.set_canonical_roster_binding(
        scene,
        "person-1",
        "player-8",
        match_snapshot={"id": "match-1"},
    )

    assert result is annotation
    assert calls == [
        ("draft", (scene, "person-1", "player-8", {"id": "match-1"})),
        ("commit", scene),
    ]


def test_clear_roster_binding_command_always_commits_planned_scene(monkeypatch) -> None:
    scene = {"id": "scene-1"}
    annotation = {"id": "unbind-1"}
    calls: list[tuple[str, object]] = []

    def draft(value, person_id):
        calls.append(("draft", (value, person_id)))
        return annotation

    monkeypatch.setattr(
        commands,
        "draft_clear_canonical_roster_binding",
        draft,
    )
    monkeypatch.setattr(
        commands,
        "commit_identity_annotation_scene",
        lambda value: calls.append(("commit", value)),
    )

    result = commands.clear_canonical_roster_binding(scene, "person-1")

    assert result is annotation
    assert calls == [
        ("draft", (scene, "person-1")),
        ("commit", scene),
    ]


def test_roster_command_does_not_commit_a_rejected_plan(monkeypatch) -> None:
    committed = False

    def reject(*_args, **_kwargs):
        raise ReconstructionError("invalid owner")

    def commit(_scene):
        nonlocal committed
        committed = True

    monkeypatch.setattr(commands, "draft_canonical_roster_binding", reject)
    monkeypatch.setattr(commands, "commit_identity_annotation_scene", commit)

    with pytest.raises(ReconstructionError, match="invalid owner"):
        commands.set_canonical_roster_binding({}, "person-1", "player-8")

    assert committed is False
