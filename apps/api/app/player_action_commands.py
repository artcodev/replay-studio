"""Always-persist commands for reviewed player-action edits."""

from __future__ import annotations

from .player_action_planning import (
    apply_player_action_delete,
    apply_player_action_upsert,
)
from .scene_repository import scenes


def upsert_player_action(scene: dict, request: dict) -> dict:
    """Apply one player-action edit and return the persisted Scene document."""

    apply_player_action_upsert(scene, request)
    return scenes.put(scene)


def delete_player_action(scene: dict, action_id: str) -> dict:
    """Delete one player action and return the persisted Scene document."""

    apply_player_action_delete(scene, action_id)
    return scenes.put(scene)
