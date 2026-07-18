"""Always-persist commands for reviewed player-action edits."""

from __future__ import annotations

from .player_action_planning import (
    apply_player_action_delete,
    apply_player_action_upsert,
)
from .scene_repository import scenes


def upsert_player_action(scene: dict, request: dict) -> dict:
    action = apply_player_action_upsert(scene, request)
    scenes.put(scene)
    return action


def delete_player_action(scene: dict, action_id: str) -> dict:
    action = apply_player_action_delete(scene, action_id)
    scenes.put(scene)
    return action
