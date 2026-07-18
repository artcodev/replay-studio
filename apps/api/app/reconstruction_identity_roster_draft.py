from __future__ import annotations

"""Prepare hydrated roster mutations for fenced queue publication."""

from typing import Mapping

from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_identity_roster_binding_planning import (
    plan_canonical_roster_binding,
)
from .reconstruction_identity_roster_clear_planning import (
    plan_clear_canonical_roster_binding,
)


def draft_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
    external_player_id: str | None,
    *,
    match_snapshot: Mapping[str, object] | None = None,
) -> dict:
    hydrate_scene_reconstruction(scene)
    return plan_canonical_roster_binding(
        scene,
        canonical_person_id,
        external_player_id,
        match_snapshot=match_snapshot,
    )


def draft_clear_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
) -> dict:
    hydrate_scene_reconstruction(scene)
    return plan_clear_canonical_roster_binding(scene, canonical_person_id)
