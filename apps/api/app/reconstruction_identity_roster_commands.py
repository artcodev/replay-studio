from __future__ import annotations

"""Always-persist commands for canonical roster decisions."""

from typing import Mapping

from .reconstruction_identity_annotation_commit import (
    commit_identity_annotation_scene,
)
from .reconstruction_identity_roster_draft import (
    draft_canonical_roster_binding,
    draft_clear_canonical_roster_binding,
)


def set_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
    external_player_id: str | None,
    *,
    match_snapshot: Mapping[str, object] | None = None,
) -> dict:
    """Persist one Set/Unbind roster decision and its dense artifacts."""

    annotation = draft_canonical_roster_binding(
        scene,
        canonical_person_id,
        external_player_id,
        match_snapshot=match_snapshot,
    )
    commit_identity_annotation_scene(scene)
    return annotation


def clear_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
) -> dict:
    """Persist removal of one explicit Unbind roster decision."""

    annotation = draft_clear_canonical_roster_binding(scene, canonical_person_id)
    commit_identity_annotation_scene(scene)
    return annotation
