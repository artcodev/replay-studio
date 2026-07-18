from __future__ import annotations

"""Pure in-memory Clear planning for canonical roster decisions."""

from .reconstruction_identity_roster_lineage import (
    active_merge_dependencies,
    active_split_dependencies,
)
from .reconstruction_identity_roster_ownership import (
    canonical_person_for_binding,
    roster_correction_for_clear,
)
from .reconstruction_identity_roster_undo_planning import (
    plan_clear_unbound_roster_correction,
)


def plan_clear_canonical_roster_binding(
    scene: dict,
    canonical_person_id: str,
) -> dict:
    """Plan Clear without publishing artifacts or writing the scene."""

    normalized_person_id = str(canonical_person_id or "").strip()
    person = canonical_person_for_binding(scene, normalized_person_id)
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction")
        or {}
    )
    annotations = list(reconstruction.get("frameAnnotations") or [])
    correction = roster_correction_for_clear(
        scene,
        annotations,
        normalized_person_id,
    )
    return plan_clear_unbound_roster_correction(
        scene,
        correction,
        active_merge_ids=active_merge_dependencies(scene, person, annotations),
        active_split_ids=active_split_dependencies(scene, person, annotations),
    )
