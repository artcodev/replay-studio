from __future__ import annotations

"""Prepare in-memory identity corrections for a fenced queue publication."""

from datetime import UTC, datetime
from uuid import uuid4

from .reconstruction_artifact_hydration import hydrate_scene_reconstruction
from .reconstruction_frame_annotation_target import resolve_frame_annotation_target
from .reconstruction_identity_annotation_undo_planning import (
    plan_frame_person_annotation_delete,
)
from .reconstruction_identity_annotation_upsert_planning import (
    plan_frame_person_annotation_upsert,
)


def draft_frame_person_annotation_upsert(scene: dict, values: dict) -> dict:
    hydrate_scene_reconstruction(scene)
    requested_id = str(values.get("annotation_id") or "").strip()
    target = resolve_frame_annotation_target(
        scene,
        scene_time=float(values["scene_time"]),
        bbox=values["bbox"],
    )
    return plan_frame_person_annotation_upsert(
        scene,
        values,
        target=target,
        annotation_id=requested_id or f"annotation-{uuid4().hex[:12]}",
        updated_at=datetime.now(UTC).isoformat(),
    )


def draft_frame_person_annotation_delete(scene: dict, annotation_id: str) -> dict:
    hydrate_scene_reconstruction(scene)
    return plan_frame_person_annotation_delete(scene, annotation_id)
