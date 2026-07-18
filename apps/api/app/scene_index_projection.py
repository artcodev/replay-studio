from __future__ import annotations

"""Relational projection of compact SceneDocument navigation metadata."""

from .database import SceneRow
from .scene_document import scene_index_values


def sync_scene_index(row: SceneRow, scene: dict) -> None:
    values = scene_index_values(scene)
    row.duration = float(values["duration"] or 0.0)
    row.kind = str(values["kind"] or "demo")
    row.parent_scene_id = (
        str(values["parent_scene_id"])
        if values["parent_scene_id"] is not None
        else None
    )
    row.selected_segment_id = (
        str(values["selected_segment_id"])
        if values["selected_segment_id"] is not None
        else None
    )
