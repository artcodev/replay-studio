from __future__ import annotations

"""Persist deletion and safe undo of one frame-person correction."""

from .reconstruction_identity_annotation_commit import commit_identity_annotation_scene
from .reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_delete,
)


def delete_frame_person_annotation(scene: dict, annotation_id: str) -> dict:
    annotation = draft_frame_person_annotation_delete(scene, annotation_id)
    commit_identity_annotation_scene(scene)
    return annotation
