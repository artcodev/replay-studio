from __future__ import annotations

"""Persist one planned frame-person annotation."""

from .reconstruction_identity_annotation_commit import commit_identity_annotation_scene
from .reconstruction_identity_annotation_draft import (
    draft_frame_person_annotation_upsert,
)


def upsert_frame_person_annotation(scene: dict, values: dict) -> dict:
    annotation = draft_frame_person_annotation_upsert(scene, values)
    commit_identity_annotation_scene(scene)
    return annotation
