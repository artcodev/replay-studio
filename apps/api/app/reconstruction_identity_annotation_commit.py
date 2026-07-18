from __future__ import annotations

"""Shared artifact and repository commit boundary for identity corrections."""

from .reconstruction_artifact_publication import publish_dense_reconstruction_artifacts
from .scene_repository import scenes


def commit_identity_annotation_scene(scene: dict) -> dict:
    publish_dense_reconstruction_artifacts(scene)
    return scenes.put(scene)
