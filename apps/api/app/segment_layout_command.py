from __future__ import annotations

"""Application command for publishing an edited timeline segment layout.

Event grouping is a sync-style edit: it is idempotent, never part of the
reconstruction input fingerprint, and has a single writer. Publishing it
through a dedicated command lets the server apply it onto the *current*
stored scene, so a client holding an older revision can still save instead
of dead-ending on the whole-document revision fence.
"""

from typing import Mapping, Sequence

from .reconstruction_errors import ReconstructionError
from .scene_repository import scenes


LAYOUT_FIELDS = ("group", "variant", "label", "role", "confidence", "motionCost")


def set_scene_segment_layout(
    scene: dict,
    entries: Sequence[Mapping[str, object]],
    status: str,
) -> dict:
    """Apply per-segment layout onto the freshly loaded scene and persist."""

    video = scene.get("payload", {}).get("videoAsset")
    if not isinstance(video, dict):
        raise ReconstructionError("Scene has no source video")
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise ReconstructionError(
            "Wait for reconstruction to finish before editing the timeline layout"
        )
    segments = video.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ReconstructionError("Scene has no timeline segments to group")

    by_id = {str(segment.get("id")): segment for segment in segments}
    unknown = [
        str(entry.get("id"))
        for entry in entries
        if str(entry.get("id")) not in by_id
    ]
    if unknown:
        # A stale client must not silently drop edits onto a rebuilt
        # timeline whose segment ids no longer exist.
        raise ReconstructionError(
            f"Unknown timeline segments: {', '.join(sorted(unknown))}"
        )

    for entry in entries:
        segment = by_id[str(entry["id"])]
        segment["layout"] = {
            field: entry[field] for field in LAYOUT_FIELDS if field in entry
        }
    layout = video.get("segmentLayout")
    if isinstance(layout, dict):
        layout["status"] = status
    return scenes.put(scene)


__all__ = ("set_scene_segment_layout",)
