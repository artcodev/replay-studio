from __future__ import annotations

"""Derive deterministic canonical-person identifiers from identity evidence."""

from hashlib import sha256

from .reconstruction_track_state import TrackState


def derive_canonical_person_id(track: TrackState) -> str:
    if track.annotation_ids:
        seed = "annotation:" + ",".join(sorted(track.annotation_ids))
    else:
        first = min(
            track.points,
            key=lambda point: (float(point["t"]), int(point["frameIndex"])),
        )
        bbox = first.get("bbox") or {}
        seed = ":".join(
            [
                str(first.get("frameIndex")),
                str(round(float(bbox.get("x") or 0.0))),
                str(round(float(bbox.get("y") or 0.0))),
                str(round(float(bbox.get("width") or 0.0))),
                str(round(float(bbox.get("height") or 0.0))),
            ]
        )
    return f"canonical-{sha256(seed.encode('utf-8')).hexdigest()[:12]}"


__all__ = ["derive_canonical_person_id"]
