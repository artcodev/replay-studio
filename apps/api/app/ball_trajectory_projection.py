"""Projection of selected image-space ball candidates onto the editor pitch."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .ball_tracking_candidates import BallCandidate, finite_number
from .ball_tracking_contract import PositionProjector


def _approximate_screen_projection(
    candidate: BallCandidate,
    frame_size: tuple[int, int],
    pitch: Mapping[str, Any],
) -> dict[str, Any]:
    width, height = frame_size
    length = finite_number(pitch.get("length"))
    pitch_width = finite_number(pitch.get("width"))
    if width <= 0 or height <= 0 or length is None or pitch_width is None:
        raise ValueError("frame_size and pitch dimensions must be positive")
    x = (candidate.image_x / width - 0.5) * length * 0.96
    z = (candidate.image_y / height - 0.5) * pitch_width * 1.05
    return {
        "x": max(-length / 2.0, min(length / 2.0, x)),
        "z": max(-pitch_width / 2.0, min(pitch_width / 2.0, z)),
        "projectionSource": "screen-approximate",
        "calibrationFrameIndex": None,
        "positionUncertaintyMetres": 14.0,
    }


def project_ball_candidate(
    candidate: BallCandidate,
    frame_size: tuple[int, int],
    pitch: Mapping[str, Any],
    projector: PositionProjector | None,
) -> dict[str, Any]:
    """Resolve direct, custom, or approximate world coordinates for a candidate."""

    if candidate.pitch_x is not None and candidate.pitch_z is not None:
        result: dict[str, Any] = {
            "x": candidate.pitch_x,
            "z": candidate.pitch_z,
            "projectionSource": str(candidate.source.get("projectionSource") or "direct"),
            "calibrationFrameIndex": candidate.source.get("calibrationFrameIndex"),
            "positionUncertaintyMetres": candidate.source.get(
                "positionUncertaintyMetres"
            ),
        }
    elif projector is None:
        result = _approximate_screen_projection(candidate, frame_size, pitch)
    else:
        projected = projector(candidate.source)
        if isinstance(projected, Mapping):
            result = deepcopy(dict(projected))
        elif isinstance(projected, (tuple, list)) and len(projected) == 2:
            result = {"x": projected[0], "z": projected[1]}
        else:
            raise ValueError("projector must return (x, z) or a mapping")
    x = finite_number(result.get("x"))
    z = finite_number(result.get("z"))
    if x is None or z is None:
        raise ValueError("projector returned no finite x/z coordinates")
    result["x"], result["z"] = x, z
    source = str(result.get("projectionSource") or "projector")
    result["projectionSource"] = source
    result.setdefault("calibrationFrameIndex", None)
    result.setdefault("positionUncertaintyMetres", None)
    result["projection"] = {
        "source": source,
        "calibrationFrameIndex": result.get("calibrationFrameIndex"),
        "uncertaintyMetres": result.get("positionUncertaintyMetres"),
    }
    return result


__all__ = ["project_ball_candidate"]
