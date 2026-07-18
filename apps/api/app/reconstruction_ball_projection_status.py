from __future__ import annotations

"""Public world-projection status derived from reconstructed ball output."""


def ball_world_projection_status(coordinate_mode: str, keyframes: list[dict]) -> str:
    if coordinate_mode == "unavailable":
        return "calibration-rejected"
    return "published" if keyframes else "no-stable-trajectory"
