from __future__ import annotations

"""Canonical read/update rules for manual frame-calibration observations."""

from copy import deepcopy


def manual_override_key(override: dict) -> tuple[str, int | float]:
    if override.get("sourceFrameIndex") is not None:
        return "source-frame", int(override["sourceFrameIndex"])
    if override.get("sampleIndex") is not None:
        return "sample", int(override["sampleIndex"])
    return "scene-time", round(float(override.get("sceneTime") or 0.0), 3)


def manual_pitch_calibration_overrides(reconstruction: dict) -> list[dict]:
    """Return the authoritative, de-duplicated manual observation collection."""

    result: list[dict] = []
    seen: set[tuple[str, int | float]] = set()
    collection = reconstruction.get("pitchCalibrationOverrides")
    if isinstance(collection, list):
        for item in collection:
            if not isinstance(item, dict) or not item.get("imageToPitch"):
                continue
            key = manual_override_key(item)
            if key in seen:
                continue
            result.append(deepcopy(item))
            seen.add(key)
    result.sort(
        key=lambda item: (
            float(item.get("sceneTime") or 0.0),
            int(item.get("sourceFrameIndex") or 0),
        )
    )
    return result


def upsert_manual_pitch_calibration_override(
    reconstruction: dict,
    override: dict,
) -> None:
    overrides = [
        item
        for item in manual_pitch_calibration_overrides(reconstruction)
        if manual_override_key(item) != manual_override_key(override)
    ]
    overrides.append(deepcopy(override))
    overrides.sort(
        key=lambda item: (
            float(item.get("sceneTime") or 0.0),
            int(item.get("sourceFrameIndex") or 0),
        )
    )
    reconstruction["pitchCalibrationOverrides"] = overrides
