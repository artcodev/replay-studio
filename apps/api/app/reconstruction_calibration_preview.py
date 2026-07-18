from __future__ import annotations

"""Persistence boundary for non-destructive frame-calibration previews."""

from copy import deepcopy
from datetime import UTC, datetime

from .scene_repository import scenes


def persist_frame_calibration_preview(scene: dict, evidence: dict) -> None:
    """Persist diagnostic evidence without changing reconstruction lifecycle state."""

    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    calibration_contract = dict(reconstruction.get("calibration") or {})
    calibration_contract.setdefault("schemaVersion", 1)
    previews = [
        deepcopy(item)
        for item in calibration_contract.get("framePreviews") or []
        if int(item.get("sourceFrameIndex") or -1)
        != int(evidence.get("sourceFrameIndex") or -2)
    ]
    persisted = {
        **deepcopy(evidence),
        "previewedAt": datetime.now(UTC).isoformat(),
    }
    previews.append(persisted)
    previews.sort(
        key=lambda item: (
            float(item.get("sceneTime") or 0.0),
            int(item.get("sourceFrameIndex") or 0),
        )
    )
    calibration_contract["framePreviews"] = previews[-240:]
    calibration_contract["lastFramePreview"] = persisted
    reconstruction["calibration"] = calibration_contract
    video["reconstruction"] = reconstruction
    scenes.put(scene)
