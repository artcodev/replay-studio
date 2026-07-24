from __future__ import annotations

"""Workflow state for staged manual calibration corrections."""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Mapping

from .reconstruction_errors import ReconstructionError


EDIT_SESSION_SCHEMA_VERSION = 1


def pending_calibration_edit_session(reconstruction: Mapping) -> dict | None:
    raw = reconstruction.get("pendingCalibrationEditSession")
    if not isinstance(raw, Mapping):
        return None
    edits = raw.get("edits")
    if (
        int(raw.get("schemaVersion") or 0) != EDIT_SESSION_SCHEMA_VERSION
        or not isinstance(edits, list)
    ):
        raise ReconstructionError("Pending calibration edit session is malformed")
    return deepcopy(dict(raw))


def register_pending_calibration_edit(
    reconstruction: dict,
    override: Mapping,
    *,
    draft_source: str,
) -> dict:
    """Upsert one staged frame edit without changing the published artifact."""

    provenance = reconstruction.get("calibrationProvenance")
    if not isinstance(provenance, Mapping):
        raise ReconstructionError(
            "A published calibration is required before frame corrections can be staged"
        )
    base_data_fingerprint = str(provenance.get("dataFingerprint") or "")
    base_input_fingerprint = str(
        provenance.get("calibrationInputFingerprint") or ""
    )
    artifact = provenance.get("artifact")
    base_artifact_sha256 = (
        str(artifact.get("sha256") or "")
        if isinstance(artifact, Mapping)
        else ""
    )
    if not base_data_fingerprint or not base_input_fingerprint or not base_artifact_sha256:
        raise ReconstructionError(
            "Published calibration provenance is incomplete; run full calibration"
        )

    current = pending_calibration_edit_session(reconstruction)
    if current is not None and (
        str(current.get("baseDataFingerprint") or "") != base_data_fingerprint
        or str(current.get("baseArtifactSha256") or "") != base_artifact_sha256
    ):
        raise ReconstructionError(
            "The published calibration changed during manual editing; reopen the timeline"
        )

    now = datetime.now(UTC).isoformat()
    edit = {
        "sampleIndex": int(override["sampleIndex"]),
        "sourceFrameIndex": override.get("sourceFrameIndex"),
        "sceneTime": float(override["sceneTime"]),
        "preset": override.get("preset"),
        "draftSource": draft_source,
        "savedAt": now,
    }
    retained = [
        item
        for item in (current or {}).get("edits") or []
        if int(item.get("sampleIndex", -1)) != edit["sampleIndex"]
    ]
    retained.append(edit)
    retained.sort(key=lambda item: int(item["sampleIndex"]))
    session = {
        "schemaVersion": EDIT_SESSION_SCHEMA_VERSION,
        "baseCalibrationInputFingerprint": base_input_fingerprint,
        "baseDataFingerprint": base_data_fingerprint,
        "baseArtifactSha256": base_artifact_sha256,
        "edits": retained,
        "editedSampleIndices": [int(item["sampleIndex"]) for item in retained],
        "createdAt": (current or {}).get("createdAt") or now,
        "updatedAt": now,
    }
    reconstruction["pendingCalibrationEditSession"] = session
    reconstruction.pop("calibrationFallbackConsent", None)
    return deepcopy(session)


def clear_pending_calibration_edit_session(reconstruction: dict) -> None:
    reconstruction.pop("pendingCalibrationEditSession", None)


__all__ = (
    "clear_pending_calibration_edit_session",
    "pending_calibration_edit_session",
    "register_pending_calibration_edit",
)
