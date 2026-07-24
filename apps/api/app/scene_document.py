from __future__ import annotations

"""Pure SceneDocument contracts and persistence projections."""

import hashlib
import json
from copy import deepcopy

from .project_match import normalized_match_snapshot_reference, project_parent_scene_id
from .scene_frame_exclusions import frame_exclusion_fingerprint_input


class SceneRevisionConflict(RuntimeError):
    """The caller tried to replace a Scene snapshot that is no longer current."""


def scene_revision(scene: dict) -> int:
    try:
        return max(0, int(scene.get("revision") or 0))
    except (TypeError, ValueError):
        return 0


def reconstruction_input_fingerprint(
    scene: dict,
    *,
    match_snapshot_ref: dict | None = None,
) -> str:
    """Digest every user-controlled input fenced by a reconstruction run."""

    payload = scene.get("payload", {})
    video = payload.get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    orientation = reconstruction.get("pitchOrientation") or {}
    visible_side_source = str(orientation.get("visiblePitchSideSource") or "")
    manual_visible_side = (
        orientation.get("visiblePitchSide")
        if visible_side_source.startswith("manual")
        else None
    )
    inputs = {
        "source": {
            "assetId": video.get("id"),
            "generationKey": video.get("generationKey"),
            "analysisFrameInput": video.get("analysisFrameInput"),
            "selectedSegmentId": video.get("selectedSegmentId"),
            "sourceStart": video.get("sourceStart"),
            "sourceEnd": video.get("sourceEnd"),
            "analysisFps": video.get("analysisFps"),
            "frameExclusions": frame_exclusion_fingerprint_input(scene),
            "samplingFrameRate": (
                reconstruction.get("samplingFrameRate") or video.get("fps")
            ),
            "directCalibrationMaxGapSeconds": (
                reconstruction.get("directCalibrationMaxGapSeconds")
                if reconstruction.get("directCalibrationMaxGapSeconds") is not None
                else 0.0
            ),
        },
        "multiPass": (
            {
                "sourcePasses": (video.get("multiPass") or {}).get("sourcePasses")
                or [],
                "manualAlignmentAnchors": (video.get("multiPass") or {}).get(
                    "manualAlignmentAnchors"
                )
                or [],
            }
            if video.get("multiPass")
            else None
        ),
        "model": reconstruction.get("model"),
        "ballDetection": {
            "backend": reconstruction.get("ballBackend"),
            "input": reconstruction.get("ballDetectionInput"),
            # The default profile is omitted so fingerprints of every scene
            # queued before profiles existed remain byte-identical.
            **(
                {"profile": reconstruction.get("ballDetectionProfile")}
                if (reconstruction.get("ballDetectionProfile") or "automatic")
                != "automatic"
                else {}
            ),
        },
        **(
            {"jerseyOcr": {"profile": reconstruction.get("jerseyOcrProfile")}}
            if (reconstruction.get("jerseyOcrProfile") or "automatic")
            != "automatic"
            else {}
        ),
        **(
            {
                "contactPoint": {
                    "profile": reconstruction.get("contactPointProfile")
                }
            }
            if (reconstruction.get("contactPointProfile") or "bbox-bottom")
            != "bbox-bottom"
            else {}
        ),
        "frameAnnotations": reconstruction.get("frameAnnotations") or [],
        "pitchCalibrationOverrides": reconstruction.get("pitchCalibrationOverrides")
        or [],
        "manualOrientation": {
            # Publication materializes "unknown" for a missing orientation;
            # digesting the absent field identically keeps the very first run
            # of a fresh scene publishable through the terminal fence.
            "attackingGoal": orientation.get("attackingGoal") or "unknown",
            "attackingGoalSource": (
                orientation.get("attackingGoalSource") or "unknown"
            ),
            "visiblePitchSide": manual_visible_side,
            "visiblePitchSideSource": (
                visible_side_source if manual_visible_side else None
            ),
        },
        "matchSnapshotRef": normalized_match_snapshot_reference(
            match_snapshot_ref
            if match_snapshot_ref is not None
            else reconstruction.get("matchSnapshotRef")
        ),
        "identityReviewDecisions": payload.get("identityReviewDecisions"),
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def annotate_reconstruction_input_state(
    scene: dict,
    current_match_snapshot_ref: dict | None,
) -> dict:
    """Decorate a project-scoped read without persisting derived staleness fields."""

    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction")
    )
    if not isinstance(reconstruction, dict):
        return scene
    current_fingerprint = reconstruction_input_fingerprint(
        scene,
        match_snapshot_ref=current_match_snapshot_ref or {},
    )
    stored_fingerprint = str(reconstruction.get("inputFingerprint") or "")
    reconstruction["currentInputFingerprint"] = current_fingerprint
    reconstruction["inputState"] = (
        "unknown"
        if not stored_fingerprint
        else "current"
        if stored_fingerprint == current_fingerprint
        else "stale"
    )
    stage = str(reconstruction.get("stage") or "")
    reconstruction["resultState"] = (
        "calibration-only"
        if stage == "calibration"
        else reconstruction["inputState"]
        if stage == "reconstruction"
        else "unavailable"
    )
    if reconstruction["inputState"] == "stale":
        reconstruction["inputStateReason"] = "reconstruction-input-changed"
    else:
        reconstruction.pop("inputStateReason", None)
    return scene


def scene_kind(scene: dict) -> str:
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video:
        return "demo"
    title = str(scene.get("title") or "").lower()
    filename = str(video.get("filename") or "").lower()
    if "smoke test" in title or "smoke" in filename:
        return "demo"
    if video.get("multiPass"):
        return "multi-pass"
    if video.get("parentSceneId") or video.get("selectedSegmentId"):
        return "segment"
    return "video"


def scene_index_values(scene: dict) -> dict[str, str | float | None]:
    try:
        duration = max(0.0, float(scene.get("duration") or 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    video = scene.get("payload", {}).get("videoAsset") or {}
    return {
        "duration": duration,
        "kind": scene_kind(scene),
        "parent_scene_id": project_parent_scene_id(scene),
        "selected_segment_id": (
            str(video.get("selectedSegmentId") or "").strip() or None
        ),
    }


def next_scene_payload(scene: dict, revision: int) -> dict:
    payload = deepcopy(scene)
    payload["revision"] = revision
    reconstruction = (
        payload.get("payload", {}).get("videoAsset", {}).get("reconstruction")
    )
    if isinstance(reconstruction, dict):
        reconstruction.pop("lease", None)
        reconstruction.pop("currentInputFingerprint", None)
        reconstruction.pop("inputState", None)
        reconstruction.pop("inputStateReason", None)
        reconstruction.pop("resultState", None)
    return payload
