from __future__ import annotations

"""Pure Scene invalidation for an analysis-frame generation cutover."""

from copy import deepcopy

from .reconstruction_ball_trajectory import (
    manual_ball_diagnostics,
    normalize_ball_payload,
)


_PRESERVED_RECONSTRUCTION_SETTINGS = frozenset(
    {
        "model",
        "ballBackend",
        "ballDetectionProfile",
        "jerseyOcrProfile",
        "contactPointProfile",
        "pitchOrientation",
        "samplingFrameRate",
    }
)


def switch_scene_analysis_frame_generation(
    scene: dict,
    *,
    generation_key: str,
    source_fps: float,
    analysis_fps: float,
    frame_count: int,
    analysis_frame_input: dict,
) -> dict:
    """Switch input pixels and discard every result tied to the old grid.

    The source timeline, teams, event bindings and manual metric ball path are
    user inputs and survive. Pixel anchors, detections, tracks, identities and
    artifacts do not: scaling them silently would manufacture precision.
    """

    updated = deepcopy(scene)
    payload = updated["payload"]
    video = payload["videoAsset"]
    previous_reconstruction = video.get("reconstruction") or {}
    preserved = {
        key: deepcopy(previous_reconstruction[key])
        for key in _PRESERVED_RECONSTRUCTION_SETTINGS
        if key in previous_reconstruction
    }
    video["generationKey"] = generation_key
    video["fps"] = source_fps
    video["analysisFps"] = analysis_fps
    video["frameCount"] = frame_count
    video["analysisFrameInput"] = deepcopy(analysis_frame_input)
    video["processingState"] = "frames-ready"
    if preserved:
        video["reconstruction"] = preserved
    else:
        video.pop("reconstruction", None)

    payload["tracks"] = []
    payload.pop("canonicalPeople", None)
    payload.pop("identityReviewDecisions", None)
    payload.pop("playerActions", None)
    payload.pop("actions", None)
    ball = payload.get("ball")
    if isinstance(ball, dict) and ball.get("mode") == "manual":
        preserved_ball = normalize_ball_payload(ball)
        preserved_ball["automaticKeyframes"] = []
        preserved_ball["automaticDiagnostics"] = {
            "trajectoryMode": "automatic",
            "source": "invalidated-analysis-frame-generation",
        }
        preserved_ball["keyframes"] = deepcopy(
            preserved_ball["manualKeyframes"]
        )
        preserved_ball["manualDiagnostics"] = manual_ball_diagnostics(
            preserved_ball["manualKeyframes"]
        )
        preserved_ball["diagnostics"] = deepcopy(
            preserved_ball["manualDiagnostics"]
        )
        payload["ball"] = preserved_ball
    else:
        payload["ball"] = {"keyframes": []}
    return updated


__all__ = ("switch_scene_analysis_frame_generation",)
