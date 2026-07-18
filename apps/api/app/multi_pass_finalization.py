from __future__ import annotations

"""Publication of one fused multi-angle reconstruction."""

from copy import deepcopy
from datetime import UTC, datetime
from time import monotonic

from .multi_pass_alignment import temporal_alignment
from .multi_pass_fusion import (
    aligned_ball_support,
    copy_reference_identity_state,
    fuse_aligned_pass_identities,
)
from .multi_pass_metrics import consensus_summary
from .multi_pass_progress import set_multi_pass_progress
from .reconstruction_artifact_publication import publish_dense_reconstruction_artifacts


def finalize_multi_pass(
    scene: dict,
    segments: list[dict],
    ready_scenes: list[tuple[dict, dict, dict]],
    pass_summaries: list[dict],
) -> dict:
    """Fuse terminal usable child artifacts and publish one compact scene."""

    progress_started = monotonic()
    video = scene["payload"]["videoAsset"]
    multi_pass = video["multiPass"]
    composite_reconstruction = deepcopy(video.get("reconstruction") or {})
    reference_scene, _reference_summary, reference_segment = max(
        ready_scenes,
        key=lambda item: item[1]["quality"],
    )
    set_multi_pass_progress(
        scene,
        segments,
        progress_started,
        "alignment",
        len(segments) + 1,
        "Aligning camera angles",
        "Comparing motion signatures and classifying replay overlap.",
        92,
        40,
        eta_seconds=6.0,
    )
    reference_payload = reference_scene["payload"]
    reference_video = reference_payload["videoAsset"]
    existing_teams = deepcopy(scene["payload"].get("teams") or [])

    scene["duration"] = reference_scene["duration"]
    scene["payload"]["pitch"] = deepcopy(reference_payload["pitch"])
    identity_copy_warnings = copy_reference_identity_state(
        scene["payload"],
        reference_payload,
    )
    scene["payload"]["ball"] = deepcopy(
        reference_payload.get("ball") or {"keyframes": []}
    )
    for index, team in enumerate(existing_teams[:2]):
        if index < len(reference_payload.get("teams") or []):
            team["color"] = reference_payload["teams"][index]["color"]
    scene["payload"]["teams"] = existing_teams or deepcopy(
        reference_payload.get("teams") or []
    )

    aligned_passes: list[tuple[dict, dict]] = []
    for pass_scene, summary, segment in ready_scenes:
        alignment = temporal_alignment(
            reference_scene,
            pass_scene,
            reference_segment,
            segment,
            multi_pass.get("manualAlignmentAnchors"),
        )
        summary["relation"] = alignment["relation"]
        summary["alignment"] = alignment
        aligned_passes.append((pass_scene, summary))
    set_multi_pass_progress(
        scene,
        segments,
        progress_started,
        "consensus",
        len(segments) + 2,
        "Fusing reconstruction evidence",
        "Selecting the strongest calibrated view and measuring cross-angle ball support.",
        97,
        45,
        eta_seconds=3.0,
    )
    identity_fusion = fuse_aligned_pass_identities(
        scene["payload"],
        reference_scene,
        aligned_passes,
    )
    ball_support = aligned_ball_support(
        reference_scene,
        aligned_passes,
        scene["payload"]["ball"].get("keyframes") or [],
    )
    consensus = consensus_summary(pass_summaries)
    warnings = [
        "Canonical trajectories currently come from the strongest calibrated pass.",
        "Aligned replay identity evidence is fused only for a unique shirt-number or external-player match.",
        "The evidence score measures reconstruction coverage; temporal overlap is reported separately.",
        *identity_copy_warnings,
    ]
    if identity_fusion.get("reviewCandidates"):
        warnings.append(
            "One or more cross-angle identities were ambiguous or lacked independent identity evidence and require review."
        )
    if len(ready_scenes) < len(segments):
        warnings.append("One or more selected angles could not be reconstructed.")
    completed_multi_pass = {
        **multi_pass,
        "status": "ready",
        "currentPass": len(segments),
        "referenceSceneId": reference_scene["id"],
        "passes": pass_summaries,
        "consensus": consensus,
        "ballSupport": ball_support,
        "identityFusion": identity_fusion,
        "warnings": warnings,
    }
    scene_video = deepcopy(reference_video)
    scene_video["processingState"] = "multi-pass-ready"
    scene_video["multiPass"] = completed_multi_pass
    reference_reconstruction = deepcopy(reference_video.get("reconstruction") or {})
    reference_reconstruction.update(
        {
            "status": "ready",
            "runId": composite_reconstruction.get("runId"),
            "runRevision": composite_reconstruction.get("runRevision", 1),
            "inputFingerprint": composite_reconstruction.get("inputFingerprint"),
            "completedAt": datetime.now(UTC).isoformat(),
            "trackCount": len(scene["payload"]["tracks"]),
            "ballSamples": len(scene["payload"]["ball"].get("keyframes") or []),
            "multiPassEvidence": consensus,
            "multiPassBallSupport": ball_support,
            "multiPassIdentityFusion": identity_fusion,
            "warnings": [
                *(reference_reconstruction.get("warnings") or []),
                *warnings,
            ],
            "progress": set_multi_pass_progress(
                scene,
                segments,
                progress_started,
                "complete",
                len(segments) + 2,
                "Multi-angle analysis complete",
                f"Analyzed {len(ready_scenes)} of {len(segments)} camera angles.",
                100,
                100,
                completed=len(ready_scenes),
                total=len(segments),
                eta_seconds=0.0,
                complete=True,
            ),
        }
    )
    if composite_reconstruction.get("matchSnapshotRef") is None:
        reference_reconstruction.pop("matchSnapshotRef", None)
    else:
        reference_reconstruction["matchSnapshotRef"] = deepcopy(
            composite_reconstruction["matchSnapshotRef"]
        )
    scene_video["reconstruction"] = reference_reconstruction
    scene["payload"]["videoAsset"] = scene_video
    publish_dense_reconstruction_artifacts(scene)
    return reference_reconstruction["progress"]
