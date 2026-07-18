from __future__ import annotations

"""Cross-angle ball and identity evidence fusion."""

from copy import deepcopy
from math import hypot

import numpy as np

from .multi_angle_identity import fuse_aligned_identity_passes
from .multi_pass_alignment import map_reference_time


def aligned_ball_support(
    reference_scene: dict,
    aligned_passes: list[tuple[dict, dict]],
    target_ball: list[dict] | None = None,
) -> dict:
    reference_ball = (
        target_ball
        if target_ball is not None
        else reference_scene.get("payload", {}).get("ball", {}).get("keyframes") or []
    )
    reference_calibration = (
        reference_scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
        .get("pitchCalibration")
        or {}
    )
    visual_passes = 0
    metric_passes = 0
    spatial_errors: list[float] = []
    for pass_scene, summary in aligned_passes:
        alignment = summary.get("alignment") or {}
        if alignment.get("relation") != "replay-overlap":
            continue
        candidate_ball = (
            pass_scene.get("payload", {}).get("ball", {}).get("keyframes") or []
        )
        if len(candidate_ball) < 3:
            continue
        pairs: list[tuple[dict, dict]] = []
        for point in reference_ball:
            pass_time = map_reference_time(alignment["anchors"], float(point["t"]))
            candidate = min(
                candidate_ball,
                key=lambda item: abs(float(item["t"]) - pass_time),
            )
            if abs(float(candidate["t"]) - pass_time) <= 0.42:
                point["support"] = int(point.get("support") or 1) + 1
                pairs.append((point, candidate))
        if len(pairs) < 3:
            continue
        visual_passes += 1
        candidate_calibration = (
            pass_scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
            .get("pitchCalibration")
            or {}
        )
        if (
            reference_calibration.get("status") != "ready"
            or candidate_calibration.get("status") != "ready"
        ):
            continue
        transforms = (
            lambda x, z: (x, z),
            lambda x, z: (-x, z),
            lambda x, z: (x, -z),
            lambda x, z: (-x, -z),
        )
        errors = [
            float(
                np.median(
                    [
                        hypot(
                            reference["x"] - transform(candidate["x"], candidate["z"])[0],
                            reference["z"] - transform(candidate["x"], candidate["z"])[1],
                        )
                        for reference, candidate in pairs
                    ]
                )
            )
            for transform in transforms
        ]
        spatial_error = min(errors)
        spatial_errors.append(round(spatial_error, 2))
        if spatial_error <= 8.0:
            metric_passes += 1
            for reference, _ in pairs:
                reference["metricSupport"] = int(reference.get("metricSupport") or 0) + 1
                reference["confidence"] = round(
                    1.0 - (1.0 - float(reference["confidence"])) * 0.75,
                    3,
                )
    return {
        "referenceSamples": len(reference_ball),
        "supportedSamples": sum(
            int(point.get("support") or 1) > 1 for point in reference_ball
        ),
        "visualPasses": visual_passes,
        "metricPasses": metric_passes,
        "spatialErrors": spatial_errors,
    }


def copy_reference_identity_state(
    target_payload: dict,
    reference_payload: dict,
) -> list[str]:
    """Copy a valid canonical identity graph from the selected pass."""

    canonical_people = deepcopy(reference_payload.get("canonicalPeople") or [])
    canonical_ids = {
        str(person.get("canonicalPersonId") or person.get("id"))
        for person in canonical_people
        if person.get("canonicalPersonId") or person.get("id")
    }
    tracks = deepcopy(reference_payload.get("tracks") or [])
    orphan_ids: set[str] = set()
    for track in tracks:
        canonical_id = str(track.get("canonicalPersonId") or "")
        if canonical_id and canonical_id not in canonical_ids:
            orphan_ids.add(canonical_id)
            track.pop("canonicalPersonId", None)
    target_payload["tracks"] = tracks
    target_payload["canonicalPeople"] = canonical_people
    if not orphan_ids:
        return []
    return [
        "The reference pass contained orphan canonical identity references; "
        f"they were detached from 3D tracks ({', '.join(sorted(orphan_ids))})."
    ]


def fuse_aligned_pass_identities(
    target_payload: dict,
    reference_scene: dict,
    aligned_passes: list[tuple[dict, dict]],
) -> dict:
    """Enrich a reference graph with independent evidence from replay views."""

    reference_scene_id = str(reference_scene.get("id") or "")
    fusion_inputs = []
    for pass_scene, summary in aligned_passes:
        source_scene_id = str(pass_scene.get("id") or "")
        alignment = summary.get("alignment") or {}
        if source_scene_id == reference_scene_id:
            continue
        if alignment.get("relation") != "replay-overlap":
            continue
        fusion_inputs.append(
            {
                "sceneId": source_scene_id,
                "segmentId": summary.get("segmentId"),
                "alignment": deepcopy(alignment),
                "canonicalPeople": deepcopy(
                    pass_scene.get("payload", {}).get("canonicalPeople") or []
                ),
            }
        )

    fused_people, diagnostics = fuse_aligned_identity_passes(
        target_payload.get("canonicalPeople") or [],
        fusion_inputs,
    )
    target_payload["canonicalPeople"] = fused_people
    return {
        **diagnostics,
        "referenceSceneId": reference_scene_id,
        "eligibleReplayPassCount": len(fusion_inputs),
    }
