from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from math import hypot, isfinite
from time import monotonic
from uuid import uuid4

import cv2
import numpy as np

from .config import get_settings
from .multi_angle_identity import fuse_aligned_identity_passes
from .project_match import copy_project_match_metadata
from .reconstruction import ReconstructionError, _frame_paths, reconstruct_scene
from .sample import make_video_scene
from .store import scene_store
from .video_processing import materialize_segment_scene


class MultiPassError(RuntimeError):
    pass


def _multi_pass_phases(segments: list[dict], current_index: int, complete: bool = False) -> list[dict]:
    labels = [
        (f"angle-{index}", f"Angle {index} · {segment.get('label') or segment['id']}")
        for index, segment in enumerate(segments, start=1)
    ]
    labels.extend([("alignment", "Align camera angles"), ("consensus", "Fuse evidence")])
    return [
        {
            "id": phase_id,
            "label": label,
            "status": (
                "completed"
                if complete or index < current_index
                else "current"
                if index == current_index
                else "pending"
            ),
        }
        for index, (phase_id, label) in enumerate(labels, start=1)
    ]


def _set_multi_pass_progress(
    scene: dict,
    segments: list[dict],
    started: float,
    phase: str,
    phase_index: int,
    label: str,
    detail: str,
    overall_percent: float,
    phase_percent: float,
    completed: int = 0,
    total: int = 0,
    eta_seconds: float | None = None,
    complete: bool = False,
) -> dict:
    payload = {
        "phase": phase,
        "phaseIndex": phase_index,
        "phaseCount": len(segments) + 2,
        "label": label,
        "detail": detail,
        "completed": completed,
        "total": total,
        "phasePercent": round(phase_percent),
        "overallPercent": round(overall_percent),
        "elapsedSeconds": round(max(0.0, monotonic() - started), 1),
        "etaSeconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "updatedAt": datetime.now(UTC).isoformat(),
        "phases": _multi_pass_phases(segments, phase_index, complete=complete),
    }
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    reconstruction["progress"] = payload
    video["reconstruction"] = reconstruction
    scene_store.put(scene)
    return payload


def pass_quality(scene: dict) -> float:
    payload = scene.get("payload") or {}
    video = payload.get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    quality = reconstruction.get("quality") or {}
    calibration = reconstruction.get("pitchCalibration") or {}
    tracks = len(payload.get("tracks") or [])
    ball_samples = len((payload.get("ball") or {}).get("keyframes") or [])
    frame_count = int(reconstruction.get("frameCount") or 0)
    verdict = str(reconstruction.get("qualityVerdict") or quality.get("verdict") or "")
    if quality and verdict:
        verdict_score = {
            "pass": 1.0,
            "review": 0.55,
            "reject": 0.0,
            "unknown": 0.15,
        }.get(verdict, 0.1)
        calibration_summary = (
            (reconstruction.get("calibration") or {}).get("summary") or {}
        )
        coverage = float(
            calibration_summary.get("usableCoverage")
            or calibration_summary.get("directCoverage")
            or 0.0
        )
        gates = quality.get("gates") or []
        if isinstance(gates, dict):
            gates = list(gates.values())
        required = [gate for gate in gates if gate.get("required", True)]
        gate_score = (
            sum(
                1.0
                if gate.get("status") == "pass"
                else 0.45
                if gate.get("status") == "review"
                else 0.0
                for gate in required
            )
            / len(required)
            if required
            else verdict_score
        )
        # Counts are evidence availability, not accuracy. They only provide a
        # small tie-breaker after quality gates and calibration coverage.
        availability = (
            min(1.0, tracks / 14.0) * 0.55
            + min(1.0, ball_samples / 18.0) * 0.30
            + min(1.0, frame_count / 30.0) * 0.15
        )
        score = verdict_score * 0.50 + coverage * 0.25 + gate_score * 0.20 + availability * 0.05
        return round(min(score, 0.24) if verdict == "reject" else score, 3)

    # Compatibility for scenes produced before the evidence-based QA contract.
    calibration_score = (
        float(calibration.get("confidence") or 0.0)
        if calibration.get("status") == "ready"
        else 0.0
    )
    score = (
        calibration_score * 0.4
        + min(1.0, tracks / 14.0) * 0.3
        + min(1.0, ball_samples / 18.0) * 0.2
        + min(1.0, frame_count / 30.0) * 0.1
    )
    return round(score, 3)


def consensus_summary(passes: list[dict]) -> dict:
    ready = [item for item in passes if item.get("status") == "ready"]
    total = max(1, len(ready))
    metric_passes = sum(
        item.get("qualityVerdict") == "pass"
        or (
            item.get("qualityVerdict") is None
            and item.get("calibrationStatus") == "ready"
        )
        for item in ready
    )
    ball_passes = sum(int(item.get("ballSamples") or 0) >= 3 for item in ready)
    track_passes = sum(int(item.get("trackCount") or 0) >= 6 for item in ready)
    overlapping_passes = sum(item.get("relation") == "replay-overlap" for item in ready)
    coverage = min(1.0, len(ready) / 3.0)
    evidence = (
        coverage * 0.25
        + metric_passes / total * 0.3
        + ball_passes / total * 0.25
        + track_passes / total * 0.2
    )
    return {
        "passesAnalyzed": len(ready),
        "metricPasses": metric_passes,
        "ballPasses": ball_passes,
        "trackPasses": track_passes,
        "overlappingPasses": overlapping_passes,
        "evidenceScore": round(evidence, 3),
    }


def motion_dtw(reference: list[float], candidate: list[float]) -> dict:
    if len(reference) < 2 or len(candidate) < 2:
        return {
            "cost": 1.0,
            "anchors": [{"reference": 0.0, "pass": 0.0}, {"reference": 1.0, "pass": 1.0}],
        }
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    distances = np.full((len(left) + 1, len(right) + 1), np.inf, dtype=np.float64)
    distances[0, 0] = 0.0
    for left_index in range(1, len(left) + 1):
        for right_index in range(1, len(right) + 1):
            distances[left_index, right_index] = abs(left[left_index - 1] - right[right_index - 1]) + min(
                distances[left_index - 1, right_index],
                distances[left_index, right_index - 1],
                distances[left_index - 1, right_index - 1],
            )

    path: list[tuple[int, int]] = []
    left_index, right_index = len(left), len(right)
    while left_index > 0 and right_index > 0:
        path.append((left_index - 1, right_index - 1))
        choices = (
            (distances[left_index - 1, right_index - 1], left_index - 1, right_index - 1),
            (distances[left_index - 1, right_index], left_index - 1, right_index),
            (distances[left_index, right_index - 1], left_index, right_index - 1),
        )
        _, left_index, right_index = min(choices, key=lambda item: item[0])
    path.reverse()

    anchors = []
    for pass_index in np.linspace(0, len(right) - 1, min(7, len(right))).round().astype(int):
        matches = [left_position for left_position, right_position in path if right_position == pass_index]
        if not matches:
            nearest = min(path, key=lambda item: abs(item[1] - pass_index))
            matches = [nearest[0]]
        anchors.append(
            {
                "reference": round(float(np.median(matches)) / max(1, len(left) - 1), 4),
                "pass": round(float(pass_index) / max(1, len(right) - 1), 4),
            }
        )
    return {
        "cost": round(float(distances[-1, -1]) / (len(left) + len(right)), 4),
        "anchors": anchors,
    }


def classify_pass_relation(motion_cost: float, segment: dict, reference_segment: dict) -> str:
    if motion_cost <= 0.07:
        return "replay-overlap"
    before_gap = float(reference_segment.get("start", 0)) - float(segment.get("end", 0))
    after_gap = float(segment.get("start", 0)) - float(reference_segment.get("end", 0))
    if -0.15 <= before_gap <= 0.4:
        return "continuation-before"
    if -0.15 <= after_gap <= 0.4:
        return "continuation-after"
    return "independent"


def _motion_signature(scene: dict, bins: int = 24) -> list[float]:
    values: list[float] = []
    times: list[float] = []
    previous: np.ndarray | None = None
    duration = max(0.1, float(scene.get("duration") or 0.1))
    for path, time in _frame_paths(scene):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        if previous is not None:
            values.append(float(cv2.absdiff(previous, image).mean() / 255.0))
            times.append(min(1.0, max(0.0, time / duration)))
        previous = image
    if len(values) < 2:
        return [0.0] * bins
    series = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(series, [10, 90])
    series = np.clip((series - low) / max(1e-6, high - low), 0.0, 1.0)
    return np.interp(np.linspace(0.0, 1.0, bins), times, series, left=series[0], right=series[-1]).tolist()


def _manual_clock_alignment(
    saved_anchors: object,
    reference_scene: dict,
    pass_scene: dict,
    pass_segment: dict,
) -> tuple[dict | None, dict | None]:
    """Build an authoritative clock map from saved, pass-scoped anchors.

    The persisted input accepts either grouped records with an ``anchors``
    array or flat anchor records.  A record must name the source scene, the
    source segment, or both.  Malformed/out-of-range points are ignored, while
    duplicate or time-reversing valid points reject the complete manual map so
    that a misleading warp is never published.
    """

    if isinstance(saved_anchors, dict):
        records = [saved_anchors]
    elif isinstance(saved_anchors, list):
        records = saved_anchors
    else:
        return None, None

    source_scene_id = str(pass_scene.get("id") or "")
    source_segment_id = str(pass_segment.get("id") or "")
    matched_records = 0
    candidates: list[object] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_scene_id = str(record.get("sourceSceneId") or "").strip()
        record_segment_id = str(record.get("segmentId") or "").strip()
        if not record_scene_id and not record_segment_id:
            continue
        if record_scene_id and record_scene_id != source_scene_id:
            continue
        if record_segment_id and record_segment_id != source_segment_id:
            continue
        matched_records += 1
        grouped = record.get("anchors")
        if isinstance(grouped, list):
            candidates.extend(grouped)
        else:
            candidates.append(record)

    if matched_records == 0:
        return None, None

    reference_duration = max(0.0, float(reference_scene.get("duration") or 0.0))
    pass_duration = max(0.0, float(pass_scene.get("duration") or 0.0))
    valid: list[dict[str, float]] = []
    rejection_reasons: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            rejection_reasons.add("anchor-not-an-object")
            continue
        reference_time = candidate.get("referenceTime")
        pass_time = candidate.get("passTime")
        if isinstance(reference_time, bool) or isinstance(pass_time, bool):
            rejection_reasons.add("anchor-time-not-numeric")
            continue
        try:
            reference_value = float(reference_time)
            pass_value = float(pass_time)
        except (TypeError, ValueError):
            rejection_reasons.add("anchor-time-not-numeric")
            continue
        if not isfinite(reference_value) or not isfinite(pass_value):
            rejection_reasons.add("anchor-time-not-finite")
            continue
        if not 0.0 <= reference_value <= reference_duration:
            rejection_reasons.add("reference-time-out-of-range")
            continue
        if not 0.0 <= pass_value <= pass_duration:
            rejection_reasons.add("pass-time-out-of-range")
            continue
        valid.append(
            {
                "referenceTime": round(reference_value, 6),
                "passTime": round(pass_value, 6),
            }
        )

    diagnostics = {
        "status": "rejected",
        "matchedRecordCount": matched_records,
        "providedAnchorCount": len(candidates),
        "validAnchorCount": len(valid),
        "rejectionReasons": sorted(rejection_reasons),
    }
    if len(valid) < 2:
        diagnostics["rejectionReasons"] = sorted(
            {*rejection_reasons, "at-least-two-valid-anchors-required"}
        )
        return None, diagnostics

    ordered = sorted(valid, key=lambda item: (item["referenceTime"], item["passTime"]))
    if any(
        right["referenceTime"] <= left["referenceTime"]
        or right["passTime"] <= left["passTime"]
        for left, right in zip(ordered, ordered[1:])
    ):
        diagnostics["rejectionReasons"] = sorted(
            {*rejection_reasons, "anchors-not-strictly-monotonic"}
        )
        return None, diagnostics

    return {
        "relation": "replay-overlap",
        "method": "manual-clock-anchors",
        "confidence": 1.0,
        "motionCost": None,
        "overlap": True,
        "anchors": ordered,
        "manualAlignment": {
            **diagnostics,
            "status": "accepted",
            "rejectionReasons": sorted(rejection_reasons),
        },
    }, None


def _temporal_alignment(
    reference_scene: dict,
    pass_scene: dict,
    reference_segment: dict,
    pass_segment: dict,
    manual_alignment_anchors: object = None,
) -> dict:
    if reference_scene["id"] == pass_scene["id"]:
        return {
            "relation": "reference",
            "method": "identity",
            "confidence": 1.0,
            "motionCost": 0.0,
            "overlap": True,
            "anchors": [
                {"referenceTime": 0.0, "passTime": 0.0},
                {"referenceTime": reference_scene["duration"], "passTime": pass_scene["duration"]},
            ],
        }
    manual_alignment, manual_diagnostics = _manual_clock_alignment(
        manual_alignment_anchors,
        reference_scene,
        pass_scene,
        pass_segment,
    )
    if manual_alignment is not None:
        return manual_alignment
    result = motion_dtw(_motion_signature(reference_scene), _motion_signature(pass_scene))
    relation = classify_pass_relation(result["cost"], pass_segment, reference_segment)
    if relation == "replay-overlap":
        confidence = max(0.55, min(0.95, 1.0 - result["cost"] / 0.12))
        method = "motion-dtw"
    elif relation.startswith("continuation"):
        confidence = 0.9
        method = "source-continuity"
    else:
        confidence = 0.2
        method = "phase-normalized"
    anchors = [
        {
            "referenceTime": round(item["reference"] * reference_scene["duration"], 3),
            "passTime": round(item["pass"] * pass_scene["duration"], 3),
        }
        for item in result["anchors"]
    ]
    alignment = {
        "relation": relation,
        "method": method,
        "confidence": round(confidence, 3),
        "motionCost": result["cost"],
        "overlap": relation == "replay-overlap",
        "anchors": anchors,
    }
    if manual_diagnostics is not None:
        alignment["manualAlignment"] = manual_diagnostics
    return alignment


def _map_reference_time(anchors: list[dict], reference_time: float) -> float:
    ordered = sorted(anchors, key=lambda item: item["referenceTime"])
    if reference_time <= ordered[0]["referenceTime"]:
        return float(ordered[0]["passTime"])
    if reference_time >= ordered[-1]["referenceTime"]:
        return float(ordered[-1]["passTime"])
    for left, right in zip(ordered, ordered[1:]):
        if left["referenceTime"] <= reference_time <= right["referenceTime"]:
            width = max(1e-6, right["referenceTime"] - left["referenceTime"])
            progress = (reference_time - left["referenceTime"]) / width
            return float(left["passTime"] + (right["passTime"] - left["passTime"]) * progress)
    return reference_time


def _aligned_ball_support(
    reference_scene: dict,
    aligned_passes: list[tuple[dict, dict]],
    target_ball: list[dict] | None = None,
) -> dict:
    reference_ball = target_ball if target_ball is not None else (
        reference_scene.get("payload", {}).get("ball", {}).get("keyframes") or []
    )
    reference_calibration = (
        reference_scene.get("payload", {}).get("videoAsset", {}).get("reconstruction", {}).get("pitchCalibration") or {}
    )
    visual_passes = 0
    metric_passes = 0
    spatial_errors: list[float] = []
    for pass_scene, summary in aligned_passes:
        alignment = summary.get("alignment") or {}
        if alignment.get("relation") != "replay-overlap":
            continue
        candidate_ball = pass_scene.get("payload", {}).get("ball", {}).get("keyframes") or []
        if len(candidate_ball) < 3:
            continue
        pairs: list[tuple[dict, dict]] = []
        for point in reference_ball:
            pass_time = _map_reference_time(alignment["anchors"], float(point["t"]))
            candidate = min(candidate_ball, key=lambda item: abs(float(item["t"]) - pass_time))
            if abs(float(candidate["t"]) - pass_time) <= 0.42:
                point["support"] = int(point.get("support") or 1) + 1
                pairs.append((point, candidate))
        if len(pairs) < 3:
            continue
        visual_passes += 1
        candidate_calibration = (
            pass_scene.get("payload", {}).get("videoAsset", {}).get("reconstruction", {}).get("pitchCalibration") or {}
        )
        if reference_calibration.get("status") != "ready" or candidate_calibration.get("status") != "ready":
            continue
        transforms = [
            lambda x, z: (x, z),
            lambda x, z: (-x, z),
            lambda x, z: (x, -z),
            lambda x, z: (-x, -z),
        ]
        errors = [
            float(np.median([
                hypot(reference["x"] - transform(candidate["x"], candidate["z"])[0], reference["z"] - transform(candidate["x"], candidate["z"])[1])
                for reference, candidate in pairs
            ]))
            for transform in transforms
        ]
        spatial_error = min(errors)
        spatial_errors.append(round(spatial_error, 2))
        if spatial_error <= 8.0:
            metric_passes += 1
            for reference, _ in pairs:
                reference["metricSupport"] = int(reference.get("metricSupport") or 0) + 1
                reference["confidence"] = round(1.0 - (1.0 - float(reference["confidence"])) * 0.75, 3)
    return {
        "referenceSamples": len(reference_ball),
        "supportedSamples": sum(int(point.get("support") or 1) > 1 for point in reference_ball),
        "visualPasses": visual_passes,
        "metricPasses": metric_passes,
        "spatialErrors": spatial_errors,
    }


def _pass_summary(scene: dict, segment: dict, status: str = "ready", error: str | None = None) -> dict:
    payload = scene.get("payload") or {}
    video = payload.get("videoAsset") or {}
    reconstruction = video.get("reconstruction") or {}
    calibration = reconstruction.get("pitchCalibration") or {}
    return {
        "sceneId": scene.get("id"),
        "segmentId": segment["id"],
        "label": segment.get("label") or segment["id"],
        "sourceStart": segment.get("start"),
        "sourceEnd": segment.get("end"),
        "status": status,
        "quality": pass_quality(scene) if status == "ready" else 0.0,
        "trackCount": len(payload.get("tracks") or []),
        "ballSamples": len((payload.get("ball") or {}).get("keyframes") or []),
        "calibrationStatus": calibration.get("status") or "fallback",
        "calibrationConfidence": calibration.get("confidence"),
        "qualityVerdict": reconstruction.get("qualityVerdict")
        or (reconstruction.get("quality") or {}).get("verdict"),
        "error": error,
    }


def _inherit_match_metadata(target: dict, source: dict) -> None:
    project_scene_id = str(
        (source.get("payload", {}).get("matchBinding") or {}).get("projectSceneId")
        or (source.get("payload", {}).get("videoAsset", {}).get("multiPass") or {}).get(
            "parentSceneId"
        )
        or source.get("id")
    )
    copy_project_match_metadata(
        target,
        source,
        project_scene_id=project_scene_id,
        inherited=target.get("id") != project_scene_id,
    )
    scene_store.put(target)


def _copy_reference_identity_state(target_payload: dict, reference_payload: dict) -> list[str]:
    """Copy the complete canonical identity graph from the selected pass.

    A render track may reference a canonical person, but it does not own that
    identity. Older/partial scenes can contain an orphan reference; carrying it
    into the composite would break video↔3D selection, so such a reference is
    removed explicitly and reported instead of being silently published.
    """

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


def _fuse_aligned_pass_identities(
    target_payload: dict,
    reference_scene: dict,
    aligned_passes: list[tuple[dict, dict]],
) -> dict:
    """Enrich the reference graph with independent evidence from replay views."""

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


def create_multi_pass_scene(
    parent: dict,
    segments: list[dict],
    title: str | None = None,
    manual_alignment_anchors: list[dict] | None = None,
) -> dict:
    if len(segments) < 2:
        raise MultiPassError("Choose at least two camera angles")
    reference_segment = max(segments, key=lambda item: (item.get("score", 0), item.get("duration", 0)))
    children = [materialize_segment_scene(parent, segment) for segment in segments]
    reference_child = next(
        child
        for child in children
        if child.get("payload", {}).get("videoAsset", {}).get("selectedSegmentId") == reference_segment["id"]
    )
    group_id = f"angles-{uuid4().hex[:8]}"
    video = deepcopy(reference_child["payload"]["videoAsset"])
    video["processingState"] = "multi-pass-queued"
    video["multiPass"] = {
        "id": group_id,
        "status": "queued",
        "matchBindingState": "current",
        "parentSceneId": parent["id"],
        "selectedSegmentIds": [segment["id"] for segment in segments],
        "referenceSceneId": None,
        "currentPass": 0,
        "passes": [],
        "consensus": None,
        "manualAlignmentAnchors": deepcopy(manual_alignment_anchors or []),
        "warnings": [],
    }
    video["reconstruction"] = {
        "status": "queued",
        "model": get_settings().reconstruction_model,
        "error": None,
        "progress": {
            "phase": "angle-1",
            "phaseIndex": 1,
            "phaseCount": len(segments) + 2,
            "label": "Waiting to analyze camera angles",
            "detail": f"Queued {len(segments)} selected views.",
            "completed": 0,
            "total": len(segments),
            "phasePercent": 0,
            "overallPercent": 0,
            "elapsedSeconds": 0.0,
            "etaSeconds": None,
            "updatedAt": datetime.now(UTC).isoformat(),
            "phases": _multi_pass_phases(segments, 1),
        },
    }
    scene = make_video_scene(
        scene_id=f"multi-{uuid4().hex[:8]}",
        title=title or f"{parent['title']} · Multi-angle ({len(segments)} passes)",
        duration=reference_child["duration"],
        video_asset=video,
    )
    copy_project_match_metadata(
        scene,
        parent,
        project_scene_id=parent["id"],
        inherited=True,
    )
    scene_store.put(parent)
    return scene_store.put(scene)


def analyze_multi_pass_by_id(scene_id: str) -> None:
    scene = scene_store.get(scene_id)
    if scene is None:
        return
    video = scene.get("payload", {}).get("videoAsset") or {}
    multi_pass = video.get("multiPass") or {}
    parent = scene_store.get(str(multi_pass.get("parentSceneId") or ""))
    if parent is None:
        _fail(scene, "Parent video scene was not found")
        return
    segment_ids = list(multi_pass.get("selectedSegmentIds") or [])
    segment_map = {
        item["id"]: item
        for item in (parent.get("payload", {}).get("videoAsset", {}).get("segments") or [])
    }
    segments = [segment_map[item] for item in segment_ids if item in segment_map]
    if len(segments) < 2:
        _fail(scene, "At least two selected camera angles are required")
        return

    progress_started = monotonic()
    multi_pass.update({"status": "processing", "currentPass": 0, "passes": []})
    video["reconstruction"].update(
        {"status": "processing", "startedAt": datetime.now(UTC).isoformat(), "error": None}
    )
    scene_store.put(scene)

    pass_summaries: list[dict] = []
    ready_scenes: list[tuple[dict, dict, dict]] = []
    for index, segment in enumerate(segments, start=1):
        multi_pass["currentPass"] = index
        _set_multi_pass_progress(
            scene,
            segments,
            progress_started,
            f"angle-{index}",
            index,
            f"Analyzing camera angle {index} of {len(segments)}",
            f"Preparing {segment.get('label') or segment['id']}.",
            (index - 1) / len(segments) * 90.0,
            0,
        )
        child = materialize_segment_scene(parent, segment)
        _inherit_match_metadata(child, scene)
        try:
            reconstruction = child.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
            if reconstruction.get("status") != "ready" or not child.get("payload", {}).get("tracks"):
                def relay_progress(child_progress: dict) -> None:
                    child_fraction = float(child_progress.get("overallPercent") or 0.0) / 100.0
                    units_done = index - 1 + child_fraction
                    elapsed = max(0.0, monotonic() - progress_started)
                    eta = (
                        elapsed / units_done * (len(segments) - units_done) + 8.0
                        if units_done > 0.0
                        else None
                    )
                    _set_multi_pass_progress(
                        scene,
                        segments,
                        progress_started,
                        f"angle-{index}",
                        index,
                        f"Analyzing camera angle {index} of {len(segments)}",
                        f"{child_progress.get('label')}: {child_progress.get('detail')}",
                        units_done / len(segments) * 90.0,
                        float(child_progress.get("overallPercent") or 0.0),
                        int(child_progress.get("completed") or 0),
                        int(child_progress.get("total") or 0),
                        eta,
                    )

                child = reconstruct_scene(child, progress_listener=relay_progress)
            summary = _pass_summary(child, segment)
            if summary.get("qualityVerdict") in {None, "pass", "review"}:
                ready_scenes.append((child, summary, segment))
            else:
                summary["error"] = "Reconstruction completed but failed quality gates"
        except ReconstructionError as exc:
            summary = _pass_summary(child, segment, status="failed", error=str(exc))
        pass_summaries.append(summary)
        multi_pass["passes"] = deepcopy(pass_summaries)
        scene_store.put(scene)

    if not ready_scenes:
        _fail(scene, "None of the selected camera angles produced a usable reconstruction", pass_summaries)
        return

    reference_scene, reference_summary, reference_segment = max(
        ready_scenes, key=lambda item: item[1]["quality"]
    )
    _set_multi_pass_progress(
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
    existing_binding = deepcopy(scene["payload"].get("matchBinding"))
    existing_teams = deepcopy(scene["payload"].get("teams") or [])

    scene["duration"] = reference_scene["duration"]
    scene["payload"]["pitch"] = deepcopy(reference_payload["pitch"])
    identity_copy_warnings = _copy_reference_identity_state(
        scene["payload"], reference_payload
    )
    scene["payload"]["ball"] = deepcopy(reference_payload.get("ball") or {"keyframes": []})
    if existing_binding:
        scene["payload"]["matchBinding"] = existing_binding
        for index, team in enumerate(existing_teams[:2]):
            if index < len(reference_payload.get("teams") or []):
                team["color"] = reference_payload["teams"][index]["color"]
        scene["payload"]["teams"] = existing_teams
    else:
        scene["payload"]["teams"] = deepcopy(reference_payload.get("teams") or existing_teams)

    aligned_passes: list[tuple[dict, dict]] = []
    for pass_scene, summary, segment in ready_scenes:
        alignment = _temporal_alignment(
            reference_scene,
            pass_scene,
            reference_segment,
            segment,
            multi_pass.get("manualAlignmentAnchors"),
        )
        summary["relation"] = alignment["relation"]
        summary["alignment"] = alignment
        aligned_passes.append((pass_scene, summary))
    _set_multi_pass_progress(
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
    identity_fusion = _fuse_aligned_pass_identities(
        scene["payload"],
        reference_scene,
        aligned_passes,
    )
    ball_support = _aligned_ball_support(
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
        "matchBindingState": "current",
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
            "completedAt": datetime.now(UTC).isoformat(),
            "trackCount": len(scene["payload"]["tracks"]),
            "ballSamples": len(scene["payload"]["ball"].get("keyframes") or []),
            "multiPassEvidence": consensus,
            "multiPassBallSupport": ball_support,
            "multiPassIdentityFusion": identity_fusion,
            "warnings": [*(reference_reconstruction.get("warnings") or []), *warnings],
            "progress": _set_multi_pass_progress(
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
    scene_video["reconstruction"] = reference_reconstruction
    scene["payload"]["videoAsset"] = scene_video
    scene_store.put(parent)
    scene_store.put(scene)


def _fail(scene: dict, message: str, passes: list[dict] | None = None) -> None:
    video = scene.get("payload", {}).get("videoAsset") or {}
    multi_pass = video.get("multiPass") or {}
    multi_pass.update({"status": "failed", "passes": passes or multi_pass.get("passes") or []})
    reconstruction = video.get("reconstruction") or {}
    reconstruction.update(
        {"status": "failed", "error": message, "completedAt": datetime.now(UTC).isoformat()}
    )
    video["processingState"] = "multi-pass-failed"
    scene_store.put(scene)
