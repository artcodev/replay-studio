"""Editor keyframes and diagnostics materialized from a solved ball path."""

from __future__ import annotations

from copy import deepcopy
from math import exp
from typing import Any, Mapping, Sequence

from .ball_tracking_candidates import NormalizedBallFrame, NormalizedBallFrames, finite_number
from .ball_tracking_contract import (
    BallState,
    BallTrackingConfig,
    BallTrajectoryResolution,
    PositionProjector,
)
from .ball_tracking_solver import BallPathSolution, BallPathStep
from .ball_trajectory_projection import project_ball_candidate


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _observed_keyframe(
    step: BallPathStep,
    projection: Mapping[str, Any],
    config: BallTrackingConfig,
) -> dict[str, Any]:
    candidate = step.candidate
    assert candidate is not None
    transition_quality = exp(-min(12.0, max(0.0, step.transition_cost)) / 6.0)
    keyframe = {
        "t": round(candidate.time, 3),
        "x": round(float(projection["x"]), 3),
        "y": round(config.rendering_ball_height_metres, 3),
        "z": round(float(projection["z"]), 3),
        # Public confidence remains measured detector evidence. The trajectory
        # heuristic is reported separately and never replaces it silently.
        "confidence": round(candidate.confidence, 6),
        "detectionConfidence": round(candidate.confidence, 6),
        "trajectoryConfidence": round(candidate.confidence * transition_quality, 6),
        "state": "observed",
        "observed": True,
        "sourceCandidateId": candidate.candidate_id,
        "candidateRank": candidate.rank,
        "candidateProvenance": deepcopy(candidate.provenance),
        "imagePosition": {
            "x": round(candidate.image_x, 3),
            "y": round(candidate.image_y, 3),
        },
        "heightSource": "rendering-placeholder",
        "projectionSource": projection.get("projectionSource"),
        "calibrationFrameIndex": projection.get("calibrationFrameIndex"),
        "positionUncertaintyMetres": projection.get("positionUncertaintyMetres"),
        "projection": deepcopy(projection.get("projection")),
    }
    if candidate.stabilised_x is not None and candidate.stabilised_y is not None:
        keyframe["stabilizedImagePosition"] = {
            "x": round(candidate.stabilised_x, 3),
            "y": round(candidate.stabilised_y, 3),
            "source": candidate.stabilised_source,
        }
    return keyframe


def _interpolated_keyframe(
    step: BallPathStep,
    left_step: BallPathStep,
    right_step: BallPathStep,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    config: BallTrackingConfig,
) -> dict[str, Any]:
    duration = max(1e-6, right_step.time - left_step.time)
    progress = min(1.0, max(0.0, (step.time - left_step.time) / duration))
    x = float(left["x"]) + (float(right["x"]) - float(left["x"])) * progress
    z = float(left["z"]) + (float(right["z"]) - float(left["z"])) * progress
    distance_to_anchor = min(step.time - left_step.time, right_step.time - step.time)
    base_confidence = min(
        float(left["detectionConfidence"]), float(right["detectionConfidence"])
    )
    confidence = base_confidence * exp(
        -config.interpolation_confidence_decay_per_second * distance_to_anchor
    )
    uncertainties = [
        float(value)
        for value in (
            left.get("positionUncertaintyMetres"),
            right.get("positionUncertaintyMetres"),
        )
        if finite_number(value) is not None
    ]
    uncertainty = (
        max(uncertainties)
        + distance_to_anchor * config.interpolation_uncertainty_metres_per_second
        if uncertainties
        else None
    )
    left_id = left["sourceCandidateId"]
    right_id = right["sourceCandidateId"]
    return {
        "t": round(step.time, 3),
        "x": round(x, 3),
        "y": round(config.rendering_ball_height_metres, 3),
        "z": round(z, 3),
        "confidence": round(confidence, 6),
        "detectionConfidence": None,
        "trajectoryConfidence": round(confidence, 6),
        "state": "inferred",
        "observed": False,
        "sourceCandidateId": None,
        "sourceCandidateIds": [left_id, right_id],
        "candidateProvenance": {
            "kind": "temporal-interpolation",
            "fromCandidateId": left_id,
            "toCandidateId": right_id,
        },
        "heightSource": "rendering-placeholder",
        "projectionSource": "temporal-interpolation",
        "calibrationFrameIndex": None,
        "positionUncertaintyMetres": _round_or_none(uncertainty, 3),
        "projection": {
            "source": "temporal-interpolation",
            "calibrationFrameIndex": None,
            "uncertaintyMetres": _round_or_none(uncertainty, 3),
        },
    }


def _gap_statistics(
    states: Sequence[BallState], frames: Sequence[NormalizedBallFrame]
) -> dict[str, Any]:
    gaps: list[tuple[BallState, int, int]] = []
    start: int | None = None
    current: BallState | None = None
    for index, state in enumerate(states):
        if state == "observed":
            if start is not None and current is not None:
                gaps.append((current, start, index - 1))
            start = None
            current = None
            continue
        if start is None or state != current:
            if start is not None and current is not None:
                gaps.append((current, start, index - 1))
            start, current = index, state
    if start is not None and current is not None:
        gaps.append((current, start, len(states) - 1))
    durations = [
        max(0.0, frames[end].time - frames[start].time)
        for _, start, end in gaps
    ]
    return {
        "gapCount": len(gaps),
        "inferredGapCount": sum(state == "inferred" for state, _, _ in gaps),
        "occludedGapCount": sum(state == "occluded" for state, _, _ in gaps),
        "longestGapFrames": max((end - start + 1 for _, start, end in gaps), default=0),
        "longestGapSeconds": round(max(durations, default=0.0), 4),
    }


def _empty_diagnostics(
    normalized: NormalizedBallFrames,
    peak_hypothesis_count: int,
    config: BallTrackingConfig,
) -> dict[str, Any]:
    frames = normalized.frames
    candidate_count = sum(len(frame.candidates) for frame in frames)
    return {
        "algorithm": "beam-viterbi-ball-v1",
        "status": "no-stable-trajectory",
        "frameCount": len(frames),
        "candidateCount": candidate_count,
        "invalidCandidateCount": normalized.invalid_candidate_count,
        "droppedByTopKCount": normalized.dropped_by_top_k_count,
        "peakHypothesisCount": peak_hypothesis_count,
        "observedFrameCount": 0,
        "inferredFrameCount": 0,
        "occludedFrameCount": len(frames),
        "observedCoverage": 0.0,
        "publishedCoverage": 0.0,
        "selectedCandidateIds": [],
        "totalCost": None,
        "runnerUpCost": None,
        "pathCostMargin": None,
        "path": [
            {
                "frameIndex": index,
                "t": round(frame.time, 3),
                "state": "occluded",
                "candidateCount": len(frame.candidates),
                "reason": "insufficient-global-evidence",
            }
            for index, frame in enumerate(frames)
        ],
        "motion": {
            "transitionCount": 0,
            "metricTransitionCount": 0,
            "stabilizedTransitionCount": 0,
            "rawImageTransitionCount": 0,
            "speedViolationCount": 0,
            "accelerationViolationCount": 0,
            "maximumMetricSpeedMetresPerSecond": None,
            "maximumImageSpeedPixelsPerSecond": None,
        },
        "gaps": {
            "gapCount": 1 if frames else 0,
            "inferredGapCount": 0,
            "occludedGapCount": 1 if frames else 0,
            "longestGapFrames": len(frames),
            "longestGapSeconds": round(frames[-1].time - frames[0].time, 4)
            if len(frames) > 1
            else 0.0,
        },
        "config": {
            "topKPerFrame": config.top_k_per_frame,
            "beamWidth": config.beam_width,
            "maxInterpolationGapSeconds": config.max_interpolation_gap_seconds,
        },
    }


def _neighboring_observations(
    frame_count: int, observed_indices: Sequence[int]
) -> tuple[list[int | None], list[int | None]]:
    previous: list[int | None] = [None] * frame_count
    following: list[int | None] = [None] * frame_count
    last: int | None = None
    observed = set(observed_indices)
    for index in range(frame_count):
        previous[index] = last
        if index in observed:
            last = index
    next_index: int | None = None
    for index in range(frame_count - 1, -1, -1):
        following[index] = next_index
        if index in observed:
            next_index = index
    return previous, following


def materialize_ball_trajectory(
    normalized: NormalizedBallFrames,
    solution: BallPathSolution,
    frame_size: tuple[int, int],
    pitch: Mapping[str, Any],
    config: BallTrackingConfig,
    projector: PositionProjector | None,
) -> BallTrajectoryResolution:
    """Convert the solver result into editor keyframes and QA diagnostics."""

    frames = normalized.frames
    selected = solution.path
    if selected is None:
        return BallTrajectoryResolution(
            keyframes=[],
            diagnostics=_empty_diagnostics(
                normalized, solution.peak_hypothesis_count, config
            ),
        )

    observed_indices = [
        index for index, step in enumerate(selected.steps) if step.state == "observed"
    ]
    observed_keyframes: dict[int, dict[str, Any]] = {}
    for index in observed_indices:
        step = selected.steps[index]
        assert step.candidate is not None
        projection = project_ball_candidate(step.candidate, frame_size, pitch, projector)
        observed_keyframes[index] = _observed_keyframe(step, projection, config)

    states: list[BallState] = ["occluded"] * len(selected.steps)
    for index in observed_indices:
        states[index] = "observed"
    for left_index, right_index in zip(observed_indices, observed_indices[1:]):
        if right_index - left_index <= 1:
            continue
        gap_duration = selected.steps[right_index].time - selected.steps[left_index].time
        if gap_duration <= config.max_interpolation_gap_seconds:
            for index in range(left_index + 1, right_index):
                states[index] = "inferred"

    previous_observed, next_observed = _neighboring_observations(
        len(selected.steps), observed_indices
    )
    keyframes: list[dict[str, Any]] = []
    path: list[dict[str, Any]] = []
    for index, (step, state) in enumerate(zip(selected.steps, states)):
        if state == "observed":
            keyframe = observed_keyframes[index]
            keyframes.append(keyframe)
            path_entry: dict[str, Any] = {
                "frameIndex": index,
                "t": round(step.time, 3),
                "state": state,
                "candidateId": keyframe["sourceCandidateId"],
                "candidateRank": keyframe["candidateRank"],
                "candidateConfidence": keyframe["detectionConfidence"],
                "candidateProvenance": deepcopy(keyframe["candidateProvenance"]),
                "candidateCount": len(frames[index].candidates),
                "emissionCost": round(step.emission_cost, 6),
                "transitionCost": round(step.transition_cost, 6),
            }
            if step.motion is not None:
                path_entry["motion"] = {
                    "source": step.motion.source,
                    "distance": round(step.motion.distance, 4),
                    "elapsedSeconds": round(step.motion.elapsed_seconds, 4),
                    "speed": round(step.motion.speed, 4),
                    "speedLimit": round(step.motion.speed_limit, 4),
                    "acceleration": _round_or_none(step.motion.acceleration, 4),
                    "accelerationLimit": _round_or_none(
                        step.motion.acceleration_limit, 4
                    ),
                    "speedViolation": step.motion.speed_violation,
                    "accelerationViolation": step.motion.acceleration_violation,
                }
            path.append(path_entry)
            continue

        left_index = previous_observed[index]
        right_index = next_observed[index]
        if state == "inferred" and left_index is not None and right_index is not None:
            keyframe = _interpolated_keyframe(
                step,
                selected.steps[left_index],
                selected.steps[right_index],
                observed_keyframes[left_index],
                observed_keyframes[right_index],
                config,
            )
            keyframes.append(keyframe)
            path.append(
                {
                    "frameIndex": index,
                    "t": round(step.time, 3),
                    "state": "inferred",
                    "candidateId": None,
                    "candidateCount": len(frames[index].candidates),
                    "confidence": keyframe["confidence"],
                    "fromCandidateId": keyframe["sourceCandidateIds"][0],
                    "toCandidateId": keyframe["sourceCandidateIds"][1],
                    "emissionCost": round(step.emission_cost, 6),
                    "transitionCost": 0.0,
                }
            )
            continue

        if left_index is None:
            reason = "before-first-observation"
        elif right_index is None:
            reason = "after-last-observation"
        else:
            reason = "gap-exceeds-interpolation-limit"
        path.append(
            {
                "frameIndex": index,
                "t": round(step.time, 3),
                "state": "occluded",
                "candidateId": None,
                "candidateCount": len(frames[index].candidates),
                "reason": reason,
                "emissionCost": round(step.emission_cost, 6),
                "transitionCost": 0.0,
            }
        )

    motions = [step.motion for step in selected.steps if step.motion is not None]
    metric_speeds = [motion.speed for motion in motions if motion.source == "pitch-metric"]
    image_speeds = [motion.speed for motion in motions if motion.source != "pitch-metric"]
    observed_confidences = [
        step.candidate.confidence
        for step in selected.steps
        if step.candidate is not None
    ]
    frame_count = len(frames)
    observed_count = states.count("observed")
    inferred_count = states.count("inferred")
    diagnostics = {
        "algorithm": "beam-viterbi-ball-v1",
        "status": "resolved",
        "frameCount": frame_count,
        "candidateCount": sum(len(frame.candidates) for frame in frames),
        "invalidCandidateCount": normalized.invalid_candidate_count,
        "droppedByTopKCount": normalized.dropped_by_top_k_count,
        "peakHypothesisCount": solution.peak_hypothesis_count,
        "observedFrameCount": observed_count,
        "inferredFrameCount": inferred_count,
        "occludedFrameCount": states.count("occluded"),
        "observedCoverage": round(observed_count / max(1, frame_count), 4),
        "publishedCoverage": round(
            (observed_count + inferred_count) / max(1, frame_count), 4
        ),
        "meanDetectionConfidence": round(
            sum(observed_confidences) / max(1, len(observed_confidences)), 6
        ),
        "peakDetectionConfidence": round(max(observed_confidences), 6),
        "selectedCandidateIds": [
            step.candidate.candidate_id
            for step in selected.steps
            if step.candidate is not None
        ],
        "totalCost": round(selected.cost, 6),
        "runnerUpCost": _round_or_none(solution.runner_up_cost, 6),
        "pathCostMargin": _round_or_none(
            solution.runner_up_cost - selected.cost
            if solution.runner_up_cost is not None
            else None,
            6,
        ),
        "path": path,
        "motion": {
            "transitionCount": len(motions),
            "metricTransitionCount": sum(
                motion.source == "pitch-metric" for motion in motions
            ),
            "stabilizedTransitionCount": sum(
                motion.source not in {"pitch-metric", "image-raw"} for motion in motions
            ),
            "rawImageTransitionCount": sum(
                motion.source == "image-raw" for motion in motions
            ),
            "speedViolationCount": sum(motion.speed_violation for motion in motions),
            "accelerationViolationCount": sum(
                motion.acceleration_violation for motion in motions
            ),
            "maximumMetricSpeedMetresPerSecond": _round_or_none(
                max(metric_speeds) if metric_speeds else None, 4
            ),
            "maximumImageSpeedPixelsPerSecond": _round_or_none(
                max(image_speeds) if image_speeds else None, 4
            ),
        },
        "gaps": _gap_statistics(states, frames),
        "config": {
            "topKPerFrame": config.top_k_per_frame,
            "beamWidth": config.beam_width,
            "maxInterpolationGapSeconds": config.max_interpolation_gap_seconds,
            "maxBallSpeedMetresPerSecond": config.max_ball_speed_metres_per_second,
            "maxImageSpeedPixelsPerSecond": config.max_image_speed_pixels_per_second,
        },
    }
    return BallTrajectoryResolution(keyframes=keyframes, diagnostics=diagnostics)


__all__ = ["materialize_ball_trajectory"]
