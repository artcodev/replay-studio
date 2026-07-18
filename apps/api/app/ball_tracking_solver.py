"""Beam-pruned Viterbi solver for a globally consistent ball path."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, log
from typing import Any, Literal, Sequence

from .ball_tracking_candidates import BallCandidate, NormalizedBallFrame
from .ball_tracking_contract import BallTrackingConfig


@dataclass(frozen=True)
class BallMotion:
    source: str
    distance: float
    elapsed_seconds: float
    speed: float
    speed_limit: float
    acceleration: float | None
    acceleration_limit: float | None
    speed_violation: bool
    acceleration_violation: bool


@dataclass(frozen=True)
class BallPathStep:
    frame_index: int
    time: float
    state: Literal["observed", "occluded"]
    candidate: BallCandidate | None
    emission_cost: float
    transition_cost: float
    motion: BallMotion | None = None


@dataclass(frozen=True, slots=True)
class _PathNode:
    """One immutable step in a shared hypothesis path."""

    step: BallPathStep
    parent: _PathNode | None
    depth: int


@dataclass(frozen=True, slots=True)
class _Hypothesis:
    cost: float
    tail: _PathNode | None
    previous_observed: BallCandidate | None
    last_observed: BallCandidate | None
    observation_count: int
    peak_confidence: float


@dataclass(frozen=True, slots=True)
class ResolvedBallPath:
    """Winning path after its backpointers have been materialised once."""

    cost: float
    steps: tuple[BallPathStep, ...]


@dataclass(frozen=True, slots=True)
class BallPathSolution:
    path: ResolvedBallPath | None
    runner_up_cost: float | None
    peak_hypothesis_count: int


def _pair_coordinates(
    left: BallCandidate, right: BallCandidate
) -> tuple[tuple[float, float], tuple[float, float], str]:
    if None not in (left.pitch_x, left.pitch_z, right.pitch_x, right.pitch_z):
        return (
            (float(left.pitch_x), float(left.pitch_z)),
            (float(right.pitch_x), float(right.pitch_z)),
            "pitch-metric",
        )
    if None not in (
        left.stabilised_x,
        left.stabilised_y,
        right.stabilised_x,
        right.stabilised_y,
    ):
        source = (
            left.stabilised_source
            if left.stabilised_source == right.stabilised_source
            else "camera-stabilized-mixed"
        )
        return (
            (float(left.stabilised_x), float(left.stabilised_y)),
            (float(right.stabilised_x), float(right.stabilised_y)),
            str(source),
        )
    return (left.image_x, left.image_y), (right.image_x, right.image_y), "image-raw"


def _triple_coordinates(
    first: BallCandidate, second: BallCandidate, third: BallCandidate
) -> tuple[
    tuple[float, float], tuple[float, float], tuple[float, float], str
]:
    if all(
        value is not None
        for candidate in (first, second, third)
        for value in (candidate.pitch_x, candidate.pitch_z)
    ):
        return (
            (float(first.pitch_x), float(first.pitch_z)),
            (float(second.pitch_x), float(second.pitch_z)),
            (float(third.pitch_x), float(third.pitch_z)),
            "pitch-metric",
        )
    if all(
        value is not None
        for candidate in (first, second, third)
        for value in (candidate.stabilised_x, candidate.stabilised_y)
    ):
        return (
            (float(first.stabilised_x), float(first.stabilised_y)),
            (float(second.stabilised_x), float(second.stabilised_y)),
            (float(third.stabilised_x), float(third.stabilised_y)),
            "camera-stabilized",
        )
    return (
        (first.image_x, first.image_y),
        (second.image_x, second.image_y),
        (third.image_x, third.image_y),
        "image-raw",
    )


def _transition(
    previous: BallCandidate,
    current: BallCandidate,
    predecessor: BallCandidate | None,
    config: BallTrackingConfig,
) -> tuple[float, BallMotion]:
    left, right, source = _pair_coordinates(previous, current)
    elapsed = max(1e-6, current.time - previous.time)
    distance = hypot(right[0] - left[0], right[1] - left[1])
    metric = source == "pitch-metric"
    speed_limit = (
        config.max_ball_speed_metres_per_second
        if metric
        else config.max_image_speed_pixels_per_second
    )
    speed = distance / elapsed
    speed_ratio = speed / speed_limit
    cost = config.motion_penalty_weight * speed_ratio * speed_ratio
    speed_violation = speed > speed_limit
    if speed_violation:
        cost += config.physical_violation_penalty * (speed_ratio - 1.0) ** 2

    acceleration: float | None = None
    acceleration_limit: float | None = None
    acceleration_violation = False
    if predecessor is not None:
        first, second, third, acceleration_source = _triple_coordinates(
            predecessor, previous, current
        )
        first_elapsed = max(1e-6, previous.time - predecessor.time)
        second_elapsed = elapsed
        first_velocity = (
            (second[0] - first[0]) / first_elapsed,
            (second[1] - first[1]) / first_elapsed,
        )
        second_velocity = (
            (third[0] - second[0]) / second_elapsed,
            (third[1] - second[1]) / second_elapsed,
        )
        acceleration = hypot(
            second_velocity[0] - first_velocity[0],
            second_velocity[1] - first_velocity[1],
        ) / max(1e-6, (first_elapsed + second_elapsed) / 2.0)
        acceleration_limit = (
            config.max_ball_acceleration_metres_per_second_squared
            if acceleration_source == "pitch-metric"
            else config.max_image_acceleration_pixels_per_second_squared
        )
        acceleration_ratio = acceleration / acceleration_limit
        cost += config.acceleration_penalty_weight * acceleration_ratio * acceleration_ratio
        acceleration_violation = acceleration > acceleration_limit
        if acceleration_violation:
            cost += config.physical_violation_penalty * (acceleration_ratio - 1.0) ** 2

    gap_seconds = max(0.0, elapsed - config.preferred_gap_seconds)
    cost += gap_seconds * config.long_gap_penalty_per_second
    return cost, BallMotion(
        source=source,
        distance=distance,
        elapsed_seconds=elapsed,
        speed=speed,
        speed_limit=speed_limit,
        acceleration=acceleration,
        acceleration_limit=acceleration_limit,
        speed_violation=speed_violation,
        acceleration_violation=acceleration_violation,
    )


def _hypothesis_signature(
    hypothesis: _Hypothesis, config: BallTrackingConfig
) -> tuple[Any, ...]:
    assert hypothesis.tail is not None
    step = hypothesis.tail.step
    return (
        step.candidate.candidate_id if step.candidate is not None else None,
        hypothesis.last_observed.frame_index
        if hypothesis.last_observed is not None
        else None,
        hypothesis.last_observed.candidate_id
        if hypothesis.last_observed is not None
        else None,
        hypothesis.previous_observed.frame_index
        if hypothesis.previous_observed is not None
        else None,
        hypothesis.previous_observed.candidate_id
        if hypothesis.previous_observed is not None
        else None,
        min(hypothesis.observation_count, config.minimum_observed_frames),
        hypothesis.peak_confidence >= config.minimum_peak_confidence,
    )


def _extend_path(hypothesis: _Hypothesis, step: BallPathStep) -> _PathNode:
    return _PathNode(
        step=step,
        parent=hypothesis.tail,
        depth=1 if hypothesis.tail is None else hypothesis.tail.depth + 1,
    )


def _materialise_steps(tail: _PathNode) -> tuple[BallPathStep, ...]:
    """Follow winner backpointers once and restore chronological ordering."""

    reversed_steps: list[BallPathStep] = []
    cursor: _PathNode | None = tail
    while cursor is not None:
        reversed_steps.append(cursor.step)
        cursor = cursor.parent
    reversed_steps.reverse()
    assert len(reversed_steps) == tail.depth
    return tuple(reversed_steps)


def solve_ball_path(
    frames: Sequence[NormalizedBallFrame], config: BallTrackingConfig
) -> BallPathSolution:
    """Choose the lowest-cost globally stable path through frame candidates."""

    if not frames:
        return BallPathSolution(None, None, 0)
    beam = [
        _Hypothesis(
            cost=0.0,
            tail=None,
            previous_observed=None,
            last_observed=None,
            observation_count=0,
            peak_confidence=0.0,
        )
    ]
    peak_hypothesis_count = 1
    for frame_index, frame in enumerate(frames):
        previous_time = frames[frame_index - 1].time if frame_index else frame.time
        frame_elapsed = max(0.0, frame.time - previous_time)
        expanded: list[_Hypothesis] = []
        for hypothesis in beam:
            previous_was_observed = bool(
                hypothesis.tail is not None
                and hypothesis.tail.step.state == "observed"
            )
            emission = (
                config.occlusion_cost_per_frame
                + config.occlusion_cost_per_second * frame_elapsed
                + (config.occlusion_start_penalty if previous_was_observed else 0.0)
            )
            occluded_step = BallPathStep(
                frame_index=frame_index,
                time=frame.time,
                state="occluded",
                candidate=None,
                emission_cost=emission,
                transition_cost=0.0,
            )
            expanded.append(
                _Hypothesis(
                    cost=hypothesis.cost + emission,
                    tail=_extend_path(hypothesis, occluded_step),
                    previous_observed=hypothesis.previous_observed,
                    last_observed=hypothesis.last_observed,
                    observation_count=hypothesis.observation_count,
                    peak_confidence=hypothesis.peak_confidence,
                )
            )
            for candidate in frame.candidates:
                candidate_emission = config.observation_cost_weight * -log(
                    max(config.confidence_floor, candidate.confidence)
                )
                transition_cost = 0.0
                motion = None
                if hypothesis.last_observed is not None:
                    transition_cost, motion = _transition(
                        hypothesis.last_observed,
                        candidate,
                        hypothesis.previous_observed,
                        config,
                    )
                    if not previous_was_observed:
                        transition_cost += config.reacquisition_penalty
                observed_step = BallPathStep(
                    frame_index=frame_index,
                    time=frame.time,
                    state="observed",
                    candidate=candidate,
                    emission_cost=candidate_emission,
                    transition_cost=transition_cost,
                    motion=motion,
                )
                expanded.append(
                    _Hypothesis(
                        cost=hypothesis.cost + candidate_emission + transition_cost,
                        tail=_extend_path(hypothesis, observed_step),
                        previous_observed=hypothesis.last_observed,
                        last_observed=candidate,
                        observation_count=hypothesis.observation_count + 1,
                        peak_confidence=max(
                            hypothesis.peak_confidence, candidate.confidence
                        ),
                    )
                )

        best_by_state: dict[tuple[Any, ...], _Hypothesis] = {}
        for hypothesis in expanded:
            signature = _hypothesis_signature(hypothesis, config)
            current = best_by_state.get(signature)
            rank = (
                hypothesis.cost,
                -hypothesis.observation_count,
                -hypothesis.peak_confidence,
            )
            if current is None or rank < (
                current.cost,
                -current.observation_count,
                -current.peak_confidence,
            ):
                best_by_state[signature] = hypothesis
        ordered = sorted(
            best_by_state.values(),
            key=lambda item: (item.cost, -item.observation_count, -item.peak_confidence),
        )
        peak_hypothesis_count = max(peak_hypothesis_count, len(ordered))
        beam = ordered[: config.beam_width]

    valid = [
        hypothesis
        for hypothesis in beam
        if hypothesis.observation_count >= config.minimum_observed_frames
        and hypothesis.peak_confidence >= config.minimum_peak_confidence
    ]
    if not valid:
        return BallPathSolution(None, None, peak_hypothesis_count)
    valid.sort(key=lambda item: (item.cost, -item.observation_count, -item.peak_confidence))
    runner_up_cost = valid[1].cost if len(valid) > 1 else None
    winner = valid[0]
    assert winner.tail is not None
    return BallPathSolution(
        path=ResolvedBallPath(
            cost=winner.cost,
            steps=_materialise_steps(winner.tail),
        ),
        runner_up_cost=runner_up_cost,
        peak_hypothesis_count=peak_hypothesis_count,
    )


__all__ = [
    "BallMotion",
    "BallPathSolution",
    "BallPathStep",
    "ResolvedBallPath",
    "solve_ball_path",
]
