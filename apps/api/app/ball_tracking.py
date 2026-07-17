"""Temporal multi-hypothesis resolution for football detections.

The detector is deliberately kept outside this module.  A detector may emit
several plausible balls per frame; this resolver chooses one globally
consistent trajectory with a beam-pruned Viterbi pass.  It prefers metric
pitch coordinates when available, then camera-stabilised image coordinates,
and finally raw image coordinates.

The public keyframes retain the scene player's legacy ``t/x/y/z/confidence``
shape.  Extra fields distinguish measured observations from short temporal
interpolations and preserve the selected detector candidate's provenance.
Long or one-sided gaps remain explicitly occluded instead of being invented.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import exp, hypot, isfinite, log
from statistics import median
from typing import Any, Callable, Literal, Mapping, Sequence


BallState = Literal["observed", "inferred", "occluded"]
MotionCoordinateSelector = Callable[
    [Mapping[str, Any], int], tuple[float, float, str] | None
]
PositionProjector = Callable[
    [Mapping[str, Any]], tuple[float, float] | Mapping[str, Any]
]


@dataclass(frozen=True)
class BallTrackingConfig:
    """Costs and physical limits used by the temporal resolver.

    Pixel limits are only fallbacks.  Metric limits take precedence whenever
    both detections have calibrated pitch coordinates.  Limits are soft: an
    impossible transition receives a large cost rather than deleting the
    hypothesis, which keeps diagnostics available for difficult clips.
    """

    top_k_per_frame: int = 6
    beam_width: int = 128
    confidence_floor: float = 1e-4
    observation_cost_weight: float = 1.0
    occlusion_cost_per_frame: float = 1.35
    occlusion_cost_per_second: float = 0.20
    occlusion_start_penalty: float = 0.20
    reacquisition_penalty: float = 0.30
    preferred_gap_seconds: float = 0.60
    long_gap_penalty_per_second: float = 1.25
    motion_penalty_weight: float = 0.80
    acceleration_penalty_weight: float = 0.25
    physical_violation_penalty: float = 24.0
    max_ball_speed_metres_per_second: float = 55.0
    max_ball_acceleration_metres_per_second_squared: float = 180.0
    max_image_speed_pixels_per_second: float = 1_800.0
    max_image_acceleration_pixels_per_second_squared: float = 8_000.0
    max_interpolation_gap_seconds: float = 0.80
    interpolation_confidence_decay_per_second: float = 0.85
    interpolation_uncertainty_metres_per_second: float = 6.0
    minimum_observed_frames: int = 2
    minimum_peak_confidence: float = 0.12
    rendering_ball_height_metres: float = 0.22

    def __post_init__(self) -> None:
        if self.top_k_per_frame < 1 or self.beam_width < 1:
            raise ValueError("top_k_per_frame and beam_width must be positive")
        if self.minimum_observed_frames < 1:
            raise ValueError("minimum_observed_frames must be positive")
        if not 0.0 < self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in (0, 1]")
        if not 0.0 <= self.minimum_peak_confidence <= 1.0:
            raise ValueError("minimum_peak_confidence must be between 0 and 1")
        non_negative = (
            self.observation_cost_weight,
            self.occlusion_cost_per_frame,
            self.occlusion_cost_per_second,
            self.occlusion_start_penalty,
            self.reacquisition_penalty,
            self.preferred_gap_seconds,
            self.long_gap_penalty_per_second,
            self.motion_penalty_weight,
            self.acceleration_penalty_weight,
            self.physical_violation_penalty,
            self.max_interpolation_gap_seconds,
            self.interpolation_confidence_decay_per_second,
            self.interpolation_uncertainty_metres_per_second,
            self.rendering_ball_height_metres,
        )
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in non_negative):
            raise ValueError("Ball-tracking costs must be finite and non-negative")
        positive = (
            self.max_ball_speed_metres_per_second,
            self.max_ball_acceleration_metres_per_second_squared,
            self.max_image_speed_pixels_per_second,
            self.max_image_acceleration_pixels_per_second_squared,
        )
        if any(not isfinite(float(value)) or float(value) <= 0.0 for value in positive):
            raise ValueError("Ball-tracking physical limits must be finite and positive")


DEFAULT_BALL_TRACKING_CONFIG = BallTrackingConfig()


@dataclass(frozen=True)
class BallTrajectoryResolution:
    """Resolved legacy keyframes plus JSON-serialisable temporal diagnostics."""

    keyframes: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "keyframes": deepcopy(self.keyframes),
            "diagnostics": deepcopy(self.diagnostics),
        }


@dataclass(frozen=True)
class _Candidate:
    frame_index: int
    time: float
    rank: int
    candidate_id: str
    image_x: float
    image_y: float
    confidence: float
    pitch_x: float | None
    pitch_z: float | None
    stabilised_x: float | None
    stabilised_y: float | None
    stabilised_source: str | None
    provenance: dict[str, Any]
    source: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class _Motion:
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
class _PathStep:
    frame_index: int
    time: float
    state: Literal["observed", "occluded"]
    candidate: _Candidate | None
    emission_cost: float
    transition_cost: float
    motion: _Motion | None = None


@dataclass(frozen=True, slots=True)
class _PathNode:
    """One immutable step in a shared hypothesis path.

    Beam hypotheses share their retained prefixes through these nodes.  This
    avoids copying an ever-growing tuple for every candidate expansion.
    """

    step: _PathStep
    parent: _PathNode | None
    depth: int


@dataclass(frozen=True, slots=True)
class _Hypothesis:
    cost: float
    tail: _PathNode | None
    previous_observed: _Candidate | None
    last_observed: _Candidate | None
    observation_count: int
    peak_confidence: float


@dataclass(frozen=True, slots=True)
class _ResolvedPath:
    """Winning path after its backpointers have been materialised once."""

    cost: float
    steps: tuple[_PathStep, ...]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _first_number(source: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(source.get(key))
        if value is not None:
            return value
    return None


def _nested_point(source: Mapping[str, Any], key: str) -> tuple[float, float] | None:
    value = source.get(key)
    if not isinstance(value, Mapping):
        return None
    x = _first_number(value, "x", "X")
    y = _first_number(value, "y", "Y")
    return (x, y) if x is not None and y is not None else None


def _pitch_point(source: Mapping[str, Any]) -> tuple[float, float] | None:
    x = _first_number(source, "pitchX", "pitch_x")
    z = _first_number(source, "pitchZ", "pitch_z")
    nested = source.get("pitch")
    if (x is None or z is None) and isinstance(nested, Mapping):
        x = _first_number(nested, "x", "X")
        z = _first_number(nested, "z", "Z", "y", "Y")
    return (x, z) if x is not None and z is not None else None


def _automatic_stabilised_point(
    source: Mapping[str, Any],
) -> tuple[float, float, str] | None:
    key_pairs = (
        ("stabilizedX", "stabilizedY", "camera-stabilized"),
        ("stabilisedX", "stabilisedY", "camera-stabilised"),
        ("stableX", "stableY", "camera-stabilized"),
        ("cameraCompensatedX", "cameraCompensatedY", "camera-compensated"),
    )
    for x_key, y_key, label in key_pairs:
        x, y = _number(source.get(x_key)), _number(source.get(y_key))
        if x is not None and y is not None:
            return x, y, label
    for key, label in (
        ("stabilized", "camera-stabilized"),
        ("stabilised", "camera-stabilised"),
        ("cameraCompensated", "camera-compensated"),
        ("motionCoordinates", "camera-motion-hook"),
    ):
        if (point := _nested_point(source, key)) is not None:
            return point[0], point[1], label
    return None


def _candidate_provenance(source: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    supplied = source.get("provenance")
    if isinstance(supplied, Mapping):
        provenance: dict[str, Any] = deepcopy(dict(supplied))
    elif supplied is not None:
        provenance = {"value": deepcopy(supplied)}
    else:
        provenance = {}
    for key in (
        "source",
        "backend",
        "detector",
        "model",
        "modelVersion",
        "inferenceId",
        "passId",
        "tile",
        "metadata",
    ):
        if key in source and key not in provenance:
            provenance[key] = deepcopy(source[key])
    provenance.setdefault("candidateId", candidate_id)
    return provenance


def _frame_contents(frame: Any, fallback_index: int) -> tuple[Sequence[Any], float]:
    if isinstance(frame, Mapping):
        time = _first_number(frame, "t", "time", "timestamp", "sceneTime")
        candidates = frame.get("candidates")
        if candidates is None:
            candidates = frame.get("detections")
        if candidates is None:
            candidates = frame.get("balls")
    elif isinstance(frame, (tuple, list)) and len(frame) == 2:
        candidates, time = frame
        time = _number(time)
    else:
        raise ValueError(
            f"Ball frame {fallback_index} must be a mapping or (candidates, time) pair"
        )
    if time is None:
        raise ValueError(f"Ball frame {fallback_index} has no finite timestamp")
    if candidates is None:
        candidates = []
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError(f"Ball frame {fallback_index} candidates must be a sequence")
    return candidates, time


def _normalise_frames(
    frames: Sequence[Any],
    config: BallTrackingConfig,
    coordinate_selector: MotionCoordinateSelector | None,
) -> tuple[list[tuple[float, list[_Candidate]]], int, int]:
    result: list[tuple[float, list[_Candidate]]] = []
    invalid_candidates = 0
    dropped_by_top_k = 0
    previous_time: float | None = None
    for frame_index, frame in enumerate(frames):
        raw_candidates, time = _frame_contents(frame, frame_index)
        if previous_time is not None and time <= previous_time:
            raise ValueError("Ball-frame timestamps must be strictly increasing")
        previous_time = time
        parsed: list[tuple[float, int, Mapping[str, Any]]] = []
        for source_index, raw in enumerate(raw_candidates):
            if not isinstance(raw, Mapping):
                invalid_candidates += 1
                continue
            x, y = _number(raw.get("x")), _number(raw.get("y"))
            confidence = _first_number(raw, "confidence", "score", "probability")
            if x is None or y is None or confidence is None:
                invalid_candidates += 1
                continue
            parsed.append((min(1.0, max(0.0, confidence)), source_index, raw))
        parsed.sort(key=lambda item: (-item[0], item[1]))
        dropped_by_top_k += max(0, len(parsed) - config.top_k_per_frame)
        normalised: list[_Candidate] = []
        for rank, (confidence, source_index, raw) in enumerate(
            parsed[: config.top_k_per_frame], start=1
        ):
            identifier = raw.get("candidateId", raw.get("id", raw.get("detectionId")))
            candidate_id = (
                str(identifier).strip()
                if identifier is not None and str(identifier).strip()
                else f"ball-f{frame_index}-c{source_index}"
            )
            pitch = _pitch_point(raw)
            stabilised: tuple[float, float, str] | None = None
            if coordinate_selector is not None:
                selected = coordinate_selector(raw, frame_index)
                if selected is not None:
                    if len(selected) != 3:
                        raise ValueError("coordinate_selector must return (x, y, source)")
                    stable_x, stable_y = _number(selected[0]), _number(selected[1])
                    if stable_x is None or stable_y is None or not str(selected[2]).strip():
                        raise ValueError("coordinate_selector returned invalid coordinates")
                    stabilised = stable_x, stable_y, str(selected[2]).strip()
            if stabilised is None:
                stabilised = _automatic_stabilised_point(raw)
            normalised.append(
                _Candidate(
                    frame_index=frame_index,
                    time=time,
                    rank=rank,
                    candidate_id=candidate_id,
                    image_x=float(raw["x"]),
                    image_y=float(raw["y"]),
                    confidence=confidence,
                    pitch_x=pitch[0] if pitch is not None else None,
                    pitch_z=pitch[1] if pitch is not None else None,
                    stabilised_x=stabilised[0] if stabilised is not None else None,
                    stabilised_y=stabilised[1] if stabilised is not None else None,
                    stabilised_source=stabilised[2] if stabilised is not None else None,
                    provenance=_candidate_provenance(raw, candidate_id),
                    source=deepcopy(dict(raw)),
                )
            )
        result.append((time, normalised))
    return result, invalid_candidates, dropped_by_top_k


def _pair_coordinates(
    left: _Candidate, right: _Candidate
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
    first: _Candidate, second: _Candidate, third: _Candidate
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
    previous: _Candidate,
    current: _Candidate,
    predecessor: _Candidate | None,
    config: BallTrackingConfig,
) -> tuple[float, _Motion]:
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
    motion = _Motion(
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
    return cost, motion


def _hypothesis_signature(
    hypothesis: _Hypothesis, config: BallTrackingConfig
) -> tuple[Any, ...]:
    assert hypothesis.tail is not None
    step = hypothesis.tail.step
    return (
        step.candidate.candidate_id if step.candidate is not None else None,
        hypothesis.last_observed.frame_index if hypothesis.last_observed is not None else None,
        hypothesis.last_observed.candidate_id if hypothesis.last_observed is not None else None,
        hypothesis.previous_observed.frame_index
        if hypothesis.previous_observed is not None
        else None,
        hypothesis.previous_observed.candidate_id
        if hypothesis.previous_observed is not None
        else None,
        min(hypothesis.observation_count, config.minimum_observed_frames),
        hypothesis.peak_confidence >= config.minimum_peak_confidence,
    )


def _extend_path(hypothesis: _Hypothesis, step: _PathStep) -> _PathNode:
    """Append ``step`` in O(1), sharing the hypothesis prefix."""

    return _PathNode(
        step=step,
        parent=hypothesis.tail,
        depth=1 if hypothesis.tail is None else hypothesis.tail.depth + 1,
    )


def _materialise_steps(tail: _PathNode) -> tuple[_PathStep, ...]:
    """Follow winner backpointers once and restore chronological ordering."""

    reversed_steps: list[_PathStep] = []
    cursor: _PathNode | None = tail
    while cursor is not None:
        reversed_steps.append(cursor.step)
        cursor = cursor.parent
    reversed_steps.reverse()
    assert len(reversed_steps) == tail.depth
    return tuple(reversed_steps)


def _resolve_path(
    frames: Sequence[tuple[float, list[_Candidate]]], config: BallTrackingConfig
) -> tuple[_ResolvedPath | None, float | None, int]:
    if not frames:
        return None, None, 0
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
    for frame_index, (time, candidates) in enumerate(frames):
        previous_time = frames[frame_index - 1][0] if frame_index else time
        frame_elapsed = max(0.0, time - previous_time)
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
            occluded_step = _PathStep(
                frame_index=frame_index,
                time=time,
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
            for candidate in candidates:
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
                observed_step = _PathStep(
                    frame_index=frame_index,
                    time=time,
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
        return None, None, peak_hypothesis_count
    valid.sort(key=lambda item: (item.cost, -item.observation_count, -item.peak_confidence))
    runner_up_cost = valid[1].cost if len(valid) > 1 else None
    winner = valid[0]
    assert winner.tail is not None
    return (
        _ResolvedPath(
            cost=winner.cost,
            steps=_materialise_steps(winner.tail),
        ),
        runner_up_cost,
        peak_hypothesis_count,
    )


def _default_projection(
    candidate: _Candidate,
    frame_size: tuple[int, int],
    pitch: Mapping[str, Any],
) -> dict[str, Any]:
    width, height = frame_size
    length = _number(pitch.get("length"))
    pitch_width = _number(pitch.get("width"))
    if width <= 0 or height <= 0 or length is None or pitch_width is None:
        raise ValueError("frame_size and pitch dimensions must be positive")
    x = (candidate.image_x / width - 0.5) * length * 0.96
    z = (candidate.image_y / height - 0.5) * pitch_width * 1.05
    return {
        "x": max(-length / 2.0, min(length / 2.0, x)),
        "z": max(-pitch_width / 2.0, min(pitch_width / 2.0, z)),
        "projectionSource": "screen-approximate",
        "calibrationFrameIndex": None,
        "positionUncertaintyMetres": 14.0,
    }


def _projection_result(
    candidate: _Candidate,
    frame_size: tuple[int, int],
    pitch: Mapping[str, Any],
    projector: PositionProjector | None,
) -> dict[str, Any]:
    if candidate.pitch_x is not None and candidate.pitch_z is not None:
        result: dict[str, Any] = {
            "x": candidate.pitch_x,
            "z": candidate.pitch_z,
            "projectionSource": str(candidate.source.get("projectionSource") or "direct"),
            "calibrationFrameIndex": candidate.source.get("calibrationFrameIndex"),
            "positionUncertaintyMetres": candidate.source.get(
                "positionUncertaintyMetres"
            ),
        }
    elif projector is None:
        result = _default_projection(candidate, frame_size, pitch)
    else:
        projected = projector(candidate.source)
        if isinstance(projected, Mapping):
            result = deepcopy(dict(projected))
        elif isinstance(projected, (tuple, list)) and len(projected) == 2:
            result = {"x": projected[0], "z": projected[1]}
        else:
            raise ValueError("projector must return (x, z) or a mapping")
    x, z = _number(result.get("x")), _number(result.get("z"))
    if x is None or z is None:
        raise ValueError("projector returned no finite x/z coordinates")
    result["x"], result["z"] = x, z
    source = str(result.get("projectionSource") or "projector")
    result["projectionSource"] = source
    result.setdefault("calibrationFrameIndex", None)
    result.setdefault("positionUncertaintyMetres", None)
    result["projection"] = {
        "source": source,
        "calibrationFrameIndex": result.get("calibrationFrameIndex"),
        "uncertaintyMetres": result.get("positionUncertaintyMetres"),
    }
    return result


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _observed_keyframe(
    step: _PathStep,
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
        # Keep the detector score as the legacy confidence; do not silently
        # replace measured evidence with a trajectory heuristic.
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
    step: _PathStep,
    left_step: _PathStep,
    right_step: _PathStep,
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
        if _number(value) is not None
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
    states: Sequence[BallState], frames: Sequence[tuple[float, list[_Candidate]]]
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
        max(0.0, frames[end][0] - frames[start][0]) for _, start, end in gaps
    ]
    return {
        "gapCount": len(gaps),
        "inferredGapCount": sum(state == "inferred" for state, _, _ in gaps),
        "occludedGapCount": sum(state == "occluded" for state, _, _ in gaps),
        "longestGapFrames": max((end - start + 1 for _, start, end in gaps), default=0),
        "longestGapSeconds": round(max(durations, default=0.0), 4),
    }


def _empty_diagnostics(
    frames: Sequence[tuple[float, list[_Candidate]]],
    invalid_candidates: int,
    dropped_by_top_k: int,
    peak_hypothesis_count: int,
    config: BallTrackingConfig,
) -> dict[str, Any]:
    candidate_count = sum(len(candidates) for _, candidates in frames)
    return {
        "algorithm": "beam-viterbi-ball-v1",
        "status": "no-stable-trajectory",
        "frameCount": len(frames),
        "candidateCount": candidate_count,
        "invalidCandidateCount": invalid_candidates,
        "droppedByTopKCount": dropped_by_top_k,
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
                "t": round(time, 3),
                "state": "occluded",
                "candidateCount": len(candidates),
                "reason": "insufficient-global-evidence",
            }
            for index, (time, candidates) in enumerate(frames)
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
            "longestGapSeconds": round(frames[-1][0] - frames[0][0], 4)
            if len(frames) > 1
            else 0.0,
        },
        "config": {
            "topKPerFrame": config.top_k_per_frame,
            "beamWidth": config.beam_width,
            "maxInterpolationGapSeconds": config.max_interpolation_gap_seconds,
        },
    }


def resolve_ball_trajectory(
    ball_frames: Sequence[Any],
    frame_size: tuple[int, int],
    pitch: Mapping[str, Any],
    *,
    config: BallTrackingConfig = DEFAULT_BALL_TRACKING_CONFIG,
    coordinate_selector: MotionCoordinateSelector | None = None,
    projector: PositionProjector | None = None,
) -> BallTrajectoryResolution:
    """Resolve top-K per-frame detections into one temporal ball trajectory.

    ``ball_frames`` accepts both the legacy ``[(detections, time), ...]`` shape
    and mappings such as ``{"t": 1.2, "candidates": [...]}``.  A candidate
    requires finite ``x``, ``y`` and ``confidence`` (``score`` is accepted).
    Optional ``pitchX/pitchZ`` coordinates enable metric physical constraints.
    Optional ``stabilizedX/stabilizedY`` coordinates are used for association
    across camera pans while the raw coordinates remain available to a custom
    frame-aware ``projector``.
    """

    frames, invalid_candidates, dropped_by_top_k = _normalise_frames(
        ball_frames, config, coordinate_selector
    )
    selected, runner_up_cost, peak_hypothesis_count = _resolve_path(frames, config)
    if selected is None:
        return BallTrajectoryResolution(
            keyframes=[],
            diagnostics=_empty_diagnostics(
                frames,
                invalid_candidates,
                dropped_by_top_k,
                peak_hypothesis_count,
                config,
            ),
        )

    observed_indices = [
        index for index, step in enumerate(selected.steps) if step.state == "observed"
    ]
    observed_projections: dict[int, dict[str, Any]] = {}
    observed_keyframes: dict[int, dict[str, Any]] = {}
    for index in observed_indices:
        step = selected.steps[index]
        assert step.candidate is not None
        projection = _projection_result(step.candidate, frame_size, pitch, projector)
        observed_projections[index] = projection
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
                "candidateCount": len(frames[index][1]),
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

        previous_observed = max(
            (candidate for candidate in observed_indices if candidate < index),
            default=None,
        )
        next_observed = min(
            (candidate for candidate in observed_indices if candidate > index),
            default=None,
        )
        if state == "inferred" and previous_observed is not None and next_observed is not None:
            keyframe = _interpolated_keyframe(
                step,
                selected.steps[previous_observed],
                selected.steps[next_observed],
                observed_keyframes[previous_observed],
                observed_keyframes[next_observed],
                config,
            )
            keyframes.append(keyframe)
            path.append(
                {
                    "frameIndex": index,
                    "t": round(step.time, 3),
                    "state": "inferred",
                    "candidateId": None,
                    "candidateCount": len(frames[index][1]),
                    "confidence": keyframe["confidence"],
                    "fromCandidateId": keyframe["sourceCandidateIds"][0],
                    "toCandidateId": keyframe["sourceCandidateIds"][1],
                    "emissionCost": round(step.emission_cost, 6),
                    "transitionCost": 0.0,
                }
            )
            continue

        if previous_observed is None:
            reason = "before-first-observation"
        elif next_observed is None:
            reason = "after-last-observation"
        else:
            reason = "gap-exceeds-interpolation-limit"
        path.append(
            {
                "frameIndex": index,
                "t": round(step.time, 3),
                "state": "occluded",
                "candidateId": None,
                "candidateCount": len(frames[index][1]),
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
        "candidateCount": sum(len(candidates) for _, candidates in frames),
        "invalidCandidateCount": invalid_candidates,
        "droppedByTopKCount": dropped_by_top_k,
        "peakHypothesisCount": peak_hypothesis_count,
        "observedFrameCount": observed_count,
        "inferredFrameCount": inferred_count,
        "occludedFrameCount": states.count("occluded"),
        "observedCoverage": round(observed_count / max(1, frame_count), 4),
        "publishedCoverage": round((observed_count + inferred_count) / max(1, frame_count), 4),
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
        "runnerUpCost": _round_or_none(runner_up_cost, 6),
        "pathCostMargin": _round_or_none(
            runner_up_cost - selected.cost if runner_up_cost is not None else None, 6
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


__all__ = [
    "BallState",
    "BallTrackingConfig",
    "BallTrajectoryResolution",
    "DEFAULT_BALL_TRACKING_CONFIG",
    "MotionCoordinateSelector",
    "PositionProjector",
    "resolve_ball_trajectory",
]
