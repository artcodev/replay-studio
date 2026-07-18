"""Clock-interval algebra used by closed-set roster identity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class TimeInterval:
    """An interval on the caller's normalized match clock, in seconds.

    Positive-duration intervals are half-open ``[start, end)``. A
    zero-duration interval is a point observation. Converting broadcast time
    to match time is deliberately outside this capability.
    """

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        start, end = float(self.start_seconds), float(self.end_seconds)
        if not isfinite(start) or not isfinite(end) or end < start:
            raise ValueError(
                "TimeInterval must be finite and end_seconds >= start_seconds"
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_payload(self) -> dict:
        return {"startTime": self.start_seconds, "endTime": self.end_seconds}


def merge_intervals(values: Iterable[TimeInterval]) -> tuple[TimeInterval, ...]:
    rows = sorted(tuple(values))
    if not rows:
        return ()
    merged: list[TimeInterval] = [rows[0]]
    for row in rows[1:]:
        previous = merged[-1]
        if previous.duration > 0.0 and row.duration > 0.0 and (
            row.start_seconds <= previous.end_seconds
        ):
            merged[-1] = TimeInterval(
                previous.start_seconds,
                max(previous.end_seconds, row.end_seconds),
            )
        elif previous.duration > 0.0 and row.duration == 0.0 and (
            previous.start_seconds <= row.start_seconds < previous.end_seconds
        ):
            continue
        elif previous.duration == 0.0 and row.duration > 0.0 and (
            row.start_seconds <= previous.start_seconds < row.end_seconds
        ):
            merged[-1] = row
        elif (
            previous.duration == 0.0
            and row.duration == 0.0
            and row.start_seconds == previous.start_seconds
        ):
            continue
        else:
            merged.append(row)
    return tuple(merged)


def intervals_overlap(
    left: Sequence[TimeInterval], right: Sequence[TimeInterval]
) -> bool:
    for first in left:
        for second in right:
            if first.duration == 0.0 and second.duration == 0.0:
                if first.start_seconds == second.start_seconds:
                    return True
            elif first.duration == 0.0:
                if second.start_seconds <= first.start_seconds < second.end_seconds:
                    return True
            elif second.duration == 0.0:
                if first.start_seconds <= second.start_seconds < first.end_seconds:
                    return True
            elif max(first.start_seconds, second.start_seconds) < min(
                first.end_seconds, second.end_seconds
            ):
                return True
    return False


def expanded_intervals(
    values: Sequence[TimeInterval], tolerance_seconds: float
) -> tuple[TimeInterval, ...]:
    tolerance = max(0.0, float(tolerance_seconds))
    if tolerance == 0.0:
        return tuple(values)
    return merge_intervals(
        TimeInterval(
            max(0.0, item.start_seconds - tolerance),
            item.end_seconds + tolerance,
        )
        for item in values
    )


def point_inside_intervals(
    timestamp: float,
    intervals: Sequence[TimeInterval],
    tolerance_seconds: float,
) -> bool:
    point = TimeInterval(float(timestamp), float(timestamp))
    return intervals_overlap((point,), expanded_intervals(intervals, tolerance_seconds))


def interval_coverage(
    observed: Sequence[TimeInterval], active: Sequence[TimeInterval]
) -> float:
    if not observed or not active:
        return 0.0
    coverage_parts: list[float] = []
    positive_duration = sum(item.duration for item in observed if item.duration > 0.0)
    if positive_duration > 0.0:
        overlap = 0.0
        for first in observed:
            if first.duration <= 0.0:
                continue
            for second in active:
                overlap += max(
                    0.0,
                    min(first.end_seconds, second.end_seconds)
                    - max(first.start_seconds, second.start_seconds),
                )
        coverage_parts.append(min(1.0, overlap / positive_duration))
    points = tuple(item for item in observed if item.duration == 0.0)
    if points:
        contained = sum(
            any(intervals_overlap((point,), (candidate,)) for candidate in active)
            for point in points
        )
        coverage_parts.append(contained / len(points))
    return min(coverage_parts) if coverage_parts else 0.0

