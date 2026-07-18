"""Numeric parsing, geometry, and summary primitives shared by evaluators."""

from __future__ import annotations

from math import isfinite, sqrt
from statistics import mean, median
from typing import Any, Iterable, Sequence


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) else None


def point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x = finite_number(value[0])
    y = finite_number(value[1])
    return (x, y) if x is not None and y is not None else None


def bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        values = (value.get("x"), value.get("y"), value.get("width"), value.get("height"))
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        values = value
    else:
        return None
    parsed = tuple(finite_number(item) for item in values)
    if any(item is None for item in parsed):
        return None
    x, y, width, height = parsed
    if width <= 0 or height <= 0:
        return None
    return float(x), float(y), float(width), float(height)


def intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    intersection_width = max(
        0.0,
        min(left_x + left_width, right_x + right_width) - max(left_x, right_x),
    )
    intersection_height = max(
        0.0,
        min(left_y + left_height, right_y + right_height) - max(left_y, right_y),
    )
    intersection = intersection_width * intersection_height
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union > 0 else 0.0


def percentile(values: Sequence[float], quantile: float) -> float | None:
    ordered = sorted(item for item in values if isfinite(item))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = min(1.0, max(0.0, quantile)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def distribution(values: Iterable[float], unit: str) -> dict[str, Any]:
    finite_values = [float(value) for value in values if isfinite(value)]
    if not finite_values:
        return {
            "available": False,
            "sampleCount": 0,
            "unit": unit,
            "mean": None,
            "median": None,
            "p95": None,
            "rmse": None,
            "maximum": None,
        }
    return {
        "available": True,
        "sampleCount": len(finite_values),
        "unit": unit,
        "mean": rounded(mean(finite_values)),
        "median": rounded(median(finite_values)),
        "p95": rounded(percentile(finite_values, 0.95)),
        "rmse": rounded(sqrt(mean(value * value for value in finite_values))),
        "maximum": rounded(max(finite_values)),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return rounded(numerator / denominator) if denominator else None


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denominator = precision + recall
    return rounded(2.0 * precision * recall / denominator) if denominator else 0.0
