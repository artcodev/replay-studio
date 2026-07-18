"""Small pure helpers shared by model-quality evaluators."""

from __future__ import annotations

from typing import Any, Sequence
import re

import numpy as np


def distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": round(float(array.min()), 6),
        "p05": round(float(np.percentile(array, 5)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "maximum": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
    }


def check(
    check_id: str,
    passed: bool,
    *,
    actual: Any,
    operator: str,
    threshold: Any,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
    }


def is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
