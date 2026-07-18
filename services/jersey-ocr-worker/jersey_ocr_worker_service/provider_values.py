from __future__ import annotations

import importlib.metadata
from typing import Any

import numpy as np


def package_version(name: str, fallback: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return fallback


def confidence(value: Any) -> float:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    array = array[np.isfinite(array)]
    if not array.size:
        return 0.0
    return float(np.clip(array.mean(), 0.0, 1.0))


def polygon(value: Any) -> list[list[float]] | None:
    try:
        points = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if points.shape[0] < 2 or not np.isfinite(points).all():
        return None
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]
