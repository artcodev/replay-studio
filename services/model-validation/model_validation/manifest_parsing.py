"""Strict JSON primitives and filesystem safety for validation manifests."""

from __future__ import annotations

import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Mapping

from .manifest_contract import ManifestError


def load_json_object(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ManifestError(f"Manifest does not exist: {source_path}")
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("Manifest root must be an object")
    return source_path, raw


def required_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    label: str,
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ManifestError(f"{label} has unsupported fields: {', '.join(unexpected)}")


def required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value.strip()


def bounded_number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result) or not minimum <= result <= maximum:
        raise ManifestError(f"{label} must be between {minimum} and {maximum}")
    return result


def safe_relative_file(base: Path, relative_path: str, label: str) -> Path:
    raw = Path(relative_path)
    if raw.is_absolute():
        raise ManifestError(f"{label} must be relative to the manifest")
    resolved = (base / raw).resolve()
    try:
        common = os.path.commonpath((str(base), str(resolved)))
    except ValueError as exc:
        raise ManifestError(f"{label} is outside the manifest directory") from exc
    if common != str(base):
        raise ManifestError(f"{label} is outside the manifest directory")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ManifestError(f"{label} does not reference a non-empty file")
    return resolved

