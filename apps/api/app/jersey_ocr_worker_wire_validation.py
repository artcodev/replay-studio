from __future__ import annotations

"""Fail-closed structural primitives for the jersey OCR wire contract."""

from typing import Any, Mapping

from .jersey_ocr_worker_contract import JerseyOcrWorkerError


def reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise JerseyOcrWorkerError(
            f"{label} has unsupported fields: {', '.join(unknown)}"
        )
