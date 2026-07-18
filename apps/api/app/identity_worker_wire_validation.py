from __future__ import annotations

"""Fail-closed structural primitives for the identity-worker wire contract."""

from typing import Any, Mapping

from .identity_worker_contract import IdentityWorkerError


def reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise IdentityWorkerError(
            f"{label} has unsupported fields: {', '.join(unknown)}"
        )
