from __future__ import annotations

"""Pure reconstruction scheduler contracts and fencing predicates."""

from dataclasses import dataclass
from typing import Any, Collection

from .scene_document import reconstruction_input_fingerprint


TERMINAL_TELEMETRY_STATUS = {
    "ready": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


@dataclass(frozen=True, slots=True)
class ReconstructionRunFence:
    scene_id: str
    run_id: str
    input_fingerprint: str


@dataclass(frozen=True, slots=True)
class QueuedReconstructionRun:
    fence: ReconstructionRunFence
    input_revision: int
    model: str | None
    progress: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class TerminalReconstructionRun:
    fence: ReconstructionRunFence
    scene_status: str
    telemetry_status: str
    model: str | None
    progress: dict[str, Any] | None
    error: str | None


def reconstruction_state(scene: dict[str, Any]) -> dict[str, Any]:
    value = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction", {})
    )
    return value if isinstance(value, dict) else {}


def queued_run_from_scene(scene: dict[str, Any]) -> QueuedReconstructionRun:
    reconstruction = reconstruction_state(scene)
    fence = ReconstructionRunFence(
        scene_id=str(scene.get("id") or ""),
        run_id=str(reconstruction.get("runId") or ""),
        input_fingerprint=str(reconstruction.get("inputFingerprint") or ""),
    )
    if not all((fence.scene_id, fence.run_id, fence.input_fingerprint)):
        raise ValueError(
            "A reconstruction queue command requires scene, run and input tokens"
        )
    if reconstruction.get("status") != "queued":
        raise ValueError("A reconstruction queue command requires queued state")
    if fence.input_fingerprint != reconstruction_input_fingerprint(scene):
        raise ValueError(
            "The queued reconstruction input fingerprint does not match its scene"
        )
    try:
        input_revision = max(1, int(reconstruction.get("runRevision") or 1))
    except (TypeError, ValueError):
        input_revision = 1
    return QueuedReconstructionRun(
        fence=fence,
        input_revision=input_revision,
        model=(
            str(reconstruction.get("model"))
            if reconstruction.get("model") is not None
            else None
        ),
        progress=(
            reconstruction.get("progress")
            if isinstance(reconstruction.get("progress"), dict)
            else None
        ),
    )


def terminal_run_from_scene(
    scene: dict[str, Any],
    expected_fence: ReconstructionRunFence,
) -> TerminalReconstructionRun | None:
    reconstruction = reconstruction_state(scene)
    scene_status = str(reconstruction.get("status") or "")
    telemetry_status = TERMINAL_TELEMETRY_STATUS.get(scene_status)
    if (
        telemetry_status is None
        or str(scene.get("id") or "") != expected_fence.scene_id
        or str(reconstruction.get("runId") or "") != expected_fence.run_id
        or str(reconstruction.get("inputFingerprint") or "")
        != expected_fence.input_fingerprint
        or reconstruction_input_fingerprint(scene)
        != expected_fence.input_fingerprint
    ):
        return None
    return TerminalReconstructionRun(
        fence=expected_fence,
        scene_status=scene_status,
        telemetry_status=telemetry_status,
        model=(
            str(reconstruction.get("model"))
            if reconstruction.get("model") is not None
            else None
        ),
        progress=(
            reconstruction.get("progress")
            if isinstance(reconstruction.get("progress"), dict)
            else None
        ),
        error=(
            str(reconstruction.get("error"))
            if telemetry_status == "failed"
            and reconstruction.get("error") is not None
            else None
        ),
    )


def job_matches_fence(
    job: Any,
    fence: ReconstructionRunFence,
    *,
    statuses: Collection[str],
) -> bool:
    return bool(
        job is not None
        and job.status in statuses
        and job.run_id == fence.run_id
        and job.input_fingerprint == fence.input_fingerprint
    )


def lease_matches_fence(
    lease: Any,
    fence: ReconstructionRunFence,
    *,
    owner_id: str,
    current_time: float,
) -> bool:
    return bool(
        lease is not None
        and lease.run_id == fence.run_id
        and lease.input_fingerprint == fence.input_fingerprint
        and lease.owner_id == owner_id
        and float(lease.expires_at) > current_time
    )


def scene_matches_fence(
    scene: dict[str, Any],
    fence: ReconstructionRunFence,
    *,
    statuses: Collection[str],
) -> bool:
    reconstruction = reconstruction_state(scene)
    return bool(
        reconstruction.get("status") in statuses
        and str(reconstruction.get("runId") or "") == fence.run_id
        and str(reconstruction.get("inputFingerprint") or "")
        == fence.input_fingerprint
        and reconstruction_input_fingerprint(scene) == fence.input_fingerprint
    )
