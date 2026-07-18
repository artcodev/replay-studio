from __future__ import annotations

"""Progress presentation for reconstruction jobs."""

from copy import deepcopy
from datetime import UTC, datetime
from time import monotonic
from typing import Callable

RECONSTRUCTION_PHASES = [
    ("preparing", "Prepare inputs"),
    ("calibration", "Calibrate pitch"),
    ("detection", "Detect objects"),
    ("tracking", "Build tracks"),
    ("projection", "Reconstruct 3D"),
    ("finalizing", "Save result"),
]

def phase_rows(current_index: int, complete: bool = False) -> list[dict]:
    return [
        {
            "id": phase_id,
            "label": label,
            "status": (
                "completed"
                if complete or index < current_index
                else "current"
                if index == current_index
                else "pending"
            ),
        }
        for index, (phase_id, label) in enumerate(RECONSTRUCTION_PHASES, start=1)
    ]


def queued_progress(frame_count: int) -> dict:
    return {
        "phase": "preparing",
        "phaseIndex": 1,
        "phaseCount": len(RECONSTRUCTION_PHASES),
        "label": "Waiting to start",
        "detail": f"Queued {frame_count} sampled frames for analysis.",
        "completed": 0,
        "total": frame_count,
        "phasePercent": 0,
        "overallPercent": 0,
        "elapsedSeconds": 0.0,
        "etaSeconds": None,
        "updatedAt": datetime.now(UTC).isoformat(),
        "phases": phase_rows(1),
    }


class ReconstructionProgress:
    def __init__(
        self,
        scene: dict,
        listener: Callable[[dict], None] | None = None,
    ) -> None:
        self.scene = scene
        self.listener = listener
        self.started = monotonic()
        self.phase_started = self.started
        self.phase = ""

    def update(
        self,
        phase: str,
        phase_index: int,
        label: str,
        detail: str,
        overall_start: float,
        overall_end: float,
        completed: int = 0,
        total: int = 0,
        fraction: float | None = None,
        eta_padding: float = 0.0,
    ) -> dict:
        now = monotonic()
        if phase != self.phase:
            self.phase = phase
            self.phase_started = now
        if fraction is None:
            fraction = completed / total if total > 0 else 0.0
        fraction = max(0.0, min(1.0, float(fraction)))
        phase_elapsed = max(0.0, now - self.phase_started)
        eta = None
        if fraction > 0.0 and fraction < 1.0:
            eta = phase_elapsed * (1.0 - fraction) / fraction + eta_padding
        elif fraction >= 1.0:
            eta = eta_padding
        payload = {
            "phase": phase,
            "phaseIndex": phase_index,
            "phaseCount": len(RECONSTRUCTION_PHASES),
            "label": label,
            "detail": detail,
            "completed": int(completed),
            "total": int(total),
            "phasePercent": round(fraction * 100),
            "overallPercent": round(overall_start + (overall_end - overall_start) * fraction),
            "elapsedSeconds": round(max(0.0, now - self.started), 1),
            "etaSeconds": round(eta, 1) if eta is not None else None,
            "updatedAt": datetime.now(UTC).isoformat(),
            "phases": phase_rows(phase_index),
        }
        video = self.scene["payload"]["videoAsset"]
        reconstruction = video.get("reconstruction") or {}
        reconstruction["status"] = "processing"
        reconstruction["processingStatus"] = "processing"
        reconstruction["progress"] = payload
        video["reconstruction"] = reconstruction
        if self.listener is not None:
            self.listener(deepcopy(payload))
        return payload

    def complete(self, track_count: int, ball_samples: int) -> dict:
        now = monotonic()
        payload = {
            "phase": "complete",
            "phaseIndex": len(RECONSTRUCTION_PHASES),
            "phaseCount": len(RECONSTRUCTION_PHASES),
            "label": "Analysis complete",
            "detail": f"Saved {track_count} tracks and {ball_samples} ball samples.",
            "completed": 1,
            "total": 1,
            "phasePercent": 100,
            "overallPercent": 100,
            "elapsedSeconds": round(max(0.0, now - self.started), 1),
            "etaSeconds": 0.0,
            "updatedAt": datetime.now(UTC).isoformat(),
            "phases": phase_rows(len(RECONSTRUCTION_PHASES), complete=True),
        }
        if self.listener is not None:
            self.listener(deepcopy(payload))
        return payload

    def failed(self, message: str) -> dict:
        current = (
            self.scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
            .get("progress")
            or queued_progress(0)
        )
        return {
            **current,
            "phase": "failed",
            "label": "Analysis failed",
            "detail": message,
            "etaSeconds": 0.0,
            "updatedAt": datetime.now(UTC).isoformat(),
        }


