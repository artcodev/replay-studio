from __future__ import annotations

"""Progress read-model for multi-angle analysis."""

from datetime import UTC, datetime
from time import monotonic


def multi_pass_phases(
    segments: list[dict],
    current_index: int,
    *,
    complete: bool = False,
) -> list[dict]:
    labels = [
        (f"angle-{index}", f"Angle {index} · {segment.get('label') or segment['id']}")
        for index, segment in enumerate(segments, start=1)
    ]
    labels.extend([("alignment", "Align camera angles"), ("consensus", "Fuse evidence")])
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
        for index, (phase_id, label) in enumerate(labels, start=1)
    ]


def set_multi_pass_progress(
    scene: dict,
    segments: list[dict],
    started: float,
    phase: str,
    phase_index: int,
    label: str,
    detail: str,
    overall_percent: float,
    phase_percent: float,
    completed: int = 0,
    total: int = 0,
    eta_seconds: float | None = None,
    complete: bool = False,
) -> dict:
    payload = {
        "phase": phase,
        "phaseIndex": phase_index,
        "phaseCount": len(segments) + 2,
        "label": label,
        "detail": detail,
        "completed": completed,
        "total": total,
        "phasePercent": round(phase_percent),
        "overallPercent": round(overall_percent),
        "elapsedSeconds": round(max(0.0, monotonic() - started), 1),
        "etaSeconds": round(eta_seconds, 1) if eta_seconds is not None else None,
        "updatedAt": datetime.now(UTC).isoformat(),
        "phases": multi_pass_phases(segments, phase_index, complete=complete),
    }
    video = scene["payload"]["videoAsset"]
    reconstruction = video.get("reconstruction") or {}
    reconstruction["progress"] = payload
    video["reconstruction"] = reconstruction
    return payload


def dependency_progress(segments: list[dict], statuses: dict[str, str]) -> dict:
    terminal = {"ready", "failed", "cancelled", "superseded", "invalid"}
    completed = sum(
        statuses.get(str(segment["sceneId"])) in terminal for segment in segments
    )
    current_index = min(completed + 1, len(segments))
    percent = 5.0 + 85.0 * completed / max(1, len(segments))
    return {
        "phase": f"angle-{current_index}",
        "phaseIndex": current_index,
        "phaseCount": len(segments) + 2,
        "label": f"Analyzing camera angles · {completed}/{len(segments)} complete",
        "detail": "Waiting for durable child reconstruction jobs.",
        "completed": completed,
        "total": len(segments),
        "phasePercent": round(100 * completed / max(1, len(segments))),
        "overallPercent": round(percent),
        "elapsedSeconds": 0.0,
        "etaSeconds": None,
        "updatedAt": datetime.now(UTC).isoformat(),
        "phases": multi_pass_phases(segments, current_index),
    }
