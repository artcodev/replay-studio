from __future__ import annotations

"""Transaction-local helpers for compact AnalysisRun telemetry mirrors.

These functions never open, commit, or roll back a transaction. Queue and
reconstruction repositories call them while holding their scheduler locks.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from .project_models import AnalysisRunRow


ACTIVE_ANALYSIS_STATUSES = frozenset({"queued", "running", "cancelling"})
TERMINAL_ANALYSIS_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
COMPACT_PROGRESS_KEYS = frozenset(
    {
        "phase",
        "phaseIndex",
        "phaseCount",
        "label",
        "detail",
        "completed",
        "total",
        "phasePercent",
        "overallPercent",
        "elapsedSeconds",
        "etaSeconds",
        "updatedAt",
    }
)


def compact_analysis_progress(
    progress: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(progress, dict):
        return {}
    return {key: progress[key] for key in COMPACT_PROGRESS_KEYS if key in progress}


def upsert_reconstruction_telemetry(
    session,
    *,
    project_id: str,
    scene_id: str,
    segment_id: str | None,
    run_id: str,
    input_fingerprint: str,
    status: str,
    now: float,
    model: str | None = None,
    progress: dict | None = None,
    error: str | None = None,
) -> AnalysisRunRow:
    """Create or repair reconstruction telemetry in the caller transaction."""

    timestamp = datetime.fromtimestamp(now, UTC)
    analysis = session.scalar(
        select(AnalysisRunRow)
        .where(AnalysisRunRow.id == run_id)
        .with_for_update()
    )
    if analysis is None:
        analysis = AnalysisRunRow(
            id=run_id,
            project_id=project_id,
            scene_id=scene_id,
            segment_id=segment_id,
            kind="reconstruction",
            status=status,
            source_run_id=run_id,
            input_fingerprint=input_fingerprint,
            model=model,
            progress=compact_analysis_progress(progress),
            diagnostics={},
            error=error,
            requested_at=timestamp,
        )
        session.add(analysis)
    else:
        analysis.project_id = project_id
        analysis.scene_id = scene_id
        analysis.segment_id = segment_id
        analysis.kind = "reconstruction"
        analysis.source_run_id = run_id
        analysis.input_fingerprint = input_fingerprint
        if model is not None:
            analysis.model = model
        analysis.status = status
        if progress is not None:
            analysis.progress = compact_analysis_progress(progress)
        analysis.error = error
    if status == "queued":
        analysis.started_at = None
        analysis.completed_at = None
    elif status == "running":
        analysis.started_at = analysis.started_at or timestamp
        analysis.completed_at = None
    elif status in TERMINAL_ANALYSIS_STATUSES:
        analysis.completed_at = timestamp
    analysis.updated_at = timestamp
    return analysis


def mark_analysis_running(session, run_id: str, now: float) -> None:
    """Mirror a claimed pipeline job inside its existing transaction."""

    run = session.get(AnalysisRunRow, str(run_id))
    if run is None:
        return
    timestamp = datetime.fromtimestamp(now, UTC)
    if run.status == "queued":
        run.status = "running"
        run.started_at = timestamp
    run.updated_at = timestamp


def cancel_active_analysis(session, run_id: str, now: float) -> None:
    """Terminalize superseded telemetry inside the queue transaction."""

    run = session.scalar(
        select(AnalysisRunRow)
        .where(AnalysisRunRow.id == str(run_id))
        .with_for_update()
    )
    if run is None or run.status not in ACTIVE_ANALYSIS_STATUSES:
        return
    timestamp = datetime.fromtimestamp(now, UTC)
    run.status = "cancelled"
    run.completed_at = timestamp
    run.updated_at = timestamp


def mark_analysis_terminal(
    session,
    run_id: str,
    status: str,
    now: float,
    *,
    error: str | None = None,
    progress: dict[str, Any] | None = None,
) -> None:
    """Mirror a terminal pipeline job inside its existing transaction."""

    run = session.get(AnalysisRunRow, str(run_id))
    if run is None or run.status == "cancelled":
        return
    timestamp = datetime.fromtimestamp(now, UTC)
    run.status = "succeeded" if status == "succeeded" else "failed"
    run.error = error if status == "failed" else None
    if progress is not None:
        run.progress = compact_analysis_progress(progress)
    run.completed_at = timestamp
    run.updated_at = timestamp
