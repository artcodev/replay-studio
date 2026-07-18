"""Reconcile historical reconstruction telemetry with the compact scheduler.

Revision ID: 20260718_0005
Revises: 20260718_0004
"""

from __future__ import annotations

from datetime import UTC, datetime

from alembic import op
import sqlalchemy as sa


revision = "20260718_0005"
down_revision = "20260718_0004"
branch_labels = None
depends_on = None


_ACTIVE_RUN_STATUSES = {"queued", "running", "cancelling"}
_ACTIVE_JOB_STATUSES = {"queued", "processing"}
_TERMINAL_JOB_STATUS = {
    "ready": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "superseded": "cancelled",
    "invalid": "cancelled",
}


def upgrade() -> None:
    """Retire telemetry rows that no authoritative job can execute.

    ``analysis_runs`` is history/UI telemetry, never a queue.  Before the
    scheduler cutover it could retain ``running`` rows after a scene had
    already completed a different run.  This one-time migration aligns those
    rows with ``reconstruction_jobs`` without reading dense scene payloads.
    Runtime polling intentionally contains no equivalent backfill.
    """

    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if not {"analysis_runs", "reconstruction_jobs"}.issubset(tables):
        return

    analysis_runs = sa.table(
        "analysis_runs",
        sa.column("id", sa.String(160)),
        sa.column("scene_id", sa.String(120)),
        sa.column("kind", sa.String(60)),
        sa.column("status", sa.String(40)),
        sa.column("source_run_id", sa.String(160)),
        sa.column("progress", sa.JSON()),
        sa.column("error", sa.Text()),
        sa.column("completed_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    jobs = sa.table(
        "reconstruction_jobs",
        sa.column("scene_id", sa.String(120)),
        sa.column("run_id", sa.String(120)),
        sa.column("status", sa.String(40)),
    )
    connection = op.get_bind()
    current_jobs = {
        str(row.scene_id): row
        for row in connection.execute(
            sa.select(jobs.c.scene_id, jobs.c.run_id, jobs.c.status)
        )
    }
    active_runs = connection.execute(
        sa.select(
            analysis_runs.c.id,
            analysis_runs.c.scene_id,
            analysis_runs.c.source_run_id,
            analysis_runs.c.status,
            analysis_runs.c.progress,
            analysis_runs.c.error,
        ).where(
            analysis_runs.c.kind == "reconstruction",
            analysis_runs.c.status.in_(sorted(_ACTIVE_RUN_STATUSES)),
        )
    ).mappings()

    completed_at = datetime.now(UTC)
    for run in active_runs:
        scene_id = str(run["scene_id"] or "")
        source_run_id = str(run["source_run_id"] or "")
        job = current_jobs.get(scene_id)
        if (
            job is not None
            and str(job.run_id) == source_run_id
            and str(job.status) in _ACTIVE_JOB_STATUSES
        ):
            continue

        if job is not None and str(job.run_id) == source_run_id:
            terminal_status = _TERMINAL_JOB_STATUS.get(
                str(job.status), "cancelled"
            )
            detail = (
                "Telemetry reconciled with the terminal authoritative "
                f"reconstruction job ({job.status})."
            )
        elif job is not None:
            terminal_status = "cancelled"
            detail = (
                "Historical reconstruction run was superseded by the current "
                f"authoritative run {job.run_id}."
            )
        else:
            terminal_status = "cancelled"
            detail = (
                "Historical reconstruction run has no authoritative scheduler "
                "record after the queue cutover."
            )

        progress = dict(run["progress"]) if isinstance(run["progress"], dict) else {}
        progress.update(
            {
                "phase": "scheduler-cutover",
                "label": "Historical run reconciled",
                "detail": detail,
                "etaSeconds": 0.0,
                "updatedAt": completed_at.isoformat(),
            }
        )
        if terminal_status == "succeeded":
            progress.update({"overallPercent": 100, "phasePercent": 100})

        connection.execute(
            analysis_runs.update()
            .where(analysis_runs.c.id == str(run["id"]))
            .values(
                status=terminal_status,
                progress=progress,
                error=(
                    run["error"]
                    if terminal_status == "failed"
                    else None
                ),
                completed_at=completed_at,
                updated_at=completed_at,
            )
        )


def downgrade() -> None:
    # Historical ownership cannot be reconstructed safely.  Reopening these
    # rows would fabricate runnable work, so the data reconciliation is
    # deliberately irreversible while the schema itself remains compatible.
    pass
