from __future__ import annotations

"""Shared compact contracts for durable pipeline persistence.

This module contains no repository or service.  It only defines the scheduler
document and the small transaction helpers used by the job repository and the
specialized atomic publication services.
"""

from dataclasses import dataclass
import json
from typing import Any

from sqlalchemy import select, text

from .database import PipelineJobLeaseRow, PipelineJobRow


PIPELINE_ACTIVE_STATUSES = frozenset({"queued", "waiting", "processing"})


class PipelineJobConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineJob:
    id: str
    project_id: str
    kind: str
    subject_id: str
    status: str
    state: dict[str, Any]
    parameters: dict[str, Any]
    available_at: float
    attempts: int
    error: str | None
    requested_at: float
    updated_at: float
    lease_token: str | None = None
    lease_owner_id: str | None = None
    lease_expires_at: float | None = None


def pipeline_job_document(
    row: PipelineJobRow,
    lease: PipelineJobLeaseRow | None = None,
) -> PipelineJob:
    return PipelineJob(
        id=str(row.id),
        project_id=str(row.project_id),
        kind=str(row.kind),
        subject_id=str(row.subject_id),
        status=str(row.status),
        state=dict(row.state or {}),
        parameters=dict(row.parameters or {}),
        available_at=float(row.available_at),
        attempts=int(row.attempts or 0),
        error=row.error,
        requested_at=float(row.requested_at),
        updated_at=float(row.updated_at),
        lease_token=str(lease.token) if lease is not None else None,
        lease_owner_id=str(lease.owner_id) if lease is not None else None,
        lease_expires_at=float(lease.expires_at) if lease is not None else None,
    )


def begin_pipeline_write(session) -> None:
    """Reserve the write transaction before reading any mutable job state."""

    if session.get_bind().dialect.name == "sqlite":
        session.execute(text("BEGIN IMMEDIATE"))
    else:
        session.begin()


def validate_compact(
    value: dict[str, Any],
    *,
    label: str,
    limit: int,
) -> dict[str, Any]:
    result = dict(value or {})
    encoded = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    if len(encoded) > limit:
        raise ValueError(f"{label} exceeds the compact scheduler limit")
    return result


def lock_owned_pipeline_job(session, job_id: str, token: str):
    """Lock Job -> Lease and return them only for the current claim token."""

    row = session.scalar(
        select(PipelineJobRow)
        .where(PipelineJobRow.id == str(job_id))
        .with_for_update()
    )
    lease = session.scalar(
        select(PipelineJobLeaseRow)
        .where(PipelineJobLeaseRow.job_id == str(job_id))
        .with_for_update()
    )
    if (
        row is None
        or lease is None
        or row.status != "processing"
        or lease.token != str(token)
    ):
        return None, None
    return row, lease
