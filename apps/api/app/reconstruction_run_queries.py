from __future__ import annotations

"""SQL statement builders for the reconstruction run repository."""

from collections.abc import Collection

from sqlalchemy import or_, select

from .database import ReconstructionJobRow, ReconstructionLeaseRow, SceneRow
from .reconstruction_run_contract import ReconstructionRunFence


def locked_job_statement(scene_id: str):
    return (
        select(ReconstructionJobRow)
        .where(ReconstructionJobRow.scene_id == scene_id)
        .with_for_update()
    )


def locked_lease_statement(scene_id: str):
    return (
        select(ReconstructionLeaseRow)
        .where(ReconstructionLeaseRow.scene_id == scene_id)
        .with_for_update()
    )


def locked_scene_statement(scene_id: str):
    return select(SceneRow).where(SceneRow.id == scene_id).with_for_update()


def recoverable_runs_statement(
    statuses: Collection[str],
    *,
    current_time: float,
):
    return (
        select(
            ReconstructionJobRow.scene_id,
            ReconstructionJobRow.run_id,
            ReconstructionJobRow.input_fingerprint,
        )
        .outerjoin(
            ReconstructionLeaseRow,
            ReconstructionLeaseRow.scene_id == ReconstructionJobRow.scene_id,
        )
        .where(ReconstructionJobRow.status.in_(statuses))
        .where(
            or_(
                ReconstructionLeaseRow.scene_id.is_(None),
                ReconstructionLeaseRow.expires_at <= current_time,
            )
        )
        .order_by(
            ReconstructionJobRow.requested_at.asc(),
            ReconstructionJobRow.scene_id.asc(),
        )
    )


def current_run_status_statement(fence: ReconstructionRunFence):
    return select(ReconstructionJobRow.status).where(
        ReconstructionJobRow.scene_id == fence.scene_id,
        ReconstructionJobRow.run_id == fence.run_id,
        ReconstructionJobRow.input_fingerprint == fence.input_fingerprint,
    )
