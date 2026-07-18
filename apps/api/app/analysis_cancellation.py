from __future__ import annotations

"""Cross-table cancellation application service for AnalysisRun jobs."""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import select

from .analysis_run_repository import (
    AnalysisRunConflict,
    AnalysisRunNotFound,
    analysis_run_document,
)
from .analysis_run_telemetry import (
    ACTIVE_ANALYSIS_STATUSES,
    TERMINAL_ANALYSIS_STATUSES,
    compact_analysis_progress,
)
from .database import (
    ReconstructionJobRow,
    ReconstructionLeaseRow,
    SceneRow,
    SessionLocal,
)
from .database_transaction import begin_write_transaction
from .project_models import AnalysisRunRow
from .analysis_run_contract import AnalysisRunDocument


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AnalysisCancellationService:
    """Atomically fence physical work and terminalize its telemetry row."""

    def __init__(self, session_factory: Callable | None = None) -> None:
        self._session_factory = session_factory

    def _session(self):
        return (self._session_factory or SessionLocal)()

    def cancel(self, run_id: str) -> AnalysisRunDocument:
        with self._session() as session:
            begin_write_transaction(session)
            seed = session.execute(
                select(
                    AnalysisRunRow.kind,
                    AnalysisRunRow.status,
                    AnalysisRunRow.scene_id,
                ).where(AnalysisRunRow.id == run_id)
            ).one_or_none()
            if seed is None:
                raise AnalysisRunNotFound(
                    f"Analysis run {run_id} was not found"
                )

            seed_kind, seed_status, seed_scene_id = seed
            job = None
            lease = None
            scene_row = None
            if (
                seed_kind == "reconstruction"
                and seed_status in ACTIVE_ANALYSIS_STATUSES
                and seed_scene_id is not None
            ):
                job = session.scalar(
                    select(ReconstructionJobRow)
                    .where(ReconstructionJobRow.scene_id == seed_scene_id)
                    .with_for_update()
                )
                if job is not None and job.run_id == run_id:
                    lease = session.scalar(
                        select(ReconstructionLeaseRow)
                        .where(ReconstructionLeaseRow.scene_id == seed_scene_id)
                        .with_for_update()
                    )
                    if job.status in {"queued", "processing"}:
                        scene_row = session.scalar(
                            select(SceneRow)
                            .where(SceneRow.id == seed_scene_id)
                            .with_for_update()
                        )

            row = session.scalar(
                select(AnalysisRunRow)
                .where(AnalysisRunRow.id == run_id)
                .with_for_update()
            )
            if row is None:
                raise AnalysisRunNotFound(
                    f"Analysis run {run_id} was not found"
                )
            cancelled_at = _utcnow()

            if row.status in TERMINAL_ANALYSIS_STATUSES:
                session.commit()
                session.refresh(row)
                return analysis_run_document(row)

            compact_terminal_status = None
            if (
                row.kind == "reconstruction"
                and row.scene_id == seed_scene_id
                and job is not None
                and job.run_id == run_id
            ):
                compact_terminal_status = {
                    "ready": "succeeded",
                    "failed": "failed",
                    "cancelled": "cancelled",
                }.get(job.status)
            if compact_terminal_status is not None:
                row.status = compact_terminal_status
                if compact_terminal_status != "failed":
                    row.error = None
                row.completed_at = cancelled_at
                row.updated_at = cancelled_at
                session.commit()
                session.refresh(row)
                return analysis_run_document(row)

            current_reconstruction = None
            exact_physical_run = (
                row.kind == "reconstruction"
                and row.scene_id == seed_scene_id
                and job is not None
                and job.run_id == run_id
                and job.status in {"queued", "processing"}
            )
            if exact_physical_run and scene_row is not None:
                current_scene = scene_row.payload
                current_video = current_scene.get("payload", {}).get(
                    "videoAsset", {}
                )
                current_reconstruction = current_video.get("reconstruction") or {}
                if str(current_reconstruction.get("runId") or "") != run_id:
                    current_reconstruction = None

            if exact_physical_run:
                if lease is not None and lease.run_id == run_id:
                    session.delete(lease)
                if current_reconstruction is not None:
                    accepted_terminal_status = {
                        "ready": "succeeded",
                        "failed": "failed",
                        "cancelled": "cancelled",
                    }.get(str(current_reconstruction.get("status") or ""))
                    if accepted_terminal_status is not None:
                        job.status = str(current_reconstruction.get("status"))
                        job.updated_at = cancelled_at.timestamp()
                        row.status = accepted_terminal_status
                        row.completed_at = cancelled_at
                        scene_progress = compact_analysis_progress(
                            current_reconstruction.get("progress")
                        )
                        if scene_progress:
                            row.progress = scene_progress
                        row.error = (
                            current_reconstruction.get("error")
                            if accepted_terminal_status == "failed"
                            else None
                        )
                        row.updated_at = cancelled_at
                        session.commit()
                        session.refresh(row)
                        return analysis_run_document(row)
                    if current_reconstruction.get("status") in {
                        "queued",
                        "processing",
                    }:
                        cancelled_scene = deepcopy(scene_row.payload)
                        cancelled_video = cancelled_scene["payload"]["videoAsset"]
                        cancelled_reconstruction = cancelled_video["reconstruction"]
                        cancelled_reconstruction.pop("lease", None)
                        cancelled_reconstruction.update(
                            {
                                "status": "cancelled",
                                "processingStatus": "cancelled",
                                "error": None,
                                "completedAt": cancelled_at.isoformat(),
                                "progress": {
                                    **dict(
                                        cancelled_reconstruction.get("progress")
                                        or {}
                                    ),
                                    "phase": "cancelled",
                                    "label": "Analysis cancelled",
                                    "detail": (
                                        "The active worker was fenced; the "
                                        "previous result was preserved."
                                    ),
                                    "etaSeconds": 0.0,
                                    "updatedAt": cancelled_at.isoformat(),
                                },
                            }
                        )
                        cancelled_video["processingState"] = "frames-ready"
                        try:
                            revision = max(
                                0,
                                int(scene_row.payload.get("revision") or 0),
                            )
                        except (TypeError, ValueError):
                            revision = 0
                        cancelled_scene["revision"] = revision + 1
                        scene_row.payload = cancelled_scene
                job.status = "cancelled"
                job.updated_at = cancelled_at.timestamp()

            if (
                row.kind == "reconstruction"
                and row.status in ACTIVE_ANALYSIS_STATUSES
            ):
                row.status = "cancelled"
                row.completed_at = cancelled_at
                row.progress = {
                    **dict(row.progress or {}),
                    "phase": "cancelled",
                    "label": "Analysis cancelled",
                    "etaSeconds": 0.0,
                    "updatedAt": cancelled_at.isoformat(),
                }
                row.updated_at = cancelled_at
            elif row.status == "queued":
                row.status = "cancelled"
                row.completed_at = cancelled_at
                row.progress = {
                    **dict(row.progress or {}),
                    "phase": "cancelled",
                    "label": "Analysis cancelled",
                    "etaSeconds": 0.0,
                    "updatedAt": cancelled_at.isoformat(),
                }
                row.updated_at = cancelled_at
            elif row.status == "running":
                row.status = "cancelling"
                row.progress = {
                    **dict(row.progress or {}),
                    "phase": "cancelling",
                    "label": "Cancelling analysis",
                    "updatedAt": cancelled_at.isoformat(),
                }
                row.updated_at = cancelled_at
            elif row.status == "cancelling":
                pass
            else:
                raise AnalysisRunConflict(
                    f"Analysis run {run_id} has unsupported status {row.status}"
                )
            session.commit()
            session.refresh(row)
            return analysis_run_document(row)


analysis_cancellation = AnalysisCancellationService()
