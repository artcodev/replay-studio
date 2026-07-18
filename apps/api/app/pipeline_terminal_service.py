from __future__ import annotations

"""Atomic application commands for terminal generic-pipeline transitions.

The scheduler repository deliberately has no knowledge of VideoAsset or Scene
read models.  Failure and cancellation are application commands because they
must fence the compact job and publish its resource-specific terminal state in
the same database transaction.
"""

from copy import deepcopy
from datetime import UTC, datetime
import time
from typing import Callable

from sqlalchemy import select

from .analysis_run_telemetry import mark_analysis_terminal
from .database import (
    PipelineJobLeaseRow,
    PipelineJobRow,
    SceneRow,
    SessionLocal,
    VideoAssetRow,
)
from .pipeline_domain import (
    PipelineJob,
    begin_pipeline_write,
    lock_owned_pipeline_job,
    pipeline_job_document,
)
from .project_models import AnalysisRunRow
from . import project_models as _project_models  # noqa: F401 - register FKs
from .scene_document import next_scene_payload, scene_revision
from .scene_index_projection import sync_scene_index


class PipelineTerminalService:
    """Coordinate fenced scheduler and owned-resource terminal publication."""

    def __init__(
        self,
        session_factory: Callable | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock

    def _session(self):
        return (self._session_factory or SessionLocal)()

    @staticmethod
    def _cancel_video_asset(session, job: PipelineJobRow) -> None:
        asset = session.get(VideoAssetRow, str(job.subject_id))
        if asset is None:
            return
        asset.status = "cancelled"
        asset.stage = "Processing cancelled"
        asset.error = None
        asset.progress = 100

    @staticmethod
    def _fail_video_asset(
        session,
        job: PipelineJobRow,
        message: str,
    ) -> None:
        asset = session.get(VideoAssetRow, str(job.subject_id))
        if asset is None:
            return
        asset.status = "failed"
        asset.stage = "Processing failed"
        asset.error = message
        asset.progress = 100

    @staticmethod
    def _cancel_multi_pass_scene(
        session,
        job: PipelineJobRow,
        timestamp: datetime,
    ) -> None:
        scene_row = session.scalar(
            select(SceneRow)
            .where(SceneRow.id == str(job.subject_id))
            .with_for_update()
        )
        if scene_row is None:
            return
        scene = deepcopy(scene_row.payload or {})
        video = scene.get("payload", {}).get("videoAsset", {})
        multi_pass = video.get("multiPass") or {}
        reconstruction = video.get("reconstruction") or {}
        multi_pass["status"] = "cancelled"
        reconstruction.update(
            {
                "status": "cancelled",
                "processingStatus": "cancelled",
                "error": None,
                "completedAt": timestamp.isoformat(),
            }
        )
        video["processingState"] = "multi-pass-cancelled"
        video["multiPass"] = multi_pass
        video["reconstruction"] = reconstruction
        next_revision = scene_revision(scene) + 1
        scene["revision"] = next_revision
        persisted = next_scene_payload(scene, next_revision)
        scene_row.payload = persisted
        sync_scene_index(scene_row, persisted)

    def fail(self, job_id: str, token: str, error: str) -> bool:
        """Fail the exact live claim and publish owned resource state atomically."""

        message = str(error)
        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            row, lease = lock_owned_pipeline_job(session, job_id, token)
            if row is None or lease is None or float(lease.expires_at) <= now:
                session.rollback()
                return False
            row.status = "failed"
            row.error = message
            row.available_at = now
            row.updated_at = now
            if row.kind == "video-processing":
                self._fail_video_asset(session, row, message)
            mark_analysis_terminal(
                session,
                str(job_id),
                "failed",
                now,
                error=message,
            )
            session.delete(lease)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def cancel(self, job_id: str) -> PipelineJob | None:
        """Fence physical work and publish its owned resource state atomically."""

        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            row = session.scalar(
                select(PipelineJobRow)
                .where(PipelineJobRow.id == str(job_id))
                .with_for_update()
            )
            if row is None:
                session.rollback()
                return None
            if row.status in {"queued", "waiting", "processing"}:
                row.status = "cancelled"
                row.available_at = now
                row.updated_at = now
                row.error = None
                timestamp = datetime.fromtimestamp(now, UTC)
                run = session.get(AnalysisRunRow, str(job_id))
                if run is not None and run.status in {
                    "queued",
                    "running",
                    "cancelling",
                }:
                    run.status = "cancelled"
                    run.error = None
                    run.completed_at = timestamp
                    run.updated_at = timestamp
                if row.kind == "video-processing":
                    self._cancel_video_asset(session, row)
                elif row.kind == "multi-pass":
                    self._cancel_multi_pass_scene(session, row, timestamp)
            lease = session.get(PipelineJobLeaseRow, str(job_id))
            if lease is not None:
                session.delete(lease)
            session.commit()
            return pipeline_job_document(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


pipeline_terminals = PipelineTerminalService()
