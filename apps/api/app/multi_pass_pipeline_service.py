from __future__ import annotations

"""Atomic queue and publication transactions for multi-angle scenes."""

from copy import deepcopy
from datetime import UTC, datetime
import time
from typing import Any, Callable

from sqlalchemy import select

from .analysis_run_telemetry import (
    compact_analysis_progress,
    mark_analysis_terminal,
)
from .database import PipelineJobRow, SceneRow, SessionLocal
from .pipeline_domain import (
    PipelineJobConflict,
    begin_pipeline_write,
    lock_owned_pipeline_job,
    validate_compact,
)
from .project_models import AnalysisRunRow
from .project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceNotFound,
    ProjectResourceRepository,
)
from .scene_document import next_scene_payload, scene_revision
from .scene_index_projection import sync_scene_index


class MultiPassPipelineService:
    """Owns the multi-angle queue and fenced Scene publication boundary."""

    def __init__(
        self,
        session_factory: Callable | None = None,
        *,
        clock: Callable[[], float] = time.time,
        resources: ProjectResourceRepository | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock
        self._resources = resources or ProjectResourceRepository(session_factory)

    def _session(self):
        return (self._session_factory or SessionLocal)()

    def enqueue(
        self,
        *,
        project_id: str,
        scene: dict[str, Any],
        source_scene_ids: list[str],
    ) -> dict[str, Any]:
        """Publish queued composite, project link, job and telemetry together."""

        prepared = deepcopy(scene)
        scene_id = str(prepared.get("id") or "")
        reconstruction = (
            prepared.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction", {})
        )
        run_id = str(reconstruction.get("runId") or "")
        if not scene_id or not run_id:
            raise ValueError("Queued multi-pass Scene requires Scene and run ids")
        if str(reconstruction.get("status") or "") != "queued":
            raise ValueError("Multi-pass Scene must be prepared in queued state")
        normalized_sources = list(
            dict.fromkeys(str(value) for value in source_scene_ids if str(value))
        )
        if len(normalized_sources) < 2:
            raise ValueError("Multi-pass Scene requires at least two source scenes")
        progress = compact_analysis_progress(reconstruction.get("progress") or {})
        now = float(self._clock())
        requested_at = datetime.fromtimestamp(now, UTC)
        session = self._session()
        try:
            begin_pipeline_write(session)
            self._resources.require_project_in_transaction(
                session,
                project_id,
                for_update=True,
            )
            owners = {
                source_id: self._resources.scene_owner_in_transaction(
                    session,
                    source_id,
                    for_update=True,
                )
                for source_id in normalized_sources
            }
            invalid_sources = [
                source_id
                for source_id in normalized_sources
                if owners.get(source_id) != str(project_id)
            ]
            if invalid_sources:
                raise PipelineJobConflict(
                    "Multi-pass source Scenes are missing or owned by another project: "
                    + ", ".join(invalid_sources)
                )
            if (
                session.get(SceneRow, scene_id) is not None
                or session.get(PipelineJobRow, run_id) is not None
                or session.get(AnalysisRunRow, run_id) is not None
            ):
                raise PipelineJobConflict(
                    f"Multi-pass Scene {scene_id} or pipeline job {run_id} already exists"
                )

            prepared["revision"] = 1
            persisted = next_scene_payload(prepared, 1)
            scene_row = SceneRow(
                id=scene_id,
                title=str(prepared.get("title") or scene_id),
                duration=float(prepared.get("duration") or 0.0),
                kind="multi-pass",
                payload=persisted,
            )
            sync_scene_index(scene_row, persisted)
            job = PipelineJobRow(
                id=run_id,
                project_id=str(project_id),
                kind="multi-pass",
                subject_id=scene_id,
                status="queued",
                state={"phase": "prepare"},
                parameters={},
                available_at=now,
                attempts=0,
                error=None,
                requested_at=now,
                updated_at=now,
            )
            run = AnalysisRunRow(
                id=run_id,
                project_id=str(project_id),
                scene_id=scene_id,
                segment_id=None,
                kind="multi-pass",
                status="queued",
                source_run_id=run_id,
                input_fingerprint=(
                    str(reconstruction.get("inputFingerprint"))
                    if reconstruction.get("inputFingerprint") is not None
                    else None
                ),
                model=(
                    str(reconstruction.get("model"))
                    if reconstruction.get("model") is not None
                    else None
                ),
                progress=progress,
                diagnostics={},
                error=None,
                requested_at=requested_at,
                started_at=None,
                completed_at=None,
            )
            session.add_all([scene_row, job, run])
            session.flush()
            self._resources.link_scene_in_transaction(
                session,
                project_id,
                scene_id,
                role="multi-pass",
            )
            session.commit()
            return prepared
        except (ProjectResourceConflict, ProjectResourceNotFound) as exc:
            session.rollback()
            raise PipelineJobConflict(str(exc)) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def publish(
        self,
        job_id: str,
        token: str,
        *,
        scene: dict[str, Any],
        status: str,
        state: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Publish a multi-pass Scene and terminal scheduler state together."""

        if status not in {"succeeded", "failed"}:
            raise ValueError(
                "Scene terminal publication requires succeeded or failed"
            )
        compact = (
            validate_compact(
                state,
                label="Pipeline job state",
                limit=64 * 1024,
            )
            if state is not None
            else None
        )
        scene_id = str(scene.get("id") or "")
        if not scene_id:
            raise ValueError("Terminal Scene has no id")
        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            job, lease = lock_owned_pipeline_job(session, job_id, token)
            if (
                job is None
                or lease is None
                or float(lease.expires_at) <= now
                or job.kind != "multi-pass"
                or job.subject_id != scene_id
            ):
                session.rollback()
                return False
            row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == scene_id)
                .with_for_update()
            )
            if row is None:
                session.rollback()
                return False

            expected_revision = scene_revision(scene)
            current_revision = scene_revision(row.payload)
            if expected_revision != current_revision:
                session.rollback()
                return False
            next_revision = current_revision + 1
            published = deepcopy(scene)
            published["revision"] = next_revision
            persisted = next_scene_payload(published, next_revision)
            row.payload = persisted
            sync_scene_index(row, persisted)
            if compact is not None:
                job.state = compact
            job.status = status
            job.error = error if status == "failed" else None
            job.available_at = now
            job.updated_at = now
            progress = (
                scene.get("payload", {})
                .get("videoAsset", {})
                .get("reconstruction", {})
                .get("progress")
            )
            mark_analysis_terminal(
                session,
                str(job_id),
                status,
                now,
                error=error,
                progress=progress if isinstance(progress, dict) else None,
            )
            session.delete(lease)
            session.commit()
            scene["revision"] = next_revision
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
