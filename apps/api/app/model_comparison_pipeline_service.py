from __future__ import annotations

"""Atomic queue and publication transactions for model comparisons."""

from copy import deepcopy
from datetime import UTC, datetime
import time
from typing import Any, Callable

from sqlalchemy import select

from .analysis_run_telemetry import (
    compact_analysis_progress,
    mark_analysis_terminal,
)
from .database import (
    PipelineJobLeaseRow,
    PipelineJobRow,
    SceneRow,
    SessionLocal,
)
from .pipeline_domain import (
    PIPELINE_ACTIVE_STATUSES,
    PipelineJob,
    PipelineJobConflict,
    begin_pipeline_write,
    lock_owned_pipeline_job,
    pipeline_job_document,
    validate_compact,
)
from .project_models import AnalysisRunRow
from .reconstruction_calibration_fingerprint import calibration_input_fingerprint
from .reconstruction_coordinate_policy import (
    METRIC_REQUIRED,
    resolve_full_run_coordinate_authorization,
)
from .reconstruction_errors import ReconstructionError
from .project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceNotFound,
    ProjectResourceRepository,
)
from .scene_document import (
    next_scene_payload,
    reconstruction_input_fingerprint,
    scene_revision,
)
from .scene_index_projection import sync_scene_index


class ModelComparisonPipelineService:
    """Owns the model-comparison queue and fenced publication boundary."""

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
        job_id: str,
        project_id: str,
        scene_id: str,
        baseline_model: str,
        candidate_model: str,
    ) -> PipelineJob:
        """Capture one scene revision and queue its model comparison."""

        now = float(self._clock())
        requested_at = datetime.fromtimestamp(now, UTC)
        session = self._session()
        try:
            begin_pipeline_write(session)
            owner = self._resources.scene_owner_in_transaction(
                session,
                str(scene_id),
                for_update=True,
            )
            if owner != str(project_id):
                raise PipelineJobConflict(
                    f"Scene {scene_id} is not owned by project {project_id}"
                )
            scene_row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == str(scene_id))
                .with_for_update()
            )
            if scene_row is None:
                raise PipelineJobConflict(f"Scene {scene_id} was not found")

            scene = deepcopy(scene_row.payload or {})
            video = scene.get("payload", {}).get("videoAsset") or {}
            if not video.get("selectedSegmentId"):
                raise PipelineJobConflict(
                    "Choose a continuous shot scene before comparing models"
                )
            reconstruction = video.get("reconstruction") or {}
            if reconstruction.get("status") in {"queued", "processing"}:
                raise PipelineJobConflict(
                    "Wait for reconstruction to finish before comparing models"
                )
            try:
                coordinate_policy, _ = resolve_full_run_coordinate_authorization(
                    reconstruction,
                    calibration_input_fingerprint=calibration_input_fingerprint(
                        scene
                    ),
                )
            except ReconstructionError as exc:
                raise PipelineJobConflict(str(exc)) from exc
            if coordinate_policy != METRIC_REQUIRED:
                raise PipelineJobConflict(
                    "Resolve every calibration frame before comparing tracking "
                    "continuity; model comparison does not use image fallback"
                )

            existing = session.scalar(
                select(PipelineJobRow)
                .where(
                    PipelineJobRow.kind == "model-comparison",
                    PipelineJobRow.subject_id == str(scene_id),
                )
                .order_by(PipelineJobRow.requested_at, PipelineJobRow.id)
                .with_for_update()
            )
            if existing is not None and existing.status in PIPELINE_ACTIVE_STATUSES:
                if str(existing.project_id) != str(project_id):
                    raise PipelineJobConflict(
                        f"Scene {scene_id} already has an active model comparison"
                    )
                session.commit()
                return pipeline_job_document(existing)
            if existing is not None:
                stale_lease = session.get(PipelineJobLeaseRow, str(existing.id))
                if stale_lease is not None:
                    session.delete(stale_lease)
                session.delete(existing)
                session.flush()
            if (
                session.get(PipelineJobRow, str(job_id)) is not None
                or session.get(AnalysisRunRow, str(job_id)) is not None
            ):
                raise PipelineJobConflict(
                    f"Model-comparison job {job_id} already exists"
                )

            input_fingerprint = reconstruction_input_fingerprint(scene)
            progress = compact_analysis_progress(
                {
                    "phase": "queued",
                    "phaseIndex": 1,
                    "phaseCount": 2,
                    "label": "Waiting to compare detection models",
                    "completed": 0,
                    "total": 2,
                    "phasePercent": 0,
                    "overallPercent": 0,
                    "etaSeconds": None,
                    "updatedAt": requested_at.isoformat(),
                }
            )
            job = PipelineJobRow(
                id=str(job_id),
                project_id=str(project_id),
                kind="model-comparison",
                subject_id=str(scene_id),
                status="queued",
                state={"phase": "queued"},
                parameters={
                    "baselineModel": str(baseline_model),
                    "candidateModel": str(candidate_model),
                    "sceneRevision": scene_revision(scene),
                },
                available_at=now,
                attempts=0,
                error=None,
                requested_at=now,
                updated_at=now,
            )
            run = AnalysisRunRow(
                id=str(job_id),
                project_id=str(project_id),
                scene_id=str(scene_id),
                segment_id=None,
                kind="model-comparison",
                status="queued",
                source_run_id=str(job_id),
                input_fingerprint=input_fingerprint,
                model=f"{baseline_model} vs {candidate_model}",
                progress=progress,
                diagnostics={},
                error=None,
                requested_at=requested_at,
                started_at=None,
                completed_at=None,
            )
            session.add_all([job, run])
            session.commit()
            return pipeline_job_document(job)
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
        report: dict[str, Any],
    ) -> bool:
        """Publish one comparison report and terminal state together."""

        compact_report = validate_compact(
            report,
            label="Model-comparison report",
            limit=64 * 1024,
        )
        now = float(self._clock())
        timestamp = datetime.fromtimestamp(now, UTC)
        session = self._session()
        try:
            begin_pipeline_write(session)
            job, lease = lock_owned_pipeline_job(session, job_id, token)
            if (
                job is None
                or lease is None
                or float(lease.expires_at) <= now
                or job.kind != "model-comparison"
            ):
                session.rollback()
                return False
            scene_id = str(job.subject_id)
            if str(compact_report.get("sceneId") or "") != scene_id:
                raise PipelineJobConflict(
                    "Model-comparison report belongs to a different Scene"
                )
            owner = self._resources.scene_owner_in_transaction(
                session,
                scene_id,
                for_update=True,
            )
            if owner != str(job.project_id):
                raise PipelineJobConflict(
                    f"Scene {scene_id} is no longer owned by project {job.project_id}"
                )
            scene_row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == scene_id)
                .with_for_update()
            )
            if scene_row is None:
                raise PipelineJobConflict(f"Scene {scene_id} was not found")

            scene = deepcopy(scene_row.payload or {})
            current_revision = scene_revision(scene)
            expected_revision = int(job.parameters.get("sceneRevision", -1))
            if current_revision != expected_revision:
                error = (
                    "Scene changed while detection models were being compared; "
                    "run the comparison again on the current revision"
                )
                job.status = "failed"
                job.error = error
                job.state = {"phase": "failed"}
                job.available_at = now
                job.updated_at = now
                mark_analysis_terminal(
                    session,
                    str(job_id),
                    "failed",
                    now,
                    error=error,
                    progress={
                        "phase": "failed",
                        "label": "Model comparison became stale",
                        "completed": 2,
                        "total": 2,
                        "phasePercent": 100,
                        "overallPercent": 100,
                        "etaSeconds": 0.0,
                        "updatedAt": timestamp.isoformat(),
                    },
                )
                session.delete(lease)
                session.commit()
                return False

            reconstruction = (
                scene.setdefault("payload", {})
                .setdefault("videoAsset", {})
                .setdefault("reconstruction", {})
            )
            reconstruction["modelComparison"] = deepcopy(compact_report)
            next_revision = current_revision + 1
            scene["revision"] = next_revision
            persisted = next_scene_payload(scene, next_revision)
            scene_row.payload = persisted
            sync_scene_index(scene_row, persisted)

            job.status = "succeeded"
            job.error = None
            job.state = {"phase": "complete"}
            job.available_at = now
            job.updated_at = now
            mark_analysis_terminal(
                session,
                str(job_id),
                "succeeded",
                now,
                progress={
                    "phase": "complete",
                    "label": "Detection model comparison ready",
                    "completed": 2,
                    "total": 2,
                    "phasePercent": 100,
                    "overallPercent": 100,
                    "etaSeconds": 0.0,
                    "updatedAt": timestamp.isoformat(),
                },
            )
            session.delete(lease)
            session.commit()
            return True
        except (ProjectResourceConflict, ProjectResourceNotFound) as exc:
            session.rollback()
            raise PipelineJobConflict(str(exc)) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
