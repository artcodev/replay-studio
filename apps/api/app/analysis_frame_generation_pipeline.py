from __future__ import annotations

"""Atomic queue/publication boundary for source-resolution analysis frames."""

from datetime import UTC, datetime
import re
import time
from typing import Callable

from sqlalchemy import select

from .analysis_frame_generation import PreparedAnalysisFrameGeneration
from .analysis_frame_generation_scene import switch_scene_analysis_frame_generation
from .analysis_run_telemetry import compact_analysis_progress, mark_analysis_terminal
from .database import (
    PipelineJobLeaseRow,
    PipelineJobRow,
    ReconstructionJobRow,
    SceneRow,
    SessionLocal,
    VideoAssetRow,
)
from .pipeline_domain import (
    PIPELINE_ACTIVE_STATUSES,
    PipelineJob,
    PipelineJobConflict,
    begin_pipeline_write,
    lock_owned_pipeline_job,
    pipeline_job_document,
)
from .project_models import AnalysisRunRow, SegmentRow
from .project_resource_repository import ProjectResourceRepository
from .scene_document import next_scene_payload, scene_revision
from .scene_index_projection import sync_scene_index


JOB_KIND = "analysis-frame-generation"


class AnalysisFrameGenerationPipelineService:
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

    @staticmethod
    def _asset_scene_ids(session, asset: VideoAssetRow) -> list[str]:
        scene_ids = {
            str(value)
            for value in session.scalars(
                select(SegmentRow.scene_id).where(
                    SegmentRow.video_asset_id == str(asset.id),
                    SegmentRow.scene_id.is_not(None),
                )
            ).all()
            if value
        }
        if asset.scene_id:
            scene_ids.add(str(asset.scene_id))
        return sorted(scene_ids)

    @staticmethod
    def _require_idle_scenes(session, scene_ids: list[str]) -> None:
        if not scene_ids:
            return
        active = session.scalars(
            select(ReconstructionJobRow.scene_id).where(
                ReconstructionJobRow.scene_id.in_(scene_ids),
                ReconstructionJobRow.status.in_(("queued", "processing")),
            )
        ).all()
        if active:
            raise PipelineJobConflict(
                "Cancel or finish reconstruction before replacing its analysis frames"
            )

    def enqueue(
        self,
        *,
        job_id: str,
        project_id: str,
        asset_id: str,
    ) -> PipelineJob:
        now = float(self._clock())
        requested_at = datetime.fromtimestamp(now, UTC)
        session = self._session()
        try:
            begin_pipeline_write(session)
            owner = self._resources.video_asset_owner_in_transaction(
                session,
                asset_id,
                for_update=True,
            )
            if owner != str(project_id):
                raise PipelineJobConflict(
                    f"Video asset {asset_id} is not owned by project {project_id}"
                )
            asset = session.scalar(
                select(VideoAssetRow)
                .where(VideoAssetRow.id == str(asset_id))
                .with_for_update()
            )
            if asset is None or not asset.generation_key:
                raise PipelineJobConflict("Video processing must finish first")
            scene_ids = self._asset_scene_ids(session, asset)
            self._require_idle_scenes(session, scene_ids)

            existing = session.scalar(
                select(PipelineJobRow)
                .where(
                    PipelineJobRow.kind == JOB_KIND,
                    PipelineJobRow.subject_id == str(asset_id),
                )
                .with_for_update()
            )
            if existing is not None and existing.status in PIPELINE_ACTIVE_STATUSES:
                session.commit()
                return pipeline_job_document(existing)
            if existing is not None:
                stale_lease = session.get(PipelineJobLeaseRow, str(existing.id))
                if stale_lease is not None:
                    session.delete(stale_lease)
                session.delete(existing)
                session.flush()
            if session.get(AnalysisRunRow, str(job_id)) is not None:
                raise PipelineJobConflict(f"Analysis-frame job {job_id} already exists")

            progress = compact_analysis_progress(
                {
                    "phase": "queued",
                    "label": "Waiting to extract source-resolution frames",
                    "detail": "The current published generation remains active until publication.",
                    "completed": 0,
                    "total": 100,
                    "phasePercent": 0,
                    "overallPercent": 0,
                    "etaSeconds": None,
                    "updatedAt": requested_at.isoformat(),
                }
            )
            job = PipelineJobRow(
                id=str(job_id),
                project_id=str(project_id),
                kind=JOB_KIND,
                subject_id=str(asset_id),
                status="queued",
                state={"phase": "queued"},
                parameters={"inputPolicy": "source-video-pixels-v1"},
                available_at=now,
                attempts=0,
                error=None,
                requested_at=now,
                updated_at=now,
            )
            run = AnalysisRunRow(
                id=str(job_id),
                project_id=str(project_id),
                scene_id=str(asset.scene_id) if asset.scene_id else None,
                segment_id=None,
                kind=JOB_KIND,
                status="queued",
                # A video may be regenerated repeatedly.  The telemetry source
                # identity therefore belongs to this immutable pipeline run,
                # not to the long-lived asset; otherwise the second explicit
                # regeneration violates uq_analysis_run_source.
                source_run_id=str(job_id),
                input_fingerprint=str(asset.generation_key),
                model=None,
                progress=progress,
                diagnostics={
                    "videoAssetId": str(asset_id),
                    "inputGenerationKey": str(asset.generation_key),
                },
                error=None,
                requested_at=requested_at,
                started_at=None,
                completed_at=None,
            )
            session.add_all((job, run))
            session.commit()
            return pipeline_job_document(job)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def publish(
        self,
        job_id: str,
        token: str,
        prepared: PreparedAnalysisFrameGeneration,
    ) -> bool:
        prepared.validate()
        generation_key = str(prepared.generation_key)
        if (
            generation_key in {"", ".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", generation_key) is None
        ):
            raise ValueError("Analysis-frame generation key is invalid")
        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            job, lease = lock_owned_pipeline_job(session, job_id, token)
            if (
                job is None
                or lease is None
                or float(lease.expires_at) <= now
                or job.kind != JOB_KIND
                or str(job.subject_id) != prepared.asset_id
            ):
                session.rollback()
                return False
            asset = session.scalar(
                select(VideoAssetRow)
                .where(VideoAssetRow.id == prepared.asset_id)
                .with_for_update()
            )
            if asset is None:
                session.rollback()
                return False
            owner = self._resources.video_asset_owner_in_transaction(
                session,
                prepared.asset_id,
                for_update=True,
            )
            if owner != str(job.project_id):
                raise PipelineJobConflict(
                    "Video asset ownership changed during analysis-frame generation"
                )
            scene_ids = self._asset_scene_ids(session, asset)
            self._require_idle_scenes(session, scene_ids)
            rows = session.scalars(
                select(SceneRow)
                .where(SceneRow.id.in_(scene_ids))
                .with_for_update()
            ).all()
            for row in rows:
                current = row.payload or {}
                video = current.get("payload", {}).get("videoAsset") or {}
                if str(video.get("id") or "") != prepared.asset_id:
                    raise PipelineJobConflict(
                        f"Scene {row.id} does not reference video asset {prepared.asset_id}"
                    )
                updated = switch_scene_analysis_frame_generation(
                    current,
                    generation_key=generation_key,
                    source_fps=prepared.source_fps,
                    analysis_fps=prepared.analysis_fps,
                    frame_count=prepared.frame_count,
                    analysis_frame_input=prepared.analysis_frame_input,
                )
                revision = scene_revision(current) + 1
                updated["revision"] = revision
                persisted = next_scene_payload(updated, revision)
                row.payload = persisted
                sync_scene_index(row, persisted)

            asset.generation_key = generation_key
            asset.frame_count = prepared.frame_count
            asset.fps = prepared.source_fps
            asset.status = "ready"
            asset.stage = "Source-resolution analysis frames ready"
            asset.progress = 100
            asset.error = None
            job.state = {"phase": "complete", "generationKey": generation_key}
            job.status = "succeeded"
            job.error = None
            job.available_at = now
            job.updated_at = now
            mark_analysis_terminal(
                session,
                job_id,
                "succeeded",
                now,
                progress={
                    "phase": "complete",
                    "label": "Source-resolution analysis frames ready",
                    "detail": f"Published {prepared.frame_count} frames; calibration is required again.",
                    "completed": 100,
                    "total": 100,
                    "phasePercent": 100,
                    "overallPercent": 100,
                    "etaSeconds": 0.0,
                    "updatedAt": datetime.fromtimestamp(now, UTC).isoformat(),
                },
            )
            session.delete(lease)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


analysis_frame_generation_pipeline = AnalysisFrameGenerationPipelineService()


__all__ = (
    "AnalysisFrameGenerationPipelineService",
    "JOB_KIND",
    "analysis_frame_generation_pipeline",
)
