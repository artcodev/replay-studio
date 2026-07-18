from __future__ import annotations

"""Atomic persistence boundary for video ingest and graph publication."""

from copy import deepcopy
from datetime import UTC, datetime
import re
import time
from typing import Any, Callable

from sqlalchemy import select

from .analysis_run_telemetry import (
    compact_analysis_progress,
    mark_analysis_terminal,
)
from .database import (
    PipelineJobRow,
    SceneRow,
    SessionLocal,
    VideoAssetRow,
)
from .pipeline_domain import (
    PipelineJob,
    PipelineJobConflict,
    begin_pipeline_write,
    lock_owned_pipeline_job,
    pipeline_job_document,
    validate_compact,
)
from .project_models import (
    AnalysisRunRow,
)
from .project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceNotFound,
    ProjectResourceRepository,
)
from .project_segment_contract import SegmentUpsert


class VideoPipelineService:
    """Owns video-specific rows and their one scheduler transaction."""

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

    def enqueue_upload(
        self,
        *,
        job_id: str,
        project_id: str,
        asset_id: str,
        filename: str,
        original_name: str,
        content_type: str,
        title: str | None = None,
    ) -> PipelineJob:
        """Commit asset, ownership, executable job and telemetry together."""

        compact_state = validate_compact(
            {"phase": "ingest"},
            label="Pipeline job state",
            limit=64 * 1024,
        )
        compact_parameters = validate_compact(
            {"title": title},
            label="Pipeline job parameters",
            limit=16 * 1024,
        )
        progress = compact_analysis_progress(
            {
                "phase": "upload-complete",
                "label": "Waiting to process video",
                "completed": 0,
                "total": 100,
                "overallPercent": 0,
            }
        )
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
            if any(
                (
                    session.get(VideoAssetRow, str(asset_id)),
                    session.get(PipelineJobRow, str(job_id)),
                    session.get(AnalysisRunRow, str(job_id)),
                )
            ):
                raise PipelineJobConflict(
                    f"Video upload {asset_id} or pipeline job {job_id} already exists"
                )
            asset = VideoAssetRow(
                id=str(asset_id),
                filename=str(filename),
                original_name=str(original_name),
                content_type=str(content_type),
                status="queued",
                stage="Waiting for pipeline runner",
                progress=2,
                frame_count=0,
            )
            job = PipelineJobRow(
                id=str(job_id),
                project_id=str(project_id),
                kind="video-processing",
                subject_id=str(asset_id),
                status="queued",
                state=compact_state,
                parameters=compact_parameters,
                available_at=now,
                attempts=0,
                error=None,
                requested_at=now,
                updated_at=now,
            )
            run = AnalysisRunRow(
                id=str(job_id),
                project_id=str(project_id),
                scene_id=None,
                segment_id=None,
                kind="video-processing",
                status="queued",
                source_run_id=str(asset_id),
                input_fingerprint=None,
                model=None,
                progress=progress,
                diagnostics={},
                error=None,
                requested_at=requested_at,
                started_at=None,
                completed_at=None,
            )
            session.add_all([asset, job, run])
            session.flush()
            self._resources.link_video_asset_in_transaction(
                session,
                project_id,
                asset_id,
                role="source",
            )
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

    def update_progress(
        self,
        job_id: str,
        token: str,
        values: dict[str, Any],
    ) -> bool:
        """Atomically fence one intermediate asset and telemetry update."""

        allowed = {
            "stage",
            "progress",
            "duration",
            "width",
            "height",
            "fps",
            "frame_count",
        }
        if any(key not in allowed for key in values):
            raise ValueError("Video progress attempted to publish terminal fields")
        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            job, lease = lock_owned_pipeline_job(session, job_id, token)
            if (
                job is None
                or lease is None
                or float(lease.expires_at) <= now
                or job.kind != "video-processing"
            ):
                session.rollback()
                return False
            asset = session.scalar(
                select(VideoAssetRow)
                .where(VideoAssetRow.id == str(job.subject_id))
                .with_for_update()
            )
            if asset is None:
                session.rollback()
                return False
            asset_owner = self._resources.video_asset_owner_in_transaction(
                session,
                str(job.subject_id),
            )
            if asset_owner != str(job.project_id):
                raise PipelineJobConflict(
                    f"Video asset {job.subject_id} is not owned by project {job.project_id}"
                )
            asset.status = "processing"
            for key, value in values.items():
                setattr(asset, key, value)
            progress_value = int(values.get("progress", asset.progress or 0))
            stage = str(values.get("stage", asset.stage or "Processing video"))
            run = session.get(AnalysisRunRow, str(job_id))
            if run is not None and run.status in {"queued", "running"}:
                timestamp = datetime.fromtimestamp(now, UTC)
                run.status = "running"
                run.started_at = run.started_at or timestamp
                run.progress = compact_analysis_progress(
                    {
                        "phase": stage,
                        "label": stage,
                        "detail": None,
                        "completed": progress_value,
                        "total": 100,
                        "overallPercent": progress_value,
                        "etaSeconds": None,
                        "updatedAt": timestamp.isoformat(),
                    }
                )
                run.updated_at = timestamp
            job.updated_at = now
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

    def publish_result(
        self,
        job_id: str,
        token: str,
        *,
        root_scene: dict[str, Any],
        child_scenes: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        frame_count: int,
        generation_key: str,
        stage: str,
        state: dict[str, Any] | None = None,
    ) -> bool:
        """Publish immutable media pointer, scene graph and terminal state."""

        compact = (
            validate_compact(
                state,
                label="Pipeline job state",
                limit=64 * 1024,
            )
            if state is not None
            else None
        )
        now = float(self._clock())
        if (
            generation_key in {"", ".", ".."}
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", generation_key) is None
        ):
            raise ValueError("Video result generation key is invalid")
        source_segment_ids = [str(segment.get("id") or "") for segment in segments]
        if (
            any(not segment_id for segment_id in source_segment_ids)
            or len(source_segment_ids) != len(set(source_segment_ids))
        ):
            raise ValueError("Video result contains empty or duplicate Segment ids")
        session = self._session()
        try:
            begin_pipeline_write(session)
            job, lease = lock_owned_pipeline_job(session, job_id, token)
            if (
                job is None
                or lease is None
                or float(lease.expires_at) <= now
                or job.kind != "video-processing"
            ):
                session.rollback()
                return False
            asset = session.scalar(
                select(VideoAssetRow)
                .where(VideoAssetRow.id == str(job.subject_id))
                .with_for_update()
            )
            if asset is None:
                session.rollback()
                return False

            from .scene_document import (
                next_scene_payload,
                scene_revision,
            )
            from .scene_index_projection import sync_scene_index

            all_scenes = [root_scene, *child_scenes]
            scene_ids = [str(scene.get("id") or "") for scene in all_scenes]
            if (
                not scene_ids
                or any(not scene_id for scene_id in scene_ids)
                or len(set(scene_ids)) != len(scene_ids)
            ):
                raise ValueError("Video result contains invalid Scene ids")
            for result_scene in all_scenes:
                video = (
                    result_scene.get("payload", {}).get("videoAsset", {})
                    if isinstance(result_scene, dict)
                    else {}
                )
                if (
                    str(video.get("id") or "") != str(job.subject_id)
                    or str(video.get("generationKey") or "") != generation_key
                ):
                    raise ValueError(
                        "Video result Scene does not reference the published generation"
                    )
            child_ids = {str(scene["id"]) for scene in child_scenes}
            if any(
                segment.get("sceneId")
                and str(segment["sceneId"]) not in child_ids
                for segment in segments
            ):
                raise ValueError(
                    "Video segment references an unpublished child Scene"
                )
            rows = session.scalars(
                select(SceneRow)
                .where(SceneRow.id.in_(scene_ids))
                .with_for_update()
            ).all()
            rows_by_id = {str(row.id): row for row in rows}
            for result_scene in all_scenes:
                scene_id = str(result_scene["id"])
                row = rows_by_id.get(scene_id)
                if row is None:
                    row = SceneRow(
                        id=scene_id,
                        title=str(result_scene.get("title") or scene_id),
                        duration=float(result_scene.get("duration") or 0.0),
                        kind="demo",
                        payload={},
                    )
                    session.add(row)
                    rows_by_id[scene_id] = row
                    current_revision = 0
                else:
                    current_revision = scene_revision(row.payload)
                next_revision = current_revision + 1
                published_scene = deepcopy(result_scene)
                published_scene["revision"] = next_revision
                persisted = next_scene_payload(
                    published_scene,
                    next_revision,
                )
                row.payload = persisted
                sync_scene_index(row, persisted)
            session.flush()

            roles = {str(root_scene["id"]): "root"}
            roles.update(
                {str(scene["id"]): "segment" for scene in child_scenes}
            )
            for scene_id, role in roles.items():
                self._resources.link_scene_in_transaction(
                    session,
                    str(job.project_id),
                    scene_id,
                    role=role,
                )

            for ordinal, segment in enumerate(segments):
                source_segment_id = source_segment_ids[ordinal]
                self._resources.upsert_segment_in_transaction(
                    session,
                    str(job.project_id),
                    SegmentUpsert(
                        video_asset_id=str(job.subject_id),
                        scene_id=(
                            str(segment.get("sceneId"))
                            if segment.get("sceneId")
                            else None
                        ),
                        source_segment_id=source_segment_id,
                        label=str(segment.get("label") or source_segment_id),
                        start_seconds=float(segment.get("start") or 0.0),
                        end_seconds=float(segment.get("end") or 0.0),
                        ordinal=ordinal,
                        replay_group=segment.get("replayGroup"),
                        replay_variant=segment.get("replayVariant"),
                        payload=deepcopy(segment),
                    ),
                )

            asset.status = "ready"
            asset.stage = str(stage)
            asset.progress = 100
            asset.scene_id = str(root_scene["id"])
            asset.frame_count = max(0, int(frame_count))
            asset.generation_key = str(generation_key)
            asset.error = None
            if compact is not None:
                job.state = compact
            job.status = "succeeded"
            job.error = None
            job.available_at = now
            job.updated_at = now
            mark_analysis_terminal(
                session,
                str(job_id),
                "succeeded",
                now,
                progress={
                    "phase": "complete",
                    "label": str(stage),
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
        except (ProjectResourceConflict, ProjectResourceNotFound) as exc:
            session.rollback()
            raise PipelineJobConflict(str(exc)) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


video_pipeline = VideoPipelineService()
