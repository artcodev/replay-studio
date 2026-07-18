from __future__ import annotations

"""CRUD and query persistence boundary for compact AnalysisRun documents."""

import hashlib
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select

from .analysis_run_telemetry import (
    ACTIVE_ANALYSIS_STATUSES,
    TERMINAL_ANALYSIS_STATUSES,
    compact_analysis_progress,
)
from .database import SessionLocal
from .database_transaction import begin_write_transaction
from .project_models import AnalysisRunRow, ProjectRow
from .project_resource_repository import (
    ProjectResourceConflict,
    ProjectResourceRepository,
)
from .analysis_run_contract import (
    AnalysisRunCreate,
    AnalysisRunDocument,
    AnalysisRunUpdate,
)


class AnalysisRunNotFound(LookupError):
    pass


class AnalysisRunConflict(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _analysis_identifier(
    project_id: str,
    request: AnalysisRunCreate,
) -> str:
    nonce = request.source_run_id or uuid4().hex
    material = "\x1f".join(
        str(value or "")
        for value in (
            project_id,
            request.scene_id,
            request.kind,
            nonce,
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"analysis-{digest}"


def analysis_run_document(row: AnalysisRunRow) -> AnalysisRunDocument:
    return AnalysisRunDocument(
        id=row.id,
        project_id=row.project_id,
        scene_id=row.scene_id,
        segment_id=row.segment_id,
        kind=row.kind,
        status=row.status,
        source_run_id=row.source_run_id,
        input_fingerprint=row.input_fingerprint,
        model=row.model,
        progress=dict(row.progress or {}),
        diagnostics=dict(row.diagnostics or {}),
        error=row.error,
        requested_at=row.requested_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class AnalysisRunRepository:
    """CRUD/query repository for compact AnalysisRun telemetry rows."""

    def __init__(self, session_factory: Callable | None = None) -> None:
        self._session_factory = session_factory
        self._resources = ProjectResourceRepository(session_factory)

    def _session(self):
        return (self._session_factory or SessionLocal)()

    def create(
        self,
        project_id: str,
        request: AnalysisRunCreate,
    ) -> tuple[AnalysisRunDocument, bool]:
        run_id = request.id or _analysis_identifier(project_id, request)
        with self._session() as session:
            begin_write_transaction(session)
            if session.get(ProjectRow, project_id) is None:
                raise AnalysisRunNotFound(
                    f"Project {project_id} was not found"
                )
            try:
                scene_owner = (
                    self._resources.scene_owner_in_transaction(
                        session,
                        request.scene_id,
                    )
                    if request.scene_id
                    else None
                )
            except ProjectResourceConflict as exc:
                raise AnalysisRunConflict(str(exc)) from exc
            if request.scene_id and scene_owner != project_id:
                raise AnalysisRunConflict(
                    f"Scene {request.scene_id} was not found in project {project_id}"
                )
            if request.segment_id and not self._resources.segment_owned_in_transaction(
                session,
                project_id,
                request.segment_id,
            ):
                raise AnalysisRunConflict(
                    f"Segment {request.segment_id} was not found in project {project_id}"
                )
            existing = session.scalar(
                select(AnalysisRunRow)
                .where(AnalysisRunRow.id == run_id)
                .with_for_update()
            )
            if existing is not None:
                raise AnalysisRunConflict(
                    f"Analysis run {run_id} already exists"
                )
            row = AnalysisRunRow(
                id=run_id,
                project_id=project_id,
                scene_id=request.scene_id,
                segment_id=request.segment_id,
                kind=request.kind,
                status=request.status,
                source_run_id=request.source_run_id,
                input_fingerprint=request.input_fingerprint,
                model=request.model,
                progress=compact_analysis_progress(request.progress),
                diagnostics=dict(request.diagnostics),
                error=request.error,
                requested_at=request.requested_at,
                started_at=request.started_at,
                completed_at=request.completed_at,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return analysis_run_document(row), True

    def list_for_project(self, project_id: str) -> list[AnalysisRunDocument]:
        with self._session() as session:
            if session.get(ProjectRow, project_id) is None:
                raise AnalysisRunNotFound(
                    f"Project {project_id} was not found"
                )
            rows = session.scalars(
                select(AnalysisRunRow)
                .where(AnalysisRunRow.project_id == project_id)
                .order_by(AnalysisRunRow.created_at.desc(), AnalysisRunRow.id)
            ).all()
            return [analysis_run_document(row) for row in rows]

    def get(self, run_id: str) -> AnalysisRunDocument | None:
        with self._session() as session:
            row = session.get(AnalysisRunRow, run_id)
            return analysis_run_document(row) if row else None

    @staticmethod
    def _validate_transition(current: str, requested: str) -> None:
        allowed = {
            "queued": {
                "queued",
                "running",
                "cancelled",
                "succeeded",
                "failed",
            },
            "running": {
                "running",
                "cancelling",
                "cancelled",
                "succeeded",
                "failed",
            },
            "cancelling": {"cancelling", "cancelled", "failed"},
            "cancelled": {"cancelled"},
            "succeeded": {"succeeded"},
            "failed": {"failed"},
        }
        if requested not in allowed.get(current, {current}):
            raise AnalysisRunConflict(
                f"Analysis status cannot move from {current} to {requested}"
            )

    def update(
        self,
        run_id: str,
        request: AnalysisRunUpdate,
    ) -> AnalysisRunDocument:
        with self._session() as session:
            begin_write_transaction(session)
            row = session.scalar(
                select(AnalysisRunRow)
                .where(AnalysisRunRow.id == run_id)
                .with_for_update()
            )
            if row is None:
                raise AnalysisRunNotFound(
                    f"Analysis run {run_id} was not found"
                )
            if request.status is not None:
                self._validate_transition(row.status, request.status)
                row.status = request.status
            elif (
                request.progress is not None
                and row.status not in ACTIVE_ANALYSIS_STATUSES
            ):
                raise AnalysisRunConflict(
                    f"Analysis run {run_id} is already {row.status}"
                )
            if request.progress is not None:
                row.progress = compact_analysis_progress(request.progress)
            if request.diagnostics is not None:
                row.diagnostics = dict(request.diagnostics)
            if "error" in request.model_fields_set:
                row.error = request.error
            if request.started_at is not None:
                row.started_at = request.started_at
            if request.completed_at is not None:
                row.completed_at = request.completed_at
            if (
                row.status in TERMINAL_ANALYSIS_STATUSES
                and row.completed_at is None
            ):
                row.completed_at = _utcnow()
            row.updated_at = _utcnow()
            session.commit()
            session.refresh(row)
            return analysis_run_document(row)

    def update_progress(
        self,
        run_id: str,
        progress: dict[str, Any],
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> AnalysisRunDocument:
        return self.update(
            run_id,
            AnalysisRunUpdate(progress=progress, diagnostics=diagnostics),
        )

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> AnalysisRunDocument:
        return self.update(
            run_id,
            AnalysisRunUpdate(
                status=status,
                error=error,
                diagnostics=diagnostics,
            ),
        )


analysis_runs = AnalysisRunRepository()
