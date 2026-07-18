from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from .database import SessionLocal
from .project_models import ProjectRow
from .project_lifecycle_contract import (
    ProjectCreate,
    ProjectHeader,
    ProjectSummary,
    ProjectUpdate,
)


class ProjectNotFound(LookupError):
    pass


class ProjectConflict(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ProjectStore:
    """Own compact project headers and lifecycle mutations only."""

    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    def _session(self):
        return (self._session_factory or SessionLocal)()

    @staticmethod
    def _begin_atomic_write(session) -> None:
        """Serialize project-header revision compare-and-swap on every backend."""

        bind = session.get_bind()
        if bind.dialect.name == "sqlite":
            session.execute(text("BEGIN IMMEDIATE"))
        else:
            session.begin()

    @staticmethod
    def _require_project(
        session,
        project_id: str,
        *,
        for_update: bool = False,
    ) -> ProjectRow:
        statement = select(ProjectRow).where(ProjectRow.id == project_id)
        if for_update:
            statement = statement.with_for_update()
        row = session.scalar(statement)
        if row is None:
            raise ProjectNotFound(f"Project {project_id} was not found")
        return row

    def create_project(self, request: ProjectCreate) -> ProjectHeader:
        project_id = request.id or f"project-{uuid4().hex[:16]}"
        with self._session() as session:
            if session.get(ProjectRow, project_id) is not None:
                raise ProjectConflict(f"Project {project_id} already exists")
            row = ProjectRow(
                id=project_id,
                title=request.title,
                status=request.status,
                revision=1,
                metadata_payload=dict(request.metadata),
            )
            session.add(row)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ProjectConflict(f"Project {project_id} already exists") from exc
        result = self.get_project(project_id)
        assert result is not None
        return result

    def list_projects(self) -> list[ProjectSummary]:
        """List dashboard headers in one compact query.

        Resource counts are intentionally omitted here because the public
        dashboard contract does not expose them. Hydrating four aggregates per
        project would turn the multi-project home screen into an N+1 query.
        """

        with self._session() as session:
            rows = session.scalars(
                select(ProjectRow).order_by(ProjectRow.updated_at.desc(), ProjectRow.id)
            ).all()
            return [
                ProjectSummary(
                    id=row.id,
                    title=row.title,
                    status=row.status,
                    revision=int(row.revision),
                    match_id=row.match_id,
                    current_match_snapshot_id=row.current_match_snapshot_id,
                    metadata=dict(row.metadata_payload or {}),
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    def update_project(
        self,
        project_id: str,
        request: ProjectUpdate,
    ) -> ProjectHeader:
        with self._session() as session:
            self._begin_atomic_write(session)
            row = self._require_project(session, project_id, for_update=True)
            if (
                request.expected_revision is not None
                and request.expected_revision != row.revision
            ):
                raise ProjectConflict(
                    f"Project {project_id} changed from revision "
                    f"{request.expected_revision} to {row.revision}"
                )
            changed = False
            if request.title is not None and request.title != row.title:
                row.title = request.title
                changed = True
            if request.status is not None and request.status != row.status:
                row.status = request.status
                changed = True
            if request.metadata is not None and request.metadata != row.metadata_payload:
                row.metadata_payload = dict(request.metadata)
                changed = True
            if changed:
                row.revision += 1
                row.updated_at = _utcnow()
            session.commit()
        result = self.get_project(project_id)
        assert result is not None
        return result

    def archive_project(
        self,
        project_id: str,
        *,
        expected_revision: int | None = None,
    ) -> ProjectHeader:
        return self.update_project(
            project_id,
            ProjectUpdate(status="archived", expected_revision=expected_revision),
        )

    def project_exists(self, project_id: str) -> bool:
        """Check project identity without hydrating its owned resource graph."""

        with self._session() as session:
            return session.scalar(
                select(ProjectRow.id).where(ProjectRow.id == project_id)
            ) is not None

    def get_project(self, project_id: str) -> ProjectHeader | None:
        """Read the compact project header without joining domain resources."""

        with self._session() as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                return None
            return ProjectHeader(
                id=row.id,
                title=row.title,
                status=row.status,
                revision=int(row.revision),
                match_id=row.match_id,
                current_match_snapshot_id=row.current_match_snapshot_id,
                metadata=dict(row.metadata_payload or {}),
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

project_store = ProjectStore()
