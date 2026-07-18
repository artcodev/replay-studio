from __future__ import annotations

"""Canonical ownership boundary for project Scenes, videos, and Segments."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Iterable, Iterator

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from .database import SceneRow, SessionLocal, VideoAssetRow
from .project_identifiers import stable_identifier
from .project_models import (
    ProjectRow,
    ProjectSceneRow,
    ProjectVideoAssetRow,
    SegmentRow,
)
from .project_segment_contract import (
    ProjectSceneLink,
    ProjectVideoAssetLink,
    SegmentDocument,
    SegmentUpsert,
)


class ProjectResourceNotFound(LookupError):
    pass


class ProjectResourceConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class ReconstructionResourceContext:
    project_id: str
    segment_id: str | None


def _utcnow() -> datetime:
    return datetime.now(UTC)


def segment_document(row: SegmentRow) -> SegmentDocument:
    return SegmentDocument(
        id=row.id,
        project_id=row.project_id,
        video_asset_id=row.video_asset_id,
        scene_id=row.scene_id,
        source_segment_id=row.source_segment_id,
        label=row.label,
        start_seconds=float(row.start_seconds),
        end_seconds=float(row.end_seconds),
        ordinal=int(row.ordinal),
        replay_group=row.replay_group,
        replay_variant=row.replay_variant,
        payload=dict(row.payload or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class ProjectResourceRepository:
    """Own project resource membership and segment metadata."""

    def __init__(self, session_factory: Callable | None = None) -> None:
        self._session_factory = session_factory

    def _session(self):
        return (self._session_factory or SessionLocal)()

    @contextmanager
    def transaction(self) -> Iterator[object]:
        with self._session() as session:
            if session.get_bind().dialect.name == "sqlite":
                session.execute(text("BEGIN IMMEDIATE"))
            else:
                session.begin()
            try:
                yield session
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise ProjectResourceConflict(
                    "Project resource ownership changed concurrently"
                ) from exc
            except Exception:
                session.rollback()
                raise

    @staticmethod
    def require_project_in_transaction(
        session,
        project_id: str,
        *,
        for_update: bool = False,
    ) -> ProjectRow:
        statement = select(ProjectRow).where(ProjectRow.id == str(project_id))
        if for_update:
            statement = statement.with_for_update()
        project = session.scalar(statement)
        if project is None:
            raise ProjectResourceNotFound(f"Project {project_id} was not found")
        return project

    @staticmethod
    def scene_owner_in_transaction(
        session,
        scene_id: str,
        *,
        for_update: bool = False,
    ) -> str | None:
        statement = (
            select(ProjectSceneRow.project_id)
            .where(ProjectSceneRow.scene_id == str(scene_id))
            .limit(2)
        )
        if for_update:
            statement = statement.with_for_update()
        owners = [str(value) for value in session.scalars(statement).all()]
        if len(owners) > 1:
            raise ProjectResourceConflict(
                f"Scene {scene_id} has multiple owning projects"
            )
        return owners[0] if owners else None

    @staticmethod
    def video_asset_owner_in_transaction(
        session,
        video_asset_id: str,
        *,
        for_update: bool = False,
    ) -> str | None:
        statement = (
            select(ProjectVideoAssetRow.project_id)
            .where(ProjectVideoAssetRow.video_asset_id == str(video_asset_id))
            .limit(2)
        )
        if for_update:
            statement = statement.with_for_update()
        owners = [str(value) for value in session.scalars(statement).all()]
        if len(owners) > 1:
            raise ProjectResourceConflict(
                f"Video asset {video_asset_id} has multiple owning projects"
            )
        return owners[0] if owners else None

    @staticmethod
    def segment_owned_in_transaction(
        session,
        project_id: str,
        segment_id: str,
    ) -> bool:
        return session.scalar(
            select(SegmentRow.id).where(
                SegmentRow.id == str(segment_id),
                SegmentRow.project_id == str(project_id),
            )
        ) is not None

    @classmethod
    def reconstruction_context_in_transaction(
        cls,
        session,
        scene_id: str,
        *,
        strict: bool = True,
    ) -> ReconstructionResourceContext | None:
        """Resolve compact run ownership inside a caller-owned transaction."""

        project_id = cls.scene_owner_in_transaction(session, scene_id)
        if project_id is None:
            if strict:
                raise ProjectResourceConflict(
                    f"Reconstruction scene {scene_id} must have exactly one "
                    "owning project; found 0"
                )
            return None
        segment_ids = [
            str(value)
            for value in session.scalars(
                select(SegmentRow.id)
                .where(
                    SegmentRow.project_id == project_id,
                    SegmentRow.scene_id == str(scene_id),
                )
                .order_by(SegmentRow.id)
                .limit(2)
            ).all()
        ]
        if len(segment_ids) > 1:
            if strict:
                raise ProjectResourceConflict(
                    f"Reconstruction scene {scene_id} is linked to multiple segments"
                )
            return None
        return ReconstructionResourceContext(
            project_id=project_id,
            segment_id=segment_ids[0] if segment_ids else None,
        )

    def scene_owner(self, scene_id: str) -> str | None:
        with self._session() as session:
            return self.scene_owner_in_transaction(session, scene_id)

    def video_asset_owner(self, video_asset_id: str) -> str | None:
        with self._session() as session:
            return self.video_asset_owner_in_transaction(session, video_asset_id)

    def link_scene(
        self,
        project_id: str,
        scene_id: str,
        *,
        role: str = "scene",
    ) -> bool:
        with self.transaction() as session:
            return self.link_scene_in_transaction(
                session,
                project_id,
                scene_id,
                role=role,
            )

    def link_scenes(
        self,
        project_id: str,
        links: Iterable[tuple[str, str]],
    ) -> int:
        normalized: dict[str, str] = {}
        for raw_scene_id, raw_role in links:
            scene_id = str(raw_scene_id)
            role = str(raw_role)
            previous = normalized.get(scene_id)
            if previous is not None and previous != role:
                raise ProjectResourceConflict(
                    f"Scene {scene_id} was assigned conflicting project roles"
                )
            normalized[scene_id] = role
        with self.transaction() as session:
            self.require_project_in_transaction(
                session,
                project_id,
                for_update=True,
            )
            return sum(
                self._link_scene_in_transaction(
                    session,
                    project_id,
                    scene_id,
                    role=role,
                )
                for scene_id, role in normalized.items()
            )

    def link_scene_in_transaction(
        self,
        session,
        project_id: str,
        scene_id: str,
        *,
        role: str = "scene",
    ) -> bool:
        self.require_project_in_transaction(
            session,
            project_id,
            for_update=True,
        )
        return self._link_scene_in_transaction(
            session,
            project_id,
            scene_id,
            role=role,
        )

    def _link_scene_in_transaction(
        self,
        session,
        project_id: str,
        scene_id: str,
        *,
        role: str,
    ) -> bool:
        scene_id = str(scene_id)
        if session.scalar(
            select(SceneRow.id)
            .where(SceneRow.id == scene_id)
            .with_for_update()
        ) is None:
            raise ProjectResourceConflict(f"Scene {scene_id} was not found")
        owner = self.scene_owner_in_transaction(
            session,
            scene_id,
            for_update=True,
        )
        if owner is not None and owner != str(project_id):
            raise ProjectResourceConflict(
                f"Scene {scene_id} already belongs to project {owner}"
            )
        key = {"project_id": str(project_id), "scene_id": scene_id}
        row = session.get(ProjectSceneRow, key)
        created = row is None
        if row is None:
            session.add(ProjectSceneRow(**key, role=str(role)))
        elif row.role != str(role):
            row.role = str(role)
        session.flush()
        return created

    def link_video_asset(
        self,
        project_id: str,
        video_asset_id: str,
        *,
        role: str = "source",
    ) -> bool:
        with self.transaction() as session:
            return self.link_video_asset_in_transaction(
                session,
                project_id,
                video_asset_id,
                role=role,
            )

    def link_video_asset_in_transaction(
        self,
        session,
        project_id: str,
        video_asset_id: str,
        *,
        role: str = "source",
    ) -> bool:
        self.require_project_in_transaction(
            session,
            project_id,
            for_update=True,
        )
        asset_id = str(video_asset_id)
        if session.scalar(
            select(VideoAssetRow.id)
            .where(VideoAssetRow.id == asset_id)
            .with_for_update()
        ) is None:
            raise ProjectResourceConflict(f"Video asset {asset_id} was not found")
        owner = self.video_asset_owner_in_transaction(
            session,
            asset_id,
            for_update=True,
        )
        if owner is not None and owner != str(project_id):
            raise ProjectResourceConflict(
                f"Video asset {asset_id} already belongs to project {owner}"
            )
        key = {"project_id": str(project_id), "video_asset_id": asset_id}
        row = session.get(ProjectVideoAssetRow, key)
        created = row is None
        if row is None:
            session.add(ProjectVideoAssetRow(**key, role=str(role)))
        elif row.role != str(role):
            row.role = str(role)
        session.flush()
        return created

    def list_scene_links(self, project_id: str) -> list[ProjectSceneLink]:
        with self._session() as session:
            self.require_project_in_transaction(session, project_id)
            rows = session.scalars(
                select(ProjectSceneRow)
                .where(ProjectSceneRow.project_id == str(project_id))
                .order_by(ProjectSceneRow.role, ProjectSceneRow.scene_id)
            ).all()
            return [
                ProjectSceneLink(
                    scene_id=row.scene_id,
                    role=row.role,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def list_video_asset_links(
        self,
        project_id: str,
    ) -> list[ProjectVideoAssetLink]:
        with self._session() as session:
            self.require_project_in_transaction(session, project_id)
            rows = session.scalars(
                select(ProjectVideoAssetRow)
                .where(ProjectVideoAssetRow.project_id == str(project_id))
                .order_by(
                    ProjectVideoAssetRow.role,
                    ProjectVideoAssetRow.video_asset_id,
                )
            ).all()
            return [
                ProjectVideoAssetLink(
                    video_asset_id=row.video_asset_id,
                    role=row.role,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def list_segments(self, project_id: str) -> list[SegmentDocument]:
        with self._session() as session:
            self.require_project_in_transaction(session, project_id)
            rows = session.scalars(
                select(SegmentRow)
                .where(SegmentRow.project_id == str(project_id))
                .order_by(SegmentRow.ordinal, SegmentRow.id)
            ).all()
            return [segment_document(row) for row in rows]

    def upsert_segment(
        self,
        project_id: str,
        request: SegmentUpsert,
    ) -> tuple[SegmentDocument, bool]:
        with self.transaction() as session:
            return self.upsert_segment_in_transaction(
                session,
                project_id,
                request,
            )

    def upsert_segment_in_transaction(
        self,
        session,
        project_id: str,
        request: SegmentUpsert,
    ) -> tuple[SegmentDocument, bool]:
        self.require_project_in_transaction(
            session,
            project_id,
            for_update=True,
        )
        project_id = str(project_id)
        if request.video_asset_id:
            asset_id = str(request.video_asset_id)
            if session.get(VideoAssetRow, asset_id) is None:
                raise ProjectResourceConflict(f"Video asset {asset_id} was not found")
            owner = self.video_asset_owner_in_transaction(session, asset_id)
            if owner != project_id:
                raise ProjectResourceConflict(
                    f"Video asset {asset_id} is not owned by project {project_id}"
                )
        if request.scene_id:
            scene_id = str(request.scene_id)
            if session.get(SceneRow, scene_id) is None:
                raise ProjectResourceConflict(f"Scene {scene_id} was not found")
            owner = self.scene_owner_in_transaction(session, scene_id)
            if owner != project_id:
                raise ProjectResourceConflict(
                    f"Scene {scene_id} is not owned by project {project_id}"
                )
        segment_id = request.id or stable_identifier(
            "segment",
            project_id,
            request.video_asset_id,
            request.source_segment_id,
            length=32,
        )
        row = session.scalar(
            select(SegmentRow)
            .where(SegmentRow.id == segment_id)
            .with_for_update()
        )
        created = row is None
        if row is None:
            row = SegmentRow(id=segment_id, project_id=project_id)
            session.add(row)
        elif row.project_id != project_id:
            raise ProjectResourceConflict(
                f"Segment {segment_id} belongs to project {row.project_id}"
            )
        values = {
            "video_asset_id": request.video_asset_id,
            "scene_id": request.scene_id,
            "source_segment_id": request.source_segment_id,
            "label": request.label,
            "start_seconds": request.start_seconds,
            "end_seconds": request.end_seconds,
            "ordinal": request.ordinal,
            "replay_group": request.replay_group,
            "replay_variant": request.replay_variant,
            "payload": dict(request.payload),
        }
        changed = False
        for field, value in values.items():
            if getattr(row, field) != value:
                setattr(row, field, value)
                changed = True
        if not created and changed:
            row.updated_at = _utcnow()
        session.flush()
        session.refresh(row)
        return segment_document(row), created


project_resources = ProjectResourceRepository()
