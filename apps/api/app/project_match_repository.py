from __future__ import annotations

"""Canonical persistence boundary for Matches and immutable snapshots."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Iterator

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from .database import SessionLocal
from .project_identifiers import stable_identifier
from .project_models import MatchRow, MatchSnapshotRow, ProjectRow
from .project_match_persistence_contract import (
    MatchDocument,
    MatchSnapshotCreate,
    MatchSnapshotDocument,
    MatchSnapshotSummary,
    MatchUpsert,
)


CANONICAL_MATCH_METADATA_KEYS = {
    "homeScore",
    "awayScore",
    "score",
    "homeTeamId",
    "awayTeamId",
    "venue",
    "round",
    "attendance",
}


class ProjectMatchNotFound(LookupError):
    pass


class ProjectMatchConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class CurrentMatchSnapshotSource:
    """Compact integration fields deliberately absent from project headers."""

    id: str
    project_id: str
    provider: str
    external_event_id: str | None


@dataclass(frozen=True)
class MatchPublication:
    match: MatchDocument
    snapshot: MatchSnapshotDocument
    match_created: bool
    snapshot_created: bool


def canonical_match_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        key: metadata[key]
        for key in CANONICAL_MATCH_METADATA_KEYS
        if key in metadata
    }


def canonical_payload_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"sha256:{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def match_document(row: MatchRow) -> MatchDocument:
    return MatchDocument(
        id=row.id,
        sport=row.sport,
        name=row.name,
        competition=row.competition,
        season=row.season,
        kickoff_at=row.kickoff_at,
        status=row.status,
        home_team_name=row.home_team_name,
        away_team_name=row.away_team_name,
        metadata=canonical_match_metadata(row.metadata_payload),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def snapshot_document(row: MatchSnapshotRow) -> MatchSnapshotDocument:
    return MatchSnapshotDocument(
        id=row.id,
        project_id=row.project_id,
        match_id=row.match_id,
        provider=row.provider,
        external_event_id=row.external_event_id,
        schema_version=row.schema_version,
        fetched_at=row.fetched_at,
        content_hash=row.content_hash,
        is_current=row.is_current,
        payload=dict(row.payload or {}),
        created_at=row.created_at,
    )


def snapshot_summary(row: Any) -> MatchSnapshotSummary:
    return MatchSnapshotSummary(
        id=row.id,
        project_id=row.project_id,
        match_id=row.match_id,
        schema_version=row.schema_version,
        fetched_at=row.fetched_at,
        content_hash=row.content_hash,
        is_current=row.is_current,
        created_at=row.created_at,
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ProjectMatchRepository:
    """Own Match rows, current selection, and immutable project snapshots."""

    def __init__(self, session_factory: Callable | None = None) -> None:
        self._session_factory = session_factory

    def _session(self):
        return (self._session_factory or SessionLocal)()

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Open one serialized transaction reusable by an application workflow."""

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
                raise ProjectMatchConflict(
                    "Match publication conflicted with another writer"
                ) from exc
            except Exception:
                session.rollback()
                raise

    @staticmethod
    def _require_project(session, project_id: str, *, for_update: bool = False):
        statement = select(ProjectRow).where(ProjectRow.id == project_id)
        if for_update:
            statement = statement.with_for_update()
        project = session.scalar(statement)
        if project is None:
            raise ProjectMatchNotFound(f"Project {project_id} was not found")
        return project

    def publish(
        self,
        project_id: str,
        match: MatchUpsert,
        snapshot: MatchSnapshotCreate,
    ) -> MatchPublication:
        """Atomically select a canonical Match and one immutable current snapshot."""

        with self.transaction() as session:
            return self.publish_in_transaction(
                session,
                project_id,
                match,
                snapshot,
            )

    def publish_in_transaction(
        self,
        session,
        project_id: str,
        match: MatchUpsert,
        snapshot: MatchSnapshotCreate,
    ) -> MatchPublication:
        """Publish through a caller-owned transaction without committing it."""

        project = self._require_project(session, project_id, for_update=True)
        match_row = session.scalar(
            select(MatchRow).where(MatchRow.id == match.id).with_for_update()
        )
        match_created = match_row is None
        if match_row is None:
            match_row = MatchRow(id=match.id, name=match.name)
            session.add(match_row)
        match_values = {
            "sport": match.sport,
            "name": match.name,
            "competition": match.competition,
            "season": match.season,
            "kickoff_at": match.kickoff_at,
            "status": match.status,
            "home_team_name": match.home_team_name,
            "away_team_name": match.away_team_name,
            "metadata_payload": canonical_match_metadata(match.metadata),
        }
        match_changed = False
        for field, value in match_values.items():
            if getattr(match_row, field) != value:
                setattr(match_row, field, value)
                match_changed = True
        if not match_created and match_changed:
            match_row.updated_at = _utcnow()
        session.flush()

        content_hash = canonical_payload_hash(snapshot.payload)
        snapshot_id = stable_identifier(
            "snapshot",
            project_id,
            content_hash,
            length=32,
        )
        snapshot_row = session.scalar(
            select(MatchSnapshotRow)
            .where(
                MatchSnapshotRow.project_id == project_id,
                MatchSnapshotRow.content_hash == content_hash,
            )
            .with_for_update()
        )
        snapshot_created = snapshot_row is None
        if snapshot_row is not None and snapshot_row.match_id != match.id:
            raise ProjectMatchConflict(
                f"Immutable snapshot {snapshot_row.id} belongs to another Match"
            )

        already_current = bool(
            snapshot_row is not None
            and snapshot_row.is_current
            and project.current_match_snapshot_id == snapshot_row.id
            and project.match_id == match.id
        )
        if not already_current:
            session.query(MatchSnapshotRow).filter(
                MatchSnapshotRow.project_id == project_id,
                MatchSnapshotRow.is_current.is_(True),
            ).update(
                {MatchSnapshotRow.is_current: False},
                synchronize_session=False,
            )
            if snapshot_row is None:
                snapshot_row = MatchSnapshotRow(
                    id=snapshot_id,
                    project_id=project_id,
                    match_id=match.id,
                    provider=snapshot.provider,
                    external_event_id=snapshot.external_event_id,
                    schema_version=snapshot.schema_version,
                    fetched_at=snapshot.fetched_at,
                    content_hash=content_hash,
                    is_current=True,
                    payload=dict(snapshot.payload),
                )
                session.add(snapshot_row)
            else:
                snapshot_row.is_current = True
            project.match_id = match.id
            project.current_match_snapshot_id = snapshot_row.id
            project.revision += 1
            project.updated_at = _utcnow()

        session.flush()
        session.refresh(match_row)
        session.refresh(snapshot_row)
        return MatchPublication(
            match=match_document(match_row),
            snapshot=snapshot_document(snapshot_row),
            match_created=match_created,
            snapshot_created=snapshot_created,
        )

    def get_match(self, match_id: str) -> MatchDocument | None:
        with self._session() as session:
            row = session.get(MatchRow, match_id)
            return match_document(row) if row is not None else None

    def current_payload(self, project_id: str) -> dict[str, Any] | None:
        snapshot = self.current_snapshot(project_id)
        return dict(snapshot.payload) if snapshot is not None else None

    def current_snapshot(self, project_id: str) -> MatchSnapshotDocument | None:
        with self._session() as session:
            project = self._require_project(session, project_id)
            if not project.current_match_snapshot_id:
                return None
            row = session.get(MatchSnapshotRow, project.current_match_snapshot_id)
            if row is None or row.project_id != project_id or not row.is_current:
                raise ProjectMatchConflict(
                    f"Project {project_id} points to an invalid match snapshot"
                )
            return snapshot_document(row)

    def current_summary(self, project_id: str) -> MatchSnapshotSummary | None:
        with self._session() as session:
            project = self._require_project(session, project_id)
            if not project.current_match_snapshot_id:
                return None
            row = session.execute(
                select(
                    MatchSnapshotRow.id,
                    MatchSnapshotRow.project_id,
                    MatchSnapshotRow.match_id,
                    MatchSnapshotRow.schema_version,
                    MatchSnapshotRow.fetched_at,
                    MatchSnapshotRow.content_hash,
                    MatchSnapshotRow.is_current,
                    MatchSnapshotRow.created_at,
                ).where(MatchSnapshotRow.id == project.current_match_snapshot_id)
            ).one_or_none()
            if row is None or row.project_id != project_id or not row.is_current:
                raise ProjectMatchConflict(
                    f"Project {project_id} points to an invalid match snapshot"
                )
            return snapshot_summary(row)

    def current_source(self, project_id: str) -> CurrentMatchSnapshotSource | None:
        with self._session() as session:
            project = self._require_project(session, project_id)
            if not project.current_match_snapshot_id:
                return None
            row = session.execute(
                select(
                    MatchSnapshotRow.id,
                    MatchSnapshotRow.project_id,
                    MatchSnapshotRow.provider,
                    MatchSnapshotRow.external_event_id,
                    MatchSnapshotRow.is_current,
                ).where(MatchSnapshotRow.id == project.current_match_snapshot_id)
            ).one_or_none()
            if row is None or row.project_id != project_id or not row.is_current:
                raise ProjectMatchConflict(
                    f"Project {project_id} points to an invalid match snapshot"
                )
            return CurrentMatchSnapshotSource(
                id=str(row.id),
                project_id=str(row.project_id),
                provider=str(row.provider),
                external_event_id=(
                    str(row.external_event_id)
                    if row.external_event_id is not None
                    else None
                ),
            )

    def get_snapshot(
        self,
        project_id: str,
        snapshot_id: str,
    ) -> MatchSnapshotDocument | None:
        with self._session() as session:
            row = session.get(MatchSnapshotRow, str(snapshot_id))
            if row is None or row.project_id != project_id:
                return None
            return snapshot_document(row)


project_matches = ProjectMatchRepository()
