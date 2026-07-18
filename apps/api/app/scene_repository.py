from __future__ import annotations

"""Revisioned SceneDocument reads and compare-and-swap writes."""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Callable

from sqlalchemy import select

from .database_transaction import begin_write_transaction
from .database import (
    ReconstructionJobRow,
    ReconstructionLeaseRow,
    SceneRow,
    SessionLocal,
)
from .scene_document import (
    SceneRevisionConflict,
    next_scene_payload,
    scene_revision,
)
from .scene_index_projection import sync_scene_index


class SceneRepository:
    def __init__(
        self,
        session_factory: Callable | None = None,
        *,
        clock: Callable[[], datetime | float] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock

    def _session(self):
        return (self._session_factory or SessionLocal)()

    def _now_timestamp(self) -> float:
        value = self._clock() if self._clock is not None else datetime.now(UTC)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return float(value.timestamp())
        return float(value)

    def list(self) -> list[dict]:
        with self._session() as session:
            rows = session.execute(
                select(
                    SceneRow.id,
                    SceneRow.title,
                    SceneRow.duration,
                    SceneRow.kind,
                    SceneRow.parent_scene_id,
                    SceneRow.updated_at,
                ).order_by(SceneRow.updated_at.desc())
            ).all()
        return [self._summary(row) for row in rows]

    def list_by_ids(self, scene_ids: list[str]) -> list[dict]:
        normalized = sorted({str(value) for value in scene_ids if str(value)})
        if not normalized:
            return []
        with self._session() as session:
            rows = session.execute(
                select(
                    SceneRow.id,
                    SceneRow.title,
                    SceneRow.duration,
                    SceneRow.kind,
                    SceneRow.parent_scene_id,
                    SceneRow.updated_at,
                ).where(SceneRow.id.in_(normalized))
            ).all()
        return [self._summary(row) for row in rows]

    @staticmethod
    def _summary(row) -> dict:
        return {
            "id": row.id,
            "title": row.title,
            "duration": float(row.duration),
            "kind": row.kind,
            "parent_scene_id": row.parent_scene_id,
            "updated_at": (
                row.updated_at.isoformat()
                if isinstance(row.updated_at, datetime)
                else None
            ),
        }

    def get(self, scene_id: str) -> dict | None:
        with self._session() as session:
            row = session.get(SceneRow, str(scene_id))
            return deepcopy(row.payload) if row is not None else None

    def put_many(self, documents: list[dict]) -> list[dict]:
        if not documents:
            return []
        scene_ids = [str(document.get("id") or "") for document in documents]
        if any(not scene_id for scene_id in scene_ids) or len(set(scene_ids)) != len(
            scene_ids
        ):
            raise ValueError("Atomic Scene writes require unique non-empty ids")

        session = self._session()
        persisted_by_id: dict[str, dict] = {}
        try:
            begin_write_transaction(session)
            session.scalars(
                select(ReconstructionJobRow)
                .where(ReconstructionJobRow.scene_id.in_(scene_ids))
                .order_by(ReconstructionJobRow.scene_id)
                .with_for_update()
            ).all()
            leases = session.scalars(
                select(ReconstructionLeaseRow)
                .where(ReconstructionLeaseRow.scene_id.in_(scene_ids))
                .order_by(ReconstructionLeaseRow.scene_id)
                .with_for_update()
            ).all()
            leases_by_id = {str(lease.scene_id): lease for lease in leases}
            rows = session.scalars(
                select(SceneRow)
                .where(SceneRow.id.in_(scene_ids))
                .order_by(SceneRow.id)
                .with_for_update()
            ).all()
            rows_by_id = {str(row.id): row for row in rows}
            missing = [scene_id for scene_id in scene_ids if scene_id not in rows_by_id]
            if missing:
                raise SceneRevisionConflict(
                    f"Scenes disappeared during atomic write: {', '.join(missing)}"
                )

            now = self._now_timestamp()
            for document, scene_id in zip(documents, scene_ids):
                row = rows_by_id[scene_id]
                lease = leases_by_id.get(scene_id)
                if lease is not None and float(lease.expires_at) > now:
                    raise SceneRevisionConflict(
                        f"Scene {scene_id} has an active reconstruction lease"
                    )
                current_revision = scene_revision(row.payload)
                supplied_revision = scene_revision(document)
                if supplied_revision != current_revision:
                    raise SceneRevisionConflict(
                        f"Scene {scene_id} changed from revision "
                        f"{supplied_revision} to {current_revision}"
                    )

            for document, scene_id in zip(documents, scene_ids):
                row = rows_by_id[scene_id]
                next_revision = scene_revision(row.payload) + 1
                persisted = next_scene_payload(document, next_revision)
                row.title = document["title"]
                row.payload = persisted
                sync_scene_index(row, persisted)
                persisted_by_id[scene_id] = persisted
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return [deepcopy(persisted_by_id[scene_id]) for scene_id in scene_ids]

    def put(self, document: dict) -> dict:
        session = self._session()
        try:
            begin_write_transaction(session)
            session.scalar(
                select(ReconstructionJobRow)
                .where(ReconstructionJobRow.scene_id == document["id"])
                .with_for_update()
            )
            lease = session.scalar(
                select(ReconstructionLeaseRow)
                .where(ReconstructionLeaseRow.scene_id == document["id"])
                .with_for_update()
            )
            row = session.scalar(
                select(SceneRow)
                .where(SceneRow.id == document["id"])
                .with_for_update()
            )
            if row is None:
                next_revision = 1
                persisted = next_scene_payload(document, next_revision)
                row = SceneRow(
                    id=document["id"],
                    title=document["title"],
                    payload=persisted,
                )
                sync_scene_index(row, persisted)
                session.add(row)
            else:
                if lease is not None and float(lease.expires_at) > self._now_timestamp():
                    raise SceneRevisionConflict(
                        f"Scene {document['id']} has an active reconstruction lease"
                    )
                current_revision = scene_revision(row.payload)
                supplied_revision = scene_revision(document)
                if supplied_revision != current_revision:
                    raise SceneRevisionConflict(
                        f"Scene {document['id']} changed from revision "
                        f"{supplied_revision} to {current_revision}"
                    )
                next_revision = current_revision + 1
                persisted = next_scene_payload(document, next_revision)
                row.title = document["title"]
                row.payload = persisted
                sync_scene_index(row, persisted)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return deepcopy(persisted)

    def find_segment_scene(
        self,
        parent_scene_id: str,
        segment_id: str,
    ) -> dict | None:
        with self._session() as session:
            scene_id = session.scalar(
                select(SceneRow.id)
                .where(
                    SceneRow.parent_scene_id == str(parent_scene_id),
                    SceneRow.selected_segment_id == str(segment_id),
                    SceneRow.kind == "segment",
                )
                .order_by(SceneRow.updated_at.desc())
                .limit(1)
            )
        return self.get(str(scene_id)) if scene_id is not None else None


scenes = SceneRepository()
