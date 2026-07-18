from __future__ import annotations

"""Persistence boundary for provider/upstream resource identifiers."""

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Callable, Iterable, Iterator

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from .database import SessionLocal
from .project_identifiers import stable_identifier
from .project_models import ExternalReferenceRow
from .project_match_persistence_contract import ExternalReferenceCreate, ExternalReferenceDocument


class ExternalReferenceConflict(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def external_reference_document(
    row: ExternalReferenceRow,
) -> ExternalReferenceDocument:
    return ExternalReferenceDocument(
        id=row.id,
        resource_type=row.resource_type,
        resource_id=row.resource_id,
        provider=row.provider,
        external_type=row.external_type,
        external_id=row.external_id,
        payload=dict(row.payload or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class ExternalReferenceRepository:
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
                raise ExternalReferenceConflict(
                    "External reference changed concurrently"
                ) from exc
            except Exception:
                session.rollback()
                raise

    def upsert(
        self,
        request: ExternalReferenceCreate,
    ) -> tuple[ExternalReferenceDocument, bool]:
        with self.transaction() as session:
            return self.upsert_in_transaction(session, request)

    def upsert_many(
        self,
        requests: Iterable[ExternalReferenceCreate],
    ) -> list[ExternalReferenceDocument]:
        with self.transaction() as session:
            return self.upsert_many_in_transaction(session, requests)

    def upsert_many_in_transaction(
        self,
        session,
        requests: Iterable[ExternalReferenceCreate],
    ) -> list[ExternalReferenceDocument]:
        return [
            self.upsert_in_transaction(session, request)[0]
            for request in requests
        ]

    def upsert_in_transaction(
        self,
        session,
        request: ExternalReferenceCreate,
    ) -> tuple[ExternalReferenceDocument, bool]:
        row = session.scalar(
            select(ExternalReferenceRow)
            .where(
                ExternalReferenceRow.resource_type == request.resource_type,
                ExternalReferenceRow.resource_id == request.resource_id,
                ExternalReferenceRow.provider == request.provider,
                ExternalReferenceRow.external_type == request.external_type,
                ExternalReferenceRow.external_id == request.external_id,
            )
            .with_for_update()
        )
        created = row is None
        if row is None:
            row = ExternalReferenceRow(
                id=stable_identifier(
                    "xref",
                    request.resource_type,
                    request.resource_id,
                    request.provider,
                    request.external_type,
                    request.external_id,
                    length=32,
                ),
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                provider=request.provider,
                external_type=request.external_type,
                external_id=request.external_id,
                payload=dict(request.payload),
            )
            session.add(row)
        else:
            payload = dict(request.payload)
            if row.payload != payload:
                row.payload = payload
                row.updated_at = _utcnow()
        session.flush()
        session.refresh(row)
        return external_reference_document(row), created

    def find(
        self,
        provider: str,
        external_type: str,
        external_id: str,
    ) -> list[ExternalReferenceDocument]:
        with self._session() as session:
            rows = session.scalars(
                select(ExternalReferenceRow)
                .where(
                    ExternalReferenceRow.provider == provider,
                    ExternalReferenceRow.external_type == external_type,
                    ExternalReferenceRow.external_id == external_id,
                )
                .order_by(
                    ExternalReferenceRow.resource_type,
                    ExternalReferenceRow.resource_id,
                )
            ).all()
            return [external_reference_document(row) for row in rows]

    def for_resource(
        self,
        resource_type: str,
        resource_id: str,
    ) -> list[ExternalReferenceDocument]:
        with self._session() as session:
            rows = session.scalars(
                select(ExternalReferenceRow)
                .where(
                    ExternalReferenceRow.resource_type == resource_type,
                    ExternalReferenceRow.resource_id == resource_id,
                )
                .order_by(
                    ExternalReferenceRow.provider,
                    ExternalReferenceRow.external_type,
                    ExternalReferenceRow.external_id,
                )
            ).all()
            return [external_reference_document(row) for row in rows]

    def for_resource_ids(
        self,
        resource_ids: Iterable[str],
    ) -> list[ExternalReferenceDocument]:
        normalized = sorted({str(value) for value in resource_ids if value})
        if not normalized:
            return []
        with self._session() as session:
            rows = session.scalars(
                select(ExternalReferenceRow)
                .where(ExternalReferenceRow.resource_id.in_(normalized))
                .order_by(
                    ExternalReferenceRow.resource_type,
                    ExternalReferenceRow.provider,
                    ExternalReferenceRow.external_type,
                    ExternalReferenceRow.external_id,
                )
            ).all()
            return [external_reference_document(row) for row in rows]


external_references = ExternalReferenceRepository()
