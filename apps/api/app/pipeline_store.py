from __future__ import annotations

"""Durable pipeline job and lease repository.

This repository owns only the compact scheduler lifecycle. Video and
scene-specific enqueue/publication transactions live in their dedicated
services; neither resource payload is consulted for discovery or claims.
"""

from datetime import UTC, datetime
import time
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import or_, select

from .analysis_run_telemetry import (
    compact_analysis_progress,
    mark_analysis_running,
)
from .config import get_settings
from .database import (
    PipelineJobLeaseRow,
    PipelineJobRow,
    SessionLocal,
)
from .pipeline_domain import (
    PIPELINE_ACTIVE_STATUSES,
    PipelineJob,
    begin_pipeline_write,
    lock_owned_pipeline_job,
    pipeline_job_document,
    validate_compact,
)
from .project_models import AnalysisRunRow
from . import project_models as _project_models  # noqa: F401 - register FKs


class PipelineStore:
    """Repository for compact jobs, claims and scheduler transitions."""

    def __init__(
        self,
        session_factory: Callable | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock

    def _session(self):
        return (self._session_factory or SessionLocal)()

    def get(self, job_id: str) -> PipelineJob | None:
        with self._session() as session:
            row = session.get(PipelineJobRow, str(job_id))
            if row is None:
                return None
            lease = session.get(PipelineJobLeaseRow, str(job_id))
            return pipeline_job_document(row, lease)

    def list_recoverable(
        self,
        *,
        now: float | None = None,
        limit: int = 100,
    ) -> list[str]:
        """Discover runnable ids from compact control columns only."""

        current = float(self._clock() if now is None else now)
        with self._session() as session:
            rows = session.scalars(
                select(PipelineJobRow.id)
                .outerjoin(
                    PipelineJobLeaseRow,
                    PipelineJobLeaseRow.job_id == PipelineJobRow.id,
                )
                .where(PipelineJobRow.status.in_(sorted(PIPELINE_ACTIVE_STATUSES)))
                .where(
                    or_(
                        PipelineJobRow.status == "processing",
                        PipelineJobRow.available_at <= current,
                    )
                )
                .where(
                    or_(
                        PipelineJobLeaseRow.job_id.is_(None),
                        PipelineJobLeaseRow.expires_at <= current,
                    )
                )
                .order_by(
                    PipelineJobRow.available_at,
                    PipelineJobRow.requested_at,
                    PipelineJobRow.id,
                )
                .limit(max(1, int(limit)))
            ).all()
            return [str(value) for value in rows]

    def claim(
        self,
        job_id: str,
        owner_id: str,
        *,
        now: float | None = None,
        ttl_seconds: float | None = None,
    ) -> PipelineJob | None:
        current = float(self._clock() if now is None else now)
        ttl = max(
            1.0,
            float(
                get_settings().pipeline_lease_ttl_seconds
                if ttl_seconds is None
                else ttl_seconds
            ),
        )
        session = self._session()
        try:
            begin_pipeline_write(session)
            row = session.scalar(
                select(PipelineJobRow)
                .where(PipelineJobRow.id == str(job_id))
                .with_for_update()
            )
            if (
                row is None
                or row.status not in PIPELINE_ACTIVE_STATUSES
                or (row.status != "processing" and row.available_at > current)
            ):
                session.rollback()
                return None
            lease = session.scalar(
                select(PipelineJobLeaseRow)
                .where(PipelineJobLeaseRow.job_id == str(job_id))
                .with_for_update()
            )
            if lease is not None and float(lease.expires_at) > current:
                session.rollback()
                return None
            token = f"lease-{uuid4().hex}"
            if lease is None:
                lease = PipelineJobLeaseRow(
                    job_id=str(job_id),
                    owner_id=str(owner_id),
                    token=token,
                    acquired_at=current,
                    heartbeat_at=current,
                    expires_at=current + ttl,
                )
                session.add(lease)
            else:
                lease.owner_id = str(owner_id)
                lease.token = token
                lease.acquired_at = current
                lease.heartbeat_at = current
                lease.expires_at = current + ttl
            row.status = "processing"
            row.attempts = int(row.attempts or 0) + 1
            row.updated_at = current
            mark_analysis_running(session, row.id, current)
            session.commit()
            return pipeline_job_document(row, lease)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def is_claim_current(
        self,
        job_id: str,
        token: str,
        *,
        now: float | None = None,
    ) -> bool:
        current = float(self._clock() if now is None else now)
        with self._session() as session:
            value = session.scalar(
                select(PipelineJobLeaseRow.job_id)
                .join(
                    PipelineJobRow,
                    PipelineJobRow.id == PipelineJobLeaseRow.job_id,
                )
                .where(
                    PipelineJobRow.id == str(job_id),
                    PipelineJobRow.status == "processing",
                    PipelineJobLeaseRow.token == str(token),
                    PipelineJobLeaseRow.expires_at > current,
                )
            )
            return value is not None

    def heartbeat(
        self,
        job_id: str,
        token: str,
        *,
        now: float | None = None,
        ttl_seconds: float | None = None,
    ) -> bool:
        current = float(self._clock() if now is None else now)
        ttl = max(
            1.0,
            float(
                get_settings().pipeline_lease_ttl_seconds
                if ttl_seconds is None
                else ttl_seconds
            ),
        )
        session = self._session()
        try:
            begin_pipeline_write(session)
            row, lease = lock_owned_pipeline_job(session, job_id, token)
            if row is None or lease is None or float(lease.expires_at) <= current:
                session.rollback()
                return False
            lease.heartbeat_at = current
            lease.expires_at = current + ttl
            row.updated_at = current
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def update_progress(
        self,
        job_id: str,
        token: str,
        progress: dict[str, Any],
    ) -> bool:
        """Guard one compact telemetry update with the authoritative lease."""

        compact = compact_analysis_progress(progress)
        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            row, lease = lock_owned_pipeline_job(session, job_id, token)
            if row is None or lease is None or float(lease.expires_at) <= now:
                session.rollback()
                return False
            run = session.get(AnalysisRunRow, str(job_id))
            if run is not None and run.status in {"queued", "running"}:
                timestamp = datetime.fromtimestamp(now, UTC)
                run.status = "running"
                run.started_at = run.started_at or timestamp
                run.progress = compact
                run.updated_at = timestamp
            row.updated_at = now
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def yield_waiting(
        self,
        job_id: str,
        token: str,
        *,
        state: dict[str, Any],
        delay_seconds: float,
        progress: dict[str, Any] | None = None,
    ) -> bool:
        compact = validate_compact(
            state,
            label="Pipeline job state",
            limit=64 * 1024,
        )
        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            row, lease = lock_owned_pipeline_job(session, job_id, token)
            if row is None or lease is None or float(lease.expires_at) <= now:
                session.rollback()
                return False
            row.state = compact
            row.status = "waiting"
            row.available_at = now + max(0.0, float(delay_seconds))
            row.updated_at = now
            run = session.get(AnalysisRunRow, str(job_id))
            if run is not None and run.status in {"queued", "running"}:
                timestamp = datetime.fromtimestamp(now, UTC)
                run.status = "running"
                run.started_at = run.started_at or timestamp
                if progress is not None:
                    run.progress = compact_analysis_progress(progress)
                run.updated_at = timestamp
            session.delete(lease)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def abandon_claim(self, job_id: str, token: str) -> bool:
        """Release a claim after a local spawn failure without losing work."""

        now = float(self._clock())
        session = self._session()
        try:
            begin_pipeline_write(session)
            row, lease = lock_owned_pipeline_job(session, job_id, token)
            if row is None or lease is None:
                session.rollback()
                return False
            row.status = "queued"
            row.available_at = now
            row.updated_at = now
            session.delete(lease)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

pipeline_store = PipelineStore()
