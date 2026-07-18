from __future__ import annotations

"""Compact read queries for reconstruction dependencies."""

from typing import Callable

from sqlalchemy import select

from .database import ReconstructionJobRow, SessionLocal


class ReconstructionJobQueries:
    def __init__(self, session_factory: Callable | None = None) -> None:
        self._session_factory = session_factory

    def _session(self):
        return (self._session_factory or SessionLocal)()

    def statuses(self, scene_ids: list[str]) -> dict[str, str]:
        """Read dependency status without hydrating any Scene document."""

        normalized = sorted({str(value) for value in scene_ids if str(value)})
        if not normalized:
            return {}
        with self._session() as session:
            rows = session.execute(
                select(
                    ReconstructionJobRow.scene_id,
                    ReconstructionJobRow.status,
                ).where(ReconstructionJobRow.scene_id.in_(normalized))
            ).all()
            return {str(scene_id): str(status) for scene_id, status in rows}


reconstruction_jobs = ReconstructionJobQueries()
