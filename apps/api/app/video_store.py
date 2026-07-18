from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select

from .database import SessionLocal, VideoAssetRow


def _serialize(row: VideoAssetRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "filename": row.filename,
        "original_name": row.original_name,
        "content_type": row.content_type,
        "status": row.status,
        "stage": row.stage,
        "progress": row.progress,
        "duration": row.duration,
        "width": row.width,
        "height": row.height,
        "fps": row.fps,
        "frame_count": row.frame_count,
        "generation_key": row.generation_key,
        "scene_id": row.scene_id,
        # HTTP URLs are project-scoped representations and are attached by
        # project routes. The persistence store does not know request scope.
        "media_url": None,
        "poster_url": None,
        "error": row.error,
        "created_at": row.created_at.isoformat() if isinstance(row.created_at, datetime) else None,
    }


class VideoStore:
    def __init__(self, session_factory=None) -> None:
        self._session_factory = session_factory

    def _session(self):
        return (self._session_factory or SessionLocal)()

    def create(self, **values: Any) -> dict:
        with self._session() as session:
            row = VideoAssetRow(**values)
            session.add(row)
            session.commit()
        return _serialize(row)

    def get(self, asset_id: str) -> dict | None:
        with self._session() as session:
            row = session.get(VideoAssetRow, asset_id)
            return _serialize(row) if row else None

    def list(self) -> list[dict]:
        with self._session() as session:
            rows = session.scalars(select(VideoAssetRow).order_by(VideoAssetRow.created_at.desc())).all()
            return [_serialize(row) for row in rows]

    def list_by_ids(self, asset_ids: list[str]) -> list[dict]:
        normalized = sorted({str(value) for value in asset_ids if str(value)})
        if not normalized:
            return []
        with self._session() as session:
            rows = session.scalars(
                select(VideoAssetRow).where(VideoAssetRow.id.in_(normalized))
            ).all()
            return [_serialize(row) for row in rows]

    def update(self, asset_id: str, **values: Any) -> dict | None:
        with self._session() as session:
            row = session.get(VideoAssetRow, asset_id)
            if row is None:
                return None
            for key, value in values.items():
                setattr(row, key, value)
            session.commit()
        return _serialize(row)


video_store = VideoStore()
